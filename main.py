"""main.py - FastAPI service for the Enterprise AI Platform.

Endpoints
  POST /v1/query               user path (Bearer JWT -> OBO -> graph)
  POST /v1/jobs/query          headless path (service principal + Airflow run id)
  GET  /v1/review/{thread_id}  what a reviewer is asked to approve
  POST /v1/review/{thread_id}  human-in-the-loop decision on a paused run
  GET  /v1/tools               MCP tool definitions
  GET  /healthz                liveness and catalog size
  POST /v1/dev/token           demo sign-in (simulated identity provider only)

Configuration (environment):
  APP_ENV              dev (default) | production
  IDENTITY_PROVIDER    simulated (default) | entra        see auth.py, entra.py
  CHECKPOINT_BACKEND   sqlite (default) | postgres | memory   see persistence.py
  AGENT_BACKEND, WAREHOUSE_BACKEND, RETRIEVER_BACKEND, CATALOG_PATH    see runtime.py
In production every development stand-in is refused at startup.

Run:  uvicorn main:app --reload
"""

import logging
import os
import threading
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from auth import (
    ACTIVE_IDP,
    API_AUDIENCE,
    APP_ROLE_TO_SNOWFLAKE_ROLE,
    IDP,
    AuthContext,
    AuthError,
    authenticate_service,
    authenticate_user,
)
from gates import MCP_TOOL_REGISTRY
from graph import build_graph
from bedrock_models import METER, request_budget
from telemetry import default_provider, export_run
from persistence import forget_thread, make_checkpointer
from runtime import RuntimeConfig, build_agent_deps, catalog_problems
from schemas import AuditEvent, Citation, GateStatus, new_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("platform.api")

APP_ENV = os.environ.get("APP_ENV", "dev").strip().lower()
# Fail closed: any environment name that is not explicitly a development one gets production rules.
IS_PRODUCTION = APP_ENV not in {"dev", "development", "local", "test"}
REVIEWER_ROLE = "Lakehouse.Admin"

CHECKPOINT_BACKEND = os.environ.get("CHECKPOINT_BACKEND", "sqlite").lower()
RUNTIME = RuntimeConfig.from_env()
DEV_TOKENS_ENABLED = not IS_PRODUCTION and ACTIVE_IDP is IDP

if IS_PRODUCTION:
    # Fail closed: a production process must not start on development stand-ins.
    problems = RUNTIME.production_problems()
    if ACTIVE_IDP is IDP:
        problems.append("IDENTITY_PROVIDER=entra (the simulated provider signs its own tokens)")
    if CHECKPOINT_BACKEND == "memory":
        problems.append("a durable CHECKPOINT_BACKEND (sqlite or postgres)")
    if problems:
        raise RuntimeError(f"APP_ENV={APP_ENV} is treated as production and requires: " + "; ".join(problems))

DEPS = build_agent_deps(RUNTIME)
if IS_PRODUCTION and (problems := catalog_problems(RUNTIME, DEPS.catalog)):
    raise RuntimeError(f"APP_ENV={APP_ENV} is treated as production and requires: " + "; ".join(problems))
CHECKPOINTER = make_checkpointer(CHECKPOINT_BACKEND)
GRAPH_CATALOG = DEPS.catalog
GRAPH = build_graph(DEPS, checkpointer=CHECKPOINTER)
logger.info(
    "startup env=%s identity_provider=%s checkpoint_backend=%s agents=%s warehouse=%s retriever=%s catalog=%s",
    APP_ENV, ACTIVE_IDP.name, CHECKPOINT_BACKEND, RUNTIME.agent_backend, RUNTIME.warehouse_backend,
    RUNTIME.retriever_backend, RUNTIME.catalog_source,
)

app = FastAPI(
    title="Enterprise AI Platform",
    version="0.1.0",
    description="Multi-agent workflow over a governed Snowflake lakehouse with fail-closed gates.",
)
bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryRequest(_Request):
    query: str = Field(min_length=1, max_length=4_000)


class JobQueryRequest(_Request):
    client_id: str = Field(min_length=1)
    client_secret: str = Field(min_length=1)
    airflow_dag_id: str = Field(min_length=1)
    airflow_dag_run_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=4_000)


class ReviewRequest(_Request):
    approved: bool
    comment: str | None = Field(default=None, max_length=2_000)


class DevTokenRequest(_Request):
    upn: str = Field(pattern=r"^[^@\s]+@[^@\s]+$")
    role: str = Field(description="One of the Entra app roles, e.g. Lakehouse.Operator")


class DevTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    audience: str = API_AUDIENCE


class HitlInfo(BaseModel):
    ticket: str
    reason: str | None


class QueryResponse(BaseModel):
    thread_id: str
    trace_id: str
    status: GateStatus
    route: str | None
    answer: str | None
    citations: list[Citation]
    refusal_reason: str | None
    hitl: HitlInfo | None = None
    audit: list[AuditEvent] | None = None
    usage: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unauthorized(err: AuthError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail=str(err),
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> AuthContext:
    """Validate the caller's JWT and run the OBO exchange."""
    if creds is None:
        raise _unauthorized(AuthError("Missing bearer token."))
    try:
        return authenticate_user(creds.credentials)
    except AuthError as err:
        raise _unauthorized(err) from err


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _to_response(thread_id: str, out: dict[str, Any], include_audit: bool, usage: dict[str, Any] | None = None) -> QueryResponse:
    interrupts = out.get("__interrupt__") or []
    hitl = None
    status_value = GateStatus(out.get("gate_status", GateStatus.REFUSED))
    answer = out.get("final_response")
    if interrupts:  # run is paused waiting for a reviewer; nothing is released
        payload = interrupts[0].value
        hitl = HitlInfo(ticket=payload["ticket"], reason=payload.get("reason"))
        status_value = GateStatus.HITL_REVIEW
        answer = f"This answer is withheld pending human review (ticket {hitl.ticket})."
    released = status_value is GateStatus.PASSED
    route = out.get("route")
    return QueryResponse(
        thread_id=thread_id,
        trace_id=out.get("trace_id", thread_id),
        status=status_value,
        route=route.value if route else None,
        answer=answer,
        citations=out.get("citations", []) if released else [],
        refusal_reason=None if released else out.get("refusal_reason"),
        hitl=hitl,
        audit=out.get("audit_telemetry") if include_audit else None,
        usage=usage if usage and usage["model_calls"] else None,
    )


def _run(query: str, ctx: AuthContext, include_audit: bool) -> QueryResponse:
    state = new_state(query)
    state["auth_context"] = ctx.model_dump(mode="json")
    thread_id = state["trace_id"]
    paused = False
    with request_budget() as budget:  # token ceiling for every model call this request makes
        try:
            out = GRAPH.invoke(state, config=_config(thread_id))
            paused = bool(out.get("__interrupt__"))
        finally:
            if not paused:  # keep state on disk only while a reviewer is pending, also when the run crashed
                forget_thread(CHECKPOINTER, thread_id)
    logger.info(
        "query.done thread_id=%s principal=%s status=%s route=%s",
        thread_id, ctx.principal, out.get("gate_status"), out.get("route"),
    )
    export_run(out, default_provider(), usage=budget.summary())
    return _to_response(thread_id, out, include_audit, budget.summary())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok", "env": APP_ENV, "catalog_tables": len(GRAPH_CATALOG),
        "identity_provider": ACTIVE_IDP.name, "checkpoint_backend": CHECKPOINT_BACKEND,
        "agent_backend": RUNTIME.agent_backend, "warehouse_backend": RUNTIME.warehouse_backend,
        "retriever_backend": RUNTIME.retriever_backend, "catalog": RUNTIME.catalog_source,
    }


@app.get("/v1/usage")
def usage_report(caller: AuthContext = Depends(current_user)) -> dict[str, Any]:
    """Token and cost totals by role and model since this process started (admins only)."""
    if caller.snowflake_role != APP_ROLE_TO_SNOWFLAKE_ROLE[REVIEWER_ROLE]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Usage reporting requires the {REVIEWER_ROLE} role.")
    return {"usage": METER.report()}


@app.get("/v1/tools")
def list_tools() -> dict[str, Any]:
    return {"tools": [t.model_dump(by_alias=True) for t in MCP_TOOL_REGISTRY.values()]}


@app.post("/v1/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(
    body: QueryRequest,
    include_audit: bool = Query(default=False),
    ctx: AuthContext = Depends(current_user),
) -> QueryResponse:
    """User path. A refused request is a normal 200 response with status REFUSED."""
    return _run(body.query, ctx, include_audit)


@app.post("/v1/jobs/query", response_model=QueryResponse, response_model_exclude_none=True)
def job_query(body: JobQueryRequest, include_audit: bool = Query(default=False)) -> QueryResponse:
    """Headless path for scheduled DAGs. Never pauses for human review."""
    try:
        ctx = authenticate_service(
            body.client_id, body.client_secret, body.airflow_dag_id, body.airflow_dag_run_id
        )
    except AuthError as err:
        raise _unauthorized(err) from err
    return _run(body.query, ctx, include_audit)


class PendingReview(BaseModel):
    thread_id: str
    ticket: str
    requested_by: str | None
    query: str | None
    draft_response: str | None
    reason: str | None
    citations: list[Citation]


def _reviewer(ctx: AuthContext = Depends(current_user)) -> AuthContext:
    if ctx.snowflake_role != APP_ROLE_TO_SNOWFLAKE_ROLE[REVIEWER_ROLE]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Reviewing requires the {REVIEWER_ROLE} role.")
    return ctx


def _pending(thread_id: str) -> Any:
    snapshot = GRAPH.get_state(_config(thread_id))
    if not snapshot.next or not any(task.interrupts for task in snapshot.tasks):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No run is awaiting review for this thread.")
    return snapshot


@app.get("/v1/review/{thread_id}", response_model=PendingReview)
def get_review(thread_id: str, reviewer: AuthContext = Depends(_reviewer)) -> PendingReview:
    """What the reviewer is being asked to approve: the draft, its citations and why it was held."""
    snapshot = _pending(thread_id)
    payload = next(i.value for task in snapshot.tasks for i in task.interrupts)
    values = snapshot.values
    return PendingReview(
        thread_id=thread_id, ticket=payload["ticket"], requested_by=values.get("principal"),
        query=values.get("query"), draft_response=values.get("draft_response"),
        reason=payload.get("reason"), citations=values.get("citations", []),
    )


_REVIEW_LOCKS: dict[str, threading.Lock] = {}
_REVIEW_LOCKS_GUARD = threading.Lock()


@app.post("/v1/review/{thread_id}", response_model=QueryResponse, response_model_exclude_none=True)
def review(
    thread_id: str,
    body: ReviewRequest,
    include_audit: bool = Query(default=False),
    reviewer: AuthContext = Depends(_reviewer),
) -> QueryResponse:
    """Approve or reject a paused run. The reviewer's identity comes from
    their token, must hold the admin role, and must not be the requester.
    Decisions on one run are serialised in this process, so the first one
    wins; run a single API replica or add a database lock before scaling out."""
    with _REVIEW_LOCKS_GUARD:
        lock = _REVIEW_LOCKS.setdefault(thread_id, threading.Lock())
    with lock:
        snapshot = _pending(thread_id)  # checked under the lock: a decided run is gone
        if snapshot.values.get("principal") == reviewer.principal:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewers cannot approve their own requests.")
        paused = False
        with request_budget() as budget:  # a resumed run can call models too
            try:
                out = GRAPH.invoke(
                    Command(resume={"approved": body.approved, "reviewer": reviewer.principal, "comment": body.comment}),
                    config=_config(thread_id),
                )
                paused = bool(out.get("__interrupt__"))
            finally:
                if not paused:
                    forget_thread(CHECKPOINTER, thread_id)
    logger.info(
        "review.done thread_id=%s reviewer=%s approved=%s status=%s",
        thread_id, reviewer.principal, body.approved, out.get("gate_status"),
    )
    export_run(out, default_provider(), usage=budget.summary())
    return _to_response(thread_id, out, include_audit, budget.summary())


@app.post("/v1/dev/token", response_model=DevTokenResponse)
def dev_token(body: DevTokenRequest) -> DevTokenResponse:
    """Stand-in for Entra sign-in so the prototype can be exercised locally."""
    if not DEV_TOKENS_ENABLED:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    if body.role not in APP_ROLE_TO_SNOWFLAKE_ROLE:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown role '{body.role}'.")
    token = IDP.issue_user_token(body.upn, f"oid-{abs(hash(body.upn)) % 10**8:08d}", [body.role])
    return DevTokenResponse(access_token=token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
