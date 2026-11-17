"""main.py - FastAPI service for the Enterprise AI Platform.

Endpoints
  POST /v1/query               user path (Bearer JWT -> OBO -> graph)
  POST /v1/jobs/query          headless path (service principal + Airflow run id)
  POST /v1/review/{thread_id}  human-in-the-loop decision on a paused run
  GET  /v1/tools               MCP tool definitions
  GET  /healthz                liveness and catalog size
  POST /v1/dev/token           demo sign-in (disabled when APP_ENV=production)

Run:  uvicorn main:app --reload
"""

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

import schemas
from auth import (
    API_AUDIENCE,
    APP_ROLE_TO_SNOWFLAKE_ROLE,
    IDP,
    AuthContext,
    AuthError,
    authenticate_service,
    authenticate_user,
)
from gates import DEFAULT_CATALOG, MCP_TOOL_REGISTRY
from graph import build_graph
from schemas import AuditEvent, Citation, GateStatus, new_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("platform.api")

APP_ENV = os.environ.get("APP_ENV", "dev")
REVIEWER_ROLE = "Lakehouse.Admin"

# Checkpointed state holds our own Pydantic models and enums; register them so
# the serializer will restore them when a paused run is resumed.
_STATE_TYPES = [
    ("schemas", name)
    for name, obj in vars(schemas).items()
    if isinstance(obj, type) and obj.__module__ == "schemas"
]
CHECKPOINTER = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES))
GRAPH = build_graph(checkpointer=CHECKPOINTER)

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


def _to_response(thread_id: str, out: dict[str, Any], include_audit: bool) -> QueryResponse:
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
    )


def _run(query: str, ctx: AuthContext, include_audit: bool) -> QueryResponse:
    state = new_state(query)
    state["auth_context"] = ctx.model_dump(mode="json")
    thread_id = state["trace_id"]
    out = GRAPH.invoke(state, config=_config(thread_id))
    logger.info(
        "query.done thread_id=%s principal=%s status=%s route=%s",
        thread_id, ctx.principal, out.get("gate_status"), out.get("route"),
    )
    return _to_response(thread_id, out, include_audit)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "env": APP_ENV, "catalog_tables": len(DEFAULT_CATALOG)}


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


@app.post("/v1/review/{thread_id}", response_model=QueryResponse, response_model_exclude_none=True)
def review(
    thread_id: str,
    body: ReviewRequest,
    include_audit: bool = Query(default=False),
    reviewer: AuthContext = Depends(current_user),
) -> QueryResponse:
    """Approve or reject a paused run. The reviewer's identity comes from
    their token, must hold the admin role, and must not be the requester."""
    if reviewer.snowflake_role != APP_ROLE_TO_SNOWFLAKE_ROLE[REVIEWER_ROLE]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Reviewing requires the {REVIEWER_ROLE} role.")
    snapshot = GRAPH.get_state(_config(thread_id))
    if not snapshot.next or not any(task.interrupts for task in snapshot.tasks):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No run is awaiting review for this thread.")
    if snapshot.values.get("principal") == reviewer.principal:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewers cannot approve their own requests.")
    out = GRAPH.invoke(
        Command(resume={"approved": body.approved, "reviewer": reviewer.principal, "comment": body.comment}),
        config=_config(thread_id),
    )
    logger.info(
        "review.done thread_id=%s reviewer=%s approved=%s status=%s",
        thread_id, reviewer.principal, body.approved, out.get("gate_status"),
    )
    return _to_response(thread_id, out, include_audit)


@app.post("/v1/dev/token", response_model=DevTokenResponse)
def dev_token(body: DevTokenRequest) -> DevTokenResponse:
    """Stand-in for Entra sign-in so the prototype can be exercised locally."""
    if APP_ENV == "production":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    if body.role not in APP_ROLE_TO_SNOWFLAKE_ROLE:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown role '{body.role}'.")
    token = IDP.issue_user_token(body.upn, f"oid-{abs(hash(body.upn)) % 10**8:08d}", [body.role])
    return DevTokenResponse(access_token=token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
