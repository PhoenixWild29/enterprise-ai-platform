"""agentcore_app.py - Entrypoint for Amazon Bedrock AgentCore Runtime.

AgentCore Runtime hosts this app in a per-session microVM and calls
POST /invocations. The handler authenticates the caller, runs the Strands
agent with its gated tools, and returns the gated result.

  user-facing      the caller's JWT arrives in the Authorization header
                   (allow-list it in the runtime's request-header settings);
                   everything runs on behalf of that person
  scheduled jobs   the payload names the job; it runs under a service
                   principal's scoped role and never waits on a human

Run locally:   python agentcore_app.py        (serves on port 8080)
Deploy:        see infra/terraform and the Dockerfile
"""

import logging
import os
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth import ACTIVE_IDP, IDP, AuthContext, AuthError, authenticate_service, authenticate_user
from memory import AgentCoreMemoryStore, MemoryService
from runtime import RuntimeConfig, build_agent_deps, catalog_problems
from schemas import GateStatus
from strands_agent import bedrock_model_from_env, run_strands_query

logger = logging.getLogger("platform.agentcore")

APP_ENV = os.environ.get("APP_ENV", "dev").strip().lower()
IS_PRODUCTION = APP_ENV not in {"dev", "development", "local", "test"}

app = BedrockAgentCoreApp()


class JobIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_id: str = Field(min_length=1)
    client_secret: str = Field(default="", repr=False)
    airflow_dag_id: str = Field(min_length=1)
    airflow_dag_run_id: str = Field(min_length=1)


class Invocation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=4_000)
    job: JobIdentity | None = None


class Platform:
    """What a request needs: tools' dependencies, the model, memory, identity."""

    def __init__(self, deps: Any, model: Any, memory: MemoryService, idp: Any) -> None:
        self.deps, self.model, self.memory, self.idp = deps, model, memory, idp


def _memory_service() -> MemoryService:
    memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if not memory_id:
        return MemoryService()
    from bedrock_agentcore.memory import MemoryClient

    return MemoryService(AgentCoreMemoryStore(MemoryClient(region_name=os.environ.get("AWS_REGION")), memory_id))


_PLATFORM: Platform | None = None


def platform() -> Platform:
    """Built once, on the first request, from the environment."""
    global _PLATFORM
    if _PLATFORM is None:
        config = RuntimeConfig.from_env()
        deps = build_agent_deps(RuntimeConfig(
            agent_backend="simulated",  # the Strands agent replaces the LangGraph agents; only the tools' dependencies are needed
            warehouse_backend=config.warehouse_backend, retriever_backend=config.retriever_backend,
            catalog_source=config.catalog_source, corpus_path=config.corpus_path,
        ))
        if IS_PRODUCTION:
            # Fail closed: the same stand-ins the API refuses in production are refused here.
            skip = ("AGENT_BACKEND", "MODEL_PROVIDER", "SYNTHESIS_MODE")
            problems = [p for p in config.production_problems() if not p.startswith(skip)]
            problems += catalog_problems(config, deps.catalog)
            if ACTIVE_IDP is IDP:
                problems.append("IDENTITY_PROVIDER=agentcore or entra (the simulated provider signs its own tokens)")
            if problems:
                raise RuntimeError(f"APP_ENV={APP_ENV} is treated as production and requires: " + "; ".join(problems))
        _PLATFORM = Platform(deps, bedrock_model_from_env(), _memory_service(), ACTIVE_IDP)
    return _PLATFORM


def configure(deps: Any, model: Any, memory: MemoryService | None = None, idp: Any = None) -> None:
    """Install explicit dependencies (tests, local runs)."""
    global _PLATFORM
    _PLATFORM = Platform(deps, model, memory or MemoryService(), idp or ACTIVE_IDP)


def _bearer(headers: dict[str, str] | None) -> str:
    for name, value in (headers or {}).items():
        if name.lower() == "authorization" and value.lower().startswith("bearer "):
            return value[7:].strip()
    raise AuthError("Missing bearer token.")


def _refused(reason: str) -> dict[str, Any]:
    return {"status": GateStatus.REFUSED.value, "answer": f"I cannot answer this request. {reason}",
            "citations": [], "refusal_reason": reason}


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    try:
        request = Invocation.model_validate(payload)
    except ValidationError as err:
        return _refused("[request] " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()))

    services = platform()
    session_id = getattr(context, "session_id", None)
    try:
        if request.job is not None:
            job = request.job
            ctx: AuthContext = authenticate_service(
                job.client_id, job.client_secret, job.airflow_dag_id, job.airflow_dag_run_id, idp=services.idp
            )
        else:
            ctx = authenticate_user(_bearer(getattr(context, "request_headers", None)), idp=services.idp)
    except AuthError as err:
        return _refused(f"[auth_gate] {err}")

    out = run_strands_query(
        request.prompt, ctx, services.deps, services.model,
        memory=services.memory if request.job is None else None, session_id=session_id,
    )
    released = out["gate_status"] is GateStatus.PASSED
    logger.info("invocation.done principal=%s status=%s trace_id=%s", ctx.principal, out["gate_status"].value, out["trace_id"])
    return {
        "status": out["gate_status"].value,
        "answer": out["final_response"],
        "citations": [c.model_dump() for c in out["citations"]] if released else [],
        "refusal_reason": None if released else out["refusal_reason"],
        "ticket": out.get("ticket"),
        "usage": out.get("usage"),
        "trace_id": out["trace_id"],
        "session_id": session_id,
    }


if __name__ == "__main__":
    app.run()
