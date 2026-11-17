"""API tests: both auth paths, refusals, and the human-review flow."""
from fastapi.testclient import TestClient

import main
from auth import TELEMETRY_SP_CLIENT_ID, TELEMETRY_SP_SECRET
from graph import AgentDeps, build_graph
from schemas import Citation

client = TestClient(main.app)


def _token(upn: str, role: str) -> dict[str, str]:
    r = client.post("/v1/dev/token", json={"upn": upn, "role": role})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


OPERATOR = _token("dana.ops@contoso.com", "Lakehouse.Operator")
ADMIN = _token("lee.admin@contoso.com", "Lakehouse.Admin")
JOB = {
    "client_id": TELEMETRY_SP_CLIENT_ID, "client_secret": TELEMETRY_SP_SECRET,
    "airflow_dag_id": "nightly_telemetry_rollup",
    "airflow_dag_run_id": "scheduled__2026-11-17T00:00:00+00:00",
}


def test_health_and_tools():
    assert client.get("/healthz").json()["catalog_tables"] == 500
    names = {t["name"] for t in client.get("/v1/tools").json()["tools"]}
    assert names == {"snowflake_query", "vector_search", "diagnostic_reason"}


def test_user_query_answers_with_citations():
    r = client.post("/v1/query", json={"query": "What is the bearing temperature alarm limit for pump P-101?"}, headers=OPERATOR)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "PASSED" and body["route"] == "rag_retriever"
    assert len(body["citations"]) >= 2 and "audit" not in body


def test_sql_query_includes_audit_and_rls():
    r = client.post("/v1/query?include_audit=true", json={"query": "List the latest telemetry readings"}, headers=OPERATOR)
    body = r.json()
    assert body["status"] == "PASSED"
    executed = next(e for e in body["audit"] if e["node"] == "sql_executor")
    assert executed["detail"]["rls_applied"] is True and "CURRENT_USER()" in executed["detail"]["sql"]


def test_refusals_return_200_with_reason():
    for q, needle in (("sql: DROP TABLE SCADA.TELEMETRY_001", "SECURITY"), ("What is the capital of France?", "thin_set"), ("total spend by cost center", "not entitled")):
        body = client.post("/v1/query", json={"query": q}, headers=OPERATOR).json()
        assert body["status"] == "REFUSED" and needle in body["refusal_reason"] and body["citations"] == []


def test_auth_failures():
    assert client.post("/v1/query", json={"query": "x"}).status_code == 401
    assert client.post("/v1/query", json={"query": "x"}, headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401
    assert client.post("/v1/jobs/query", json=JOB | {"client_secret": "wrong", "query": "x"}).status_code == 401
    assert client.post("/v1/jobs/query", json=JOB | {"airflow_dag_id": "rogue", "query": "x"}).status_code == 401
    assert client.post("/v1/query", json={"query": "x", "extra": 1}, headers=OPERATOR).status_code == 422


def test_headless_job_logs_service_identity():
    r = client.post("/v1/jobs/query?include_audit=true", json=JOB | {"query": "List the latest telemetry readings"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "PASSED"
    detail = body["audit"][0]["detail"]
    assert detail["service_account_id"].startswith("svc-telemetry-rollup") and detail["airflow_dag_run_id"] == JOB["airflow_dag_run_id"]


def test_human_review_flow(monkeypatch):
    def fabricating(query, passages):
        return "Pump is fine.", [Citation(passage_id=passages[0].passage_id, snippet="this snippet was invented by the model, not the source")]

    monkeypatch.setattr(main, "GRAPH", build_graph(AgentDeps(synthesizer=fabricating), checkpointer=main.CHECKPOINTER))
    ask = {"query": "What is the bearing temperature alarm limit for pump P-101?"}

    paused = client.post("/v1/query", json=ask, headers=OPERATOR).json()
    assert paused["status"] == "HITL_REVIEW" and paused["hitl"]["ticket"].startswith("HITL-") and paused["citations"] == []
    tid = paused["thread_id"]

    assert client.post(f"/v1/review/{tid}", json={"approved": True}, headers=OPERATOR).status_code == 403  # not admin
    assert client.post("/v1/review/unknown-thread", json={"approved": True}, headers=ADMIN).status_code == 404
    approved = client.post(f"/v1/review/{tid}?include_audit=true", json={"approved": True}, headers=ADMIN).json()
    assert approved["status"] == "PASSED" and approved["answer"] == "Pump is fine."
    assert any(e["detail"].get("reviewer") == "lee.admin@contoso.com" for e in approved["audit"])
    assert client.post(f"/v1/review/{tid}", json={"approved": True}, headers=ADMIN).status_code == 404  # already resolved

    tid2 = client.post("/v1/query", json=ask, headers=OPERATOR).json()["thread_id"]
    rejected = client.post(f"/v1/review/{tid2}", json={"approved": False}, headers=ADMIN).json()
    assert rejected["status"] == "REFUSED" and "rejected" in rejected["refusal_reason"]

    # four-eyes: an admin cannot approve their own request
    tid3 = client.post("/v1/query", json=ask, headers=ADMIN).json()["thread_id"]
    assert client.post(f"/v1/review/{tid3}", json={"approved": True}, headers=ADMIN).status_code == 403
