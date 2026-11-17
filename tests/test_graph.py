import time
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from auth import *
from auth import IDP
from graph import *
from schemas import *
from gates import DEFAULT_CATALOG as CAT
import base64, json
from gates import sql_ast_gate


def test_auth_and_graph():

    op = authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]))
    fin = authenticate_user(IDP.issue_user_token("fay.fin@contoso.com", "oid-2", ["Lakehouse.Finance"]))
    svc = authenticate_service(TELEMETRY_SP_CLIENT_ID, TELEMETRY_SP_SECRET, "nightly_telemetry_rollup", "scheduled__2026-11-17T00:00:00+00:00")
    assert op.snowflake_user == "DANA.OPS@CONTOSO.COM" and op.snowflake_role == "OPS_ANALYST"
    assert svc.service_account_id and svc.airflow_dag_run_id and svc.snowflake_role == "SVC_TELEMETRY_ROLLUP_ROLE"
    print(svc.to_audit_event().detail)

    def denied(fn):
        try: fn()
        except AuthError as e: print("AUTH DENIED:", e); return True
        return False
    tok = IDP.issue_user_token("x@contoso.com", "o", ["Lakehouse.Operator"])
    h, p, s = tok.split(".")
    forged = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode() + "." + p + "."
    assert denied(lambda: authenticate_user(forged))
    assert denied(lambda: authenticate_user(tok[:-3] + "abc"))
    assert denied(lambda: authenticate_user(IDP.issue_user_token("x@c.com", "o", ["Lakehouse.Operator"], ttl=-5)))
    assert denied(lambda: authenticate_user(IDP.issue_user_token("x@c.com", "o", ["Nope"])))
    assert denied(lambda: authenticate_user(SimulatedEntraID().issue_user_token("x@c.com", "o", ["Lakehouse.Admin"])))
    assert denied(lambda: authenticate_service(TELEMETRY_SP_CLIENT_ID, "wrong", "nightly_telemetry_rollup", "r1"))
    assert denied(lambda: authenticate_service(TELEMETRY_SP_CLIENT_ID, TELEMETRY_SP_SECRET, "rogue_dag", "r1"))
    assert denied(lambda: authenticate_service(TELEMETRY_SP_CLIENT_ID, TELEMETRY_SP_SECRET, "nightly_telemetry_rollup", " "))
    # OBO token must not be accepted as a user token (audience)
    obo = IDP.acquire_token_on_behalf_of(tok, PLATFORM_CLIENT_ID, PLATFORM_CLIENT_SECRET, "session:role:OPS_ANALYST")
    assert denied(lambda: authenticate_user(obo))

    rls = apply_row_level_security("SELECT t.value FROM SCADA.TELEMETRY_001 t JOIN MAINT.EQUIPMENT e ON t.equipment_id = e.equipment_id WHERE t.reading_ts > '2026-01-01' LIMIT 5", op, CAT)
    print(rls); assert "T.USER_ID = CURRENT_USER()" in rls
    assert sql_ast_gate(rls).passed
    assert "CURRENT_USER" not in apply_row_level_security("SELECT value FROM SCADA.TELEMETRY_001 WHERE reading_ts > '2026-01-01' LIMIT 5", svc, CAT)

    app = build_graph()
    def run(q, ctx, g=app, **k):
        out = run_query(q, ctx, g, **k)
        path = [e.node for e in out["audit_telemetry"]]
        print(f"\n> {q!r}\n  [{out['gate_status']}] {' > '.join(path)}\n  {str(out['final_response'])[:230]}")
        return out

    o = run("What is the bearing temperature alarm limit for pump P-101?", op); assert o["gate_status"] == "PASSED" and o["route"] == Route.RAG
    o = run("List the latest telemetry readings", op); assert o["gate_status"] == "PASSED" and "CURRENT_USER()" in o["generated_sql"] and o["execution_results"]
    o = run("Why did the P-101 pump bearing fail?", op); assert o["gate_status"] == "PASSED" and o["route"] == Route.DIAGNOSTIC and o["execution_results"]
    o = run("sql: DROP TABLE SCADA.TELEMETRY_001", op); assert o["gate_status"] == "REFUSED" and any(e.event_type == "security" for e in o["audit_telemetry"]) and not o["execution_results"]
    o = run("sql: SELECT * FROM SCADA.TELEMETRY_001", op); assert o["gate_status"] == "REFUSED"
    o = run("sql: SELECT value FROM SCADA.TELEMETRY_001 WHERE reading_ts > '2026-01-01'", op); assert o["gate_status"] == "REFUSED" and "payload" in o["refusal_reason"]
    o = run("What is the capital of France?", op); assert o["gate_status"] == "REFUSED" and "thin_set" in o["refusal_reason"]
    o = run("total spend by cost center", op); assert o["gate_status"] == "REFUSED" and "authorization" in o["refusal_reason"]
    o = run("total spend by cost center", fin); assert o["gate_status"] == "PASSED"
    o = run("", op); assert o["gate_status"] == "REFUSED"
    o = run("List the latest telemetry readings", None); assert o["gate_status"] == "REFUSED" and o["audit_telemetry"][0].event_type == "security"
    o = run("List the latest telemetry readings", svc); assert o["gate_status"] == "PASSED" and "CURRENT_USER" not in o["generated_sql"]
    assert o["audit_telemetry"][0].detail["airflow_dag_run_id"].startswith("scheduled__")
    o = run("list work orders", svc); assert o["gate_status"] == "REFUSED"
    exp_ctx = op.model_copy(update={"expires_at": op.expires_at.replace(year=2020)})
    o = run("list work orders", exp_ctx); assert o["gate_status"] == "REFUSED"

    # citation failure -> HITL (user), refusal (service)
    def bad_synth(q, passages):
        return "Pump is fine, trust me.", [Citation(passage_id=passages[0].passage_id, snippet="this text was fabricated by the model and is long")]
    bad = AgentDeps(synthesizer=bad_synth)
    o = run("What is the bearing temperature alarm limit for pump P-101?", op, build_graph(bad)); assert o["gate_status"] == "HITL_REVIEW"
    o = run("List the latest telemetry readings", svc, build_graph(bad)); assert o["gate_status"] == "REFUSED"

    # interrupt-based HITL
    g = build_graph(bad, checkpointer=InMemorySaver())
    for approved, reviewer, want in ((True, "lead@contoso.com", "PASSED"), (False, "lead@contoso.com", "REFUSED"), (True, "", "REFUSED")):
        tid = f"t-{approved}-{reviewer}"
        o = run_query("What is the bearing temperature alarm limit for pump P-101?", op, g, thread_id=tid)
        assert "__interrupt__" in o, o.keys()
        print("\nINTERRUPT:", o["__interrupt__"][0].value["ticket"], o["__interrupt__"][0].value["reason"][:80])
        o = g.invoke(Command(resume={"approved": approved, "reviewer": reviewer}), config={"configurable": {"thread_id": tid}})
        print("  resumed ->", o["gate_status"], "|", str(o["final_response"])[:90]); assert o["gate_status"] == want
    print("\nALL OK")
