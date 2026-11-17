import asyncio

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace import TracerProvider

import mcp_server
from anomaly import robust_z_anomalies
from auth import IDP, authenticate_user
from graph import KNOWLEDGE_BASE, AgentDeps, build_graph, run_query
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from retrieval import BM25Index, HashingEmbedder, HybridRetriever, VectorIndex
from telemetry import build_tracer_provider, export_run

SQL = ("SELECT t.sensor_id, t.value FROM SCADA.TELEMETRY_001 AS t WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) LIMIT 5")


def _token(role="Lakehouse.Operator"):
    return IDP.issue_user_token("dana.ops@contoso.com", "oid-1", [role])


def _call(server, name, args, token=None):
    async def run():
        reset = None
        if token is not None:
            access = await server_verifier.verify_token(token)
            if access is not None:
                reset = auth_context_var.set(AuthenticatedUser(access))
        try:
            return await server.call_tool(name, args)
        finally:
            if reset is not None:
                auth_context_var.reset(reset)
    return asyncio.run(run())


server_verifier = mcp_server.PlatformTokenVerifier(idp=IDP)


def _text(result):
    blocks = result.content if hasattr(result, "content") else result[0] if isinstance(result, tuple) else result
    return " ".join(getattr(b, "text", str(b)) for b in blocks)


def test_mcp_tools_are_listed_and_run_as_the_caller():
    server = mcp_server.build_server(AgentDeps(), server_verifier)
    assert sorted(t.name for t in asyncio.run(server.list_tools())) == ["find_tables", "run_sql", "search_manuals"]
    rows = _text(_call(server, "run_sql", {"sql": SQL}, _token()))
    assert '<passage id="row-1">' in rows and "DANA.OPS@CONTOSO.COM" in rows
    assert "REFUSED" in _text(_call(server, "run_sql", {"sql": "SELECT * FROM SCADA.TELEMETRY_001"}, _token()))
    assert "SECURITY" in _text(_call(server, "run_sql", {"sql": "DROP TABLE SCADA.TELEMETRY_001"}, _token()))
    assert "FIN." not in _text(_call(server, "find_tables", {"question": "ledger spend by cost center"}, _token()))


def test_mcp_tools_refuse_without_a_verified_identity():
    server = mcp_server.build_server(AgentDeps(), server_verifier)
    for token in (None, "not.a.jwt"):
        try:
            text = _text(_call(server, "run_sql", {"sql": SQL}, token))
        except Exception as err:  # the SDK may raise or return an error result
            text = str(err)
        assert "row-1" not in text and "DANA" not in text and ("Not authenticated" in text or "Error executing tool" in text)
    assert asyncio.run(server_verifier.verify_token("not.a.jwt")) is None
    access = asyncio.run(server_verifier.verify_token(_token()))
    assert access.subject == "dana.ops@contoso.com"


def test_run_is_exported_as_a_trace_without_data():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ctx = authenticate_user(_token(), idp=IDP)
    out = run_query("List the latest telemetry readings", ctx, build_graph())
    trace_id = export_run(out, provider, usage={"total_tokens": 360, "estimated_cost_usd": None})
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert names[-1] == "platform.request" and "gate.sql_ast_parser" in names and "tool_call.sql_executor" in names
    root = spans[-1]
    assert root.attributes["platform.gate_status"] == "PASSED" and root.attributes["gen_ai.usage.total_tokens"] == 360
    assert format(root.context.trace_id, "032x") == trace_id and all(s.context.trace_id == root.context.trace_id for s in spans)
    dumped = " ".join(f"{k}={v}" for s in spans for k, v in s.attributes.items())
    assert "SELECT" not in dumped and "sensor_id-001" not in dumped and "telemetry readings" not in dumped

    exporter.clear()
    refused = run_query("sql: DROP TABLE SCADA.TELEMETRY_001", ctx, build_graph())
    export_run(refused, provider)
    spans = exporter.get_finished_spans()
    assert spans[-1].status.status_code.name == "ERROR" and any(s.name == "security.sql_ast_parser" for s in spans)
    assert export_run(out, None) is None and build_tracer_provider() is None


def _exported_text(outcome):
    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    assert export_run(outcome, provider) is not None
    spans = exporter.get_finished_spans()
    return " ".join(f"{s.name} {s.status.description} " + " ".join(f"{k}={v}" for k, v in s.attributes.items()) for s in spans)


def test_gate_messages_that_quote_callers_or_models_are_never_exported():
    from scripted_model import ScriptedModel
    from strands_agent import run_strands_query
    ctx = authenticate_user(_token(), idp=IDP)
    bad_sql = [
        "SELECT t.sensor_id FROM SCADA.TELEMETRY_001 t WHERE t.user_id = 'bob.ops@contoso.com' AND t.equipment_id = 'SECRET-EQUIP-77' LIMIT 5 ORDER",
        'SELECT t."ssn 123-45-6789 of bob" FROM SCADA.TELEMETRY_001 t',
    ]
    for sql in bad_sql:
        dumped = _exported_text(run_query(f"sql: {sql}", ctx, build_graph()))
        assert "refused_by=sql_ast_parser" in dumped
        for leak in ("SECRET-EQUIP-77", "bob.ops", "123-45-6789", "SELECT", "dana.ops"):
            assert leak not in dumped, leak
    out = run_strands_query("What is the alarm limit for pump P-101?", ctx, AgentDeps(), ScriptedModel([
        {"tool": ("search_manuals", {"query": "bearing temperature alarm limit pump P-101"})},
        {"tool": ("EvidenceAnswer", {"evidence": [{"passage_id": "sop-pump-014", "quote": "MODEL-WRITTEN claim: limit is 185 C"}]})}, {"text": "x"}]))
    dumped = _exported_text(out)
    assert out["gate_status"] == "HITL_REVIEW" and "185" not in dumped and "MODEL-WRITTEN" not in dumped and "dana.ops" not in dumped


def test_both_runtimes_export_their_runs(monkeypatch):
    import agentcore_app
    import main
    import telemetry
    from fastapi.testclient import TestClient as ApiClient
    from scripted_model import ScriptedModel
    from starlette.testclient import TestClient

    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_DEFAULT_PROVIDER", provider)
    monkeypatch.setattr(telemetry, "_DEFAULT_BUILT", True)
    headers = {"Authorization": f"Bearer {_token()}"}

    agentcore_app.configure(AgentDeps(), ScriptedModel([{"tool": ("EvidenceAnswer", {"evidence": []})}, {"text": "x"}]), idp=IDP)
    TestClient(agentcore_app.app).post("/invocations", json={"prompt": "List telemetry"}, headers=headers)
    agentcore_app._PLATFORM = None
    assert [s.name for s in exporter.get_finished_spans()][-1] == "platform.request"

    exporter.clear()
    assert ApiClient(main.app).post("/v1/query", json={"query": "List the latest telemetry readings"}, headers=headers).status_code == 200
    names = [s.name for s in exporter.get_finished_spans()]
    assert names[-1] == "platform.request" and "tool_call.sql_executor" in names


def test_hybrid_retrieval_finds_exact_identifiers():
    corpus = list(KNOWLEDGE_BASE)
    dense, sparse = VectorIndex(HashingEmbedder()), BM25Index()
    dense.add(corpus)
    sparse.add(corpus)
    hybrid = HybridRetriever(dense, sparse)
    assert hybrid("C-200")[0].passage_id == "man-comp-003"
    hits = hybrid("What is the bearing temperature alarm limit for pump P-101?")
    assert hits[0].passage_id == "sop-pump-014" and all(0 < p.score <= 1 for p in hits) and len({p.passage_id for p in hits}) == len(hits)
    assert hybrid("capital of France") == [] and hybrid("  ") == []
    # One common word matching is not confidence: it must not clear the thin-set gate on its own.
    from gates import thin_set_refusal_gate
    weak = hybrid("pump")
    assert not thin_set_refusal_gate(weak, [p.score for p in weak]).passed


def test_robust_anomaly_detection():
    readings = [{"SENSOR_ID": "s-1", "READING_TS": i, "VALUE": v} for i, v in enumerate([70, 71, 69, 70, 72, 71, 70, 140, 69, 70])]
    report = robust_z_anomalies(readings + [{"SENSOR_ID": "s-2", "READING_TS": 0, "VALUE": 5}, {"SENSOR_ID": "s-3", "READING_TS": 0, "VALUE": "bad"}]
                                + [{"SENSOR_ID": "flat", "READING_TS": i, "VALUE": 10} for i in range(10)])
    assert [(a["sensor"], a["value"]) for a in report["anomalies"]] == [("s-1", 140.0)]
    assert report["sensors_not_assessed"] == ["flat", "s-2"] and report["readings_scanned"] == 21


def test_reserved_evidence_ids_cannot_come_from_the_corpus_and_wide_scans_stay_releasable():
    from platform_tools import PlatformTools, RunLedger
    from schemas import DataDomain, Passage
    ctx = authenticate_user(_token(), idp=IDP)
    planted = [Passage(passage_id=pid, source="kb://planted", text=f"Planted passage {pid} about pump bearing temperature alarm limit",
                       score=0.9, domain=DataDomain.MAINTENANCE) for pid in ("row-1", "anomaly-scan-1", "sop-real-001", "sop-real-002")]
    ledger = RunLedger(query="pump bearing temperature alarm limit", ctx=ctx, trace_id="t")
    tools = PlatformTools(AgentDeps(retriever=lambda q: planted), ledger)
    tools.search_manuals("pump bearing temperature alarm limit")
    assert {p.passage_id for p in ledger.passages} == {"sop-real-001", "sop-real-002"}

    wide = AgentDeps(executor=lambda sql, c, cat: [{"_table": "SCADA.TELEMETRY_001", "SENSOR_ID": f"sensor-{i:04d}", "READING_TS": 0, "VALUE": 1.0} for i in range(400)])
    ledger = RunLedger(query="q", ctx=ctx, trace_id="t")
    tools = PlatformTools(wide, ledger)
    tools.run_sql(SQL)
    tools.flag_anomalies()
    scan = next(p for p in ledger.passages if p.passage_id == "anomaly-scan-1")
    assert len(scan.text) < 1200 and '"sensors_not_assessed_count": 400' in scan.text
