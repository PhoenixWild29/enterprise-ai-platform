import sqlglot, pydantic
from gates import *
from schemas import *


def test_gates():

    print("sqlglot", sqlglot.__version__, "pydantic", pydantic.VERSION, "tables", len(DEFAULT_CATALOG))
    T = "SCADA.TELEMETRY_001"
    ok = [
        f"SELECT sensor_id, AVG(value) FROM {T} WHERE reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) GROUP BY 1 LIMIT 100",
        f"SELECT t.value, e.site_id FROM {T} t JOIN MAINT.EQUIPMENT e ON t.equipment_id = e.equipment_id WHERE t.reading_ts BETWEEN '2026-01-01' AND '2026-02-01' LIMIT 50",
        f"SELECT t.value FROM {T} t JOIN MAINT.WORK_ORDERS_003 w ON t.equipment_id = w.equipment_id WHERE t.reading_ts > '2026-01-01' AND w.opened_at > '2026-01-01' LIMIT 5",
        f"WITH r AS (SELECT equipment_id, value FROM {T} WHERE reading_ts >= '2026-01-01') SELECT * FROM r LIMIT 10",
        f"SELECT value FROM LAKEHOUSE.{T} JOIN MAINT.EQUIPMENT USING (equipment_id) WHERE reading_ts >= '2026-01-01' LIMIT 5",
    ]
    bad = [
        f"SELECT * FROM {T}",
        f"SELECT * FROM {T} WHERE reading_ts > '2026-01-01' OR 1=1",
        f"SELECT * FROM {T} WHERE reading_ts >= reading_ts",
        "SELECT * FROM SECRET.PAYROLL WHERE ts > '2026-01-01'",
        "SELECT * FROM TELEMETRY_001 WHERE reading_ts > '2026-01-01'",
        f"SELECT * FROM {T} t JOIN MAINT.EQUIPMENT e ON t.sensor_id = e.site_id WHERE t.reading_ts > '2026-01-01'",
        f"SELECT * FROM {T} t, MAINT.EQUIPMENT e WHERE t.reading_ts > '2026-01-01'",
        f"SELECT * FROM {T} t CROSS JOIN MAINT.EQUIPMENT e WHERE t.reading_ts > '2026-01-01'",
        f"SELECT * FROM {T} t JOIN MAINT.EQUIPMENT e ON t.equipment_id = e.equipment_id OR 1=1 WHERE t.reading_ts > '2026-01-01'",
        f"SELECT * FROM MAINT.EQUIPMENT WHERE equipment_id IN (SELECT equipment_id FROM {T})",
        "SELECT 1", "", "SELEC nope FROM",
    ]
    sec = [
        f"DROP TABLE {T}", f"DELETE FROM {T} WHERE 1=1", f"UPDATE {T} SET value = 0",
        f"INSERT INTO {T} (value) VALUES (1)", f"SELECT 1; DROP TABLE {T}",
        f"TRUNCATE TABLE {T}", f"CREATE TABLE x AS SELECT * FROM {T}", f"MERGE INTO {T} t USING MAINT.EQUIPMENT e ON t.equipment_id=e.equipment_id WHEN MATCHED THEN DELETE",
        f"ALTER TABLE {T} ADD COLUMN z INT", "GRANT ALL ON DATABASE LAKEHOUSE TO ROLE PUBLIC", "CALL sp()",

    ]
    for s in ok:
        r = sql_ast_gate(s); assert r.passed, (s, r.reasons)
    for s in bad:
        r = sql_ast_gate(s); assert not r.passed, s; print("REFUSED:", r.reasons[0][:90])
    for s in sec:
        try:
            sql_ast_gate(s); raise AssertionError("not raised: " + s)
        except SQLSecurityError as e:
            print("SECURITY:", e)
        st = new_state("q"); st["generated_sql"] = s
        u = sql_ast_gate_node(st); assert u["gate_status"] is GateStatus.REFUSED and u["audit_telemetry"][0].event_type == "security"

    # thin set
    P = [Passage(passage_id=f"p{i}", source="s", text=f"Pump P-10{i} bearing temperature exceeded 85C during the night shift.", score=0.8) for i in range(3)]
    assert thin_set_refusal_gate(P, [0.8, 0.7, 0.9]).passed
    assert not thin_set_refusal_gate(P[:1], [0.9]).passed
    assert not thin_set_refusal_gate(P, [0.3, 0.3, 0.5]).passed
    assert not thin_set_refusal_gate(P, [0.9]).passed
    assert not thin_set_refusal_gate(P, [0.9, float("nan"), 0.9]).passed

    # payload
    mk = lambda sql, **a: {"tool_name": "snowflake_query", "arguments": {"sql": sql, **a}, "principal": "u1"}
    assert payload_safety_gate(mk(ok[0])).passed
    assert not payload_safety_gate(mk(f"SELECT * FROM {T} WHERE reading_ts > '2026-01-01'")).passed
    assert not payload_safety_gate(mk(f"SELECT * FROM {T} WHERE reading_ts > '2026-01-01' LIMIT 50000", max_rows=10000)).passed
    assert not payload_safety_gate(mk(ok[0], max_rows=10)).passed
    assert not payload_safety_gate(mk(ok[0], evil=1)).passed
    assert not payload_safety_gate({"tool_name": "shell", "arguments": {}, "principal": "u"}).passed
    assert not payload_safety_gate({"tool_name": "snowflake_query", "arguments": {"sql": ok[0]}}).passed
    assert payload_safety_gate(mk("SELECT * FROM MAINT.EQUIPMENT")).passed
    assert payload_safety_gate({"tool_name": "vector_search", "arguments": {"query": "x"}, "principal": "u"}).passed
    assert not payload_safety_gate({"tool_name": "diagnostic_reason", "arguments": {"equipment_id": "a;drop", "symptom": "x"}, "principal": "u"}).passed
    st = new_state("q", "u1"); st["generated_sql"] = f"DROP TABLE {T}"
    assert payload_safety_gate_node(st)["gate_status"] is GateStatus.REFUSED

    # citations
    cits = [Citation.from_passage(p) for p in P[:2]]
    draft = f'Bearing temps were high. "{P[0].text[:40]}" [p0] and "{P[1].text[:40]}" [p1].'
    assert verbatim_citation_gate(draft, cits, P).passed
    assert not verbatim_citation_gate("No quotes here.", cits, P).passed
    assert not verbatim_citation_gate(draft, [], P).passed
    assert not verbatim_citation_gate(draft, [Citation(passage_id="zz", snippet="x")], P).passed
    assert not verbatim_citation_gate(draft, [Citation(passage_id="p0", snippet="Pump")], P).passed
    assert not verbatim_citation_gate(draft + " fabricated text that is forty chars long!!", [Citation(passage_id="p0", snippet="fabricated text that is forty chars long!!")], P).passed

    st = new_state("q"); st["raw_passages"] = P[:1]; st["confidence_scores"] = [0.9]
    u = thin_set_gate_node(st); assert u["gate_status"] is GateStatus.REFUSED and u["refusal_reason"]
    print(u["audit_telemetry"][0].model_dump_json()[:300])
    print(MCP_TOOL_REGISTRY["snowflake_query"].model_dump(by_alias=True)["inputSchema"]["required"])
    print("ALL OK")


def test_cte_names_are_resolved_in_scope_only():
    from gates import sql_ast_gate

    E, T, B = "MAINT.EQUIPMENT", "SCADA.TELEMETRY_000", "reading_ts >= '2026-01-01'"
    in_scope = [
        f"WITH a AS (SELECT equipment_id FROM {E}), b AS (SELECT equipment_id FROM a) SELECT * FROM b",
        f"WITH a AS (SELECT equipment_id FROM {E}) SELECT * FROM (SELECT equipment_id FROM a) s",
        f"WITH a AS (SELECT equipment_id FROM {E}), b AS (WITH c AS (SELECT equipment_id FROM a) SELECT equipment_id FROM c) SELECT * FROM b",
        f"WITH a AS (SELECT equipment_id FROM {E}) SELECT equipment_id FROM a UNION ALL SELECT equipment_id FROM a",
        f"SELECT e.site_id FROM {E} e WHERE e.equipment_id IN (WITH x AS (SELECT equipment_id FROM {T} WHERE {B}) SELECT equipment_id FROM x)",
    ]
    out_of_scope = [
        "WITH a AS (SELECT * FROM a) SELECT * FROM a",
        f"WITH a AS (SELECT * FROM b), b AS (SELECT equipment_id FROM {E}) SELECT * FROM a",
        f"SELECT * FROM (WITH p AS (SELECT equipment_id FROM {E}) SELECT * FROM p) s JOIN p ON 1=1",
        f"SELECT equipment_id FROM (WITH p AS (SELECT equipment_id FROM {E}) SELECT equipment_id FROM p) s UNION ALL SELECT equipment_id FROM p",
        f"WITH a AS (WITH inner_only AS (SELECT equipment_id FROM {E}) SELECT equipment_id FROM inner_only) SELECT * FROM inner_only",
        f"SELECT e.site_id, (SELECT COUNT(*) FROM x) FROM {E} e WHERE e.equipment_id IN (WITH x AS (SELECT equipment_id FROM {E}) SELECT equipment_id FROM x)",
    ]
    for sql in in_scope:
        assert sql_ast_gate(sql).passed, sql
    for sql in out_of_scope:
        assert not sql_ast_gate(sql).passed, sql
