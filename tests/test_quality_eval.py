"""The quality evaluation: its warehouse, grader, stand-in models and audit."""
import dataclasses

import pytest

import quality_eval as qe
from auth import IDP, apply_row_level_security, authenticate_user
from gates import SQLSecurityError
from local_warehouse import FROZEN_NOW, LocalWarehouse

CATALOG = qe.build_eval_catalog()
READINGS = ("SELECT t.reading_id, t.user_id FROM SCADA.TELEMETRY_001 AS t "
            "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) ORDER BY 1 LIMIT 1000")


def _user(upn, role="Lakehouse.Operator"):
    return authenticate_user(IDP.issue_user_token(upn, f"oid-{upn}", [role]), idp=IDP)


@pytest.fixture(scope="module")
def run():
    return qe.run(repeats=2)


def test_the_warehouse_really_executes_and_row_level_security_really_filters():
    dana, bob = _user("dana.ops@contoso.com"), _user("bob.ops@contoso.com")
    mine = LocalWarehouse(CATALOG)(apply_row_level_security(READINGS, dana, CATALOG), dana, CATALOG)
    theirs = LocalWarehouse(CATALOG)(apply_row_level_security(READINGS, bob, CATALOG), bob, CATALOG)
    everyone = LocalWarehouse(CATALOG)(READINGS, dana, CATALOG)
    assert mine and theirs and {r["USER_ID"] for r in mine} == {"DANA.OPS@CONTOSO.COM"} and {r["USER_ID"] for r in theirs} == {"BOB.OPS@CONTOSO.COM"}
    assert len(mine) < len(everyone) and not {r["READING_ID"] for r in mine} & {r["READING_ID"] for r in theirs}
    narrower = READINGS.replace("-7", "-2")
    assert len(LocalWarehouse(CATALOG)(narrower, dana, CATALOG)) < len(everyone)        # a different query, a different answer
    assert LocalWarehouse(CATALOG)(READINGS, dana, CATALOG) == everyone                # same query, same rows, every time
    assert FROZEN_NOW.year == 2026
    with pytest.raises(SQLSecurityError):                                              # nothing unvalidated runs here either
        LocalWarehouse(CATALOG)("DROP TABLE SCADA.TELEMETRY_001", dana, CATALOG)


def test_every_reference_query_passes_the_gates_and_returns_rows():
    assert len(qe.CASES) == 46 and len({c.case_id for c in qe.CASES}) == 46
    for case in qe.CASES:
        if case.expect == "rows":
            rows = qe.reference_rows(case, LocalWarehouse(CATALOG), CATALOG)
            assert rows and len(rows) <= 20, case.case_id
    tiers = {t: sum(c.tier == t and c.expect == "rows" for c in qe.CASES) for t in (1, 2, 3)}
    assert tiers == {1: 14, 2: 14, 3: 12} and sum(c.expect == "refusal" for c in qe.CASES) == 6
    assert {c.tier for c in qe.CASES if c.split == "heldout" and c.expect == "rows"} == {1, 2, 3}
    # Tier 3 questions name a place, never a table: the table has to be found from its description.
    for case in qe.CASES:
        if case.tier == 3:
            assert not any(word in case.question for word in ("TELEMETRY", "WORK_ORDERS", "LEDGER", "117", "204", "251", "055", "088"))
    assert "Ridgeback" in CATALOG.get("SCADA.TELEMETRY_117").description


def test_the_grader_compares_rows_not_names_or_order():
    case = next(c for c in qe.CASES if c.expect == "rows")
    expected = [{"SENSOR_ID": "TT-101", "AVG_VALUE": 70.129}, {"SENSOR_ID": "PT-102", "AVG_VALUE": 310.0}]
    passed = {"gate_status": "PASSED"}
    same_values = [{"mean": 310.001, "sensor": "PT-102", "_table": "x"}, {"mean": 70.13, "sensor": "TT-101", "_table": "x"}]
    assert qe.grade(case, expected, passed | {"execution_results": same_values}).reason == "correct"
    for wrong in ([{"SENSOR_ID": "TT-101", "AVG_VALUE": 71.0}, expected[1]], expected[:1], expected + expected[:1], []):
        assert qe.grade(case, expected, passed | {"execution_results": wrong}).reason == "wrong rows"
    assert qe.grade(case, expected, {"gate_status": "REFUSED", "execution_results": []}).reason == "refused"
    assert qe.grade(case, expected, {"gate_status": "HITL_REVIEW", "execution_results": expected}).passed is False
    assert qe.grade(case, expected, {"gate_status": "ERROR", "error": "KeyError"}).reason == "error"
    refusal = next(c for c in qe.CASES if c.expect == "refusal")
    assert qe.grade(refusal, [], {"gate_status": "REFUSED", "execution_results": []}).reason == "correctly refused"
    assert qe.grade(refusal, [], passed | {"execution_results": expected}).passed is False


def test_both_runtimes_run_every_case_and_refuse_what_must_be_refused(run):
    results, _ = run
    assert [(r.runtime, r.model) for r in results] == [("langgraph", "example-small"), ("strands", "example-small"),
                                                       ("langgraph", "example-large"), ("strands", "example-large")]
    for result in results:
        assert len(result.cases) == 46 and result.score(expect="refusal") == (6, 6)
        assert not any(c.reason == "error" for c in result.cases) and all(c.stable for c in result.cases)
        refused = [c for c in result.cases if c.expect == "refusal"]
        assert all(c.status == "REFUSED" and c.rows == 0 for c in refused)
    # The stand-ins are scripted, so the runtimes must agree case by case; a difference would be a wiring bug.
    for a, b in ((results[0], results[1]), (results[2], results[3])):
        assert [(c.case_id, c.passed, c.reason) for c in a.cases] == [(c.case_id, c.passed, c.reason) for c in b.cases]


def test_the_example_models_differ_and_leave_headroom(run):
    results, _ = run
    small, large = results[0], results[2]
    assert small.score()[0] < large.score()[0] < 46
    assert all(not s.passed or l.passed for s, l in zip(small.cases, large.cases))       # the larger one never loses a case the smaller wins
    assert large.score(tier=1)[0] > large.score(tier=3)[0] and "wrong rows" in {c.reason for c in large.cases}
    repaired = [c for c in qe.CASES if len(qe.ExampleModel("example-large").attempts(c)) == 2]
    assert repaired, "some cases exercise the refuse-then-repair path"


def test_the_audit_passes_on_a_sound_eval(run):
    _, checks = run
    by_name = {c.name: c for c in checks}
    assert {c.status for c in checks} == {"ok", "not checked"}, [(c.name, c.detail) for c in checks if c.status in ("fail", "warn")]
    assert by_name["Cases look like real questions"].status == "not checked" and "invented" in by_name["Cases look like real questions"].detail
    assert "scripted stand-ins" in by_name["Stronger model scores higher"].detail


def test_the_audit_catches_planted_bugs(monkeypatch):
    original = list(qe.CASES)
    case = original[0]

    def audit_with(cases, models=("example-small", "example-large")):
        monkeypatch.setattr(qe, "CASES", cases)
        expected = {c.case_id: (qe.reference_rows(c, LocalWarehouse(CATALOG), CATALOG) if c.expect == "rows" else []) for c in cases}
        results = [qe.run_system("langgraph", qe.ExampleModel(m), CATALOG, expected, repeats=1) for m in models]
        return {c.name: c for c in qe.audit(results, CATALOG, expected, simulated=True)}

    # A reference query that returns nothing: every system "fails" it for no reason.
    empty = dataclasses.replace(case, case_id="bug-empty", reference_sql=case.reference_sql.replace("-7,", "-0,"))
    assert audit_with(original[:6] + [empty])["Reference answers"].status == "fail"
    # The same question asked twice.
    assert audit_with(original[:6] + [dataclasses.replace(case, case_id="bug-dup")])["Reference answers"].status == "fail"
    # A question copied from a prompt.
    leaked = dataclasses.replace(case, case_id="bug-leak", question="One read-only SELECT statement. No INSERT, UPDATE, DELETE, MERGE or DDL.")
    assert audit_with(original[:6] + [leaked])["Leakage into prompts"].status == "fail"
    # Only easy cases: the score is at its ceiling.
    easy = [c for c in original if c.tier == 1 and c.expect == "rows"]
    assert audit_with(easy, models=("example-large",))["Headroom"].status == "warn"
    # One model only: the ordering check cannot run and says so.
    assert audit_with(original[:8], models=("example-large",))["Stronger model scores higher"].status == "not checked"
    # No held-out cases.
    assert audit_with([dataclasses.replace(c, split="dev") for c in original[:12]])["Held-out set"].status == "warn"


def test_interval_and_report(run, tmp_path):
    low, high = qe.wilson(39, 46)
    assert 0.70 < low < 0.85 < high < 0.95 and qe.wilson(0, 0) == (0.0, 0.0) and qe.wilson(46, 46)[1] == 1.0
    results, checks = run
    path = qe.write_report(results, checks, simulated=True, directory=tmp_path)
    page = path.read_text()
    assert "Simulated models" in page and "Is the evaluation sound?" in page and page.count("<tr>") > 50
    assert (tmp_path / "quality_eval.json").exists()
    assert qe.main(["--model", "llm"]) == 2      # a real model is not wired in yet, and the tool says so instead of pretending
