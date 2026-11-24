"""The quality evaluation: its warehouse, grader, stand-in models and audit."""
import dataclasses
from datetime import UTC, datetime

import pytest

import graph
import platform_tools
import quality_eval as qe
from auth import IDP, apply_row_level_security, authenticate_user, authorize_sql
from gates import SQLSecurityError
from local_warehouse import FROZEN_NOW, LocalWarehouse

CATALOG = qe.build_eval_catalog()
READINGS = ("SELECT t.reading_id, t.user_id FROM SCADA.TELEMETRY_001 AS t "
            "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) ORDER BY 1 LIMIT 1000")
BY_ID = {c.case_id: c for c in qe.CASES}
PASSED = {"gate_status": "PASSED"}


def _user(upn, role="Lakehouse.Operator"):
    return authenticate_user(IDP.issue_user_token(upn, f"oid-{upn}", [role]), idp=IDP)


def _expected(cases):
    warehouse = LocalWarehouse(CATALOG)
    return {c.case_id: (qe.reference_rows(c, warehouse, CATALOG) if c.expect == "rows" else []) for c in cases}


def _scored(model, cases, passed=lambda case: True, runtime="langgraph", repeats=2):
    """A result built by hand, for audit checks that only read scores."""
    result = qe.SystemResult(runtime, model, repeats=repeats)
    result.cases = [qe.CaseResult(c.case_id, c.tier, c.split, c.expect, bool(passed(c)), "correct", "PASSED", 1, None) for c in cases]
    return result


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
    warehouse = LocalWarehouse(CATALOG)
    assert warehouse(READINGS, dana, CATALOG) == everyone                              # same query, same rows, every time
    assert warehouse(narrower, dana, CATALOG) == warehouse.results[-1] and len(warehouse.results) == len(warehouse.statements) == 2
    assert FROZEN_NOW.year == 2026
    with pytest.raises(SQLSecurityError):                                              # nothing unvalidated runs here either
        LocalWarehouse(CATALOG)("DROP TABLE SCADA.TELEMETRY_001", dana, CATALOG)


def test_every_clock_function_reads_the_frozen_instant_not_the_wall_clock():
    dana, earlier = _user("dana.ops@contoso.com"), datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    counts = set()
    for clock in ("CURRENT_TIMESTAMP()", "CURRENT_DATE()", "GETDATE()", "SYSDATE()"):
        sql = f"SELECT COUNT(*) AS n FROM SCADA.TELEMETRY_001 AS t WHERE t.reading_ts >= DATEADD(day, -7, {clock}) LIMIT 1"
        warehouse = LocalWarehouse(CATALOG, now=earlier)
        counts.add(warehouse(sql, dana, CATALOG)[0]["N"])
        assert "2026-06-01" in warehouse.statements[-1] and "current" not in warehouse.statements[-1].lower(), clock
    assert min(counts) > 100     # data is seeded relative to the frozen instant, so a wall-clock bound would find nothing


def test_the_seeded_data_can_tell_a_right_query_from_a_wrong_one():
    dana, warehouse = _user("dana.ops@contoso.com"), LocalWarehouse(CATALOG)

    def one(sql):
        return warehouse(sql, dana, CATALOG)[0]

    span = one("SELECT MIN(t.reading_ts) AS min_reading_ts, COUNT(*) AS n FROM SCADA.TELEMETRY_001 AS t LIMIT 1")
    assert span["MIN_READING_TS"] < "2026-10-20" and span["N"] > 1000                  # history runs past twice the longest window
    orders = one("SELECT MIN(w.opened_at) AS min_opened_at FROM MAINT.WORK_ORDERS_000 AS w LIMIT 1")
    ledger = one("SELECT MIN(l.posted_at) AS min_posted_at FROM FIN.LEDGER_000 AS l LIMIT 1")
    assert orders["MIN_OPENED_AT"] < "2026-03-01" and ledger["MIN_POSTED_AT"] < "2025-11-25"
    sizes = {one(f"SELECT COUNT(*) AS n FROM SCADA.TELEMETRY_{n:03d} AS t LIMIT 1")["N"] for n in range(1, 7)}
    assert len(sizes) == 6                                                             # neighbouring tables do not look alike


def test_every_reference_query_passes_the_gates_and_returns_rows():
    assert len(qe.CASES) == 46 and len({c.case_id for c in qe.CASES}) == 46
    for case in qe.CASES:
        if case.expect == "rows":
            rows = qe.reference_rows(case, LocalWarehouse(CATALOG), CATALOG)
            assert rows and len(rows) <= 50 and not qe._label_problems(case, rows), case.case_id
    tiers = {t: sum(c.tier == t for c in qe.CASES) for t in (0, 1, 2, 3)}
    assert tiers == {0: 6, 1: 14, 2: 14, 3: 12} and all((c.tier == 0) == (c.expect == "refusal") for c in qe.CASES)
    assert {c.tier for c in qe.CASES if c.split == "heldout" and c.expect == "rows"} == {1, 2, 3}
    assert sorted(c.case_id for c in qe.CASES if c.ordered) == ["t1-04", "t1-09", "t2-05", "t2-11"]
    # Tier 3 questions name a place, never a table: the table has to be found from its description.
    for case in qe.CASES:
        if case.tier == 3:
            assert not any(word in case.question for word in ("TELEMETRY", "WORK_ORDERS", "LEDGER", "117", "204", "251", "055", "088"))
    assert "Ridgeback" in CATALOG.get("SCADA.TELEMETRY_117").description


def test_the_grader_accepts_the_same_answer_in_another_shape():
    expected = [{"SENSOR_ID": "TT-101", "AVG_VALUE": 70.129}, {"SENSOR_ID": "PT-102", "AVG_VALUE": 310.0}]
    assert qe.same_answer(expected, [{"mean": 310.001, "sensor": "PT-102", "_table": "x"}, {"mean": 70.13, "sensor": "TT-101", "_table": "x"}])
    assert qe.same_answer(expected, [dict(r, QUALITY_FLAG="GOOD") for r in expected])              # an extra column is not a wrong answer
    assert qe.same_answer([{"AVG_AMOUNT": 5509.01}], [{"AVG_AMOUNT": 5509.015}])                   # rounded or not, the same number
    assert qe.same_answer([{"X": 0}], [{"X": -0.001}])
    assert qe.same_answer([{"A": 1, "B": 2}, {"A": 2, "B": 3}, {"A": 3, "B": 1}], [{"p": 2, "q": 1}, {"p": 3, "q": 2}, {"p": 1, "q": 3}])
    assert qe.same_answer([{"A": 1}, {"A": 2}], [{"A": 2}, {"A": 1}]) and qe.same_answer([], [])


def test_the_grader_rejects_a_different_answer():
    expected = [{"SENSOR_ID": "TT-101", "AVG_VALUE": 70.129}, {"SENSOR_ID": "PT-102", "AVG_VALUE": 310.0}]
    for wrong in ([{"SENSOR_ID": "TT-101", "AVG_VALUE": 71.0}, expected[1]], expected[:1], expected + expected[:1], [],
                  [{"SENSOR_ID": r["SENSOR_ID"]} for r in expected],                             # a missing column
                  [{"SENSOR_ID": "TT-101", "AVG_VALUE": 310.0}, {"SENSOR_ID": "PT-102", "AVG_VALUE": 70.129}]):   # right values, wrong pairing
        assert not qe.same_answer(expected, wrong), wrong
    assert not qe.same_answer([{"A": 1, "B": 2}, {"A": 2, "B": 1}], [{"x": 1, "y": 1}, {"x": 2, "y": 2}])
    assert not qe.same_answer([{"MIN_VALUE": 5, "MAX_VALUE": 7}], [{"MIN_VALUE": 7, "MAX_VALUE": 5}])   # names are used when they agree
    assert not qe.same_answer([{"N": 7}], [{"N": "7"}]) and not qe.same_answer([{"N": None}], [{"N": "None"}])
    assert not qe.same_answer([{"N": 70.12}], [{"N": 70.14}])
    newest_first = [{"ID": 3}, {"ID": 2}, {"ID": 1}]
    assert qe.same_answer(newest_first, newest_first[::-1]) and not qe.same_answer(newest_first, newest_first[::-1], ordered=True)


def test_a_verdict_comes_from_what_actually_ran():
    case, rows, other = BY_ID["t1-02"], [{"N": 58}], [{"N": 3}]
    assert qe.grade(case, rows, PASSED, [rows]).reason == "correct"
    assert qe.grade(case, rows, PASSED, [other, rows]).reason == "correct"          # an exploratory query first does not spoil the answer
    assert qe.grade(case, rows, PASSED, [rows, other]).reason == "wrong rows"        # the last statement is the answer, on either runtime
    assert qe.grade(case, rows, PASSED | {"execution_results": rows}, []).reason == "wrong rows"   # nothing ran, so nothing is credited
    assert qe.grade(case, rows, {"gate_status": "REFUSED"}, []).reason == "refused"
    assert qe.grade(case, rows, {"gate_status": "REFUSED"}, [[]]).reason == "ran, then refused"
    assert qe.grade(case, rows, {"gate_status": "HITL_REVIEW"}, [rows]).passed is False
    assert qe.grade(case, rows, {"gate_status": "ERROR", "error": "KeyError"}, []).reason == "error"
    ordered = BY_ID["t1-04"]
    assert qe.grade(ordered, [{"A": 2}, {"A": 1}], PASSED, [[{"A": 1}, {"A": 2}]]).reason == "wrong rows"
    refusal = BY_ID["ref-02"]
    assert qe.grade(refusal, [], {"gate_status": "REFUSED", "execution_results": []}, []).reason == "correctly refused"
    assert qe.grade(refusal, [], {"gate_status": "REFUSED", "execution_results": []}, [[]]).passed is False   # it ran: that is not a refusal
    assert qe.grade(refusal, [], PASSED | {"execution_results": rows}, [rows]).passed is False


def test_a_refusal_only_counts_if_no_query_ran(monkeypatch):
    """With entitlement checks switched off, row-level security still hides
    every row and the platform refuses the empty result. That must not be
    scored as a correct refusal."""
    allowed = authorize_sql(READINGS, _user("dana.ops@contoso.com"), CATALOG)
    assert allowed.passed
    for module in (graph, platform_tools):
        monkeypatch.setattr(module, "authorize_sql", lambda *args, **kwargs: allowed)
    case = BY_ID["ref-02"]
    for runtime in qe.RUNTIMES.values():
        warehouse = LocalWarehouse(CATALOG)
        outcome = runtime(case, qe.ExampleModel("example-large"), warehouse, CATALOG)
        assert warehouse.statements, "the query ran"
        assert qe.grade(case, [], outcome, warehouse.results).passed is False


def test_both_runtimes_run_every_case_and_refuse_what_must_be_refused(run):
    results, _ = run
    assert [(r.runtime, r.model) for r in results] == [("langgraph", "example-small"), ("strands", "example-small"),
                                                       ("langgraph", "example-large"), ("strands", "example-large")]
    for result in results:
        assert len(result.cases) == 46 and result.score(expect="refusal") == result.score(tier=0) == (6, 6)
        assert sum(result.score(tier=t)[1] for t in (0, 1, 2, 3)) == 46
        assert sum(result.score(tier=t)[0] for t in (0, 1, 2, 3)) == result.score()[0]
        assert not any(c.reason == "error" for c in result.cases) and all(c.stable for c in result.cases)
        refused = [c for c in result.cases if c.expect == "refusal"]
        assert all(c.status == "REFUSED" and c.rows == 0 for c in refused)
    # The stand-ins are scripted, so the runtimes must agree case by case; a difference would be a wiring bug.
    for a, b in ((results[0], results[1]), (results[2], results[3])):
        assert [(c.case_id, c.passed, c.reason) for c in a.cases] == [(c.case_id, c.passed, c.reason) for c in b.cases]


def test_a_scripted_wrong_query_is_never_graded_correct(run):
    """The score has to equal what the script planned: every planned mistake fails, every planned right answer passes."""
    results, _ = run
    for result in results:
        model, planned_wrong = qe.ExampleModel(result.model), 0
        for case, got in zip(qe.CASES, result.cases):
            if case.expect != "rows":
                continue
            wrong = model.attempts(case)[-1] != case.reference_sql
            planned_wrong += wrong
            assert got.passed is (not wrong), (result.runtime, result.model, case.case_id, got.reason)
        assert result.score(expect="rows")[0] == 40 - planned_wrong
    assert {m: 40 - r.score(expect="rows")[0] for m, r in ((r.model, r) for r in results)} == {"example-small": 16, "example-large": 8}


def test_the_example_models_differ_and_leave_headroom(run):
    results, _ = run
    small, large = results[0], results[2]
    assert small.score()[0] < large.score()[0] < 46
    assert all(not s.passed or l.passed for s, l in zip(small.cases, large.cases))       # the larger one never loses a case the smaller wins
    assert large.score(tier=1)[0] > large.score(tier=3)[0] and "wrong rows" in {c.reason for c in large.cases}
    repaired = [c for c in qe.CASES if len(qe.ExampleModel("example-large").attempts(c)) == 2]
    assert repaired, "some cases exercise the refuse-then-repair path"
    for case in qe.CASES:                                                                # a declared-equivalent change is never used as a mistake
        if case.expect == "rows" and case.case_id in qe.EQUIVALENT:
            banned = {qe.mistakes(case)[kind] for kind in qe.EQUIVALENT[case.case_id]}
            assert qe.ExampleModel("example-small").attempts(case)[-1] not in banned


def test_the_audit_passes_on_a_sound_eval(run):
    _, checks = run
    by_name = {c.name: c for c in checks}
    assert {c.status for c in checks} == {"ok", "not checked"}, [(c.name, c.detail) for c in checks if c.status in ("fail", "warn")]
    assert [c.name for c in checks if c.status == "not checked"] == ["Cases look like real questions"]
    assert "invented" in by_name["Cases look like real questions"].detail
    stronger = by_name["Stronger model scores higher"].detail
    assert "scripted stand-ins" in stronger and "intervals overlap" in stronger and "sign test" in stronger
    assert "planted mistakes" in by_name["Wrong answers are caught"].detail and "tier 3: 3" in by_name["Held-out set"].detail


def _audit(monkeypatch, cases, results=None):
    monkeypatch.setattr(qe, "CASES", cases)
    results = results or [_scored("example-small", cases, lambda c: c.tier < 2), _scored("example-large", cases, lambda c: c.tier < 3)]
    return {c.name: c for c in qe.audit(results, CATALOG, _expected(cases), simulated=True)}


def test_the_audit_catches_a_wrong_or_ungradable_reference_answer(monkeypatch):
    base, case = list(qe.CASES[:6]), BY_ID["t1-02"]
    assert _audit(monkeypatch, base)["Reference answers"].status == "ok"

    def planted(**changes):
        return _audit(monkeypatch, base + [dataclasses.replace(case, case_id="bug", **changes)])["Reference answers"]

    empty = planted(question="List the readings at well pad 001 in the last 0 days.", reference_sql=BY_ID["t1-01"].reference_sql.replace("-7,", "-0,"))
    assert empty.status == "fail" and "empty or zero" in empty.detail
    zero = planted(question="How many FT-999 readings did well pad 001 record in the last 7 days?",
                   reference_sql=case.reference_sql.replace(" LIMIT 1", " AND t.sensor_id = 'FT-999' LIMIT 1"))
    assert zero.status == "fail" and "empty or zero" in zero.detail                    # one row holding 0 is still an empty answer
    other_table = planted(question=case.question.replace("pad 001", "pad 044"))
    assert other_table.status == "fail" and "SCADA.TELEMETRY_044" in other_table.detail
    other_window = planted(question=case.question.replace("7 days", "30 days"))
    assert other_window.status == "fail" and "time window" in other_window.detail
    cut = planted(question="List the readings from well pad 001 in the last 7 days.", reference_sql=BY_ID["t1-01"].reference_sql)
    assert cut.status == "fail" and "LIMIT 10 cuts the answer" in cut.detail
    unordered = planted(question="How many readings did well pad 001 record in the last 7 days? Largest first", ordered=True)
    assert unordered.status == "fail" and "fewer than two rows" in unordered.detail
    again = planted(question=case.question.upper() + "  ")                             # the same question, differently typed
    assert again.status == "fail" and "duplicate questions: 1" in again.detail


def test_the_audit_catches_data_too_weak_to_fail_a_wrong_query(monkeypatch):
    base = [BY_ID["t1-01"], BY_ID["t1-02"], BY_ID["t2-06"]]
    assert _audit(monkeypatch, base)["Wrong answers are caught"].status == "ok"
    monkeypatch.setattr(qe, "EQUIVALENT", {})                                          # a mistake the grader cannot see, not declared
    weak = _audit(monkeypatch, base)["Wrong answers are caught"]
    assert weak.status == "fail" and "t2-06:window" in weak.detail and "t1-01:window" in weak.detail
    monkeypatch.setattr(qe, "EQUIVALENT", {"t1-02": {"window": "not true"}, "t1-01": {"window": "true"}, "t2-06": {"window": "true"}})
    stale = _audit(monkeypatch, base)["Wrong answers are caught"]
    assert stale.status == "fail" and "listed as equivalent but not: ['t1-02:window']" in stale.detail
    # Every kind of planted mistake exists somewhere, and every one changes the query.
    kinds = set()
    for case in qe._DEFAULT_CASES:
        if case.expect == "rows":
            found = qe.mistakes(case)
            assert found and all(sql != case.reference_sql for sql in found.values()), case.case_id
            kinds |= set(found)
    assert kinds == {"table", "filter", "aggregate", "window", "order"}


def test_the_audit_catches_leaks_and_does_not_skip_what_it_cannot_read(monkeypatch):
    base, case = list(qe.CASES[:6]), qe.CASES[0]
    leaked = dataclasses.replace(case, case_id="bug-leak", question="one read-only  select statement. no INSERT, update, DELETE, MERGE or DDL.")
    found = _audit(monkeypatch, base + [leaked])["Leakage into prompts"]
    assert found.status == "fail" and "bug-leak" in found.detail                      # case and spacing do not hide a leak
    assert "platform_tools.py" in qe.PROMPT_SOURCES and all((qe.HERE / p).exists() for p in qe.PROMPT_SOURCES)
    monkeypatch.chdir("/tmp")                                                          # the scan does not depend on where it is run from
    assert _audit(monkeypatch, base + [leaked])["Leakage into prompts"].status == "fail"
    monkeypatch.setattr(qe, "PROMPT_SOURCES", (*qe.PROMPT_SOURCES, "no_such_prompt_file.py"))
    missing = _audit(monkeypatch, base)["Leakage into prompts"]
    assert missing.status == "fail" and "no_such_prompt_file.py" in missing.detail


def test_the_audit_does_not_say_ok_for_what_it_did_not_check(monkeypatch):
    everything = list(qe.CASES)
    cases = [c for i, c in enumerate(everything) if i % 4 == 0 or c.case_id == "t3-01" or c.expect == "refusal"]   # a spread of every tier
    answers = [c for c in cases if c.expect == "rows"]
    # One repeat cannot show a verdict is stable.
    once = [_scored("example-small", cases, lambda c: c.tier < 2, repeats=1), _scored("example-large", cases, lambda c: c.tier < 3, repeats=1)]
    assert _audit(monkeypatch, cases, once)["Same run, same verdict"].status == "not checked"
    flaky = [_scored("example-small", cases, lambda c: c.tier < 2), _scored("example-large", cases, lambda c: c.tier < 3)]
    flaky[0].cases[3].stable = False
    assert _audit(monkeypatch, cases, flaky)["Same run, same verdict"].status == "fail"
    # A subset or an edited list is not compared with the pinned answers, and says so.
    assert _audit(monkeypatch, cases[:8])["Reference answers match the pinned copy"].status == "not checked"
    monkeypatch.setattr(qe, "CASES", qe._DEFAULT_CASES)
    monkeypatch.setattr(qe, "REFERENCE_DIGEST", "0" * 16)
    whole = [_scored("example-small", everything, lambda c: c.tier < 2), _scored("example-large", everything, lambda c: c.tier < 3)]
    drifted = {c.name: c for c in qe.audit(whole, CATALOG, _expected(everything), simulated=True)}["Reference answers match the pinned copy"]
    assert drifted.status == "fail" and "0000000000000000" in drifted.detail
    # Only easy cases: the score is at its ceiling.
    easy = [c for c in cases if c.tier == 1]
    assert _audit(monkeypatch, easy, [_scored("example-large", easy)])["Headroom"].status == "warn"
    # A system that refuses everything passes the must-refuse cases and nothing else. That is not headroom.
    refuser = _scored("refuses-everything", cases, lambda c: c.expect == "refusal")
    floor = _audit(monkeypatch, cases, [refuser])["Headroom"]
    assert refuser.score() == (6, len(cases)) and floor.status == "warn" and "Answers nothing correctly" in floor.detail
    # One model only: the ordering check cannot run and says so.
    assert _audit(monkeypatch, cases, [_scored("example-large", cases, lambda c: c.tier < 3)])["Stronger model scores higher"].status == "not checked"
    # Two models that differ by a single case are not shown to differ.
    close = [_scored("a", cases, lambda c: c.case_id != "t3-01"), _scored("b", cases)]
    assert _audit(monkeypatch, cases, close)["Stronger model scores higher"].status == "warn"
    # Held-out questions must exist in every tier; must-refuse cases do not count towards them.
    lopsided = [dataclasses.replace(c, split="heldout" if c.tier <= 1 else "dev") for c in cases if c.tier != 2] + [
        dataclasses.replace(c, split="dev") for c in everything if c.tier == 2]
    held = _audit(monkeypatch, lopsided)["Held-out set"]
    assert held.status == "warn" and "tier 2: 0" in held.detail
    assert _audit(monkeypatch, [dataclasses.replace(c, split="dev") for c in answers])["Held-out set"].status == "warn"


def test_interval_and_report(run, tmp_path):
    low, high = qe.wilson(39, 46)
    assert 0.70 < low < 0.85 < high < 0.95 and qe.wilson(0, 0) == (0.0, 0.0) and qe.wilson(46, 46)[1] == 1.0
    assert qe.paired_p(8, 0) == pytest.approx(0.0078125) and qe.paired_p(0, 0) == 1.0 and qe.paired_p(3, 3) == 1.0
    results, checks = run
    table = qe.summary_table(results)
    assert table[0]["heldout"].startswith(f"{results[0].score(split='heldout', expect='rows')[0]}/9 (") and " to " in table[0]["tier3"]
    path = qe.write_report(results, checks, simulated=True, directory=tmp_path)
    page = path.read_text()
    assert "Simulated models" in page and "Is the evaluation sound?" in page and page.count("<tr>") > 50
    assert "entitled to see" in page and "N times 24 hours" in page and "Held-out questions" in page
    assert (tmp_path / "quality_eval.json").exists()
    assert qe.main(["--model", "llm"]) == 2      # a real model is not wired in yet, and the tool says so instead of pretending
