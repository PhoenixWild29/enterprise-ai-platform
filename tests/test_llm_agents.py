"""Model-backed agents, driven by scripted completions (no network)."""
import json

from auth import IDP, authenticate_user
from eval_harness import JudgeInput, RubricDimension, make_llm_judge
from gates import DEFAULT_CATALOG, sql_ast_gate
from graph import build_graph, run_query
from llm_agents import (
    LLMRouter, LLMSQLGenerator, LLMSynthesizer, build_llm_deps, select_schema_context, strip_sql,
)
from schemas import Passage, Route

GOOD_SQL = ("SELECT t.sensor_id, t.value FROM SCADA.TELEMETRY_000 AS t "
            "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) LIMIT 100")
UNBOUNDED_SQL = "SELECT t.sensor_id, t.value FROM SCADA.TELEMETRY_000 AS t LIMIT 100"
P = [Passage(passage_id=f"p{i}", source="sop://x", score=0.9,
             text=f"Pump P-10{i} bearing temperature alarm limit is 85 C under normal load.") for i in (1, 2)]


class Script:
    """Completion that replays canned replies and records prompts."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _operator():
    return authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]), idp=IDP)


def test_schema_context_is_a_small_relevant_slice():
    tables = select_schema_context("average vibration from telemetry sensors", DEFAULT_CATALOG)
    names = {t.fqn for t in tables}
    assert len(tables) <= 9 and "MAINT.EQUIPMENT" in names  # FK parent pulled in
    assert all(n.startswith(("SCADA.", "MAINT.EQUIPMENT")) for n in names)
    assert select_schema_context("zzz qqq", DEFAULT_CATALOG) == []


def test_sql_generator_strips_fences_and_accepts_valid_sql():
    llm = Script(f"```sql\n{GOOD_SQL};\n```")
    assert LLMSQLGenerator(llm)("recent telemetry", Route.TEXT_TO_SQL) == GOOD_SQL and len(llm.prompts) == 1
    assert "FIN.LEDGER" not in llm.prompts[0] and "<question>recent telemetry</question>" in llm.prompts[0]
    assert strip_sql("SELECT 1;") == "SELECT 1"


def test_sql_generator_repairs_from_gate_feedback():
    llm = Script(UNBOUNDED_SQL, GOOD_SQL)
    sql = LLMSQLGenerator(llm)("recent telemetry", Route.TEXT_TO_SQL)
    assert sql == GOOD_SQL and len(llm.prompts) == 2 and sql_ast_gate(sql).passed
    assert "previous query was rejected" in llm.prompts[1] and "lower bound on READING_TS" in llm.prompts[1]


def test_sql_generator_gives_up_after_bounded_repairs():
    llm = Script(UNBOUNDED_SQL)
    assert LLMSQLGenerator(llm, max_repairs=2)("recent telemetry", Route.TEXT_TO_SQL) == UNBOUNDED_SQL
    assert len(llm.prompts) == 3


def test_sql_generator_never_repairs_a_security_violation():
    llm = Script("DROP TABLE SCADA.TELEMETRY_000", GOOD_SQL)
    assert LLMSQLGenerator(llm)("delete everything", Route.TEXT_TO_SQL).startswith("DROP") and len(llm.prompts) == 1


def test_sql_generator_survives_model_outage():
    def down(prompt):
        raise TimeoutError
    assert LLMSQLGenerator(down)("recent telemetry", Route.TEXT_TO_SQL) == ""


def test_synthesizer_attaches_verbatim_quotes_itself():
    llm = Script(json.dumps({"evidence": [{"passage_id": "p1", "claim": "The alarm limit is 85 C."},
                                          {"passage_id": "p2", "claim": "Same limit applies to P-102."}]}))
    draft, cites = LLMSynthesizer(llm)("alarm limit?", P)
    assert [c.snippet for c in cites] == [p.text[:40] for p in P]
    assert all(c.snippet in draft for c in cites) and "The alarm limit is 85 C." in draft


def test_synthesizer_fails_closed_on_bad_output():
    for reply in ("not json", '{"evidence": "nope"}', '{"evidence": []}', '{"evidence": [{"passage_id": "", "claim": "x"}]}'):
        assert LLMSynthesizer(Script(reply))("q", P) == ("", [])


def test_router_labels_and_fallback():
    assert LLMRouter(Script(" SQL. "))("how many readings") is Route.TEXT_TO_SQL
    assert LLMRouter(Script("diagnostic"))("why did it fail") is Route.DIAGNOSTIC
    assert LLMRouter(Script("I think this is rag maybe"))("q") is Route.REFUSE
    assert LLMRouter(Script("rag"))("   ") is Route.REFUSE


def _fake_llm(synth_reply, sql_reply=GOOD_SQL, route="sql"):
    def complete(prompt):
        if prompt.startswith("Classify"):
            return route
        if prompt.startswith("You write Snowflake SQL"):
            return sql_reply
        return synth_reply
    return complete


def test_graph_end_to_end_on_model_backed_agents():
    reply = json.dumps({"evidence": [{"passage_id": "row-1", "claim": "Latest reading is 71.5."},
                                     {"passage_id": "row-2", "claim": "The prior reading is 77.75."}]})
    out = run_query("latest telemetry readings", _operator(), build_graph(build_llm_deps(_fake_llm(reply))))
    assert out["gate_status"] == "PASSED" and "USER_ID = CURRENT_USER()" in out["generated_sql"]
    assert "Latest reading is 71.5." in out["final_response"] and len(out["citations"]) == 2


def test_graph_refuses_model_sql_that_stays_invalid():
    out = run_query("latest telemetry readings", _operator(), build_graph(build_llm_deps(_fake_llm("{}", sql_reply=UNBOUNDED_SQL))))
    assert out["gate_status"] == "REFUSED" and "sql_ast_parser" in out["refusal_reason"] and not out["execution_results"]


def test_graph_catches_citation_to_unretrieved_passage():
    reply = json.dumps({"evidence": [{"passage_id": "sop-secret-999", "claim": "The limit is 120 C per the vendor."}]})
    out = run_query("latest telemetry readings", _operator(), build_graph(build_llm_deps(_fake_llm(reply))))
    assert out["gate_status"] == "HITL_REVIEW" and "does not match any retrieved passage" in out["final_response"]


def test_llm_judge_parses_scores_and_fails_closed():
    inp = JudgeInput(query="q", response="r", passages=P, citations=[])
    good = json.dumps({"scores": [{"dimension": d.value, "score": 0.9, "rationale": "ok"} for d in RubricDimension]})
    verdict = make_llm_judge(Script(f"Here you go:\n{good}"))(inp)
    assert {s.score for s in verdict.scores} == {0.9}
    for bad in ("no json here", '{"scores": []}', json.dumps({"scores": [{"dimension": "groundedness", "score": 7}]})):
        assert {s.score for s in make_llm_judge(Script(bad))(inp).scores} == {0.0}

    def down(prompt):
        raise ConnectionError
    assert {s.score for s in make_llm_judge(down)(inp).scores} == {0.0}
