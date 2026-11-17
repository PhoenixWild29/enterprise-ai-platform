"""graph.py - LangGraph topology for the Enterprise AI Platform.

    START -> auth_gate -> supervisor_router
        RAG        : rag_retriever -> thin_set_gate -> synthesizer
        TEXT-TO-SQL: text_to_sql -> sql_ast_gate -> payload_safety_gate
                     -> authz_gate -> sql_executor -> synthesizer
        DIAGNOSTIC : diagnostic_reasoner -> thin_set_gate -> (SQL chain above)
    synthesizer -> citation_gate -> respond | hitl_review | refusal -> END

Any gate that does not return PASSED sends the run to an output node:
security and thin-evidence failures go to `refusal`; a citation failure on
the user path goes to `hitl_review`; headless runs never wait on a human.

The model-facing pieces (router, retriever, SQL generator, synthesizer) and
the warehouse executor are injected through `AgentDeps`. The defaults are
deterministic simulations so the graph runs with no LLM or Snowflake account.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlglot import exp

from auth import (
    GATE_AUTHZ,
    AuthContext,
    AuthError,
    AuthPath,
    apply_row_level_security,
    authorize_sql,
    load_auth_context,
)
from gates import (
    DEFAULT_CATALOG,
    TOOL_SNOWFLAKE_QUERY,
    _limit_value,
    _run_gate,
    _table_fqn,
    citation_gate_node,
    parse_read_only_sql,
    payload_safety_gate_node,
    sql_ast_gate_node,
    thin_set_gate_node,
)
from schemas import (
    AgentState,
    AuditEvent,
    Catalog,
    Citation,
    DataDomain,
    GateStatus,
    Passage,
    Route,
    ToolPayload,
    new_state,
)

# Node names
N_AUTH = "auth_gate"
N_SUPERVISOR = "supervisor_router"
N_RAG = "rag_retriever"
N_SQL = "text_to_sql"
N_DIAG = "diagnostic_reasoner"
N_THIN = "thin_set_gate"
N_AST = "sql_ast_gate"
N_PAYLOAD = "payload_safety_gate"
N_AUTHZ = "authz_gate"
N_EXEC = "sql_executor"
N_SYNTH = "synthesizer"
N_CITE = "citation_gate"
N_HITL = "hitl_review"
N_RESPOND = "respond"
N_REFUSAL = "refusal"

logger = logging.getLogger("platform.graph")

type RouterFn = Callable[[str], Route]
type Retriever = Callable[[str], list[Passage]]
type SQLGenerator = Callable[[str, Route, frozenset[DataDomain]], str]
type SQLExecutor = Callable[[str, AuthContext, Catalog], list[dict[str, Any]]]
type Synthesizer = Callable[[str, list[Passage]], tuple[str, list[Citation]]]


# ---------------------------------------------------------------------------
# Default (simulated) dependencies
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were what which who how why when where do does did for of on in to "
    "and or with at by from this that it its be as can should would me my our show tell give "
    "please about any there".split()
)

KNOWLEDGE_BASE: tuple[Passage, ...] = (
    Passage(
        passage_id="sop-pump-014", source="sop://pumps/P-101/bearing-monitoring", score=1.0,
        domain=DataDomain.MAINTENANCE,
        text="Pump P-101 bearing temperature alarm limit is 85 C; trip the pump at 95 C and "
             "inspect lubrication before restart.",
    ),
    Passage(
        passage_id="sop-pump-022", source="sop://pumps/vibration-limits", score=1.0,
        domain=DataDomain.MAINTENANCE,
        text="Centrifugal pump vibration above 7.1 mm/s RMS indicates bearing wear or "
             "misalignment and requires a work order within 24 hours.",
    ),
    Passage(
        passage_id="man-comp-003", source="manual://compressors/C-200/surge", score=1.0,
        domain=DataDomain.MAINTENANCE,
        text="Compressor C-200 surge is preceded by a rapid discharge pressure drop; open the "
             "anti-surge valve and reduce load immediately.",
    ),
    Passage(
        passage_id="scada-guide-007", source="guide://scada/quality-flags", score=1.0,
        domain=DataDomain.SCADA,
        text="SCADA telemetry readings with quality flag BAD must be excluded from averages; "
             "sensor temperature drift is corrected at calibration.",
    ),
    Passage(
        passage_id="maint-hist-031", source="report://maintenance/P-101/2026-Q3", score=1.0,
        domain=DataDomain.MAINTENANCE,
        text="In Q3 the P-101 pump bearing failure was traced to degraded lubrication after "
             "bearing temperature exceeded the alarm limit for six hours.",
    ),
    Passage(
        passage_id="fin-policy-002", source="policy://finance/maintenance-capex", score=1.0,
        domain=DataDomain.FINANCE,
        text="Maintenance spend above 50,000 USD per work order requires cost center owner "
             "approval before the ledger entry is posted.",
    ),
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if t not in _STOPWORDS}


def keyword_retriever(query: str, top_k: int = 4) -> list[Passage]:
    """Lexical retriever: score = share of query terms found in the passage."""
    q = _tokens(query)
    if not q:
        return []
    scored = [
        p.model_copy(update={"score": round(len(q & _tokens(p.text)) / len(q), 4)})
        for p in KNOWLEDGE_BASE
    ]
    hits = sorted((p for p in scored if p.score > 0), key=lambda p: p.score, reverse=True)
    return hits[:top_k]


_DIAG_TERMS = ("why", "diagnose", "root cause", "troubleshoot", "what caused", "failure")
_SQL_TERMS = (
    "sql:", "how many", "average", "avg", "count", "list", "trend", "latest", "readings",
    "total", "sum of", "spend", "work orders",
)


def keyword_router(query: str) -> Route:
    """Supervisor policy. Empty input is refused rather than guessed at."""
    q = query.lower().strip()
    if not q:
        return Route.REFUSE
    if q.startswith("sql:"):
        return Route.TEXT_TO_SQL
    if any(t in q for t in _DIAG_TERMS):
        return Route.DIAGNOSTIC
    if any(t in q for t in _SQL_TERMS):
        return Route.TEXT_TO_SQL
    return Route.RAG


def template_sql_generator(query: str, route: Route, allowed_domains: frozenset[DataDomain] | None = None) -> str:
    """Stand-in for the Text-to-SQL model. A query prefixed with `sql:` is
    passed through verbatim so tests can simulate a compromised generator."""
    q = query.strip()
    if q.lower().startswith("sql:"):
        return q[4:].strip()
    q = q.lower()
    if route is Route.DIAGNOSTIC:
        return (
            "SELECT t.sensor_id, t.reading_ts, t.value, t.quality_flag, e.equipment_type "
            "FROM SCADA.TELEMETRY_001 AS t "
            "JOIN MAINT.EQUIPMENT AS e ON t.equipment_id = e.equipment_id "
            "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) "
            "ORDER BY t.reading_ts DESC LIMIT 100"
        )
    if any(t in q for t in ("telemetry", "sensor", "reading", "temperature", "vibration")):
        return (
            "SELECT t.sensor_id, t.reading_ts, t.value, t.quality_flag "
            "FROM SCADA.TELEMETRY_001 AS t "
            "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) "
            "ORDER BY t.reading_ts DESC LIMIT 100"
        )
    if any(t in q for t in ("spend", "cost", "ledger", "amount")):
        return (
            "SELECT l.cost_center_id, SUM(l.amount) AS total_amount "
            "FROM FIN.LEDGER_000 AS l "
            "WHERE l.posted_at >= DATEADD(day, -30, CURRENT_TIMESTAMP()) "
            "GROUP BY l.cost_center_id LIMIT 100"
        )
    return (
        "SELECT w.work_order_id, w.equipment_id, w.opened_at, w.failure_code "
        "FROM MAINT.WORK_ORDERS_000 AS w "
        "WHERE w.opened_at >= DATEADD(day, -30, CURRENT_TIMESTAMP()) "
        "ORDER BY w.opened_at DESC LIMIT 100"
    )


def simulated_executor(sql: str, ctx: AuthContext, catalog: Catalog) -> list[dict[str, Any]]:
    """Deterministic stand-in for Snowflake. Returns up to 3 synthetic rows
    shaped like the first governed table; USER_ID is always the session user,
    mirroring what Row-Level Security would return."""
    root = parse_read_only_sql(sql)
    spec = next(
        (s for t in root.find_all(exp.Table) if (s := catalog.get(_table_fqn(t))) is not None),
        None,
    )
    if spec is None:
        return []
    n = min(_limit_value(root) or 3, 3)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"_table": spec.fqn}
        for col in spec.columns:
            if col == "USER_ID":
                row[col] = ctx.snowflake_user
            elif col.endswith(("_TS", "_AT")):
                row[col] = f"2026-11-{16 - i:02d}T08:00:00Z"
            elif col in ("VALUE", "AMOUNT"):
                row[col] = round(71.5 + 6.25 * i, 2)
            else:
                row[col] = f"{col.lower()}-{i + 1:03d}"
        rows.append(row)
    return rows


def extractive_synthesizer(query: str, passages: list[Passage]) -> tuple[str, list[Citation]]:
    """Builds a draft that quotes the first 40 characters of the strongest
    evidence, balancing documents and SQL rows when both are present."""
    ranked = sorted(passages, key=lambda p: p.score, reverse=True)
    docs = [p for p in ranked if not p.source.startswith("snowflake://")]
    rows = [p for p in ranked if p.source.startswith("snowflake://")]
    top = docs[: 2 if rows else 3] + rows[: 2 if docs else 3]
    citations = [Citation.from_passage(p) for p in top]
    lines = [f'- "{c.snippet}" [{c.passage_id}]' for c in citations]
    draft = f"Findings for: {query}\n" + "\n".join(lines)
    return draft, citations


@dataclass(frozen=True)
class AgentDeps:
    catalog: Catalog = field(default_factory=lambda: DEFAULT_CATALOG)
    router: RouterFn = keyword_router
    retriever: Retriever = keyword_retriever
    sql_generator: SQLGenerator = template_sql_generator
    executor: SQLExecutor = simulated_executor
    synthesizer: Synthesizer = extractive_synthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(node: str, event_type: str, message: str, status: GateStatus | None = None, **detail: Any) -> AuditEvent:
    return AuditEvent(node=node, event_type=event_type, status=status, message=message, detail=detail)  # type: ignore[arg-type]


def _refuse(node: str, reason: str, event_type: str = "gate", **detail: Any) -> dict[str, Any]:
    return {
        "gate_status": GateStatus.REFUSED,
        "refusal_reason": f"[{node}] {reason}",
        "audit_telemetry": [_event(node, event_type, reason, GateStatus.REFUSED, **detail)],
    }


def _passed(state: AgentState) -> bool:
    return state.get("gate_status") == GateStatus.PASSED


def _ctx(state: AgentState) -> AuthContext:
    return load_auth_context(state.get("auth_context"))


def _rows_to_passages(rows: list[dict[str, Any]]) -> list[Passage]:
    """SQL rows become evidence passages so the citation gate covers them too."""
    return [
        Passage(
            passage_id=f"row-{i + 1}",
            source=f"snowflake://{row.get('_table', 'result')}",
            text=json.dumps({k: v for k, v in row.items() if k != "_table"}, sort_keys=True),
            score=1.0,
        )
        for i, row in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def auth_gate(state: AgentState) -> dict[str, Any]:
    try:
        ctx = _ctx(state)
    except AuthError as err:
        return _refuse(N_AUTH, str(err), event_type="security")
    return {
        "gate_status": GateStatus.PASSED,
        "principal": ctx.principal,
        "audit_telemetry": [ctx.to_audit_event(N_AUTH)],
    }


def make_supervisor(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def supervisor_router(state: AgentState) -> dict[str, Any]:
        route = deps.router(state.get("query", ""))
        update: dict[str, Any] = {
            "route": route,
            "audit_telemetry": [_event(N_SUPERVISOR, "route", f"Routed to {route.value}", route=route.value)],
        }
        if route is Route.REFUSE:
            update |= {"gate_status": GateStatus.REFUSED, "refusal_reason": f"[{N_SUPERVISOR}] Query is empty or unroutable."}
        return update

    return supervisor_router


MAX_PASSAGES = 4


def _retrieve(state: AgentState, deps: AgentDeps, node: str) -> dict[str, Any]:
    ctx = _ctx(state)
    try:
        found = deps.retriever(state["query"])
    except Exception as err:  # noqa: BLE001 - a retriever outage is a refusal, not a crash
        logger.exception("retriever.failed node=%s", node)
        return _refuse(node, f"Evidence retrieval failed ({type(err).__name__}).")
    # A passage with no domain cannot be entitlement-checked, so it is not shown to anyone.
    entitled = [p for p in found if p.domain is not None and p.domain in ctx.allowed_domains]
    passages = sorted(entitled, key=lambda p: p.score, reverse=True)[:MAX_PASSAGES]
    return {
        "gate_status": GateStatus.PASSED,
        "raw_passages": passages,
        "confidence_scores": [p.score for p in passages],
        "audit_telemetry": [
            _event(node, "tool_call", f"Retrieved {len(passages)} passage(s)",
                   retrieved=len(found), dropped_by_entitlement=len(found) - len(passages))
        ],
    }


def _generate_sql(state: AgentState, deps: AgentDeps, node: str, route: Route) -> dict[str, Any]:
    ctx = _ctx(state)
    sql = deps.sql_generator(state["query"], route, ctx.allowed_domains)
    payload = ToolPayload(tool_name=TOOL_SNOWFLAKE_QUERY, arguments={"sql": sql}, principal=ctx.principal)
    return {
        "generated_sql": sql,
        "tool_payload": payload,
        "audit_telemetry": [_event(node, "tool_call", "Generated SQL", call_id=payload.call_id, sql=sql)],
    }


def make_rag_retriever(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def rag_retriever(state: AgentState) -> dict[str, Any]:
        return _retrieve(state, deps, N_RAG)

    return rag_retriever


def make_text_to_sql(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def text_to_sql(state: AgentState) -> dict[str, Any]:
        return _generate_sql(state, deps, N_SQL, Route.TEXT_TO_SQL)

    return text_to_sql


def make_diagnostic_reasoner(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    """Gathers both kinds of evidence: documents (thin-set gated) and a
    telemetry query (SQL gated). The synthesizer reasons over the union."""

    def diagnostic_reasoner(state: AgentState) -> dict[str, Any]:
        update = _retrieve(state, deps, N_DIAG)
        if update["gate_status"] != GateStatus.PASSED:
            return update
        sql_update = _generate_sql(state, deps, N_DIAG, Route.DIAGNOSTIC)
        events = update["audit_telemetry"] + sql_update["audit_telemetry"]
        return update | sql_update | {"audit_telemetry": events}

    return diagnostic_reasoner


def make_authz_gate(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def authz_gate(state: AgentState) -> dict[str, Any]:
        return _run_gate(
            state, GATE_AUTHZ,
            lambda: authorize_sql(state.get("generated_sql") or "", _ctx(state), deps.catalog),
        )

    return authz_gate


def make_sql_executor(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def sql_executor(state: AgentState) -> dict[str, Any]:
        try:
            ctx = _ctx(state)
            secured_sql = apply_row_level_security(state["generated_sql"] or "", ctx, deps.catalog)
            rows = deps.executor(secured_sql, ctx, deps.catalog)
        except Exception as err:  # noqa: BLE001 - execution errors never leak partial data
            # Warehouse messages can name objects the caller may not know exist: log them, do not return them.
            logger.exception("executor.failed")
            return _refuse(N_EXEC, f"Execution failed ({type(err).__name__}).")
        if not rows and state.get("route") == Route.TEXT_TO_SQL:
            return _refuse(N_EXEC, "Query returned no rows; there is no evidence to answer from.")
        row_passages = _rows_to_passages(rows)
        passages = list(state.get("raw_passages", [])) + row_passages
        return {
            "generated_sql": secured_sql,
            "execution_results": rows,
            "raw_passages": passages,
            "confidence_scores": [p.score for p in passages],
            "gate_status": GateStatus.PASSED,
            "audit_telemetry": [
                _event(N_EXEC, "tool_call", f"Executed query, {len(rows)} row(s)",
                       snowflake_user=ctx.snowflake_user, snowflake_role=ctx.snowflake_role,
                       rls_applied=secured_sql != state["generated_sql"], sql=secured_sql)
            ],
        }

    return sql_executor


def make_synthesizer(deps: AgentDeps) -> Callable[[AgentState], dict[str, Any]]:
    def synthesizer(state: AgentState) -> dict[str, Any]:
        draft, citations = deps.synthesizer(state["query"], state.get("raw_passages", []))
        return {
            "draft_response": draft,
            "citations": citations,
            "audit_telemetry": [_event(N_SYNTH, "output", "Drafted response", citations=len(citations))],
        }

    return synthesizer


def make_hitl_review(wait_for_human: bool) -> Callable[[AgentState], dict[str, Any]]:
    def hitl_review(state: AgentState) -> dict[str, Any]:
        ticket = f"HITL-{state.get('trace_id', '')[:8].upper()}"
        if not wait_for_human:
            return {
                "gate_status": GateStatus.HITL_REVIEW,
                "final_response": (
                    f"This answer was withheld pending human review (ticket {ticket}). "
                    f"Reason: {state.get('refusal_reason')}"
                ),
                "audit_telemetry": [_event(N_HITL, "output", "Queued for human review", GateStatus.HITL_REVIEW, ticket=ticket)],
            }
        # Pauses the run; resume with Command(resume={"approved": bool, "reviewer": str}).
        decision = interrupt(
            {
                "ticket": ticket,
                "query": state.get("query"),
                "draft_response": state.get("draft_response"),
                "reason": state.get("refusal_reason"),
            }
        )
        decision = decision if isinstance(decision, dict) else {}
        reviewer = str(decision.get("reviewer") or "").strip()
        if decision.get("approved") is True and reviewer:
            return {
                "gate_status": GateStatus.PASSED,
                "refusal_reason": None,
                "audit_telemetry": [_event(N_HITL, "gate", "Human reviewer approved release", GateStatus.PASSED, ticket=ticket, reviewer=reviewer)],
            }
        return _refuse(N_HITL, "Human reviewer rejected the draft.", ticket=ticket, reviewer=reviewer or "unknown")

    return hitl_review


def respond(state: AgentState) -> dict[str, Any]:
    return {
        "gate_status": GateStatus.PASSED,
        "final_response": state.get("draft_response"),
        "audit_telemetry": [_event(N_RESPOND, "output", "Response released", GateStatus.PASSED)],
    }


def refusal(state: AgentState) -> dict[str, Any]:
    reason = state.get("refusal_reason") or "A validation gate did not pass."
    return {
        "gate_status": GateStatus.REFUSED,
        "draft_response": None,
        "final_response": f"I cannot answer this request. {reason}",
        "audit_telemetry": [_event(N_REFUSAL, "output", "Request refused", GateStatus.REFUSED, reason=reason)],
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def _pass_or_refuse(next_node: str) -> Callable[[AgentState], str]:
    return lambda state: next_node if _passed(state) else N_REFUSAL


def _after_supervisor(state: AgentState) -> str:
    route = state.get("route")
    return route.value if route in (Route.RAG, Route.TEXT_TO_SQL, Route.DIAGNOSTIC) else N_REFUSAL


def _after_thin_set(state: AgentState) -> str:
    if not _passed(state):
        return N_REFUSAL
    return N_AST if state.get("route") == Route.DIAGNOSTIC else N_SYNTH


def _after_citation(state: AgentState) -> str:
    if _passed(state):
        return N_RESPOND
    try:
        is_user = _ctx(state).auth_path is AuthPath.USER_OBO
    except AuthError:
        return N_REFUSAL
    return N_HITL if is_user else N_REFUSAL


def _after_hitl(state: AgentState) -> str:
    status = state.get("gate_status")
    if status == GateStatus.PASSED:
        return N_RESPOND
    return END if status == GateStatus.HITL_REVIEW else N_REFUSAL


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    deps: AgentDeps | None = None, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile the workflow. With a checkpointer, `hitl_review` pauses the run
    for a human decision; without one it ends the run in HITL_REVIEW status."""
    deps = deps or AgentDeps()
    g = StateGraph(AgentState)

    g.add_node(N_AUTH, auth_gate)
    g.add_node(N_SUPERVISOR, make_supervisor(deps))
    g.add_node(N_RAG, make_rag_retriever(deps))
    g.add_node(N_SQL, make_text_to_sql(deps))
    g.add_node(N_DIAG, make_diagnostic_reasoner(deps))
    g.add_node(N_THIN, thin_set_gate_node)
    g.add_node(N_AST, partial(sql_ast_gate_node, catalog=deps.catalog))
    g.add_node(N_PAYLOAD, partial(payload_safety_gate_node, catalog=deps.catalog))
    g.add_node(N_AUTHZ, make_authz_gate(deps))
    g.add_node(N_EXEC, make_sql_executor(deps))
    g.add_node(N_SYNTH, make_synthesizer(deps))
    g.add_node(N_CITE, citation_gate_node)
    g.add_node(N_HITL, make_hitl_review(wait_for_human=checkpointer is not None))
    g.add_node(N_RESPOND, respond)
    g.add_node(N_REFUSAL, refusal)

    g.add_edge(START, N_AUTH)
    g.add_conditional_edges(N_AUTH, _pass_or_refuse(N_SUPERVISOR), [N_SUPERVISOR, N_REFUSAL])
    g.add_conditional_edges(N_SUPERVISOR, _after_supervisor, [N_RAG, N_SQL, N_DIAG, N_REFUSAL])

    g.add_conditional_edges(N_RAG, _pass_or_refuse(N_THIN), [N_THIN, N_REFUSAL])
    g.add_conditional_edges(N_DIAG, _pass_or_refuse(N_THIN), [N_THIN, N_REFUSAL])
    g.add_conditional_edges(N_THIN, _after_thin_set, [N_SYNTH, N_AST, N_REFUSAL])

    g.add_edge(N_SQL, N_AST)
    g.add_conditional_edges(N_AST, _pass_or_refuse(N_PAYLOAD), [N_PAYLOAD, N_REFUSAL])
    g.add_conditional_edges(N_PAYLOAD, _pass_or_refuse(N_AUTHZ), [N_AUTHZ, N_REFUSAL])
    g.add_conditional_edges(N_AUTHZ, _pass_or_refuse(N_EXEC), [N_EXEC, N_REFUSAL])
    g.add_conditional_edges(N_EXEC, _pass_or_refuse(N_SYNTH), [N_SYNTH, N_REFUSAL])

    g.add_edge(N_SYNTH, N_CITE)
    g.add_conditional_edges(N_CITE, _after_citation, [N_RESPOND, N_HITL, N_REFUSAL])
    g.add_conditional_edges(N_HITL, _after_hitl, [N_RESPOND, N_REFUSAL, END])

    g.add_edge(N_RESPOND, END)
    g.add_edge(N_REFUSAL, END)
    return g.compile(checkpointer=checkpointer)


def run_query(
    query: str,
    auth_context: AuthContext | None,
    graph: CompiledStateGraph | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run one request through the workflow and return the final state."""
    state = new_state(query)
    state["auth_context"] = auth_context.model_dump(mode="json") if auth_context else {}
    config = {"configurable": {"thread_id": thread_id or state["trace_id"]}}
    return (graph or build_graph()).invoke(state, config=config)


if __name__ == "__main__":
    from auth import IDP, authenticate_user

    operator_ctx = authenticate_user(
        IDP.issue_user_token("dana.ops@contoso.com", "oid-001", ["Lakehouse.Operator"])
    )
    app = build_graph()
    for q in (
        "What is the bearing temperature alarm limit for pump P-101?",
        "List the latest telemetry readings",
        "Why did the P-101 pump bearing fail?",
        "sql: DROP TABLE SCADA.TELEMETRY_001",
        "What is the capital of France?",
    ):
        out = run_query(q, operator_ctx, app)
        print(f"\n> {q}\n[{out['gate_status']}] {out['final_response']}")
