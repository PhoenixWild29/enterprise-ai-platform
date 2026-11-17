"""platform_tools.py - The platform's gated tools, independent of any agent framework.

These are the only ways an agent can touch data. Each tool runs the same
fail-closed gates as the LangGraph workflow and writes what it did to a
per-request ledger. The Strands agent (strands_agent.py) and the MCP server
(mcp_server.py) both expose exactly these functions, so a tool behaves the
same whichever runtime calls it.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from anomaly import robust_z_anomalies
from auth import AuthContext, apply_row_level_security, authorize_sql
from gates import (
    GATE_PAYLOAD,
    GATE_SQL_AST,
    GATE_THIN_SET,
    TOOL_SNOWFLAKE_QUERY,
    SQLSecurityError,
    payload_safety_gate,
    sql_ast_gate,
    thin_set_refusal_gate,
)
from graph import MAX_PASSAGES, AgentDeps, _rows_to_passages
from llm_agents import as_data, describe_table, select_schema_context
from schemas import AuditEvent, GateResult, GateStatus, Passage

logger = logging.getLogger("platform.tools")

MAX_ROWS_SHOWN = 20
MAX_ANOMALIES_CITED = 6
REFUSED = "REFUSED"


@dataclass
class RunLedger:
    """Everything one request retrieved, ran and decided. The final answer
    may only cite evidence recorded here."""

    query: str
    ctx: AuthContext
    trace_id: str
    passages: list[Passage] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    executed_sql: list[str] = field(default_factory=list)
    audit: list[AuditEvent] = field(default_factory=list)
    halted: str | None = None  # set by a security violation: the run ends REFUSED whatever happens next
    tool_calls: int = 0

    def record(self, node: str, event_type: str, message: str, status: GateStatus | None = None, **detail: Any) -> None:
        self.audit.append(AuditEvent(node=node, event_type=event_type, status=status, message=message, detail=detail))  # type: ignore[arg-type]

    def record_gate(self, result: GateResult) -> None:
        self.record(result.gate, "gate", "; ".join(result.reasons) or "passed", result.status, **result.details)

    def add_passages(self, passages: list[Passage]) -> None:
        known = {p.passage_id for p in self.passages}
        self.passages.extend(p for p in passages if p.passage_id not in known)


class PlatformTools:
    def __init__(self, deps: AgentDeps, ledger: RunLedger) -> None:
        self._deps = deps
        self._ledger = ledger

    # -- discovery ----------------------------------------------------------

    def find_tables(self, question: str) -> str:
        """Tables relevant to a question that the caller is entitled to query."""
        ledger = self._ledger
        tables = select_schema_context(question, self._deps.catalog, allowed_domains=ledger.ctx.allowed_domains)
        ledger.record("find_tables", "tool_call", f"{len(tables)} table(s) offered", tables=[t.fqn for t in tables])
        if not tables:
            return "No governed tables match that question for your role."
        return "\n".join(f"- {describe_table(t)}" for t in tables)

    # -- documents ----------------------------------------------------------

    def search_manuals(self, query: str) -> str:
        """Retrieve passages from manuals and procedures. Returns nothing
        usable unless the evidence clears the thin-set gate."""
        ledger = self._ledger
        try:
            found = self._deps.retriever(query)
        except Exception as err:  # noqa: BLE001 - an outage is a refusal, not a crash
            logger.exception("retriever.failed")
            ledger.record("search_manuals", "tool_call", f"Retrieval failed ({type(err).__name__})", GateStatus.REFUSED)
            return f"{REFUSED}: evidence retrieval is unavailable."
        entitled = [p for p in found if p.domain is not None and p.domain in ledger.ctx.allowed_domains]
        passages = sorted(entitled, key=lambda p: p.score, reverse=True)[:MAX_PASSAGES]
        gate = thin_set_refusal_gate(passages, [p.score for p in passages])
        ledger.record_gate(gate)
        if not gate.passed:
            return f"{REFUSED} [{GATE_THIN_SET}]: " + " ".join(gate.reasons) + " Do not answer from memory."
        ledger.add_passages(passages)
        return "\n".join(f'<passage id="{as_data(p.passage_id)}">{as_data(p.text)}</passage>' for p in passages)

    # -- warehouse ----------------------------------------------------------

    def run_sql(self, sql: str) -> str:
        """Run one read-only SELECT through every SQL gate, as the caller."""
        ledger, catalog = self._ledger, self._deps.catalog
        if ledger.halted:
            return f"{REFUSED}: {ledger.halted}"
        try:
            ast = sql_ast_gate(sql, catalog)
            ledger.record_gate(ast)
            if not ast.passed:
                return f"{REFUSED} [{GATE_SQL_AST}]: " + " ".join(ast.reasons)
            payload = payload_safety_gate(
                {"tool_name": TOOL_SNOWFLAKE_QUERY, "arguments": {"sql": sql}, "principal": ledger.ctx.principal}, catalog
            )
            ledger.record_gate(payload)
            if not payload.passed:
                return f"{REFUSED} [{GATE_PAYLOAD}]: " + " ".join(payload.reasons)
            authz = authorize_sql(sql, ledger.ctx, catalog)
            ledger.record_gate(authz)
            if not authz.passed:
                return f"{REFUSED} [authorization]: " + " ".join(authz.reasons)
            secured = apply_row_level_security(sql, ledger.ctx, catalog)
        except SQLSecurityError as err:
            ledger.halted = f"SECURITY: {err}"
            ledger.record(GATE_SQL_AST, "security", ledger.halted, GateStatus.REFUSED, security_violation=True)
            return f"{REFUSED}: {ledger.halted} This run is over."
        except Exception as err:  # noqa: BLE001 - a crashing gate never passes
            ledger.record(GATE_SQL_AST, "gate", f"Gate error ({type(err).__name__})", GateStatus.REFUSED)
            return f"{REFUSED}: the query could not be validated."

        try:
            rows = self._deps.executor(secured, ledger.ctx, catalog)
        except Exception as err:  # noqa: BLE001 - warehouse messages stay in the server log
            logger.exception("executor.failed")
            ledger.record("sql_executor", "tool_call", f"Execution failed ({type(err).__name__})", GateStatus.REFUSED)
            return f"{REFUSED}: execution failed ({type(err).__name__})."

        offset = len(ledger.rows)
        evidence = [
            p.model_copy(update={"passage_id": f"row-{offset + i + 1}"}) for i, p in enumerate(_rows_to_passages(rows))
        ]
        ledger.rows.extend(rows)
        ledger.executed_sql.append(secured)
        ledger.add_passages(evidence)
        ledger.record(
            "sql_executor", "tool_call", f"Executed query, {len(rows)} row(s)", GateStatus.PASSED,
            sql=secured, rls_applied=secured != sql, snowflake_role=ledger.ctx.snowflake_role,
        )
        if not rows:
            return "The query ran and returned no rows."
        shown = evidence[:MAX_ROWS_SHOWN]
        more = f"\n({len(evidence) - len(shown)} more rows recorded as evidence.)" if len(evidence) > len(shown) else ""
        return "\n".join(f'<passage id="{p.passage_id}">{as_data(p.text)}</passage>' for p in shown) + more

    def flag_anomalies(self) -> str:
        """Run classical anomaly detection over the sensor readings already
        fetched in this request. The result is evidence that can be cited."""
        ledger = self._ledger
        readings = [r for r in ledger.rows if "VALUE" in r]
        if not readings:
            return "No sensor readings have been fetched yet. Run a telemetry query first."
        report = robust_z_anomalies(readings)
        report["anomaly_count"] = len(report["anomalies"])
        report["anomalies"] = report["anomalies"][:MAX_ANOMALIES_CITED]  # keep the evidence passage small enough to release whole
        scan_id = f"anomaly-scan-{sum(p.passage_id.startswith('anomaly-scan-') for p in ledger.passages) + 1}"
        passage = Passage(passage_id=scan_id, source="analysis://robust-z", text=json.dumps(report, sort_keys=True, default=str), score=1.0)
        ledger.add_passages([passage])
        ledger.record("flag_anomalies", "tool_call", f"{report['anomaly_count']} anomalous reading(s)", GateStatus.PASSED,
                      readings_scanned=report["readings_scanned"])
        return f'<passage id="{scan_id}">{as_data(passage.text)}</passage>'
