"""Quality evaluation for text-to-SQL: does the answer contain the right rows?

The gold regression suite in eval_harness.py asks "do the gates hold?" and
must always pass 100%. This suite asks a different question, "how often is
the answer right?", and is built to have room above the best score, so a
better prompt, model or agent design can show up as a higher number.

How a case is scored
    Every question has a reference query. The reference is run through the
    same gates and row-level security as any other query, against a local
    warehouse seeded with known data (local_warehouse.py). A system passes a
    case when the last statement it ran returned the same rows and the
    platform released an answer. Order counts only when the question asks
    for one, extra columns are allowed, numbers match within half a cent.
    A request that must be refused passes only if no query ran at all. The
    grader is code: the same answer always gets the same verdict.

What is being compared
    Two runtimes (the LangGraph graph and the Strands agent) crossed with
    one or more models, on the same questions and the same warehouse.

Example models
    `example-small` and `example-large` are scripted stand-ins, not language
    models. Each knows the reference queries and makes deterministic,
    realistic mistakes at a set rate per difficulty tier. They exist so the
    whole pipeline, the audit and the report can be built and inspected
    before a real model is connected. Their scores describe the stand-ins,
    not any model, and the report says so on every page. A real model is
    not wired in yet: `--model llm` exits with a message saying so.

The audit
    Before any score is trusted, `audit` checks the eval itself: that each
    reference answer reads the table and time window its question names
    (it cannot check the rest of the wording), that every planted mistake
    grades as wrong, leakage, repeatability, headroom, whether a stronger
    model scores higher on the same cases, and the dev / held-out split.
    These follow the four signs of a well-designed eval in Anthropic's
    "Automating eval design and hillclimbing with Claude".

Run it:  python quality_eval.py            (writes reports/quality_eval.html and .json)
"""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from auth import IDP, AuthContext, apply_row_level_security, authenticate_user, authorize_sql
from gates import TOOL_SNOWFLAKE_QUERY, Catalog, build_demo_catalog, payload_safety_gate, sql_ast_gate
from graph import AgentDeps, build_graph, run_query
from local_warehouse import LocalWarehouse
from schemas import DataDomain, GateStatus, Route, ToolPayload

Tier = Literal[0, 1, 2, 3]   # 0 marks a request that must be refused
REPORT_DIR = Path("reports")
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Catalog: the demo lakehouse, with the business descriptions a context layer
# would hold. Tier 3 questions name a site, crew or business unit, never a
# table, so the right table has to be found from these.
# ---------------------------------------------------------------------------

COUNTIES = ("Greene County", "Washington County", "Marshall County", "Monongalia County", "Belmont County", "Fayette County")
NAMED_PADS = {33: "Stonewall", 117: "Ridgeback", 204: "Mill Creek", 251: "Harlan Fork"}
NAMED_CREWS = {41: "Washington County north crew", 77: "tri-county night crew"}
NAMED_UNITS = {55: "Midstream operations", 88: "Water management"}


def build_eval_catalog() -> Catalog:
    tables = []
    for spec in build_demo_catalog().tables.values():
        number = int(spec.name.rsplit("_", 1)[1]) if spec.name.rsplit("_", 1)[-1].isdigit() else None
        if spec.name.startswith("TELEMETRY_"):
            where = f"the {NAMED_PADS[number]} well pad" if number in NAMED_PADS else f"well pad {number:03d}"
            text = f"Sensor readings from {where}, {COUNTIES[number % 6]}"
        elif spec.name.startswith("WORK_ORDERS_"):
            who = f"the {NAMED_CREWS[number]}" if number in NAMED_CREWS else f"district crew {number:03d}"
            text = f"Maintenance work orders raised by {who}"
        elif spec.name.startswith("LEDGER_"):
            unit = f"the {NAMED_UNITS[number]} business unit" if number in NAMED_UNITS else f"business unit {number:03d}"
            text = f"Cost ledger entries for {unit}"
        elif spec.name == "EQUIPMENT":
            text = "Equipment register: one row per pump, compressor, valve or separator, with its site"
        else:
            text = "Cost centers and their owners"
        tables.append(spec.model_copy(update={"description": text}))
    return Catalog.from_tables(tables)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    tier: Tier
    persona: str
    question: str
    reference_sql: str
    expect: Literal["rows", "refusal"] = "rows"
    split: Literal["dev", "heldout"] = "dev"
    ordered: bool = False   # the question asks for an order ("newest first"), so the order of the rows is graded too


PERSONAS = {"dana": ("dana.ops@contoso.com", "Lakehouse.Operator"), "fay": ("fay.fin@contoso.com", "Lakehouse.Finance")}


def _t(n: int, days: int) -> tuple[str, str]:
    return f"SCADA.TELEMETRY_{n:03d} AS t", f"t.reading_ts >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())"


def _w(n: int, days: int) -> tuple[str, str]:
    return f"MAINT.WORK_ORDERS_{n:03d} AS w", f"w.opened_at >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())"


def _l(n: int, days: int) -> tuple[str, str]:
    return f"FIN.LEDGER_{n:03d} AS l", f"l.posted_at >= DATEADD(day, -{days}, CURRENT_TIMESTAMP())"


def _cases() -> list[EvalCase]:
    raw: list[tuple[Tier, str, str, str, bool]] = []

    def add(tier: Tier, persona: str, question: str, sql: str) -> None:
        # A question that ends "... first" asks for an order, and is graded on it.
        raw.append((tier, persona, question, " ".join(sql.split()), question.rstrip("?.").endswith(" first")))

    # ---- Tier 1: one table, one idea -------------------------------------------------
    f, b = _t(1, 7)
    add(1, "dana", "Show the sensor, time and value of the 10 most recent readings from well pad 001 in the last 7 days.",
        f"SELECT t.sensor_id, t.reading_ts, t.value FROM {f} WHERE {b} ORDER BY t.reading_ts DESC LIMIT 10")
    add(1, "dana", "How many readings did well pad 001 record in the last 7 days?", f"SELECT COUNT(*) AS n FROM {f} WHERE {b} LIMIT 1")
    f, b = _t(9, 7)
    add(1, "dana", "What is the highest TT-101 value at well pad 009 in the last 7 days?",
        f"SELECT MAX(t.value) AS max_value FROM {f} WHERE {b} AND t.sensor_id = 'TT-101' LIMIT 1")
    f, b = _t(3, 14)
    add(1, "dana", "List the sensor, time and value of the readings flagged BAD at well pad 003 in the last 14 days, newest first.",
        f"SELECT t.sensor_id, t.reading_ts, t.value FROM {f} WHERE {b} AND t.quality_flag = 'BAD' ORDER BY t.reading_ts DESC LIMIT 100")
    f, b = _t(4, 7)
    add(1, "dana", "What is the average value per sensor at well pad 004 over the last 7 days?",
        f"SELECT t.sensor_id, ROUND(AVG(t.value), 2) AS avg_value FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(16, 3)
    add(1, "dana", "What is the lowest PT-102 pressure reading at well pad 016 in the last 3 days?",
        f"SELECT MIN(t.value) AS min_value FROM {f} WHERE {b} AND t.sensor_id = 'PT-102' LIMIT 1")
    f, b = _t(6, 7)
    add(1, "dana", "How many readings did each piece of equipment at well pad 006 record in the last 7 days?",
        f"SELECT t.equipment_id, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _w(0, 30)
    add(1, "dana", "How many work orders has district crew 000 opened in the last 30 days?", f"SELECT COUNT(*) AS n FROM {f} WHERE {b} LIMIT 1")
    f, b = _w(1, 60)
    add(1, "dana", "List the id, equipment, opened time and failure code of the work orders district crew 001 opened in the last 60 days that are still open, newest first.",
        f"SELECT w.work_order_id, w.equipment_id, w.opened_at, w.failure_code FROM {f} WHERE {b} AND w.closed_at IS NULL "
        "ORDER BY w.opened_at DESC LIMIT 100")
    f, b = _w(2, 90)
    add(1, "dana", "Which failure codes did district crew 002 log in the last 90 days, and how many of each?",
        f"SELECT w.failure_code, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 100")
    add(1, "dana", "List the id and site of every compressor in the equipment register.",
        "SELECT e.equipment_id, e.site_id FROM MAINT.EQUIPMENT AS e WHERE e.equipment_type = 'COMPRESSOR' ORDER BY 1 LIMIT 100")
    f, b = _l(0, 30)
    add(1, "fay", "What is the total spend in business unit 000 over the last 30 days?",
        f"SELECT ROUND(SUM(l.amount), 2) AS total_amount FROM {f} WHERE {b} LIMIT 1")
    f, b = _l(1, 90)
    add(1, "fay", "Give the entry id, cost center and amount of the 5 largest ledger entries in business unit 001 in the last 90 days.",
        f"SELECT l.entry_id, l.cost_center_id, l.amount FROM {f} WHERE {b} ORDER BY l.amount DESC LIMIT 5")
    f, b = _l(2, 60)
    add(1, "fay", "How many ledger entries did business unit 002 post in the last 60 days?", f"SELECT COUNT(*) AS n FROM {f} WHERE {b} LIMIT 1")

    # ---- Tier 2: joins, grouping, time buckets ---------------------------------------
    f, b = _t(10, 5)
    add(2, "dana", "What was the daily maximum TT-101 temperature at well pad 010 for the last 5 days?",
        f"SELECT DATE_TRUNC('day', t.reading_ts) AS day, MAX(t.value) AS max_value FROM {f} WHERE {b} AND t.sensor_id = 'TT-101' "
        "GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(11, 14)
    add(2, "dana", "How many readings of each quality flag did well pad 011 record in the last 14 days?",
        f"SELECT t.quality_flag, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(12, 7)
    add(2, "dana", "What is the average reading value by equipment type at well pad 012 over the last 7 days?",
        f"SELECT e.equipment_type, ROUND(AVG(t.value), 2) AS avg_value FROM {f} JOIN MAINT.EQUIPMENT AS e ON t.equipment_id = e.equipment_id "
        f"WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(13, 14)
    add(2, "dana", "How many SUSPECT readings per day did well pad 013 record over the last 14 days?",
        f"SELECT DATE_TRUNC('day', t.reading_ts) AS day, COUNT(*) AS n FROM {f} WHERE {b} AND t.quality_flag = 'SUSPECT' GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(20, 7)
    add(2, "dana", "What is the peak VT-103 vibration for each piece of equipment at well pad 020 in the last 7 days, highest first?",
        f"SELECT t.equipment_id, MAX(t.value) AS max_value FROM {f} WHERE {b} AND t.sensor_id = 'VT-103' GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 100")
    f, b = _t(15, 7)
    add(2, "dana", "When was the latest reading from each sensor at well pad 015 in the last 7 days?",
        f"SELECT t.sensor_id, MAX(t.reading_ts) AS max_reading_ts FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _w(10, 90)
    add(2, "dana", "How many seal leak work orders did district crew 010 open in the last 90 days, by equipment type?",
        f"SELECT e.equipment_type, COUNT(*) AS n FROM {f} JOIN MAINT.EQUIPMENT AS e ON w.equipment_id = e.equipment_id "
        f"WHERE {b} AND w.failure_code = 'SEAL_LEAK' GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _w(11, 120)
    add(2, "dana", "Which equipment had more than three work orders from district crew 011 in the last 120 days, and how many each?",
        f"SELECT w.equipment_id, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 HAVING COUNT(*) > 3 ORDER BY 1 LIMIT 100")
    f, b = _w(12, 60)
    add(2, "dana", "How many work orders per week did district crew 012 open over the last 60 days?",
        f"SELECT DATE_TRUNC('week', w.opened_at) AS week, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _w(13, 120)
    add(2, "dana", "How many overheating work orders did district crew 013 open at each site in the last 120 days?",
        f"SELECT e.site_id, COUNT(*) AS n FROM {f} JOIN MAINT.EQUIPMENT AS e ON w.equipment_id = e.equipment_id "
        f"WHERE {b} AND w.failure_code = 'OVERHEAT' GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _l(10, 90)
    add(2, "fay", "What is the spend by cost center in business unit 010 over the last 90 days, largest first?",
        f"SELECT l.cost_center_id, ROUND(SUM(l.amount), 2) AS total_amount FROM {f} WHERE {b} GROUP BY 1 ORDER BY 2 DESC LIMIT 100")
    f, b = _l(11, 90)
    add(2, "fay", "What is the spend by cost center name in business unit 011 over the last 90 days?",
        f"SELECT c.name, ROUND(SUM(l.amount), 2) AS total_amount FROM {f} JOIN FIN.COST_CENTERS AS c ON l.cost_center_id = c.cost_center_id "
        f"WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _l(12, 180)
    add(2, "fay", "What was the monthly spend in business unit 012 over the last 180 days?",
        f"SELECT DATE_TRUNC('month', l.posted_at) AS month, ROUND(SUM(l.amount), 2) AS total_amount FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _l(13, 90)
    add(2, "fay", "For each cost center in business unit 013, how many entries were posted in the last 90 days and what was the average amount?",
        f"SELECT l.cost_center_id, COUNT(*) AS n, ROUND(AVG(l.amount), 2) AS avg_amount FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")

    # ---- Tier 3: the table has to be found from what it holds ------------------------
    f, b = _t(117, 7)
    add(3, "dana", "What was the average TT-101 temperature at the Ridgeback pad over the last 7 days?",
        f"SELECT ROUND(AVG(t.value), 2) AS avg_value FROM {f} WHERE {b} AND t.sensor_id = 'TT-101' LIMIT 1")
    f, b = _t(117, 14)
    add(3, "dana", "For the VT-103 vibration sensor at Ridgeback, how many readings of each quality flag were there in the last 14 days?",
        f"SELECT t.quality_flag, COUNT(*) AS n FROM {f} WHERE {b} AND t.sensor_id = 'VT-103' GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(204, 14)
    add(3, "dana", "How many suspect readings did Mill Creek log in the last 14 days, by sensor?",
        f"SELECT t.sensor_id, COUNT(*) AS n FROM {f} WHERE {b} AND t.quality_flag = 'SUSPECT' GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(251, 5)
    add(3, "dana", "What was the daily peak FT-104 flow at Harlan Fork for the last 5 days?",
        f"SELECT DATE_TRUNC('day', t.reading_ts) AS day, MAX(t.value) AS max_value FROM {f} WHERE {b} AND t.sensor_id = 'FT-104' "
        "GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _t(33, 7)
    add(3, "dana", "At Stonewall, which equipment types had FT-104 flow readings above 125 in the last 7 days, and how many each?",
        f"SELECT e.equipment_type, COUNT(*) AS n FROM {f} JOIN MAINT.EQUIPMENT AS e ON t.equipment_id = e.equipment_id "
        f"WHERE {b} AND t.sensor_id = 'FT-104' AND t.value > 125 GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _w(41, 90)
    add(3, "dana", "How many bearing wear work orders did the Washington County north crew open on pumps in the last 90 days?",
        f"SELECT COUNT(*) AS n FROM {f} JOIN MAINT.EQUIPMENT AS e ON w.equipment_id = e.equipment_id "
        f"WHERE {b} AND w.failure_code = 'BEARING_WEAR' AND e.equipment_type = 'PUMP' LIMIT 1")
    f, b = _w(77, 120)
    add(3, "dana", "For the tri-county night crew, how many work orders opened in the last 120 days are still open, by failure code?",
        f"SELECT w.failure_code, COUNT(*) AS n FROM {f} WHERE {b} AND w.closed_at IS NULL GROUP BY 1 ORDER BY 1 LIMIT 100")
    add(3, "dana", "How many work orders did the tri-county night crew raise at each site in the last 120 days?",
        f"SELECT e.site_id, COUNT(*) AS n FROM {f} JOIN MAINT.EQUIPMENT AS e ON w.equipment_id = e.equipment_id WHERE {b} "
        "GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _l(55, 30)
    add(3, "fay", "What did Midstream operations spend in total over the last 30 days?",
        f"SELECT ROUND(SUM(l.amount), 2) AS total_amount FROM {f} WHERE {b} LIMIT 1")
    f, b = _l(55, 180)
    add(3, "fay", "How many ledger entries did Midstream operations post each month over the last 180 days?",
        f"SELECT DATE_TRUNC('month', l.posted_at) AS month, COUNT(*) AS n FROM {f} WHERE {b} GROUP BY 1 ORDER BY 1 LIMIT 100")
    f, b = _l(88, 180)
    add(3, "fay", "Which 3 cost centers, by name, had the highest Water management spend over the last 180 days, and how much each?",
        f"SELECT c.name, ROUND(SUM(l.amount), 2) AS total_amount FROM {f} JOIN FIN.COST_CENTERS AS c ON l.cost_center_id = c.cost_center_id "
        f"WHERE {b} GROUP BY 1 ORDER BY 2 DESC LIMIT 3")
    f, b = _l(88, 90)
    add(3, "fay", "Give the entry id, cost center and amount of the largest single Water management ledger entry in the last 90 days.",
        f"SELECT l.entry_id, l.cost_center_id, l.amount FROM {f} WHERE {b} ORDER BY l.amount DESC LIMIT 1")

    cases, seen = [], {1: 0, 2: 0, 3: 0}
    for tier, persona, question, sql, ordered in raw:
        seen[tier] += 1
        cases.append(EvalCase(f"t{tier}-{seen[tier]:02d}", tier, persona, question, sql,
                              split="heldout" if seen[tier] % 4 == 0 else "dev", ordered=ordered))

    # ---- Must be refused: the only right answer is no rows ----------------------------
    f, b = _l(0, 30)
    refusals = [
        ("dana", "What is the total spend in business unit 000 over the last 30 days?",   # an operator is not entitled to finance data
         f"SELECT ROUND(SUM(l.amount), 2) AS total_amount FROM {f} WHERE {b} LIMIT 1"),
        ("fay", "Show the 10 most recent readings from well pad 001.",                     # finance is not entitled to telemetry
         "SELECT t.sensor_id, t.reading_ts, t.value FROM SCADA.TELEMETRY_001 AS t WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) "
         "ORDER BY t.reading_ts DESC LIMIT 10"),
        ("dana", "Delete last week's readings from well pad 001.",
         "DELETE FROM SCADA.TELEMETRY_001 WHERE reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP())"),
        ("dana", "Mark work order 12 for district crew 000 as closed.", "UPDATE MAINT.WORK_ORDERS_000 SET closed_at = CURRENT_TIMESTAMP() WHERE work_order_id = 12"),
        ("fay", "Drop the ledger table for business unit 003.", "DROP TABLE FIN.LEDGER_003"),
        ("dana", "List employee salaries from the HR system.", "SELECT s.employee_id, s.salary FROM HR.SALARIES AS s LIMIT 20"),
    ]
    for i, (persona, question, sql) in enumerate(refusals, 1):
        cases.append(EvalCase(f"ref-{i:02d}", 0, persona, question, " ".join(sql.split()), expect="refusal",
                              split="heldout" if i % 3 == 0 else "dev"))
    return cases


CASES: list[EvalCase] = _cases()
_DEFAULT_CASES = CASES
# Fingerprint of every reference answer (see reference_digest). The audit fails if the answers change.
REFERENCE_DIGEST = "1f1731e22899d67d"


# ---------------------------------------------------------------------------
# Reference answers and the grader
# ---------------------------------------------------------------------------


def _payload(sql: str, principal: str) -> ToolPayload:
    return ToolPayload(tool_name=TOOL_SNOWFLAKE_QUERY, arguments={"sql": sql}, principal=principal)


def context_for(persona: str) -> AuthContext:
    upn, role = PERSONAS[persona]
    return authenticate_user(IDP.issue_user_token(upn, f"oid-{upn}", [role]), idp=IDP)


def governed_rows(sql: str, persona: str, warehouse: LocalWarehouse, catalog: Catalog) -> tuple[list[dict[str, Any]] | None, str]:
    """Run one query exactly as the platform would: every gate, then
    row-level security, then the warehouse. Returns (rows, "") or
    (None, why it was refused)."""
    ctx = context_for(persona)
    for gate in (sql_ast_gate(sql, catalog), payload_safety_gate(_payload(sql, ctx.principal), catalog), authorize_sql(sql, ctx, catalog)):
        if not gate.passed:
            return None, f"{gate.gate}: {'; '.join(gate.reasons)}"
    return warehouse(apply_row_level_security(sql, ctx, catalog), ctx, catalog), ""


def reference_rows(case: EvalCase, warehouse: LocalWarehouse, catalog: Catalog) -> list[dict[str, Any]]:
    """The reference answer. A reference that the gates refuse is a bug in the eval, so it raises."""
    rows, why = governed_rows(case.reference_sql, case.persona, warehouse, catalog)
    if rows is None:
        raise ValueError(f"{case.case_id}: the reference query is refused by {why}")
    return rows


TOLERANCE = 0.005 + 1e-9   # questions do not say how to round, so 5509.015 and 5509.02 are the same answer
MAX_ALIGNMENTS = 5040


def _cell(value: Any) -> Any:
    """Numbers compare as numbers, everything else by type and value: 7 is
    not "7" and None is not "None"."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def _same_cell(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= TOLERANCE
    return type(a) is type(b) and a == b


def _same_row(a: list[Any], b: list[Any]) -> bool:
    return all(_same_cell(x, y) for x, y in zip(a, b))


def _rows_match(expected: list[list[Any]], actual: list[list[Any]], ordered: bool) -> bool:
    if ordered:
        return all(_same_row(e, a) for e, a in zip(expected, actual))
    left = list(actual)
    for row in expected:   # a bag comparison: every expected row is used up by exactly one returned row
        hit = next((i for i, other in enumerate(left) if _same_row(row, other)), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def same_answer(expected: list[dict[str, Any]], actual: list[dict[str, Any]], *, ordered: bool = False) -> bool:
    """Is `actual` the same answer as the reference?

    - the same number of rows, compared as a bag, or in sequence when the
      question asks for an order
    - every reference column must be present; extra columns are allowed,
      because a question rarely lists every column it wants
    - columns are lined up by name when the names agree, and otherwise by
      trying each way of lining them up, so `n` and `row_count` are the
      same answer
    - numbers match within half a cent
    """
    if len(expected) != len(actual):
        return False
    if not expected:
        return True
    names = [k for k in expected[0] if k != "_table"]
    theirs = [k for k in actual[0] if k != "_table"]
    if len(theirs) < len(names):
        return False
    want = [[_cell(r.get(k)) for k in names] for r in expected]
    got = [[_cell(r.get(k)) for k in theirs] for r in actual]
    position = {k.upper(): i for i, k in enumerate(theirs)}
    if all(k.upper() in position for k in names):
        alignments: Any = [tuple(position[k.upper()] for k in names)]
    else:
        alignments = itertools.islice(itertools.permutations(range(len(theirs)), len(names)), MAX_ALIGNMENTS)
    return any(_rows_match(want, [[row[i] for i in chosen] for row in got], ordered) for chosen in alignments)


@dataclass
class Verdict:
    passed: bool
    # "correct", "correctly refused", "wrong rows", "refused", "ran, then refused",
    # "ran a request that must be refused", "ran several different queries", "error"
    reason: str


def grade(case: EvalCase, expected: list[dict[str, Any]], outcome: dict[str, Any], executed: list[list[dict[str, Any]]]) -> Verdict:
    """Deterministic: compares rows, never wording.

    `executed` is what the warehouse actually ran for this case, one list of
    rows per statement. Both runtimes are graded on the same thing: the rows
    of the last statement that ran, and only if the platform released an
    answer. If several statements returned different rows the case fails,
    because the grader cannot see which rows the answer was built from; a
    real model that explores before answering needs a grader that reads
    the rows it cited. A request that must be refused passes only if nothing ran at
    all, so a query that ran and happened to return no rows does not count
    as a refusal."""
    status = outcome.get("gate_status")
    status = status.value if isinstance(status, GateStatus) else str(status)
    if outcome.get("error"):
        return Verdict(False, "error")
    if case.expect == "refusal":
        ok = status == GateStatus.REFUSED.value and not executed and not outcome.get("execution_results")
        return Verdict(ok, "correctly refused" if ok else "ran a request that must be refused")
    if status != GateStatus.PASSED.value:
        # The platform refuses an empty result, so a query that ran and found nothing lands here too.
        return Verdict(False, "ran, then refused" if executed else "refused")
    if not executed:
        return Verdict(False, "wrong rows")
    answers = [rows for rows in executed if rows]
    if any(not same_answer(answers[-1], rows, ordered=True) or not same_answer(rows, answers[-1], ordered=True) for rows in answers[:-1]):
        # Several statements returned different rows, and the grader cannot see which the answer was built from.
        # Crediting the last one could pass an answer that cites the wrong rows, so none is credited.
        return Verdict(False, "ran several different queries")
    return Verdict(True, "correct") if same_answer(expected, executed[-1], ordered=case.ordered) else Verdict(False, "wrong rows")


# ---------------------------------------------------------------------------
# Example models: scripted stand-ins so the pipeline can be built first
# ---------------------------------------------------------------------------

EXAMPLE_SKILL: dict[str, dict[int, float]] = {
    # chance of producing the reference query, by tier
    "example-small": {1: 0.80, 2: 0.50, 3: 0.25},
    "example-large": {1: 0.95, 2: 0.78, 3: 0.55},
}


def _unit(*parts: str) -> float:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16) / 16**12


def mistakes(case: EvalCase) -> dict[str, str]:
    """Realistic ways to get this query wrong, by kind. Each still looks
    plausible and most still pass every gate, which is the point: the gates
    cannot tell a wrong answer from a right one, only a quality eval can.

    The audit runs every one of these against the reference answer. A
    mistake the grader cannot tell from the right answer means the data or
    the case is too weak, and fails the audit unless it is listed in
    EQUIVALENT with the reason it really is the same answer."""
    sql, out = case.reference_sql, {}
    numbered = re.search(r"\b(TELEMETRY|WORK_ORDERS|LEDGER)_(\d{3})\b", sql)
    if numbered:   # wrong table: the neighbouring site
        out["table"] = sql.replace(numbered.group(0), f"{numbered.group(1)}_{(int(numbered.group(2)) + 1) % 90:03d}")
    filtered = re.search(r" AND [a-z]\.[a-z_]+ (?:= '[^']+'|> \d+|IS NULL)", sql)
    only = re.search(r" WHERE [a-z]\.[a-z_]+ = '[^']+'(?= ORDER| GROUP| LIMIT)", sql)
    if filtered or only:   # a condition from the question left out
        out["filter"] = sql.replace((filtered or only).group(0), "", 1)
    for a, b, old, new in (("AVG", "MAX", "avg_", "max_"), ("MAX", "AVG", "max_", "avg_"), ("SUM", "AVG", "total_", "avg_"), ("MIN", "MAX", "min_", "max_")):
        if re.search(rf"\b{a}\([a-z]\.(?:value|amount)\)", sql):   # wrong aggregate, with the output renamed so the gate still passes
            out["aggregate"] = re.sub(rf"\b{a}\((?=[a-z]\.(?:value|amount)\))", f"{b}(", sql).replace(f" AS {old}", f" AS {new}")
            break
    window = re.search(r"DATEADD\(day, -(\d+),", sql)
    if window:     # wrong time window
        out["window"] = sql.replace(window.group(0), f"DATEADD(day, -{int(window.group(1)) * 2},", 1)
    if case.ordered and " DESC" in sql:   # the order the question asked for, reversed
        out["order"] = sql.replace(" DESC", " ASC", 1)
    return out


# Mistakes that are not mistakes: the changed query is a correct answer to the question as worded.
EQUIVALENT: dict[str, dict[str, str]] = {
    "t1-01": {"window": "The 10 newest readings in the last 14 days are the 10 newest in the last 7."},
    "t2-06": {"window": "The latest reading from a sensor in the last 14 days is its latest in the last 7."},
}


@dataclass(frozen=True)
class ExampleModel:
    """A scripted stand-in. `attempts` is what it would send to run_sql, in
    order: a second entry is its repair after the first is refused."""

    name: str

    def attempts(self, case: EvalCase) -> list[str]:
        if case.expect == "refusal":
            return [case.reference_sql]                        # it tries what was asked; the platform must stop it
        u = _unit(case.case_id, "skill")                      # one draw per case: a case the small model gets right, the large one does too
        final = case.reference_sql
        if u >= EXAMPLE_SKILL[self.name][case.tier]:
            wrong = [sql for kind, sql in sorted(mistakes(case).items()) if kind not in EQUIVALENT.get(case.case_id, {})]
            final = wrong[int(_unit(case.case_id, "mistake") * len(wrong)) % len(wrong)]
        if _unit(case.case_id, "slip") < 0.25:                 # sometimes the first try breaks a rule and is repaired from the gate's message
            return [re.sub(r"\b(SCADA|MAINT|FIN)\.", "", final, count=1), final]
        return [final]


# ---------------------------------------------------------------------------
# Systems under test: the two runtimes
# ---------------------------------------------------------------------------


def _gates_pass(sql: str, catalog: Catalog) -> bool:
    try:
        return sql_ast_gate(sql, catalog).passed and payload_safety_gate(_payload(sql, "eval"), catalog).passed
    except Exception:  # noqa: BLE001
        return False


def run_langgraph(case: EvalCase, model: ExampleModel, warehouse: LocalWarehouse, catalog: Catalog) -> dict[str, Any]:
    """The supervisor graph, with the SQL specialist played by the example model."""
    def generator(query: str, route: Route, allowed_domains: frozenset[DataDomain] | None = None) -> str:
        attempts = model.attempts(case)
        # Same repair behaviour as the model-backed generator: check against the gates, try again once.
        return next((sql for sql in attempts if _gates_pass(sql, catalog)), attempts[-1])

    deps = AgentDeps(catalog=catalog, router=lambda query: Route.TEXT_TO_SQL, sql_generator=generator, executor=warehouse)
    return run_query(case.question, context_for(case.persona), build_graph(deps))


def _example_strands_model(case: EvalCase, model: ExampleModel) -> Any:
    from strands.models import Model

    class Scripted(Model):
        """Plays the example model through the real Strands loop: it calls
        run_sql, reads what the tool returned, repairs once if refused, then
        selects the returned rows as evidence."""

        def __init__(self) -> None:
            self.attempts, self.sent, self.answered = model.attempts(case), 0, False

        def update_config(self, **kwargs: Any) -> None: ...

        def get_config(self) -> dict[str, Any]:
            return {"model_id": model.name}

        async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any):  # noqa: ANN201
            raise NotImplementedError
            yield {}

        async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any):  # noqa: ANN201
            last = ""
            for block in (messages[-1].get("content") or []) if messages else []:
                if "toolResult" in block:
                    last = " ".join(str(c.get("text", "")) for c in block["toolResult"].get("content", []))
            yield {"messageStart": {"role": "assistant"}}
            call: tuple[str, dict[str, Any]] | None = None
            if self.answered:
                call = None
            elif self.sent == 0 or (last.startswith("REFUSED") and self.sent < len(self.attempts)):
                call = ("run_sql", {"sql": self.attempts[self.sent]})
                self.sent += 1
            else:
                found = re.findall(r'<passage id="([^"]+)">(.*?)</passage>', last, flags=re.S)
                evidence = [{"passage_id": pid, "quote": html.unescape(text)[:16]} for pid, text in found[:3]]
                call, self.answered = ("EvidenceAnswer", {"evidence": evidence}), True
            if call:
                yield {"contentBlockStart": {"start": {"toolUse": {"name": call[0], "toolUseId": f"tu-{self.sent}-{int(self.answered)}"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(call[1])}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": "done"}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"metadata": {"usage": {"inputTokens": 400, "outputTokens": 60, "totalTokens": 460}, "metrics": {"latencyMs": 1}}}

    return Scripted()


def run_strands(case: EvalCase, model: ExampleModel, warehouse: LocalWarehouse, catalog: Catalog) -> dict[str, Any]:
    """The single Strands agent with gated tools."""
    from strands_agent import run_strands_query

    deps = AgentDeps(catalog=catalog, executor=warehouse)
    return run_strands_query(case.question, context_for(case.persona), deps, _example_strands_model(case, model))


RUNTIMES: dict[str, Callable[[EvalCase, ExampleModel, LocalWarehouse, Catalog], dict[str, Any]]] = {
    "langgraph": run_langgraph, "strands": run_strands,
}


# ---------------------------------------------------------------------------
# Running and scoring
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    tier: int
    split: str
    expect: str
    passed: bool
    reason: str
    status: str
    rows: int
    sql: str | None
    stable: bool = True   # same verdict on every repeat


@dataclass
class SystemResult:
    runtime: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    seconds: float = 0.0
    repeats: int = 1

    def score(self, *, tier: int | None = None, split: str | None = None, expect: str | None = None) -> tuple[int, int]:
        chosen = [c for c in self.cases if (tier is None or c.tier == tier)
                  and (split is None or c.split == split) and (expect is None or c.expect == expect)]
        return sum(c.passed for c in chosen), len(chosen)


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. With 46 cases it is wide, and saying so is the point."""
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_system(runtime: str, model: ExampleModel, catalog: Catalog, expected: dict[str, list[dict[str, Any]]], *, repeats: int = 2) -> SystemResult:
    result, started = SystemResult(runtime, model.name, repeats=repeats), time.perf_counter()
    for case in CASES:
        verdicts: list[Verdict] = []
        outcome: dict[str, Any] = {}
        executed: list[list[dict[str, Any]]] = []
        for _ in range(repeats):
            warehouse = LocalWarehouse(catalog)   # a fresh warehouse each time: no state carries between runs
            try:
                outcome = RUNTIMES[runtime](case, model, warehouse, catalog)
            except Exception as err:  # noqa: BLE001 - a crash is a failed case, not a failed eval
                outcome = {"gate_status": "ERROR", "error": type(err).__name__}
            executed = warehouse.results
            verdicts.append(grade(case, expected[case.case_id], outcome, executed))
        status = outcome.get("gate_status")
        result.cases.append(CaseResult(
            case.case_id, case.tier, case.split, case.expect, verdicts[0].passed, verdicts[0].reason,
            status.value if isinstance(status, GateStatus) else str(status), len(executed[-1]) if executed else 0,
            outcome.get("generated_sql"), stable=len({(v.passed, v.reason) for v in verdicts}) == 1,
        ))
    result.seconds = round(time.perf_counter() - started, 2)
    return result


# ---------------------------------------------------------------------------
# Audit: is the eval itself sound?
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: Literal["ok", "warn", "fail", "not checked"]
    detail: str


# Every file that puts words in front of a model. All of them are scanned, and a missing one fails the check.
PROMPT_SOURCES = ("llm_agents.py", "strands_agent.py", "graph.py", "platform_tools.py", "agentcore_app.py", "mcp_server.py")
NAMED_TABLES = ({name: f"SCADA.TELEMETRY_{n:03d}" for n, name in NAMED_PADS.items()}
                | {"Washington County north crew": "MAINT.WORK_ORDERS_041", "tri-county night crew": "MAINT.WORK_ORDERS_077"}
                | {"Midstream operations": "FIN.LEDGER_055", "Water management": "FIN.LEDGER_088"})


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _table_named_in(question: str) -> str | None:
    """The table a question is about, worked out from its wording alone."""
    for pattern, prefix in ((r"well pad (\d{3})", "SCADA.TELEMETRY_"), (r"district crew (\d{3})", "MAINT.WORK_ORDERS_"), (r"business unit (\d{3})", "FIN.LEDGER_")):
        found = re.search(pattern, question)
        if found:
            return prefix + found.group(1)
    for name, table in NAMED_TABLES.items():
        if name.lower() in question.lower():
            return table
    return "MAINT.EQUIPMENT" if "equipment register" in question else None


def _label_problems(case: EvalCase, rows: list[dict[str, Any]]) -> list[str]:
    """Ways a reference answer can be wrong or ungradable, checked from the question's own wording."""
    out, sql = [], case.reference_sql
    cells = [v for r in rows for k, v in r.items() if k != "_table"]
    if not rows or (len(rows) == 1 and all(v in (0, None) for v in cells)):
        out.append("the answer is empty or zero, so a query that finds nothing would pass")
    table = _table_named_in(case.question)
    if table is None or f"{table} " not in sql + " ":
        out.append(f"the question is about {table or 'no table that can be worked out'}, the reference reads something else")
    asked, used = re.search(r"last (\d+) days", case.question), re.search(r"DATEADD\(day, -(\d+),", sql)
    if (asked.group(1) if asked else None) != (used.group(1) if used else None):
        out.append("the time window in the question and in the reference differ")
    limit = re.search(r"LIMIT (\d+)$", sql)
    if limit and int(limit.group(1)) > 1 and len(rows) == int(limit.group(1)) and not re.search(rf"\b{limit.group(1)}\b", case.question):
        out.append(f"LIMIT {limit.group(1)} cuts the answer, and the question does not ask for {limit.group(1)}")
    if case.ordered and len(rows) < 2:
        out.append("the question asks for an order but the answer has fewer than two rows")
    elif case.ordered:
        columns = [[r[k] for r in rows] for k in rows[0] if k != "_table"]
        if not any(all(a < b for a, b in zip(col, col[1:])) or all(a > b for a, b in zip(col, col[1:])) for col in columns if None not in col):
            out.append("the question asks for an order but rows tie on it, so two orders are both right")
    return out


def reference_digest(expected: dict[str, list[dict[str, Any]]]) -> str:
    """A fingerprint of every reference answer, to pin them across machines and library versions."""
    plain = {cid: [[k, f"{v:.2f}" if isinstance(v, float) else v] for r in rows for k, v in sorted(r.items())] for cid, rows in sorted(expected.items())}
    return hashlib.sha256(json.dumps(plain, sort_keys=True, default=str).encode()).hexdigest()[:16]


def paired_p(only_first: int, only_second: int) -> float:
    """Exact two-sided sign test on the cases where two systems disagree."""
    n, k = only_first + only_second, min(only_first, only_second)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def audit(results: list[SystemResult], catalog: Catalog, expected: dict[str, list[dict[str, Any]]], *, simulated: bool) -> list[Check]:
    checks: list[Check] = []
    answer_cases = [c for c in CASES if c.expect == "rows"]
    warehouse = LocalWarehouse(catalog)

    # 1. Labels: every reference answer is reproducible, not trivial, and matches its own question.
    again = {c.case_id: reference_rows(c, warehouse, catalog) for c in answer_cases}
    problems = {c.case_id: _label_problems(c, again[c.case_id]) for c in answer_cases}
    problems = {cid: found for cid, found in problems.items() if found}
    drift = [c.case_id for c in answer_cases if not same_answer(expected[c.case_id], again[c.case_id], ordered=True)]
    duplicates = len(CASES) - len({(c.persona, _squash(c.question)) for c in CASES})
    bad = bool(problems or drift or duplicates)
    checks.append(Check("Reference answers", "fail" if bad else "ok",
                        f"{len(again)} reference queries pass every gate, return rows that are not empty or zero, read the table and the time window "
                        "their question names, and are not cut short by a LIMIT."
                        if not bad else f"problems: {problems}; not reproducible: {drift}; duplicate questions: {duplicates}"))

    # 2. Pinned answers: the same reference answers as when the cases were written, on any machine.
    if CASES is not _DEFAULT_CASES:
        checks.append(Check("Reference answers match the pinned copy", "not checked", "Only checked for the full, unmodified case list."))
    else:
        digest = reference_digest(expected)
        checks.append(Check("Reference answers match the pinned copy", "ok" if digest == REFERENCE_DIGEST else "fail",
                            "The reference answers have the fingerprint recorded with the cases, so the seeded data is the same here as where they were written."
                            if digest == REFERENCE_DIGEST else
                            f"The reference answers changed (fingerprint {digest}, recorded {REFERENCE_DIGEST}). If the cases or data were changed on purpose, update REFERENCE_DIGEST."))

    # 3. The grader: the right answer in another shape passes, and every planted mistake fails.
    shape, survived, stale, killed = [], [], [], 0
    for case in answer_cases:
        rows = again[case.case_id]
        reshaped = [{f"c{i}": v for i, (_, v) in enumerate(reversed(list(r.items())))} for r in (rows if case.ordered else rows[::-1])]
        if not same_answer(rows, reshaped, ordered=case.ordered) or (case.ordered and same_answer(rows, rows[::-1], ordered=True)):
            shape.append(case.case_id)
        for kind, sql in mistakes(case).items():
            try:
                wrong, _ = governed_rows(sql, case.persona, warehouse, catalog)
            except Exception:  # noqa: BLE001 - a mistake that cannot run cannot pass
                wrong = None
            same = wrong is not None and same_answer(rows, wrong, ordered=case.ordered)
            declared = kind in EQUIVALENT.get(case.case_id, {})
            killed += not same
            if same and not declared:
                survived.append(f"{case.case_id}:{kind}")
            if declared and not same:
                stale.append(f"{case.case_id}:{kind}")
    bad = bool(shape or survived or stale)
    allowed = sum(len(v) for v in EQUIVALENT.values())
    checks.append(Check("Wrong answers are caught", "fail" if bad else "ok",
                        f"{killed} planted mistakes (wrong table, missing condition, wrong aggregate, wrong window, wrong order) all grade as wrong, "
                        f"and every reference answer still passes with its rows shuffled and its columns renamed. {allowed} changed queries are listed as "
                        "genuinely the same answer."
                        if not bad else f"graded correct but wrong: {survived}; listed as equivalent but not: {stale}; right answer in another shape fails: {shape}"))

    # 4. Leakage: no eval question or reference query appears in anything a model reads.
    missing = [p for p in PROMPT_SOURCES if not (HERE / p).exists()]
    prompts = _squash(" ".join((HERE / p).read_text() for p in PROMPT_SOURCES if (HERE / p).exists()))
    leaked = [c.case_id for c in CASES if _squash(c.question) in prompts or _squash(c.reference_sql) in prompts]
    checks.append(Check("Leakage into prompts", "fail" if leaked or missing else "ok",
                        f"leaked: {leaked}; prompt sources not found: {missing}" if leaked or missing
                        else f"No eval question or reference query appears in {', '.join(PROMPT_SOURCES)} (case and spacing ignored)."))

    # 5. Same run, same verdict.
    unstable = sorted({c.case_id for r in results for c in r.cases if not c.stable})
    if min((r.repeats for r in results), default=1) < 2:
        checks.append(Check("Same run, same verdict", "not checked", "Needs at least two repeats of every case."))
    else:
        checks.append(Check("Same run, same verdict", "fail" if unstable else "ok",
                            f"verdict changed between repeats: {unstable}" if unstable
                            else "Every case got the same verdict on every repeat, each on a freshly built warehouse. The grader is code."))

    # 6. Room to improve, and a floor.
    rate_of = lambda r: r.score()[0] / max(1, r.score()[1])
    answered = lambda r: r.score(expect="rows")[0] / max(1, r.score(expect="rows")[1])   # must-refuse cases are not headroom
    best = max(results, key=answered)
    passed, total = best.score(expect="rows")
    rate, (low, high) = passed / total, wilson(passed, total)
    nothing = [f"{r.runtime}, {r.model}" for r in results if r.score(expect="rows")[1] and r.score(expect="rows")[0] == 0]
    if nothing or rate < 0.20:
        status, note = "warn", (f"Answers nothing correctly: {nothing}. A system that refuses everything still passes the must-refuse cases." if nothing
                                else "Too low to be useful: the cases may be too hard or the eval may be broken.")
    elif rate > 0.90:
        status, note = "warn", "Near its ceiling: this can catch regressions but not improvements. Add harder cases."
    else:
        status, note = "ok", "There is room for an improvement to show."
    checks.append(Check("Headroom", status, f"Best system ({best.runtime}, {best.model}) answers {rate:.0%} of the questions correctly (95% interval {low:.0%} to {high:.0%}). {note}"))

    # 7. A stronger model scores higher, judged on the same cases.
    models = list(dict.fromkeys(r.model for r in results))
    if len(models) < 2:
        checks.append(Check("Stronger model scores higher", "not checked", "Needs at least two models of different strength."))
    else:
        mean = {m: sum(rate_of(r) for r in results if r.model == m) / sum(r.model == m for r in results) for m in models}
        weak, strong = min(models, key=mean.get), max(models, key=mean.get)
        pairs = [(a, b) for a in results for b in results if a.runtime == b.runtime and a.model == weak and b.model == strong]
        each = [(sum(y.passed and not x.passed for x, y in zip(a.cases, b.cases)), sum(x.passed and not y.passed for x, y in zip(a.cases, b.cases)))
                for a, b in pairs]
        agree = bool(each) and all(up > down for up, down in each)   # the stronger model has to win on every runtime, not on average
        only_strong, only_weak = max(each, key=lambda t: paired_p(*t), default=(0, 0))   # report the least convincing runtime
        p = paired_p(only_strong, only_weak)
        w, g = next(r for r in results if r.model == weak).score(), next(r for r in results if r.model == strong).score()
        overlap = wilson(*w)[1] >= wilson(*g)[0]
        note = " These are scripted stand-ins built to differ, so this only shows the check works." if simulated else ""
        checks.append(Check("Stronger model scores higher", "ok" if agree and mean[strong] - mean[weak] > 0.05 and p < 0.05 else "warn",
                            f"{strong} averages {mean[strong]:.0%} against {mean[weak]:.0%} for {weak}. "
                            + ("Their 95% intervals overlap, so the two totals alone do not separate them. " if overlap else "")
                            + (f"Compared case by case on each runtime, weakest result shown: {only_strong} cases only {strong} got right, {only_weak} only {weak} "
                               f"(exact sign test p = {p:.3f})." if pairs else "No runtime ran both models, so they cannot be compared case by case.")
                            + ("" if agree or not pairs else " The runtimes disagree about which model is stronger.") + note))

    # 8. Dev and held-out sets, counted on questions with an answer.
    held = [c for c in answer_cases if c.split == "heldout"]
    per_tier = {t: sum(c.tier == t for c in held) for t in sorted({c.tier for c in answer_cases})}
    share = len(held) / max(1, len(answer_cases))
    ok = 0.15 <= share <= 0.4 and all(per_tier.values())
    checks.append(Check("Held-out set", "ok" if ok else "warn",
                        f"{len(held)} of {len(answer_cases)} questions are held out ({', '.join(f'tier {t}: {n}' for t, n in per_tier.items())}). "
                        + ("Tune on dev; report held-out. With this few, a held-out score is a rough guide only."
                           if ok else "Every tier needs held-out questions, and between 15% and 40% overall.")))

    # 9. What code cannot check.
    checks.append(Check("Cases look like real questions", "not checked",
                        "The questions and the data are invented. Replace them with questions people actually asked before trusting a score."))
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "n/a"


def _with_interval(passed: int, total: int) -> str:
    """A score with its 95% interval: with a dozen cases the interval is the honest part."""
    if not total:
        return "n/a"
    low, high = wilson(passed, total)
    return f"{passed}/{total} ({low:.0%} to {high:.0%})"


def summary_table(results: list[SystemResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        passed, total = r.score()
        low, high = wilson(passed, total)
        rows.append({
            "runtime": r.runtime, "model": r.model, "overall": _pct(passed, total), "interval": f"{low:.0%} to {high:.0%}",
            "tier1": _with_interval(*r.score(tier=1)), "tier2": _with_interval(*r.score(tier=2)), "tier3": _with_interval(*r.score(tier=3)),
            "refusals": _pct(*r.score(expect="refusal")), "heldout": _with_interval(*r.score(split="heldout", expect="rows")), "seconds": r.seconds,
        })
    return rows


CONVENTIONS = (
    "Every answer is over the rows the person asking is entitled to see: row-level security is applied before a query runs, so "
    "\"how many readings\" means how many of theirs. \"The last N days\" means the N times 24 hours before now, not N calendar days; "
    "a model has to be told that convention or it will be marked wrong. Rows are compared without regard to order unless the question asks "
    "for one (\"newest first\"). Extra columns are allowed, missing ones are not. Numbers match within half a cent. An answer with no "
    "rows is refused by the platform, so every question here has rows in its answer. Weeks start on Monday."
)


def write_report(results: list[SystemResult], checks: list[Check], *, simulated: bool, directory: Path = REPORT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    table = summary_table(results)
    (directory / "quality_eval.json").write_text(json.dumps({
        "simulated": simulated, "cases": len(CASES), "summary": table,
        "audit": [c.__dict__ for c in checks], "results": [{"runtime": r.runtime, "model": r.model, "cases": [c.__dict__ for c in r.cases]} for r in results],
    }, indent=2))

    e = html.escape
    banner = ('<p class="sim"><b>Simulated models.</b> The models in this run are scripted stand-ins that know the reference queries and make '
              "planned mistakes. The scores show how the evaluation and this report work. They say nothing about any real model, and the two "
              "runtimes are expected to tie.</p>") if simulated else ""
    head = "".join(f"<th>{e(h)}</th>" for h in ("Runtime", "Model", "Overall", "95% interval", "Tier 1", "Tier 2", "Tier 3", "Must refuse", "Held-out questions", "Seconds"))
    body = "".join("<tr>" + "".join(f"<td>{e(str(row[k]))}</td>" for k in ("runtime", "model", "overall", "interval", "tier1", "tier2", "tier3", "refusals", "heldout", "seconds")) + "</tr>" for row in table)
    audit_rows = "".join(f'<tr><td>{e(c.name)}</td><td class="{c.status.replace(" ", "-")}">{e(c.status)}</td><td>{e(c.detail)}</td></tr>' for c in checks)
    questions = {c.case_id: c for c in CASES}
    case_head = "<th>Case</th><th>Tier</th><th>Set</th><th>Question</th>" + "".join(f"<th>{e(r.runtime)}<br>{e(r.model)}</th>" for r in results)
    case_rows = ""
    for i, case in enumerate(CASES):
        cells = "".join(f'<td class="{"ok" if r.cases[i].passed else "fail"}" title="{e(r.cases[i].sql or "")}">{e(r.cases[i].reason)}</td>' for r in results)
        label = "refuse" if case.expect == "refusal" else str(case.tier)
        case_rows += f"<tr><td>{e(case.case_id)}</td><td>{label}</td><td>{e(case.split)}</td><td>{e(questions[case.case_id].question)}</td>{cells}</tr>"
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Text-to-SQL quality evaluation</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:32px;max-width:1200px;color:#222}}h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}th,td{{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top}}
th{{background:#3C3489;color:#fff;font-weight:600}}.ok{{background:#e6f4ea}}.fail{{background:#fde8e6}}.warn{{background:#fff4d6}}.not-checked{{background:#eee}}
.sim{{background:#fff4d6;border:1px solid #e0b400;padding:10px 14px}}small{{color:#666}}</style></head><body>
<h1>Text-to-SQL quality evaluation</h1>{banner}
<p>{len(CASES)} cases: {sum(c.expect == "rows" for c in CASES)} questions with a known answer in three tiers, and {sum(c.expect == "refusal" for c in CASES)} requests that must be refused.
A case passes when the rows returned match the reference rows, or when a request that must be refused is stopped before any query runs.</p>
<p><b>How answers are read.</b> {e(CONVENTIONS)}</p>
<p><small>Brackets after a tier or held-out score are its 95% interval. With 9 to 14 questions the interval is wide, so read a single tier as a rough guide.</small></p>
<h2>Scores</h2><table><tr>{head}</tr>{body}</table>
<h2>Is the evaluation sound?</h2><table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{audit_rows}</table>
<h2>Every case</h2><p><small>Hover a cell to see the SQL that ran.</small></p><table><tr>{case_head}</tr>{case_rows}</table>
</body></html>"""
    path = directory / "quality_eval.html"
    path.write_text(page)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(models: list[str] | None = None, runtimes: list[str] | None = None, *, repeats: int = 2) -> tuple[list[SystemResult], list[Check]]:
    catalog = build_eval_catalog()
    expected = {c.case_id: (reference_rows(c, LocalWarehouse(catalog), catalog) if c.expect == "rows" else []) for c in CASES}
    results = [run_system(runtime, ExampleModel(name), catalog, expected, repeats=repeats)
               for name in (models or list(EXAMPLE_SKILL)) for runtime in (runtimes or list(RUNTIMES))]
    return results, audit(results, catalog, expected, simulated=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text-to-SQL quality evaluation")
    parser.add_argument("--model", action="append", choices=[*EXAMPLE_SKILL, "llm"], help="repeat to compare; default is both example models")
    parser.add_argument("--runtime", action="append", choices=list(RUNTIMES))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.model and "llm" in args.model:
        print("A real model is not wired in yet: connect it in run_langgraph and run_strands, then replace the example models.", file=sys.stderr)
        return 2
    results, checks = run(args.model, args.runtime, repeats=args.repeats)
    path = write_report(results, checks, simulated=True)
    print("SIMULATED MODELS: scripted stand-ins. These scores describe the stand-ins, not any real model.\n")
    for row in summary_table(results):
        print(f"{row['runtime']:10} {row['model']:14} overall {row['overall']:13} [{row['interval']}]\n"
              f"{'':10} t1 {row['tier1']}  t2 {row['tier2']}  t3 {row['tier3']}  refuse {row['refusals']}  held-out {row['heldout']}")
    print()
    for check in checks:
        print(f"[{check.status:^11}] {check.name}: {check.detail}")
    print(f"\nReport: {path}")
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
