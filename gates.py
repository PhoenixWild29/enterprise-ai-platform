"""gates.py - The four fail-closed validation gates.

Each gate has two layers:
  * a pure function returning a GateResult (easy to unit test and reuse in
    the eval harness), and
  * a `*_node` wrapper that takes AgentState and returns a partial state
    update for LangGraph (gate_status, gate_results, audit_telemetry).

Fail-closed rule: anything a gate cannot positively verify is REFUSED.
"""

import math
from collections.abc import Callable, Iterator
from typing import Any

import sqlglot
from pydantic import BaseModel, ValidationError
from sqlglot import exp
from sqlglot.errors import SqlglotError

from schemas import (
    AgentState,
    AuditEvent,
    Catalog,
    Citation,
    DataDomain,
    DiagnosticArgs,
    ForeignKey,
    GateResult,
    GateStatus,
    MCPToolDefinition,
    OTelSpan,
    Passage,
    SnowflakeQueryArgs,
    TableSpec,
    ToolPayload,
    VectorSearchArgs,
)

SQL_DIALECT = "snowflake"

MIN_PASSAGES = 2
MIN_AVG_CONFIDENCE = 0.4
MAX_SENSOR_LIMIT = 10_000
CITATION_SNIPPET_CHARS = 40

GATE_THIN_SET = "thin_set_refusal"
GATE_SQL_AST = "sql_ast_parser"
GATE_PAYLOAD = "payload_safety"
GATE_CITATION = "verbatim_citation"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GateError(Exception):
    """Base class for gate failures."""


class SQLValidationError(GateError):
    """SQL could not be parsed or verified."""


class SQLSecurityError(GateError):
    """SQL attempted a write, DDL, or statement stacking. Always fatal."""


# ---------------------------------------------------------------------------
# Demo catalog (500 tables)
# ---------------------------------------------------------------------------


def build_demo_catalog() -> Catalog:
    """500-table lakehouse catalog: 300 SCADA telemetry tables, 101 maintenance
    tables and 99 finance tables, with declared PK/FK relationships."""
    tables: list[TableSpec] = [
        TableSpec(
            schema_name="MAINT",
            name="EQUIPMENT",
            domain=DataDomain.MAINTENANCE,
            columns=("EQUIPMENT_ID", "SITE_ID", "EQUIPMENT_TYPE", "COMMISSIONED_AT"),
            primary_key=("EQUIPMENT_ID",),
        ),
        TableSpec(
            schema_name="FIN",
            name="COST_CENTERS",
            domain=DataDomain.FINANCE,
            columns=("COST_CENTER_ID", "NAME", "OWNER_USER_ID"),
            primary_key=("COST_CENTER_ID",),
        ),
    ]
    equipment_fk = ForeignKey(
        column="EQUIPMENT_ID", ref_table="MAINT.EQUIPMENT", ref_column="EQUIPMENT_ID"
    )
    for i in range(300):
        tables.append(
            TableSpec(
                schema_name="SCADA",
                name=f"TELEMETRY_{i:03d}",
                domain=DataDomain.SCADA,
                columns=(
                    "READING_ID", "EQUIPMENT_ID", "SENSOR_ID", "READING_TS",
                    "VALUE", "QUALITY_FLAG", "USER_ID",
                ),
                primary_key=("READING_ID",),
                foreign_keys=(equipment_fk,),
                time_column="READING_TS",
                is_sensor=True,
            )
        )
    for i in range(100):
        tables.append(
            TableSpec(
                schema_name="MAINT",
                name=f"WORK_ORDERS_{i:03d}",
                domain=DataDomain.MAINTENANCE,
                columns=(
                    "WORK_ORDER_ID", "EQUIPMENT_ID", "OPENED_AT", "CLOSED_AT",
                    "FAILURE_CODE", "TECHNICIAN_NOTES", "USER_ID",
                ),
                primary_key=("WORK_ORDER_ID",),
                foreign_keys=(equipment_fk,),
                time_column="OPENED_AT",
            )
        )
    for i in range(98):
        tables.append(
            TableSpec(
                schema_name="FIN",
                name=f"LEDGER_{i:03d}",
                domain=DataDomain.FINANCE,
                columns=(
                    "ENTRY_ID", "COST_CENTER_ID", "EQUIPMENT_ID", "POSTED_AT",
                    "AMOUNT", "CURRENCY", "USER_ID",
                ),
                primary_key=("ENTRY_ID",),
                foreign_keys=(
                    ForeignKey(
                        column="COST_CENTER_ID",
                        ref_table="FIN.COST_CENTERS",
                        ref_column="COST_CENTER_ID",
                    ),
                    equipment_fk,
                ),
                time_column="POSTED_AT",
            )
        )
    return Catalog.from_tables(tables)


DEFAULT_CATALOG: Catalog = build_demo_catalog()


# ---------------------------------------------------------------------------
# Gate 1: Thin-Set Refusal
# ---------------------------------------------------------------------------


def thin_set_refusal_gate(
    passages: list[Passage],
    confidence_scores: list[float],
    min_passages: int = MIN_PASSAGES,
    min_avg_confidence: float = MIN_AVG_CONFIDENCE,
) -> GateResult:
    """Refuse when evidence is too thin: < 2 passages or mean confidence < 0.4."""
    reasons: list[str] = []
    n = len(passages)
    if n < min_passages:
        reasons.append(f"Only {n} passage(s) retrieved; at least {min_passages} required.")
    if len(confidence_scores) != n:
        reasons.append(
            f"Confidence scores ({len(confidence_scores)}) do not align with passages ({n})."
        )
    if any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in confidence_scores):
        reasons.append("Confidence scores contain values outside [0, 1].")
        avg = 0.0
    else:
        avg = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    if avg < min_avg_confidence:
        reasons.append(
            f"Average confidence {avg:.3f} is below the {min_avg_confidence:.2f} threshold."
        )
    return GateResult(
        gate=GATE_THIN_SET,
        status=GateStatus.REFUSED if reasons else GateStatus.PASSED,
        reasons=reasons,
        details={"passage_count": n, "avg_confidence": round(avg, 4)},
    )


# ---------------------------------------------------------------------------
# Gate 2: Text-to-SQL AST Parser
# ---------------------------------------------------------------------------


def _forbidden_node_types() -> tuple[type[exp.Expression], ...]:
    names = (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable",
        "Merge", "TruncateTable", "Command", "Copy", "Grant", "Revoke", "Into",
        "Use", "Set", "Transaction", "Commit", "Rollback", "Put", "Get",
        "Execute", "Call",
    )
    return tuple(t for n in names if isinstance(t := getattr(exp, n, None), type))


_FORBIDDEN_NODES = _forbidden_node_types()
_QUERY_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)

# Tables are allow-listed by the catalog; functions are allow-listed here.
# Anything else (system functions, table functions, context functions that
# reveal session details, UDFs) is refused.
_ALLOWED_FUNCTION_NAMES = (
    "And", "Or", "Xor", "Case", "If", "Coalesce", "Nullif", "Cast", "TryCast",
    "Avg", "Sum", "Count", "Min", "Max", "Median", "Stddev", "StddevPop", "StddevSamp",
    "Variance", "VariancePop", "ApproxDistinct",
    "Round", "Abs", "Floor", "Ceil", "Greatest", "Least",
    "Lower", "Upper", "Trim", "Length", "Concat", "Substring",
    "DateAdd", "DateSub", "DateDiff", "DateTrunc", "TimestampTrunc", "TimestampAdd",
    "TimestampSub", "TimestampDiff", "Extract", "Year", "Month", "Day",
    "CurrentTimestamp", "CurrentDate", "CurrentUser",
    "RowNumber", "Rank", "DenseRank", "Lag", "Lead", "FirstValue", "LastValue",
)
_ALLOWED_FUNCTIONS: frozenset[type] = frozenset(
    t for n in _ALLOWED_FUNCTION_NAMES if isinstance(t := getattr(exp, n, None), type)
)
# Clauses that widen what a SELECT can reach or do: time travel and change
# tracking, bind/session variables, row-pattern and hierarchical queries,
# pivots, sampling and semi-structured path access.
_FORBIDDEN_CLAUSES: tuple[type[exp.Expression], ...] = tuple(
    t
    for n in ("HistoricalData", "Changes", "Parameter", "Placeholder", "SessionParameter", "Pivot",
              "MatchRecognize", "Connect", "TableSample", "Bracket", "Lock")
    if isinstance(t := getattr(exp, n, None), type)
)
_FORBIDDEN_ROW_SOURCES: tuple[type[exp.Expression], ...] = tuple(
    t for n in ("Lateral", "Unnest", "TableFromRows", "Values") if isinstance(t := getattr(exp, n, None), type)
)


def _check_expression_policy(root: exp.Expression) -> None:
    """Refuse functions and row sources that are not explicitly allowed."""
    for node in root.find_all(exp.Func):
        if type(node) not in _ALLOWED_FUNCTIONS:
            name = node.name if isinstance(node, exp.Anonymous) else type(node).__name__
            raise SQLValidationError(f"Function '{name.upper()}' is not on the allowed list.")
    for node in root.find_all(*_FORBIDDEN_ROW_SOURCES):
        raise SQLValidationError(
            f"{type(node).__name__} row sources are not allowed; query catalog tables only."
        )
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLValidationError("Table functions are not allowed; query catalog tables only.")
    for node in root.find_all(*_FORBIDDEN_CLAUSES):
        raise SQLValidationError(f"{type(node).__name__} is not allowed in governed queries.")
    for with_ in root.find_all(exp.With):
        if with_.args.get("recursive"):
            raise SQLValidationError("Recursive CTEs are not allowed.")
    for ident in root.find_all(exp.Identifier):
        if ident.quoted:
            # A quoted name is a different, case-sensitive object in Snowflake.
            raise SQLValidationError(f"Quoted identifier \"{ident.name}\" is not allowed.")
    _check_projection_policy(root)
    for column in root.find_all(exp.Column):
        name = column.name.upper()
        if column.args.get("db") or column.args.get("catalog"):
            raise SQLValidationError(f"Column reference '{column.sql()}' must use a table alias, not a schema path.")
        if name in ("NEXTVAL", "CURRVAL") or name.startswith("$"):
            raise SQLValidationError(f"'{column.sql()}' is not allowed in governed queries.")
        if not column.table and name.startswith("CURRENT_"):
            raise SQLValidationError(f"Session context value '{name}' is not on the allowed list.")


# Functions whose own arguments may be constants in a SELECT list (a rounding
# precision, a date part, an interval). Everywhere else in a SELECT list a
# constant is refused.
_CONSTANT_ARGUMENT_FUNCTIONS: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in ("Round", "DateTrunc", "TimestampTrunc", "DateAdd", "DateSub", "DateDiff", "TimestampAdd",
                 "TimestampSub", "TimestampDiff", "TsOrDsAdd", "Extract", "Interval", "TimeSlice")
    if hasattr(exp, name)
)
_ALIAS_WORDS = frozenset(
    "total avg average min max sum count cnt n num number first last latest earliest mean median stddev distinct "
    "daily hourly weekly monthly day hour week month year date bucket per by of rows".split()
)
MAX_ALIAS_CHARS = 40


def _check_projection_policy(root: exp.Expression) -> None:
    """Every value a query returns must come from table data.

    Result rows are released to the user as evidence, so a query must not be
    able to return text or numbers its author made up: no constants in a
    SELECT list (outside a date part or rounding precision), every selected
    expression must read a column, and an output name is built only from
    column names in the query and a fixed set of aggregate words."""
    column_words = {word for column in root.find_all(exp.Column) for word in column.name.lower().split("_") if word}
    for select in root.find_all(exp.Select):
        for projection in select.expressions:
            value = projection.unalias()
            for constant in projection.find_all(exp.Literal, exp.Boolean, exp.Null):
                holder = constant.parent
                while isinstance(holder, (exp.Paren, exp.Neg)):
                    holder = holder.parent
                if not isinstance(holder, _CONSTANT_ARGUMENT_FUNCTIONS):
                    raise SQLValidationError(
                        "Constants are not allowed in the SELECT list; select columns and aggregates of columns only."
                    )
            if not isinstance(value, exp.Star) and value.find(exp.Column, exp.Star) is None:
                raise SQLValidationError("Every selected expression must read a table column.")
            alias = projection.alias if isinstance(projection, exp.Alias) else ""
            if alias:
                words = [word for word in alias.lower().split("_") if word]
                if len(alias) > MAX_ALIAS_CHARS or not all(w in _ALIAS_WORDS or w in column_words for w in words):
                    raise SQLValidationError(
                        f"Output name '{alias[:MAX_ALIAS_CHARS]}' is not allowed; name outputs after the columns they come "
                        "from, with words such as total, avg, min, max or count."
                    )


def parse_read_only_sql(sql: str) -> exp.Expression:
    """Parse SQL into an AST and enforce the SELECT-only policy.

    Raises SQLSecurityError for DML/DDL, statement stacking, or any
    non-SELECT root. Raises SQLValidationError when the SQL cannot be parsed
    or uses a function or row source that is not on the allowed list.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=SQL_DIALECT) if s is not None]
    except SqlglotError as err:
        raise SQLValidationError(f"SQL failed to parse: {err}") from err
    except RecursionError as err:
        raise SQLValidationError("SQL is nested too deeply to validate.") from err
    if not statements:
        raise SQLValidationError("SQL contains no statements.")

    for stmt in statements:
        for node in stmt.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SQLSecurityError(
                    f"Forbidden {type(node).__name__.upper()} operation; "
                    "only read-only SELECT statements are permitted."
                )
    if len(statements) > 1:
        raise SQLSecurityError(f"Statement stacking detected ({len(statements)} statements).")

    root = statements[0]
    if isinstance(root, exp.Subquery):
        # Clauses attached outside the parentheses (ORDER BY, OFFSET) would escape every check.
        raise SQLValidationError("Remove the parentheses around the whole query.")
    if not isinstance(root, _QUERY_ROOTS):
        raise SQLSecurityError(
            f"Root statement is {type(root).__name__}; only SELECT is permitted."
        )
    try:
        _check_expression_policy(root)
    except RecursionError as err:
        raise SQLValidationError("SQL is nested too deeply to validate.") from err
    return root


def _table_fqn(table: exp.Table) -> str:
    return f"{table.db}.{table.name}".upper() if table.db else table.name.upper()


def resolve_table(table: exp.Table, catalog: Catalog) -> TableSpec | None:
    """The catalog entry a table reference names, or None. A reference to
    another database is never a catalog table, whatever its schema and name."""
    if table.catalog and table.catalog.upper() != catalog.database.upper():
        return None
    return catalog.get(_table_fqn(table)) if table.db else None


def _with_of(node: exp.Expression) -> exp.With | None:
    found = node.args.get("with_") or node.args.get("with")
    return found if isinstance(found, exp.With) else None


def _is_cte_ref(table: exp.Table) -> bool:
    """True only when an unqualified table name refers to a CTE that is
    actually in scope at that point: defined by a WITH on an enclosing query,
    or, inside a CTE body, by an earlier sibling in the same WITH. A CTE
    defined in some other branch of the statement does not make the name a CTE."""
    if table.args.get("db") is not None or table.args.get("catalog") is not None:
        return False  # any qualifier, even an empty one (".NAME"), means it is not a CTE reference
    name = table.name.upper()

    def defines(ctes: list[exp.Expression]) -> bool:
        return any(c.alias and c.alias.upper() == name for c in ctes)

    node: exp.Expression = table
    while node.parent is not None:
        parent = node.parent
        if isinstance(parent, exp.With) and isinstance(node, exp.CTE):
            # Climbing out of a CTE body: only siblings defined before it are visible.
            siblings = list(parent.expressions)
            position = next(i for i, c in enumerate(siblings) if c is node)
            if defines(siblings[:position]):
                return True
        else:
            with_ = _with_of(parent)
            # `node` is part of the query's body, not its WITH clause: every CTE of that WITH is visible.
            if with_ is not None and node is not with_ and defines(list(with_.expressions)):
                return True
        node = parent
    return False


def _conjuncts(node: exp.Expression | None) -> Iterator[exp.Expression]:
    """Top-level AND-ed predicates. Anything under an OR is not a guarantee."""
    if node is None:
        return
    node = node.unnest()
    if isinstance(node, exp.And):
        yield from _conjuncts(node.left)
        yield from _conjuncts(node.right)
    else:
        yield node


type Scope = dict[str, TableSpec]  # alias (upper) -> catalog table


def _scope_of(select: exp.Select, catalog: Catalog) -> Scope:
    """Catalog tables referenced directly by this SELECT (not nested ones)."""
    scope: Scope = {}
    for table in select.find_all(exp.Table):
        if table.find_ancestor(exp.Select) is not select or _is_cte_ref(table):
            continue
        if spec := resolve_table(table, catalog):
            scope[table.alias_or_name.upper()] = spec
    return scope


def _resolve_column(col: exp.Expression, scope: Scope) -> tuple[str, TableSpec, str] | None:
    """Resolve a column to (alias, table, column). None when ambiguous/unknown."""
    if not isinstance(col, exp.Column):
        return None
    name = col.name.upper()
    if col.table:
        alias = col.table.upper()
        spec = scope.get(alias)
        return (alias, spec, name) if spec and name in spec.columns else None
    owners = [(a, s) for a, s in scope.items() if name in s.columns]
    return (owners[0][0], owners[0][1], name) if len(owners) == 1 else None


def _check_tables(root: exp.Expression, catalog: Catalog) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    referenced: list[str] = []
    for table in root.find_all(exp.Table):
        if _is_cte_ref(table):
            continue
        fqn = _table_fqn(table)
        if table.catalog and table.catalog.upper() != catalog.database:
            violations.append(f"Database '{table.catalog}' is not the governed lakehouse.")
        elif not table.db:
            violations.append(f"Table '{fqn}' must be schema-qualified (SCHEMA.TABLE).")
        elif catalog.get(fqn) is None:
            violations.append(f"Table '{fqn}' is not in the allowed catalog.")
        elif fqn not in referenced:
            referenced.append(fqn)
    return violations, referenced


def _check_joins(select: exp.Select, scope: Scope, catalog: Catalog) -> list[str]:
    violations: list[str] = []
    for join in select.args.get("joins") or []:
        target = join.this
        if not isinstance(target, exp.Table) or _is_cte_ref(target):
            violations.append(
                "JOIN to a derived relation (subquery/CTE/lateral) cannot be key-verified."
            )
            continue
        alias = target.alias_or_name.upper()
        spec = scope.get(alias)
        if spec is None:
            continue  # unknown table: already reported by the catalog check
        on, using = join.args.get("on"), join.args.get("using")

        verified = False
        if on is not None:
            for pred in _conjuncts(on):
                if not isinstance(pred, exp.EQ):
                    continue
                left, right = _resolve_column(pred.left, scope), _resolve_column(pred.right, scope)
                if left is None or right is None or alias not in (left[0], right[0]):
                    continue
                if left[0] != right[0] and catalog.is_key_join(left[1], left[2], right[1], right[2]):
                    verified = True
                    break
        elif using:
            for ident in using:
                col = ident.name.upper()
                if any(
                    a != alias and col in other.columns and catalog.is_key_join(spec, col, other, col)
                    for a, other in scope.items()
                ):
                    verified = True
                    break
        else:
            violations.append(f"JOIN on '{spec.fqn}' has no ON/USING condition (cartesian join).")
            continue
        if not verified:
            violations.append(
                f"JOIN on '{spec.fqn}' does not follow a declared primary/foreign key relationship."
            )
    return violations


def _is_time_col(node: exp.Expression, alias: str, spec: TableSpec, scope: Scope) -> bool:
    resolved = _resolve_column(node, scope)
    return resolved is not None and resolved[0] == alias and resolved[2] == spec.time_column


def _is_constant(node: exp.Expression) -> bool:
    return node.find(exp.Column) is None and node.find(exp.Select) is None


def _has_lower_time_bound(where: exp.Expression | None, alias: str, spec: TableSpec, scope: Scope) -> bool:
    """A top-level WHERE conjunct must bound the time column from below with a
    constant expression (>=, >, =, BETWEEN). Bounds under OR do not count."""
    if where is None:
        return False
    for pred in _conjuncts(where.this):
        if isinstance(pred, exp.Between):
            if (
                _is_time_col(pred.this, alias, spec, scope)
                and _is_constant(pred.args["low"])
                and _is_constant(pred.args["high"])
            ):
                return True
        elif isinstance(pred, (exp.GT, exp.GTE, exp.EQ)):
            if _is_time_col(pred.left, alias, spec, scope) and _is_constant(pred.right):
                return True
            if isinstance(pred, exp.EQ) and _is_time_col(pred.right, alias, spec, scope) and _is_constant(pred.left):
                return True
        elif isinstance(pred, (exp.LT, exp.LTE)):
            if _is_time_col(pred.right, alias, spec, scope) and _is_constant(pred.left):
                return True
    return False


def _check_time_bounds(select: exp.Select, scope: Scope) -> list[str]:
    where = select.args.get("where")
    return [
        f"Table '{spec.fqn}' requires a WHERE lower bound on {spec.time_column}."
        for alias, spec in scope.items()
        if spec.time_column and not _has_lower_time_bound(where, alias, spec, scope)
    ]


def sql_ast_gate(sql: str | None, catalog: Catalog = DEFAULT_CATALOG) -> GateResult:
    """Validate generated SQL against the catalog and governance policy.

    Checks: SELECT-only, allowed tables, PK/FK join conditions, mandatory
    WHERE time bounds. Raises SQLSecurityError on DML/DDL (never swallowed
    here; the node wrapper converts it to REFUSED and logs a security event).
    """
    try:
        root = parse_read_only_sql(sql or "")
    except SQLValidationError as err:
        return GateResult(gate=GATE_SQL_AST, status=GateStatus.REFUSED, reasons=[str(err)])

    violations, referenced = _check_tables(root, catalog)
    for select in root.find_all(exp.Select):
        scope = _scope_of(select, catalog)
        violations += _check_joins(select, scope, catalog)
        violations += _check_time_bounds(select, scope)

    if not referenced and not violations:
        violations.append("Query references no governed catalog tables.")

    return GateResult(
        gate=GATE_SQL_AST,
        status=GateStatus.REFUSED if violations else GateStatus.PASSED,
        reasons=list(dict.fromkeys(violations)),
        details={
            "tables": referenced,
            "normalized_sql": root.sql(dialect=SQL_DIALECT),
        },
    )


# ---------------------------------------------------------------------------
# Gate 3: Payload & Safety
# ---------------------------------------------------------------------------

TOOL_SNOWFLAKE_QUERY = "snowflake_query"
TOOL_VECTOR_SEARCH = "vector_search"
TOOL_DIAGNOSTIC = "diagnostic_reason"

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    TOOL_SNOWFLAKE_QUERY: SnowflakeQueryArgs,
    TOOL_VECTOR_SEARCH: VectorSearchArgs,
    TOOL_DIAGNOSTIC: DiagnosticArgs,
}

MCP_TOOL_REGISTRY: dict[str, MCPToolDefinition] = {
    TOOL_SNOWFLAKE_QUERY: MCPToolDefinition.from_args_model(
        TOOL_SNOWFLAKE_QUERY,
        "Run a governed, read-only SELECT against the Snowflake lakehouse.",
        SnowflakeQueryArgs,
    ),
    TOOL_VECTOR_SEARCH: MCPToolDefinition.from_args_model(
        TOOL_VECTOR_SEARCH,
        "Retrieve passages from maintenance manuals and operating procedures.",
        VectorSearchArgs,
    ),
    TOOL_DIAGNOSTIC: MCPToolDefinition.from_args_model(
        TOOL_DIAGNOSTIC,
        "Reason over telemetry and maintenance history for one equipment item.",
        DiagnosticArgs,
    ),
}


def _limit_value(root: exp.Expression) -> int | None:
    """Integer LIMIT on the outermost query, or None if absent/non-literal."""
    limit = root.args.get("limit")
    if limit is None:
        return None
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int:
        return int(value.this)
    return None


def payload_safety_gate(
    payload: ToolPayload | dict[str, Any],
    catalog: Catalog = DEFAULT_CATALOG,
    max_sensor_limit: int = MAX_SENSOR_LIMIT,
) -> GateResult:
    """Validate a tool call: registered tool, schema-valid arguments, and a
    literal LIMIT <= cap on any query touching sensor tables."""

    def refuse(*reasons: str, **details: Any) -> GateResult:
        return GateResult(gate=GATE_PAYLOAD, status=GateStatus.REFUSED, reasons=list(reasons), details=details)

    try:
        envelope = payload if isinstance(payload, ToolPayload) else ToolPayload.model_validate(payload)
    except ValidationError as err:
        return refuse(*(f"payload.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()))

    args_model = TOOL_ARG_MODELS.get(envelope.tool_name)
    if args_model is None or envelope.tool_name not in MCP_TOOL_REGISTRY:
        return refuse(f"Tool '{envelope.tool_name}' is not a registered MCP tool.")

    try:
        args = args_model.model_validate(envelope.arguments)
    except ValidationError as err:
        return refuse(
            *(f"arguments.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()),
            tool=envelope.tool_name,
        )

    details: dict[str, Any] = {"tool": envelope.tool_name, "call_id": envelope.call_id}
    if isinstance(args, SnowflakeQueryArgs):
        try:
            root = parse_read_only_sql(args.sql)
        except SQLValidationError as err:
            return refuse(str(err), **details)
        sensor_tables = sorted(
            {
                spec.fqn
                for t in root.find_all(exp.Table)
                if not _is_cte_ref(t)
                and (spec := resolve_table(t, catalog)) is not None
                and spec.is_sensor
            }
        )
        details["sensor_tables"] = sensor_tables
        limit = _limit_value(root)
        details["limit"] = limit
        if root.args.get("limit") is not None and (limit is None or limit > max_sensor_limit):
            return refuse(f"LIMIT must be a literal no greater than {max_sensor_limit}.", **details)
        if sensor_tables:
            if limit is None:
                return refuse(
                    "Queries on sensor tables require a literal LIMIT on the outermost query.",
                    **details,
                )
            if limit > max_sensor_limit:
                return refuse(
                    f"LIMIT {limit} exceeds the sensor-table cap of {max_sensor_limit}.", **details
                )
            if limit > args.max_rows:
                return refuse(
                    f"LIMIT {limit} exceeds the payload max_rows of {args.max_rows}.", **details
                )

    return GateResult(gate=GATE_PAYLOAD, status=GateStatus.PASSED, details=details)


# ---------------------------------------------------------------------------
# Gate 4: Verbatim Citation Validator
# ---------------------------------------------------------------------------


def verbatim_citation_gate(
    draft: str | None,
    citations: list[Citation],
    passages: list[Passage],
    snippet_chars: int = CITATION_SNIPPET_CHARS,
) -> GateResult:
    """Every citation must (a) point at a retrieved passage, (b) carry a snippet
    that exists verbatim in that passage, and (c) appear verbatim in the draft.

    The canonical snippet is `p.text[:40]`, so check (c) is exactly
    `p.text[:40] in draft`. Matching is exact: no case or whitespace folding.
    """
    reasons: list[str] = []
    if not draft or not draft.strip():
        reasons.append("Draft response is empty.")
    if not citations:
        reasons.append("Draft contains no citations; uncited answers are not released.")

    by_id = {p.passage_id: p for p in passages}
    verified: list[str] = []
    for c in citations:
        passage = by_id.get(c.passage_id)
        if passage is None:
            reasons.append(f"Citation '{c.passage_id}' does not match any retrieved passage.")
            continue
        required = min(snippet_chars, len(passage.text))
        if len(c.snippet) < required:
            reasons.append(
                f"Citation '{c.passage_id}' snippet is shorter than {required} characters."
            )
        elif c.snippet not in passage.text:
            reasons.append(f"Citation '{c.passage_id}' snippet is not verbatim in its source passage.")
        elif draft and c.snippet not in draft:
            reasons.append(f"Citation '{c.passage_id}' snippet does not appear verbatim in the draft.")
        else:
            verified.append(c.passage_id)

    return GateResult(
        gate=GATE_CITATION,
        status=GateStatus.REFUSED if reasons else GateStatus.PASSED,
        reasons=reasons,
        details={"citations_checked": len(citations), "citations_verified": verified},
    )


# ---------------------------------------------------------------------------
# LangGraph node wrappers
# ---------------------------------------------------------------------------


def _run_gate(
    state: AgentState, gate: str, check: Callable[[], GateResult]
) -> dict[str, Any]:
    """Run a gate, fail closed on any exception, and emit state + telemetry."""
    span = OTelSpan.start(f"gate.{gate}", trace_id=state.get("trace_id"), attributes={"gate.name": gate})
    event_type = "gate"
    try:
        result = check()
    except SQLSecurityError as err:
        event_type = "security"
        result = GateResult(
            gate=gate, status=GateStatus.REFUSED, reasons=[f"SECURITY: {err}"],
            details={"security_violation": True},
        )
    except Exception as err:  # noqa: BLE001 - a crashing gate must never pass
        result = GateResult(
            gate=gate, status=GateStatus.REFUSED,
            reasons=[f"Gate error ({type(err).__name__}): {err}"],
        )

    span.attributes["gate.status"] = result.status.value
    span.end(ok=result.passed, message="; ".join(result.reasons) or None)
    update: dict[str, Any] = {
        "gate_status": result.status,
        "gate_results": [result],
        "audit_telemetry": [
            AuditEvent(
                node=gate, event_type=event_type, status=result.status,
                message="; ".join(result.reasons) or "passed",
                detail=result.details, span=span,
            )
        ],
    }
    if not result.passed:
        update["refusal_reason"] = f"[{gate}] " + " ".join(result.reasons)
    return update


def thin_set_gate_node(state: AgentState) -> dict[str, Any]:
    return _run_gate(
        state,
        GATE_THIN_SET,
        lambda: thin_set_refusal_gate(
            state.get("raw_passages", []), state.get("confidence_scores", [])
        ),
    )


def sql_ast_gate_node(state: AgentState, catalog: Catalog = DEFAULT_CATALOG) -> dict[str, Any]:
    return _run_gate(state, GATE_SQL_AST, lambda: sql_ast_gate(state.get("generated_sql"), catalog))


def payload_safety_gate_node(state: AgentState, catalog: Catalog = DEFAULT_CATALOG) -> dict[str, Any]:
    def check() -> GateResult:
        payload = state.get("tool_payload")
        if payload is None:
            payload = {
                "tool_name": TOOL_SNOWFLAKE_QUERY,
                "arguments": {"sql": state.get("generated_sql") or ""},
                "principal": state.get("principal") or "",
            }
        return payload_safety_gate(payload, catalog)

    return _run_gate(state, GATE_PAYLOAD, check)


def citation_gate_node(state: AgentState) -> dict[str, Any]:
    return _run_gate(
        state,
        GATE_CITATION,
        lambda: verbatim_citation_gate(
            state.get("draft_response"), state.get("citations", []), state.get("raw_passages", [])
        ),
    )
