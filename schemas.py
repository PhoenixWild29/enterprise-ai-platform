"""schemas.py - State, catalog, tool-payload, MCP and telemetry contracts.

Everything that crosses a node boundary in the LangGraph workflow is defined
here so that gates, auth, the graph and the eval harness share one contract.
"""

import operator
import secrets
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GateStatus(StrEnum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    REFUSED = "REFUSED"
    HITL_REVIEW = "HITL_REVIEW"


class DataDomain(StrEnum):
    SCADA = "SCADA"
    MAINTENANCE = "MAINTENANCE"
    FINANCE = "FINANCE"


class Route(StrEnum):
    RAG = "rag_retriever"
    TEXT_TO_SQL = "text_to_sql"
    DIAGNOSTIC = "diagnostic_reasoner"
    REFUSE = "refusal"
    HITL = "hitl_review"


class SpanKind(StrEnum):
    INTERNAL = "INTERNAL"
    CLIENT = "CLIENT"
    SERVER = "SERVER"


class SpanStatusCode(StrEnum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class _Strict(BaseModel):
    """Base for all contracts: unknown fields are rejected (fail closed)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Retrieval and citation models
# ---------------------------------------------------------------------------


class Passage(_Strict):
    # Passage text is evidence: never strip or alter it.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    passage_id: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Document URI or table of origin")
    text: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0, description="Retriever confidence")
    domain: DataDomain | None = None


class Citation(_Strict):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    passage_id: str = Field(min_length=1)
    snippet: str = Field(min_length=1, description="Verbatim text copied from the passage")

    @classmethod
    def from_passage(cls, passage: Passage, length: int = 40) -> Self:
        """Canonical citation: the first `length` characters of the passage."""
        return cls(passage_id=passage.passage_id, snippet=passage.text[:length])


# ---------------------------------------------------------------------------
# Snowflake catalog models
# ---------------------------------------------------------------------------


class ForeignKey(_Strict):
    column: str
    ref_table: str = Field(description="Fully qualified SCHEMA.TABLE of the parent")
    ref_column: str

    @field_validator("column", "ref_table", "ref_column")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class TableSpec(_Strict):
    schema_name: str
    name: str
    domain: DataDomain
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    time_column: str | None = Field(
        default=None, description="Column that every query on this table must bound"
    )
    is_sensor: bool = Field(default=False, description="High-volume telemetry table")
    description: str = Field(default="", max_length=300, description="What the table holds, in business terms")

    @field_validator("schema_name", "name")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("columns", "primary_key")
    @classmethod
    def _upper_all(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(c.upper() for c in v)

    @field_validator("time_column")
    @classmethod
    def _upper_opt(cls, v: str | None) -> str | None:
        return v.upper() if v else None

    @model_validator(mode="after")
    def _columns_consistent(self) -> Self:
        cols = set(self.columns)
        referenced = set(self.primary_key) | {fk.column for fk in self.foreign_keys}
        if self.time_column:
            referenced.add(self.time_column)
        if missing := referenced - cols:
            raise ValueError(f"{self.fqn}: key/time columns not in table: {sorted(missing)}")
        return self

    @property
    def fqn(self) -> str:
        return f"{self.schema_name}.{self.name}"


class Catalog(_Strict):
    database: str = "LAKEHOUSE"
    tables: dict[str, TableSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _foreign_keys_resolve(self) -> Self:
        for spec in self.tables.values():
            for fk in spec.foreign_keys:
                parent = self.tables.get(fk.ref_table)
                if parent is None or fk.ref_column not in parent.primary_key:
                    raise ValueError(
                        f"{spec.fqn}.{fk.column} references unknown key "
                        f"{fk.ref_table}.{fk.ref_column}"
                    )
        return self

    @classmethod
    def from_tables(cls, tables: list[TableSpec], database: str = "LAKEHOUSE") -> Self:
        return cls(database=database.upper(), tables={t.fqn: t for t in tables})

    def get(self, fqn: str) -> TableSpec | None:
        return self.tables.get(fqn.upper())

    def __len__(self) -> int:
        return len(self.tables)

    def is_key_join(self, a: TableSpec, col_a: str, b: TableSpec, col_b: str) -> bool:
        """True when a.col_a = b.col_b follows a declared key relationship.

        Accepted: FK -> parent PK (either direction), or two FKs that
        reference the same parent key (sibling join through a shared key).
        """
        col_a, col_b = col_a.upper(), col_b.upper()
        for child, c_col, parent, p_col in ((a, col_a, b, col_b), (b, col_b, a, col_a)):
            for fk in child.foreign_keys:
                if fk.column == c_col and fk.ref_table == parent.fqn and fk.ref_column == p_col:
                    return True
        refs_a = {(fk.ref_table, fk.ref_column) for fk in a.foreign_keys if fk.column == col_a}
        refs_b = {(fk.ref_table, fk.ref_column) for fk in b.foreign_keys if fk.column == col_b}
        return bool(refs_a & refs_b)


# ---------------------------------------------------------------------------
# Tool payloads and MCP tool definitions
# ---------------------------------------------------------------------------


class SnowflakeQueryArgs(_Strict):
    sql: str = Field(min_length=1, max_length=20_000)
    max_rows: int = Field(default=1_000, ge=1, le=10_000)
    timeout_seconds: int = Field(default=60, ge=1, le=300)


class VectorSearchArgs(_Strict):
    query: str = Field(min_length=1, max_length=4_000)
    top_k: int = Field(default=8, ge=1, le=50)
    domain: DataDomain | None = None


class DiagnosticArgs(_Strict):
    equipment_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    symptom: str = Field(min_length=1, max_length=2_000)
    window_hours: int = Field(default=24, ge=1, le=720)


class ToolPayload(_Strict):
    """Envelope for every tool invocation an agent wants to make."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    call_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    principal: str = Field(min_length=1, description="User or service account making the call")


class ToolResult(_Strict):
    call_id: str
    tool_name: str
    ok: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class MCPToolDefinition(BaseModel):
    """Model Context Protocol tool descriptor (tools/list entry)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    read_only: bool = True

    @classmethod
    def from_args_model(cls, name: str, description: str, args_model: type[BaseModel]) -> Self:
        return cls(name=name, description=description, input_schema=args_model.model_json_schema())


# ---------------------------------------------------------------------------
# OpenTelemetry trace spans and audit telemetry
# ---------------------------------------------------------------------------


class SpanEvent(_Strict):
    name: str
    time_unix_nano: int = Field(default_factory=time.time_ns)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class OTelSpan(_Strict):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    name: str
    kind: SpanKind = SpanKind.INTERNAL
    start_time_unix_nano: int
    end_time_unix_nano: int | None = None
    status_code: SpanStatusCode = SpanStatusCode.UNSET
    status_message: str | None = None
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)

    @classmethod
    def start(
        cls,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Self:
        return cls(
            trace_id=trace_id or secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time_unix_nano=time.time_ns(),
            attributes=attributes or {},
        )

    def end(self, ok: bool, message: str | None = None) -> Self:
        self.end_time_unix_nano = time.time_ns()
        self.status_code = SpanStatusCode.OK if ok else SpanStatusCode.ERROR
        self.status_message = message
        return self


class GateResult(_Strict):
    gate: str
    status: GateStatus
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASSED


class AuditEvent(_Strict):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node: str
    event_type: Literal["gate", "route", "tool_call", "auth", "security", "output"]
    status: GateStatus | None = None
    message: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    span: OTelSpan | None = None


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared workflow state. Nodes return partial dicts; list fields annotated
    with operator.add are appended to rather than overwritten."""

    # Required by the spec
    query: str
    raw_passages: list[Passage]
    confidence_scores: list[float]
    generated_sql: str | None
    execution_results: list[dict[str, Any]]
    draft_response: str | None
    citations: list[Citation]
    gate_status: GateStatus
    audit_telemetry: Annotated[list[AuditEvent], operator.add]

    # Supporting fields used by routing, auth and output nodes
    route: Route | None
    tool_payload: ToolPayload | None
    gate_results: Annotated[list[GateResult], operator.add]
    refusal_reason: str | None
    principal: str | None
    auth_context: dict[str, Any]
    trace_id: str
    final_response: str | None


def new_state(query: str, principal: str | None = None) -> AgentState:
    """Fresh state with every spec field initialised."""
    return AgentState(
        query=query,
        raw_passages=[],
        confidence_scores=[],
        generated_sql=None,
        execution_results=[],
        draft_response=None,
        citations=[],
        gate_status=GateStatus.PENDING,
        audit_telemetry=[],
        route=None,
        tool_payload=None,
        gate_results=[],
        refusal_reason=None,
        principal=principal,
        auth_context={},
        trace_id=secrets.token_hex(16),
        final_response=None,
    )
