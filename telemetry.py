"""telemetry.py - Export the workflow's audit trail as OpenTelemetry spans.

Every gate and tool call already records an AuditEvent. This turns one run
into a trace: a root span for the request and a child span per event, with
the gate status, refusal reason and token usage as attributes. With the
OTLP exporter pointed at the AWS Distro for OpenTelemetry collector, the
traces land in Amazon CloudWatch next to AgentCore's own spans.

  OTEL_EXPORTER_OTLP_ENDPOINT   collector endpoint; when unset, nothing is exported
  OTEL_SERVICE_NAME             defaults to enterprise-ai-platform

Span attributes carry identifiers and decisions, never query text, SQL
results or passage text.
"""

import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

from schemas import AuditEvent, GateStatus

_SAFE_DETAIL_KEYS = frozenset({
    "auth_path", "snowflake_role", "service_account_id", "airflow_dag_id", "airflow_dag_run_id", "route",
    "passage_count", "avg_confidence", "tables", "sensor_tables", "limit", "citations_checked", "rls_applied",
    "security_violation", "ticket", "retrieved", "dropped_by_entitlement", "readings_scanned", "tool", "call_id",
})


def build_tracer_provider(exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Provider that exports to `exporter`, or to OTLP when an endpoint is configured."""
    if exporter is None:
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return None
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "enterprise-ai-platform")}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _attributes(event: AuditEvent) -> dict[str, Any]:
    attrs: dict[str, Any] = {"platform.node": event.node, "platform.event_type": event.event_type}
    if event.status is not None:
        attrs["platform.gate_status"] = event.status.value
    if event.status is GateStatus.REFUSED and event.event_type in ("gate", "security"):
        attrs["platform.refusal_reason"] = event.message[:300]  # gate reasons name rules and tables, not data
    for key, value in event.detail.items():
        if key in _SAFE_DETAIL_KEYS and isinstance(value, (str, int, float, bool)):
            attrs[f"platform.{key}"] = value
        elif key in _SAFE_DETAIL_KEYS and isinstance(value, list):
            attrs[f"platform.{key}"] = [str(v) for v in value]
    return attrs


def export_run(outcome: dict[str, Any], provider: TracerProvider | None, usage: dict[str, Any] | None = None) -> str | None:
    """Emit one trace for a finished run. Returns the trace id, or None when
    export is not configured. Never raises: telemetry cannot fail a request."""
    if provider is None:
        return None
    try:
        tracer = provider.get_tracer("enterprise_ai_platform")
        status = outcome.get("gate_status")
        status_value = status.value if isinstance(status, GateStatus) else str(status)
        route = outcome.get("route")
        with tracer.start_as_current_span("platform.request") as root:
            root.set_attribute("platform.gate_status", status_value)
            root.set_attribute("platform.route", getattr(route, "value", str(route)))
            root.set_attribute("platform.trace_ref", str(outcome.get("trace_id", "")))
            for key, value in (usage or outcome.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    root.set_attribute(f"gen_ai.usage.{key}", value)
            if status_value != GateStatus.PASSED.value:
                root.set_status(Status(StatusCode.ERROR, status_value))
            for event in outcome.get("audit_telemetry", []):
                with tracer.start_as_current_span(f"{event.event_type}.{event.node}") as span:
                    span.set_attributes(_attributes(event))
                    if event.status is GateStatus.REFUSED:
                        span.set_status(Status(StatusCode.ERROR, "REFUSED"))
            return format(root.get_span_context().trace_id, "032x")
    except Exception:  # noqa: BLE001
        return None
