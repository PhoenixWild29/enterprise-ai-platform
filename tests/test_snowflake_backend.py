"""Snowflake executor and catalog loader against a fake DB-API connection."""
import datetime as dt
import decimal

import pytest

from auth import IDP, TOKEN_VAULT, AuthPath, TokenVault, authenticate_user
from gates import DEFAULT_CATALOG, SQLSecurityError, sql_ast_gate
from graph import AgentDeps, build_graph, run_query
from schemas import DataDomain
from snowflake_backend import (
    GovernanceOverlay, SnowflakeExecutor, SnowflakeSettings, TableGovernance, load_catalog, load_catalog_file,
)

SQL = ("SELECT t.sensor_id, t.reading_ts, t.value FROM SCADA.TELEMETRY_001 AS t "
       "WHERE t.reading_ts >= DATEADD(day, -7, CURRENT_TIMESTAMP()) AND T.USER_ID = CURRENT_USER() LIMIT 100")


class FakeCursor:
    def __init__(self, conn):
        self.conn, self.description, self._rows, self.closed = conn, None, [], False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if self.conn.fail:
            raise RuntimeError("warehouse unavailable")
        self.description, self._rows = self.conn.results.pop(0) if self.conn.results else ([], [])

    def fetchmany(self, n):
        return self._rows[:n]

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class FakeConn:
    def __init__(self, results=None, fail=False):
        self.results, self.fail, self.executed, self.closed, self.cursors = list(results or []), fail, [], False, []

    def cursor(self):
        self.cursors.append(FakeCursor(self))
        return self.cursors[-1]

    def close(self):
        self.closed = True


def _cols(*names):
    return [(n,) for n in names]


TELEMETRY = (_cols("sensor_id", "reading_ts", "value"),
             [("s-1", dt.datetime(2026, 11, 16, 8, 0, tzinfo=dt.UTC), decimal.Decimal("71.50")),
              ("s-2", dt.datetime(2026, 11, 15, 8, 0, tzinfo=dt.UTC), decimal.Decimal("77"))])


def _operator():
    return authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]), idp=IDP)


def test_executor_runs_with_the_callers_own_token_and_closes_everything():
    ctx, seen = _operator(), []
    conn = FakeConn([TELEMETRY])

    def factory(c, token):
        seen.append((c.snowflake_role, token))
        return conn

    rows = SnowflakeExecutor(factory)(SQL, ctx, DEFAULT_CATALOG)
    role, token = seen[0]
    assert role == "OPS_ANALYST" and token == TOKEN_VAULT.get(ctx.token_fingerprint).access_token and token.count(".") == 2
    assert conn.executed == [(SQL, None)] and conn.closed and conn.cursors[0].closed
    assert rows == [
        {"_table": "SCADA.TELEMETRY_001", "SENSOR_ID": "s-1", "READING_TS": "2026-11-16T08:00:00+00:00", "VALUE": 71.5},
        {"_table": "SCADA.TELEMETRY_001", "SENSOR_ID": "s-2", "READING_TS": "2026-11-15T08:00:00+00:00", "VALUE": 77},
    ]


def test_executor_refuses_before_connecting():
    ctx, calls = _operator(), []
    executor = SnowflakeExecutor(lambda c, t: calls.append(1) or FakeConn([TELEMETRY]), vault=TokenVault())
    with pytest.raises(PermissionError, match="session unavailable"):
        executor(SQL, ctx, DEFAULT_CATALOG)  # token not in this vault (e.g. another replica)
    executor = SnowflakeExecutor(lambda c, t: calls.append(1) or FakeConn([TELEMETRY]))
    with pytest.raises(SQLSecurityError):
        executor("DELETE FROM SCADA.TELEMETRY_001", ctx, DEFAULT_CATALOG)
    with pytest.raises(PermissionError, match="outside the catalog"):
        executor("SELECT * FROM SECRET.PAYROLL", ctx, DEFAULT_CATALOG)
    with pytest.raises(PermissionError, match="outside the catalog"):
        executor("SELECT CURRENT_USER()", ctx, DEFAULT_CATALOG)
    assert calls == []


def test_executor_closes_connection_on_failure_and_caps_rows():
    ctx, conn = _operator(), FakeConn(fail=True)
    with pytest.raises(RuntimeError):
        SnowflakeExecutor(lambda c, t: conn)(SQL, ctx, DEFAULT_CATALOG)
    assert conn.closed and conn.cursors[0].closed
    many = (_cols("sensor_id"), [(f"s-{i}",) for i in range(50)])
    assert len(SnowflakeExecutor(lambda c, t: FakeConn([many]), max_rows=7)(SQL, ctx, DEFAULT_CATALOG)) == 7


def test_expired_vault_token_is_refused_and_purged():
    from auth import DownstreamToken
    vault = TokenVault()
    fp = vault.put(DownstreamToken(access_token="tok", expires_at=dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)))
    with pytest.raises(Exception, match="sign in again"):
        vault.get(fp)
    assert len(vault) == 0


def test_graph_end_to_end_on_snowflake_executor_and_warehouse_failure():
    conns = []

    def factory(c, token):
        conns.append(FakeConn([TELEMETRY]))
        return conns[-1]

    graph = build_graph(AgentDeps(executor=SnowflakeExecutor(factory)))
    out = run_query("List the latest telemetry readings", _operator(), graph)
    assert out["gate_status"] == "PASSED" and len(out["execution_results"]) == 2
    assert "USER_ID = CURRENT_USER()" in conns[0].executed[0][0]  # RLS predicate reached the warehouse
    assert '"READING_TS": "2026-11-16' in out["final_response"]

    graph = build_graph(AgentDeps(executor=SnowflakeExecutor(lambda c, t: FakeConn(fail=True))))
    out = run_query("List the latest telemetry readings", _operator(), graph)
    assert out["gate_status"] == "REFUSED" and "Execution failed" in out["refusal_reason"] and not out["execution_results"]


OVERLAY = GovernanceOverlay(
    schema_domains={"SCADA": DataDomain.SCADA, "MAINT": DataDomain.MAINTENANCE},
    default_time_columns=["READING_TS", "OPENED_AT"], sensor_schemas=["SCADA"],
    tables={"MAINT.EQUIPMENT": TableGovernance()},
)
COLUMNS = (_cols("TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME"), [
    ("SCADA", "PUMP_READINGS", "READING_ID"), ("SCADA", "PUMP_READINGS", "EQUIPMENT_ID"),
    ("SCADA", "PUMP_READINGS", "READING_TS"), ("SCADA", "PUMP_READINGS", "VALUE"),
    ("MAINT", "EQUIPMENT", "EQUIPMENT_ID"), ("MAINT", "EQUIPMENT", "SITE_ID"),
])
PKS = (_cols("created_on", "database_name", "schema_name", "table_name", "column_name", "key_sequence"), [
    (None, "LAKEHOUSE", "SCADA", "PUMP_READINGS", "READING_ID", 1), (None, "LAKEHOUSE", "MAINT", "EQUIPMENT", "EQUIPMENT_ID", 1),
])
FKS = (_cols("pk_schema_name", "pk_table_name", "pk_column_name", "fk_schema_name", "fk_table_name", "fk_column_name"), [
    ("MAINT", "EQUIPMENT", "EQUIPMENT_ID", "SCADA", "PUMP_READINGS", "EQUIPMENT_ID"),
    ("HR", "PEOPLE", "PERSON_ID", "SCADA", "PUMP_READINGS", "VALUE"),       # parent not governed: dropped
    ("MAINT", "EQUIPMENT", "SITE_ID", "SCADA", "PUMP_READINGS", "VALUE"),   # parent column is not a key: dropped
])


def test_catalog_loads_from_metadata_and_drives_the_gates(tmp_path):
    conn = FakeConn([COLUMNS, PKS, FKS])
    catalog = load_catalog(conn, "LAKEHOUSE", OVERLAY)
    assert conn.executed[0][1] == ["SCADA", "MAINT"] and "INFORMATION_SCHEMA.COLUMNS" in conn.executed[0][0]
    readings = catalog.get("SCADA.PUMP_READINGS")
    assert readings.is_sensor and readings.time_column == "READING_TS" and readings.primary_key == ("READING_ID",)
    assert [(fk.column, fk.ref_table) for fk in readings.foreign_keys] == [("EQUIPMENT_ID", "MAINT.EQUIPMENT")]
    assert catalog.get("MAINT.EQUIPMENT").time_column is None and len(catalog) == 2

    good = ("SELECT r.value FROM SCADA.PUMP_READINGS r JOIN MAINT.EQUIPMENT e ON r.equipment_id = e.equipment_id "
            "WHERE r.reading_ts >= '2026-01-01' LIMIT 10")
    assert sql_ast_gate(good, catalog).passed
    assert not sql_ast_gate("SELECT value FROM SCADA.PUMP_READINGS LIMIT 10", catalog).passed
    assert not sql_ast_gate(good.replace("PUMP_READINGS", "TELEMETRY_001"), catalog).passed  # demo table is not in this catalog

    path = tmp_path / "catalog.json"
    path.write_text(catalog.model_dump_json())
    assert load_catalog_file(str(path)).get("SCADA.PUMP_READINGS").foreign_keys == readings.foreign_keys


def test_catalog_loader_fails_closed():
    no_time = OVERLAY.model_copy(update={"default_time_columns": []})
    with pytest.raises(ValueError, match="sensor tables must declare a time column"):
        load_catalog(FakeConn([COLUMNS, PKS, FKS]), "LAKEHOUSE", no_time)
    bad_col = OVERLAY.model_copy(update={"tables": {"MAINT.EQUIPMENT": TableGovernance(time_column="NOPE")}})
    with pytest.raises(ValueError, match="does not exist"):
        load_catalog(FakeConn([COLUMNS, PKS, FKS]), "LAKEHOUSE", bad_col)
    with pytest.raises(ValueError, match="Invalid database"):
        load_catalog(FakeConn(), "LAKEHOUSE; DROP TABLE X", OVERLAY)
    with pytest.raises(ValueError, match="No tables"):
        load_catalog(FakeConn([(COLUMNS[0], []), PKS, FKS]), "LAKEHOUSE", OVERLAY)


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        SnowflakeSettings.from_env()
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acme-prod")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "AI_WH")
    assert SnowflakeSettings.from_env().database == "LAKEHOUSE"
