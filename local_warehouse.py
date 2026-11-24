"""A small local warehouse so governed SQL actually runs.

The simulated executor returns the same three rows whatever a query asks,
which is enough to test the gates and useless for measuring whether a query
is right. This module seeds every catalog table with deterministic data in
an in-memory DuckDB and executes the secured SQL against it, so two
different queries return two different results and a result can be graded.

It is a test double for Snowflake, not a replacement:

  - the SQL is re-validated, then translated from the Snowflake dialect
  - CURRENT_TIMESTAMP() is a frozen instant, so results do not drift from
    one run to the next
  - CURRENT_USER() is the caller's Snowflake user, so the row-level security
    predicate the platform adds really does filter rows
"""
from __future__ import annotations

import hashlib
import random
import threading
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import duckdb
from sqlglot import exp

from auth import AuthContext
from gates import SQL_DIALECT, parse_read_only_sql, resolve_table
from schemas import Catalog, TableSpec

FROZEN_NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
SEED = "quality-eval-v2"
# Rows are owned by a small set of people, so row-level security has something to filter.
OPS_USERS = ("DANA.OPS@CONTOSO.COM", "BOB.OPS@CONTOSO.COM", "RAVI.OPS@CONTOSO.COM")
FINANCE_USERS = ("FAY.FIN@CONTOSO.COM", "OMAR.FIN@CONTOSO.COM")
EQUIPMENT_TYPES = ("PUMP", "COMPRESSOR", "VALVE", "SEPARATOR")
FAILURE_CODES = ("SEAL_LEAK", "BEARING_WEAR", "VIBRATION_HIGH", "OVERHEAT", "SENSOR_FAULT")
SENSORS = ("TT-101", "PT-102", "VT-103", "FT-104")
EQUIPMENT_COUNT, COST_CENTER_COUNT = 40, 12
# Days of history per kind of table: more than twice the longest window any question uses (14, 120 and 180 days).
TELEMETRY_DAYS, WORK_ORDER_DAYS, LEDGER_DAYS = 40, 300, 400


def _rng(name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:16], 16))


_ROWS: dict[tuple[str, datetime], list[tuple[Any, ...]]] = {}


def _rows_for(spec: TableSpec, now: datetime) -> list[tuple[Any, ...]]:
    """Deterministic rows for one table, derived only from its name.

    The data is shaped so the planted mistakes in quality_eval.py return a
    different answer from the right query (the audit checks every one). It
    does not catch every wrong query: a window that is off by a day can
    return the same maximum or the same top five. History runs well past twice the longest window any question
    asks about, each table has its own volume and its own mix of row owners,
    and every sensor reports several times a day, so MAX, AVG and MIN differ."""
    key = (spec.fqn, now)
    if key not in _ROWS:
        _ROWS[key] = _build_rows(spec, now)
    return _ROWS[key]


def _build_rows(spec: TableSpec, now: datetime) -> list[tuple[Any, ...]]:
    rng, naive_now = _rng(spec.fqn), now.replace(tzinfo=None)
    if spec.fqn == "MAINT.EQUIPMENT":
        return [(i, f"SITE-{1 + i % 8:02d}", EQUIPMENT_TYPES[i % 4], naive_now - timedelta(days=400 + 37 * i)) for i in range(1, EQUIPMENT_COUNT + 1)]
    if spec.fqn == "FIN.COST_CENTERS":
        return [(i, f"Cost center {i:02d}", FINANCE_USERS[i % 2]) for i in range(1, COST_CENTER_COUNT + 1)]
    rows: list[tuple[Any, ...]] = []
    if spec.name.startswith("TELEMETRY_"):
        equipment = rng.sample(range(1, EQUIPMENT_COUNT + 1), 6)
        owners = [rng.uniform(0.6, 1.6) for _ in OPS_USERS]
        for i in range(rng.randint(TELEMETRY_DAYS * 30, TELEMETRY_DAYS * 50)):
            when = naive_now - timedelta(seconds=rng.uniform(0, TELEMETRY_DAYS * 86400))
            sensor = rng.choice(SENSORS)
            base = {"TT-101": 70.0, "PT-102": 310.0, "VT-103": 4.0, "FT-104": 125.0}[sensor]
            value = round(base + rng.gauss(0, base * 0.04) + (base * 0.6 if rng.random() < 0.02 else 0), 2)
            flag = "BAD" if rng.random() < 0.05 else "SUSPECT" if rng.random() < 0.08 else "GOOD"
            rows.append((i + 1, rng.choice(equipment), sensor, when, value, flag, rng.choices(OPS_USERS, owners)[0]))
    elif spec.name.startswith("WORK_ORDERS_"):
        owners = [rng.uniform(0.6, 1.6) for _ in OPS_USERS]
        for i in range(rng.randint(500, 900)):
            opened = naive_now - timedelta(days=rng.uniform(0, WORK_ORDER_DAYS))
            closed = opened + timedelta(hours=rng.uniform(2, 96)) if rng.random() < 0.8 else None
            code = rng.choice(FAILURE_CODES)
            rows.append((i + 1, rng.randint(1, EQUIPMENT_COUNT), opened, closed, code, f"Inspected and logged {code.lower().replace('_', ' ')}.",
                         rng.choices(OPS_USERS, owners)[0]))
    elif spec.name.startswith("LEDGER_"):
        owners = [rng.uniform(0.7, 1.4) for _ in FINANCE_USERS]
        for i in range(rng.randint(500, 900)):
            posted = naive_now - timedelta(days=rng.uniform(0, LEDGER_DAYS))
            rows.append((i + 1, rng.randint(1, COST_CENTER_COUNT), rng.randint(1, EQUIPMENT_COUNT), posted,
                         round(rng.uniform(150, 9500), 2), "USD", rng.choices(FINANCE_USERS, owners)[0]))
    return rows


def _column_type(column: str) -> str:
    if column.endswith(("_TS", "_AT")):
        return "TIMESTAMP"
    if column in ("VALUE", "AMOUNT"):
        return "DOUBLE"
    if column.endswith("_ID") and column not in ("SENSOR_ID", "USER_ID", "SITE_ID", "OWNER_USER_ID"):
        return "BIGINT"
    return "VARCHAR"


class LocalWarehouse:
    """In-memory DuckDB behind the platform's SQLExecutor signature. Tables
    are created and seeded the first time a query reads them."""

    def __init__(self, catalog: Catalog, *, now: datetime = FROZEN_NOW) -> None:
        self._catalog, self._now = catalog, now
        self._db = duckdb.connect(":memory:")
        self._seeded: set[str] = set()
        self._lock = threading.Lock()
        self.statements: list[str] = []   # what was executed, newest last, for the report
        self.results: list[list[dict[str, Any]]] = []   # the rows each statement returned, in the same order

    def _ensure(self, spec: TableSpec) -> None:
        if spec.fqn in self._seeded:
            return
        self._db.execute(f"CREATE SCHEMA IF NOT EXISTS {spec.schema_name}")
        columns = ", ".join(f"{c} {_column_type(c)}" for c in spec.columns)
        self._db.execute(f"CREATE TABLE {spec.fqn} ({columns})")
        rows = _rows_for(spec, self._now)
        one = f"({', '.join('?' * len(spec.columns))})"
        for start in range(0, len(rows), 500):   # multi-row inserts: executemany is far too slow for a per-run warehouse
            chunk = rows[start:start + 500]
            self._db.execute(f"INSERT INTO {spec.fqn} VALUES {', '.join([one] * len(chunk))}", [v for row in chunk for v in row])
        self._seeded.add(spec.fqn)

    def __call__(self, sql: str, ctx: AuthContext, catalog: Catalog) -> list[dict[str, Any]]:
        root = parse_read_only_sql(sql)   # the same policy as the real executor: nothing unvalidated runs
        def literal(text: str, kind: str) -> exp.Expression:
            return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build(kind))

        def freeze(node: exp.Expression) -> exp.Expression:
            # Every clock function reads the frozen instant, never the wall clock, or verdicts would drift by the day.
            if isinstance(node, (exp.CurrentTimestamp, exp.CurrentDatetime)):
                return literal(self._now.strftime("%Y-%m-%d %H:%M:%S"), "TIMESTAMP")
            if isinstance(node, exp.CurrentDate):
                return literal(self._now.strftime("%Y-%m-%d"), "DATE")
            if isinstance(node, exp.CurrentTime):
                return literal(self._now.strftime("%H:%M:%S"), "TIME")
            if isinstance(node, exp.CurrentUser):
                return exp.Literal.string(ctx.snowflake_user)
            return node

        root = root.transform(freeze)
        statement = root.sql(dialect="duckdb")
        with self._lock:
            for table in root.find_all(exp.Table):
                spec = resolve_table(table, self._catalog)
                if spec is not None:
                    self._ensure(spec)
            cursor = self._db.execute(statement)
            names = [d[0].upper() for d in cursor.description]
            rows = cursor.fetchall()
            result = [dict(zip(names, (_plain(v) for v in row))) for row in rows]
            self.statements.append(statement)
            self.results.append(result)
        return [dict(row) for row in result]


def _plain(value: Any) -> Any:
    """Values as JSON-friendly scalars, rounded so equal results compare equal."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (float, Decimal)):
        return round(float(value), 4)
    return value


__all__ = ["FROZEN_NOW", "LocalWarehouse", "SQL_DIALECT"]
