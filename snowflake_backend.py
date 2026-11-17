"""snowflake_backend.py - Snowflake warehouse executor and catalog loader.

SnowflakeExecutor
    Runs gated, read-only SQL in a session opened with the caller's own
    OAuth access token (the OBO token on the user path, the client-credentials
    token on the headless path), so Snowflake itself evaluates CURRENT_USER
    and the session role. The token is looked up in the in-process
    TokenVault by fingerprint; it is never part of workflow state.

load_catalog
    Builds the governed `Catalog` from Snowflake metadata (columns, primary
    and foreign keys) plus a governance overlay that supplies what Snowflake
    does not know: which schemas are governed, each table's time column and
    which tables are high-volume sensor tables.

Configuration: SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE.
Requires `snowflake-connector-python` (imported on first connection).

Export a catalog file for the API to load at startup (CATALOG_PATH):
    python snowflake_backend.py export-catalog --overlay governance.json --out catalog.json
"""

import argparse
import datetime as dt
import decimal
import logging
import os
import re
import sys
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from auth import TOKEN_VAULT, AuthContext, AuthError, TokenVault
from gates import _cte_names, _is_cte_ref, _table_fqn, parse_read_only_sql
from schemas import Catalog, DataDomain, ForeignKey, TableSpec
from sqlglot import exp

logger = logging.getLogger("platform.snowflake")

type ConnectionFactory = Callable[[AuthContext, str], Any]  # (identity, access token) -> DB-API connection

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class SnowflakeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account: str = Field(min_length=1)
    warehouse: str = Field(min_length=1)
    database: str = Field(default="LAKEHOUSE", pattern=_IDENTIFIER.pattern)
    statement_timeout_seconds: int = Field(default=60, ge=1, le=600)

    @classmethod
    def from_env(cls) -> "SnowflakeSettings":
        try:
            return cls(
                account=os.environ.get("SNOWFLAKE_ACCOUNT", ""),
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
                database=os.environ.get("SNOWFLAKE_DATABASE", "LAKEHOUSE"),
            )
        except ValueError as err:
            raise RuntimeError(f"Snowflake settings are incomplete: {err}") from err


def oauth_connection_factory(settings: SnowflakeSettings) -> ConnectionFactory:
    """Opens one short-lived session per query with the caller's OAuth token."""

    def connect(ctx: AuthContext, access_token: str) -> Any:
        import snowflake.connector  # imported lazily: optional dependency

        return snowflake.connector.connect(
            account=settings.account,
            authenticator="oauth",
            token=access_token,
            role=ctx.snowflake_role,
            warehouse=settings.warehouse,
            database=settings.database,
            client_session_keep_alive=False,
            session_parameters={
                "QUERY_TAG": f"enterprise-ai-platform:{ctx.auth_path.value}:{ctx.token_fingerprint}",
                "STATEMENT_TIMEOUT_IN_SECONDS": settings.statement_timeout_seconds,
            },
        )

    return connect


def _jsonable(value: Any) -> Any:
    """Rows are stored in workflow state and quoted as evidence, so every
    value must serialise to JSON deterministically."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return str(value)


class SnowflakeExecutor:
    """`graph.SQLExecutor` backed by a real warehouse."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        vault: TokenVault = TOKEN_VAULT,
        max_rows: int = 1_000,
    ) -> None:
        self._connect = connection_factory
        self._vault = vault
        self._max_rows = max_rows

    def __call__(self, sql: str, ctx: AuthContext, catalog: Catalog) -> list[dict[str, Any]]:
        # Defence in depth: the gates ran already, but nothing that is not a
        # single read-only SELECT over catalog tables ever reaches the wire.
        root = parse_read_only_sql(sql)
        cte_names = _cte_names(root)
        tables = [t for t in root.find_all(exp.Table) if not _is_cte_ref(t, cte_names)]
        unknown = [_table_fqn(t) for t in tables if catalog.get(_table_fqn(t)) is None]
        if unknown or not tables:
            raise PermissionError(f"Query references tables outside the catalog: {unknown or 'none'}")
        try:
            token = self._vault.get(ctx.token_fingerprint)
        except AuthError as err:
            raise PermissionError(f"Warehouse session unavailable: {err}") from err

        conn = self._connect(ctx, token.access_token)
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                columns = [str(col[0]).upper() for col in cursor.description or []]
                fetched = cursor.fetchmany(self._max_rows)
            finally:
                cursor.close()
        finally:
            conn.close()
        source = _table_fqn(tables[0])
        rows = [{"_table": source} | {c: _jsonable(v) for c, v in zip(columns, row, strict=True)} for row in fetched]
        logger.info(
            "snowflake.query.ok principal=%s role=%s rows=%d tables=%s",
            ctx.principal, ctx.snowflake_role, len(rows), sorted({_table_fqn(t) for t in tables}),
        )
        return rows


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


class TableGovernance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_column: str | None = None
    is_sensor: bool = False


class GovernanceOverlay(BaseModel):
    """Governance metadata Snowflake does not hold."""

    model_config = ConfigDict(extra="forbid")

    schema_domains: dict[str, DataDomain] = Field(description="Governed schemas and their data domain")
    default_time_columns: list[str] = Field(
        default_factory=list, description="Tried in order when a table has no explicit time column"
    )
    sensor_schemas: list[str] = Field(default_factory=list, description="Every table here is a sensor table")
    tables: dict[str, TableGovernance] = Field(default_factory=dict, description="Per-table overrides by SCHEMA.TABLE")


def _dict_rows(cursor: Any) -> list[dict[str, Any]]:
    names = [str(col[0]).lower() for col in cursor.description or []]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def load_catalog(conn: Any, database: str, overlay: GovernanceOverlay) -> Catalog:
    """Read columns and key constraints for the governed schemas and merge the
    governance overlay. Fails closed: a sensor table with no time column is an
    error, and foreign keys that cannot be verified are dropped."""
    if not _IDENTIFIER.match(database):
        raise ValueError(f"Invalid database identifier: {database!r}")
    domains = {schema.upper(): domain for schema, domain in overlay.schema_domains.items()}
    if not domains:
        raise ValueError("Governance overlay lists no governed schemas.")
    per_table = {name.upper(): gov for name, gov in overlay.tables.items()}
    sensor_schemas = {s.upper() for s in overlay.sensor_schemas}
    default_time = [c.upper() for c in overlay.default_time_columns]

    cursor = conn.cursor()
    try:
        placeholders = ", ".join(["%s"] * len(domains))
        cursor.execute(
            f"SELECT table_schema, table_name, column_name FROM {database}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE table_schema IN ({placeholders}) ORDER BY table_schema, table_name, ordinal_position",
            list(domains),
        )
        column_rows = _dict_rows(cursor)
        cursor.execute(f"SHOW PRIMARY KEYS IN DATABASE {database}")
        pk_rows = _dict_rows(cursor)
        cursor.execute(f"SHOW IMPORTED KEYS IN DATABASE {database}")
        fk_rows = _dict_rows(cursor)
    finally:
        cursor.close()

    columns: dict[str, list[str]] = {}
    for row in column_rows:
        columns.setdefault(f"{row['table_schema']}.{row['table_name']}".upper(), []).append(str(row["column_name"]).upper())

    primary: dict[str, list[tuple[int, str]]] = {}
    for row in pk_rows:
        fqn = f"{row['schema_name']}.{row['table_name']}".upper()
        primary.setdefault(fqn, []).append((int(row.get("key_sequence") or 0), str(row["column_name"]).upper()))
    primary_keys = {fqn: tuple(col for _, col in sorted(cols)) for fqn, cols in primary.items()}

    foreign: dict[str, list[ForeignKey]] = {}
    dropped_fks = 0
    for row in fk_rows:
        child = f"{row['fk_schema_name']}.{row['fk_table_name']}".upper()
        parent = f"{row['pk_schema_name']}.{row['pk_table_name']}".upper()
        parent_col = str(row["pk_column_name"]).upper()
        if child in columns and parent in columns and parent_col in primary_keys.get(parent, ()):
            foreign.setdefault(child, []).append(
                ForeignKey(column=str(row["fk_column_name"]), ref_table=parent, ref_column=parent_col)
            )
        else:
            dropped_fks += 1

    specs: list[TableSpec] = []
    for fqn, cols in columns.items():
        schema, name = fqn.split(".", 1)
        gov = per_table.get(fqn, TableGovernance())
        is_sensor = gov.is_sensor or schema in sensor_schemas
        time_column = (gov.time_column or "").upper() or next((c for c in default_time if c in cols), None)
        if time_column and time_column not in cols:
            raise ValueError(f"{fqn}: governance time column {time_column} does not exist.")
        if is_sensor and not time_column:
            raise ValueError(f"{fqn}: sensor tables must declare a time column.")
        specs.append(
            TableSpec(
                schema_name=schema, name=name, domain=domains[schema], columns=tuple(cols),
                primary_key=primary_keys.get(fqn, ()), foreign_keys=tuple(foreign.get(fqn, ())),
                time_column=time_column, is_sensor=is_sensor,
            )
        )
    if not specs:
        raise ValueError("No tables found in the governed schemas.")
    logger.info("snowflake.catalog.loaded tables=%d foreign_keys_dropped=%d", len(specs), dropped_fks)
    return Catalog.from_tables(specs, database=database)


def load_catalog_file(path: str) -> Catalog:
    with open(path, encoding="utf-8") as fh:
        return Catalog.model_validate_json(fh.read())


def service_connection() -> Any:
    """Platform-owned connection for metadata and retrieval, taken from the
    named entry in the Snowflake connections.toml (SNOWFLAKE_CONNECTION_NAME)."""
    import snowflake.connector

    name = os.environ.get("SNOWFLAKE_CONNECTION_NAME")
    if not name:
        raise RuntimeError("Set SNOWFLAKE_CONNECTION_NAME to a connection in connections.toml.")
    return snowflake.connector.connect(connection_name=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snowflake utilities for the Enterprise AI Platform")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-catalog", help="Write the governed catalog to a JSON file")
    export.add_argument("--overlay", required=True, help="Governance overlay JSON file")
    export.add_argument("--out", required=True, help="Catalog JSON file to write")
    export.add_argument("--database", default=os.environ.get("SNOWFLAKE_DATABASE", "LAKEHOUSE"))
    args = parser.parse_args(argv)

    with open(args.overlay, encoding="utf-8") as fh:
        overlay = GovernanceOverlay.model_validate_json(fh.read())
    conn = service_connection()
    try:
        catalog = load_catalog(conn, args.database, overlay)
    finally:
        conn.close()
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(catalog.model_dump_json(indent=2))
    print(f"Wrote {len(catalog)} tables to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
