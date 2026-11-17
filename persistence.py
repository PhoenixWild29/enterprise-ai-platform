"""persistence.py - Checkpointer selection for the LangGraph workflow.

A checkpointer is what lets a run pause for human review and resume later.
Checkpoints contain the run's full state, including query results, so the
API deletes a thread's checkpoints as soon as the run finishes (see
`forget_thread`); only runs waiting on a reviewer stay on disk. Encrypt the
store at rest in production.
Backends (CHECKPOINT_BACKEND):
  sqlite   (default) single-file, survives restarts; one API process
  postgres           shared store for multiple API replicas (CHECKPOINT_DSN)
  memory             tests only; paused runs are lost on restart
"""

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import schemas

Backend = Literal["sqlite", "postgres", "memory"]
_NO_LOCK = contextlib.nullcontext()
DEFAULT_SQLITE_PATH = ".data/checkpoints.sqlite"


def state_serializer() -> JsonPlusSerializer:
    """Serializer that is allowed to restore our own state models and enums.
    Anything else found in a checkpoint is not deserialized into objects."""
    allowed = [
        ("schemas", name)
        for name, obj in vars(schemas).items()
        if isinstance(obj, type) and obj.__module__ == "schemas"
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def sqlite_checkpointer(path: str | os.PathLike[str] = DEFAULT_SQLITE_PATH) -> BaseCheckpointSaver:
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch(mode=0o600, exist_ok=True)  # paused runs hold query results: owner-only
    os.chmod(db_path, 0o600)
    # FastAPI runs sync endpoints on a thread pool; SqliteSaver serialises access itself.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA secure_delete = ON")  # deleted checkpoints are overwritten, not just unlinked
    saver = SqliteSaver(conn, serde=state_serializer())
    saver.setup()
    return saver


def postgres_checkpointer(dsn: str) -> BaseCheckpointSaver:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # A pool replaces dead connections, so a dropped connection is not an outage.
    pool = ConnectionPool(
        dsn, min_size=1, max_size=10, open=True, check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool, serde=state_serializer())
    saver.setup()  # creates the checkpoint tables on first use
    return saver


def make_checkpointer(backend: str | None = None) -> BaseCheckpointSaver:
    """Build the checkpointer named by `backend` or CHECKPOINT_BACKEND."""
    choice = (backend or os.environ.get("CHECKPOINT_BACKEND", "sqlite")).lower()
    match choice:
        case "memory":
            return InMemorySaver(serde=state_serializer())
        case "sqlite":
            return sqlite_checkpointer(os.environ.get("CHECKPOINT_SQLITE_PATH", DEFAULT_SQLITE_PATH))
        case "postgres":
            dsn = os.environ.get("CHECKPOINT_DSN")
            if not dsn:
                raise RuntimeError("CHECKPOINT_BACKEND=postgres requires CHECKPOINT_DSN.")
            return postgres_checkpointer(dsn)
        case _:
            raise RuntimeError(f"Unknown CHECKPOINT_BACKEND '{choice}' (use sqlite, postgres or memory).")


def forget_thread(checkpointer: BaseCheckpointSaver, thread_id: str) -> None:
    """Delete every checkpoint of a finished run, including from the SQLite
    write-ahead log, where deleted rows otherwise stay readable until the
    connection closes."""
    checkpointer.delete_thread(thread_id)
    conn = getattr(checkpointer, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        with getattr(checkpointer, "lock", _NO_LOCK):
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
