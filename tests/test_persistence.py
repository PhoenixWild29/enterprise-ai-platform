"""A paused human review must survive a process restart."""
import pytest
from langgraph.types import Command

from auth import IDP, authenticate_user
from graph import AgentDeps, build_graph, run_query
from persistence import make_checkpointer, sqlite_checkpointer
from schemas import Citation

ASK = "What is the bearing temperature alarm limit for pump P-101?"


def _fabricating(query, passages):
    return "Pump is fine.", [Citation(passage_id=passages[0].passage_id, snippet="this snippet was invented by the model, not the source")]


def _operator():
    return authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]), idp=IDP)


def _pause_then_resume_after_restart(make_saver):
    deps = AgentDeps(synthesizer=_fabricating)
    paused = run_query(ASK, _operator(), build_graph(deps, checkpointer=make_saver()), thread_id="t-restart")
    assert "__interrupt__" in paused

    # "Restart": brand-new saver and graph objects over the same store.
    graph = build_graph(deps, checkpointer=make_saver())
    config = {"configurable": {"thread_id": "t-restart"}}
    snapshot = graph.get_state(config)
    assert snapshot.next == ("hitl_review",) and snapshot.values["query"] == ASK
    out = graph.invoke(Command(resume={"approved": True, "reviewer": "lee.admin@contoso.com"}), config=config)
    assert out["gate_status"] == "PASSED" and out["final_response"] == "Pump is fine."
    assert out["audit_telemetry"][0].node == "auth_gate"  # full audit trail restored as typed objects


def test_sqlite_checkpoint_survives_restart(tmp_path):
    _pause_then_resume_after_restart(lambda: sqlite_checkpointer(tmp_path / "cp.sqlite"))


def test_postgres_checkpoint_survives_restart():
    import os

    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set")
    from persistence import postgres_checkpointer

    with __import__("psycopg").connect(dsn, autocommit=True) as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = 't-restart'") if conn.execute(
                "SELECT to_regclass(%s)", (table,)
            ).fetchone()[0] else None
    _pause_then_resume_after_restart(lambda: postgres_checkpointer(dsn))


def test_sqlite_store_is_owner_only_and_threads_can_be_forgotten(tmp_path):
    import stat
    from persistence import forget_thread

    path = tmp_path / "cp.sqlite"
    saver = sqlite_checkpointer(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    graph = build_graph(AgentDeps(), checkpointer=saver)
    config = {"configurable": {"thread_id": "t-done"}}
    run_query(ASK, _operator(), graph, thread_id="t-done")
    assert graph.get_state(config).values["query"] == ASK
    forget_thread(saver, "t-done")
    assert graph.get_state(config).values == {}
    assert b"bearing" not in path.read_bytes() or saver.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0


def test_backend_selection(monkeypatch, tmp_path):
    assert type(make_checkpointer("memory")).__name__ == "InMemorySaver"
    monkeypatch.setenv("CHECKPOINT_SQLITE_PATH", str(tmp_path / "x" / "cp.sqlite"))
    assert type(make_checkpointer("sqlite")).__name__ == "SqliteSaver"
    monkeypatch.delenv("CHECKPOINT_DSN", raising=False)
    with pytest.raises(RuntimeError, match="CHECKPOINT_DSN"):
        make_checkpointer("postgres")
    with pytest.raises(RuntimeError, match="Unknown"):
        make_checkpointer("redis")
