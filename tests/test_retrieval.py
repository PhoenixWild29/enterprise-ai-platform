import json

import pytest

from auth import IDP, authenticate_user
from graph import KNOWLEDGE_BASE, AgentDeps, build_graph, run_query
from retrieval import (
    HashingEmbedder, SnowflakeVectorRetriever, VectorIndex, VectorRetriever, chunk_text, load_corpus,
)
from runtime import RuntimeConfig, build_agent_deps
from schemas import DataDomain


def _index():
    index = VectorIndex(HashingEmbedder())
    index.add(list(KNOWLEDGE_BASE))
    return index


def test_hashing_embedder_is_deterministic_and_normalised():
    a, b = HashingEmbedder().embed(["Pump bearing temperature", "Pump bearing temperature"])
    assert a == b and abs(sum(v * v for v in a) - 1.0) < 1e-9
    assert HashingEmbedder().embed([""])[0] == [0.0] * 1024


def test_vector_search_ranks_relevant_passages_first():
    hits = _index().search("What is the bearing temperature alarm limit for pump P-101?")
    assert hits[0].passage_id == "sop-pump-014" and hits[0].score > hits[1].score > 0
    assert _index().search("What is the capital of France?") == [] and _index().search("   ") == []
    assert all(0.0 <= p.score <= 1.0 for p in hits)


def test_vector_retriever_through_the_graph():
    ctx = authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]), idp=IDP)
    graph = build_graph(AgentDeps(retriever=VectorRetriever(_index())))
    out = run_query("What is the bearing temperature alarm limit for pump P-101?", ctx, graph)
    assert out["gate_status"] == "PASSED" and out["citations"][0].passage_id == "sop-pump-014"
    out = run_query("What is the capital of France?", ctx, graph)
    assert out["gate_status"] == "REFUSED" and "thin_set" in out["refusal_reason"]


def test_graph_refuses_on_retriever_outage_and_hides_unlabelled_passages():
    from schemas import Passage
    ctx = authenticate_user(IDP.issue_user_token("dana.ops@contoso.com", "oid-1", ["Lakehouse.Operator"]), idp=IDP)

    def down(query):
        raise ConnectionError("vector store unreachable at 10.0.0.5")

    out = run_query("What is the bearing alarm limit?", ctx, build_graph(AgentDeps(retriever=down)))
    assert out["gate_status"] == "REFUSED" and "ConnectionError" in out["refusal_reason"] and "10.0.0.5" not in out["final_response"]
    out = run_query("Why did the pump bearing fail?", ctx, build_graph(AgentDeps(retriever=down)))  # diagnostic route
    assert out["gate_status"] == "REFUSED" and not out["execution_results"]

    def mixed(query):
        text = "Quarterly finance memo: bearing replacement budget is frozen until further notice."
        return ([Passage(passage_id="memo-null", source="doc://memo", text=text, score=0.99, domain=None),
                 Passage(passage_id="memo-fin", source="doc://memo2", text=text, score=0.98, domain=DataDomain.FINANCE)]
                + [Passage(passage_id=f"m{i}", source="sop://m", text=f"Maintenance procedure number {i} covers bearing checks in detail.",
                           score=0.9 - i / 100, domain=DataDomain.MAINTENANCE) for i in range(8)])

    out = run_query("What is the bearing alarm limit?", ctx, build_graph(AgentDeps(retriever=mixed)))
    ids = [p.passage_id for p in out["raw_passages"]]
    assert ids == ["m0", "m1", "m2", "m3"] and "memo" not in out["final_response"]


def test_chunking_and_corpus_loading(tmp_path):
    text = " ".join(f"Sentence number {i} about pump maintenance." for i in range(40))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 5 and all(len(c) <= 260 for c in chunks) and chunks[1].startswith(chunks[0].split(". ")[-1][:20])
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps({"passage_id": f"d{i}", "source": "doc://x", "text": c, "domain": "MAINTENANCE"})
                              for i, c in enumerate(chunks)) + "\n\n")
    corpus = load_corpus(str(path))
    assert len(corpus) == len(chunks) and corpus[0].domain is DataDomain.MAINTENANCE
    path.write_text('{"passage_id": "a", "source": "s", "text": "x", "domain": "SCADA"}\n'
                    '{"passage_id": "a", "source": "s", "text": "y", "domain": "SCADA"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(str(path))
    path.write_text('{"passage_id": "a", "source": "s", "text": "x"}\n')
    with pytest.raises(ValueError, match="no domain"):
        load_corpus(str(path))
    path.write_text('{"passage_id": "a"}\n')
    with pytest.raises(ValueError, match="invalid passage"):
        load_corpus(str(path))


class _Cur:
    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    def execute(self, sql, params):
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows

    def close(self):
        self.log.append("cursor closed")


class _Conn:
    def __init__(self, rows, log):
        self.rows, self.log = rows, log

    def cursor(self):
        return _Cur(self.rows, self.log)

    def close(self):
        self.log.append("conn closed")


def test_snowflake_vector_retriever_binds_inputs_and_maps_rows():
    log = []
    rows = [("p1", "sop://a", "Bearing limit is 85 C.", "MAINTENANCE", 0.83), ("p2", "sop://b", "Vibration limit.", "maintenance", 1.2),
            ("p3", "sop://c", "Mystery.", "LEGAL", 0.9), ("p4", "sop://d", "", "SCADA", 0.9), ("p5", "sop://e", "Opposite.", None, -0.4)]
    retriever = SnowflakeVectorRetriever(lambda: _Conn(rows, log), "LAKEHOUSE.DOCS.PASSAGES", top_k=5)
    hits = retriever("bearing limit'; DROP TABLE x; --")
    sql, params = log[0]
    assert params == ("snowflake-arctic-embed-m-v1.5", "bearing limit'; DROP TABLE x; --") and "DROP" not in sql
    assert "VECTOR_COSINE_SIMILARITY(embedding, SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s))" in sql and sql.endswith("WHERE domain IS NOT NULL ORDER BY score DESC LIMIT 25")
    assert log[-2:] == ["cursor closed", "conn closed"]
    assert [(p.passage_id, p.score, p.domain) for p in hits] == [
        ("p1", 0.83, DataDomain.MAINTENANCE), ("p2", 1.0, DataDomain.MAINTENANCE)]  # unlabelled p5 is never returned
    with pytest.raises(ValueError, match="Invalid passages table"):
        SnowflakeVectorRetriever(lambda: None, "passages; DROP TABLE x")
    assert retriever("  ") == []


def test_runtime_assembly_from_environment(monkeypatch, tmp_path):
    for name in ("AGENT_BACKEND", "WAREHOUSE_BACKEND", "RETRIEVER_BACKEND", "CATALOG_PATH", "AGENT_MODEL",
                 "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_WAREHOUSE", "PASSAGES_TABLE", "SNOWFLAKE_CONNECTION_NAME", "CORPUS_PATH"):
        monkeypatch.delenv(name, raising=False)
    config = RuntimeConfig.from_env()
    assert len(config.production_problems()) == 4
    assert type(build_agent_deps(config).executor).__name__ == "function"

    monkeypatch.setenv("AGENT_BACKEND", "llm")
    with pytest.raises(RuntimeError, match="AGENT_MODEL"):
        build_agent_deps()
    monkeypatch.setenv("AGENT_MODEL", "some-model")
    monkeypatch.setenv("WAREHOUSE_BACKEND", "snowflake")
    with pytest.raises(RuntimeError, match="incomplete"):
        build_agent_deps()
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acme")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "AI_WH")
    monkeypatch.setenv("RETRIEVER_BACKEND", "snowflake")
    with pytest.raises(RuntimeError, match="PASSAGES_TABLE"):
        build_agent_deps()
    monkeypatch.setenv("RETRIEVER_BACKEND", "vector")
    from gates import DEFAULT_CATALOG
    from runtime import catalog_problems
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(DEFAULT_CATALOG.model_dump_json())
    monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
    assert any("CORPUS_PATH" in p for p in RuntimeConfig.from_env().production_problems())
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"passage_id": "d1", "source": "doc://x", "text": "Pump seals are replaced yearly.", "domain": "MAINTENANCE"}\n')
    monkeypatch.setenv("CORPUS_PATH", str(corpus))
    monkeypatch.setenv("SYNTHESIS_MODE", "abstractive")
    assert any("SYNTHESIS_MODE" in p for p in RuntimeConfig.from_env().production_problems())
    monkeypatch.setenv("SYNTHESIS_MODE", "extractive")
    deps = build_agent_deps()
    assert any("demo catalog" in p for p in catalog_problems(RuntimeConfig.from_env(), deps.catalog))
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "OTHER_DB")
    assert any("SNOWFLAKE_DATABASE" in p for p in catalog_problems(RuntimeConfig.from_env(), deps.catalog))
    assert [type(x).__name__ for x in (deps.router, deps.sql_generator, deps.executor, deps.retriever)] == [
        "LLMRouter", "LLMSQLGenerator", "SnowflakeExecutor", "VectorRetriever"]
    assert len(deps.catalog) == 500 and RuntimeConfig.from_env().production_problems() == []
    monkeypatch.setenv("RETRIEVER_BACKEND", "elastic")
    with pytest.raises(RuntimeError, match="Unknown RETRIEVER_BACKEND"):
        build_agent_deps()
