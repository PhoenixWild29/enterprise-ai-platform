"""runtime.py - Assemble the workflow's dependencies from the environment.

  CATALOG_PATH        JSON catalog exported by snowflake_backend.py (default: demo catalog)
  AGENT_BACKEND       simulated (default) | llm            needs AGENT_MODEL
  MODEL_PROVIDER      bedrock (default) | anthropic        see bedrock_models.py for per-role models and budgets
  SYNTHESIS_MODE      extractive (default) | abstractive   see llm_agents.LLMSynthesizer
  WAREHOUSE_BACKEND   simulated (default) | snowflake      needs SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE
  RETRIEVER_BACKEND   keyword (default) | vector | hybrid | snowflake
                      vector, hybrid: CORPUS_PATH (JSONL); snowflake: PASSAGES_TABLE, SNOWFLAKE_CONNECTION_NAME

Settings are validated at startup; optional client libraries (anthropic,
snowflake-connector-python) are imported on first use.
"""

import os
from dataclasses import dataclass, replace

from gates import DEFAULT_CATALOG
from graph import KNOWLEDGE_BASE, AgentDeps
from schemas import Catalog


@dataclass(frozen=True)
class RuntimeConfig:
    agent_backend: str
    warehouse_backend: str
    retriever_backend: str
    catalog_source: str
    synthesis_mode: str = "extractive"
    corpus_path: str | None = None
    model_provider: str = "bedrock"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            agent_backend=os.environ.get("AGENT_BACKEND", "simulated").lower(),
            warehouse_backend=os.environ.get("WAREHOUSE_BACKEND", "simulated").lower(),
            retriever_backend=os.environ.get("RETRIEVER_BACKEND", "keyword").lower(),
            catalog_source=os.environ.get("CATALOG_PATH") or "demo",
            synthesis_mode=os.environ.get("SYNTHESIS_MODE", "extractive").lower(),
            corpus_path=os.environ.get("CORPUS_PATH") or None,
            model_provider=os.environ.get("MODEL_PROVIDER", "bedrock").lower(),
        )

    def production_problems(self) -> list[str]:
        """Development stand-ins that a production process must not run on."""
        problems: list[str] = []
        if self.agent_backend == "simulated":
            problems.append("AGENT_BACKEND=llm (the simulated SQL generator passes raw SQL through)")
        if self.warehouse_backend == "simulated":
            problems.append("WAREHOUSE_BACKEND=snowflake (the simulated executor returns synthetic rows)")
        if self.retriever_backend == "keyword":
            problems.append("RETRIEVER_BACKEND=vector or snowflake (the keyword retriever uses sample passages)")
        if self.retriever_backend in ("vector", "hybrid") and not self.corpus_path:
            problems.append("CORPUS_PATH (without it the vector retriever indexes the sample passages)")
        if self.agent_backend == "llm" and self.model_provider != "bedrock":
            problems.append("MODEL_PROVIDER=bedrock (models must run inside your own AWS account)")
        if self.synthesis_mode != "extractive":
            problems.append("SYNTHESIS_MODE=extractive (abstractive mode releases model-written text)")
        if self.catalog_source == "demo":
            problems.append("CATALOG_PATH (the demo catalog is generated, not your lakehouse)")
        return problems


def _lazy_anthropic_completion():
    model = os.environ.get("AGENT_MODEL")
    if not model:
        raise RuntimeError("AGENT_BACKEND=llm requires AGENT_MODEL.")
    client = None

    def complete(prompt: str) -> str:
        nonlocal client
        if client is None:
            from llm_agents import anthropic_completion

            client = anthropic_completion(model)
        return client(prompt)

    return complete


def model_completions(config: "RuntimeConfig") -> dict[str, object]:
    """Completion functions by role for the configured model provider."""
    if config.model_provider == "bedrock":
        from bedrock_models import ModelPortfolio, portfolio_completions

        return portfolio_completions(ModelPortfolio.from_env())
    if config.model_provider == "anthropic":
        single = _lazy_anthropic_completion()
        return {role: single for role in ("router", "sql", "synth", "verifier", "judge")}
    raise RuntimeError(f"Unknown MODEL_PROVIDER '{config.model_provider}' (use bedrock or anthropic).")


def load_catalog(config: RuntimeConfig) -> Catalog:
    if config.catalog_source == "demo":
        return DEFAULT_CATALOG
    from snowflake_backend import load_catalog_file

    return load_catalog_file(config.catalog_source)


def catalog_problems(config: RuntimeConfig, catalog: Catalog) -> list[str]:
    """Catalog checks a production process must pass."""
    problems: list[str] = []
    demo = DEFAULT_CATALOG.tables.keys()
    if len(catalog.tables.keys() & demo) > len(demo) // 2:
        problems.append("a catalog exported from your lakehouse (CATALOG_PATH holds the demo catalog)")
    if config.warehouse_backend == "snowflake":
        database = os.environ.get("SNOWFLAKE_DATABASE", "LAKEHOUSE").upper()
        if catalog.database.upper() != database:
            problems.append(f"SNOWFLAKE_DATABASE to match the catalog's database ({catalog.database})")
    return problems


def build_agent_deps(config: RuntimeConfig | None = None) -> AgentDeps:
    config = config or RuntimeConfig.from_env()
    catalog = load_catalog(config)

    match config.agent_backend:
        case "simulated":
            deps = AgentDeps(catalog=catalog)
        case "llm":
            from llm_agents import build_llm_deps

            mode = config.synthesis_mode
            if mode not in ("extractive", "abstractive"):
                raise RuntimeError(f"Unknown SYNTHESIS_MODE '{mode}' (use extractive or abstractive).")
            models = model_completions(config)
            deps = build_llm_deps(
                models["synth"], catalog, synthesis_mode=mode, router_complete=models["router"],
                sql_complete=models["sql"], verifier_complete=models["verifier"],
            )
        case other:
            raise RuntimeError(f"Unknown AGENT_BACKEND '{other}' (use simulated or llm).")

    match config.warehouse_backend:
        case "simulated":
            pass
        case "snowflake":
            from snowflake_backend import SnowflakeExecutor, SnowflakeSettings, oauth_connection_factory

            deps = replace(deps, executor=SnowflakeExecutor(oauth_connection_factory(SnowflakeSettings.from_env())))
        case other:
            raise RuntimeError(f"Unknown WAREHOUSE_BACKEND '{other}' (use simulated or snowflake).")

    match config.retriever_backend:
        case "keyword":
            pass
        case "vector":
            from retrieval import HashingEmbedder, VectorIndex, VectorRetriever, load_corpus

            index = VectorIndex(HashingEmbedder())
            index.add(load_corpus(config.corpus_path) if config.corpus_path else list(KNOWLEDGE_BASE))
            deps = replace(deps, retriever=VectorRetriever(index))
        case "hybrid":
            from retrieval import BM25Index, HashingEmbedder, HybridRetriever, VectorIndex, load_corpus

            corpus = load_corpus(config.corpus_path) if config.corpus_path else list(KNOWLEDGE_BASE)
            dense, sparse = VectorIndex(HashingEmbedder()), BM25Index()
            dense.add(corpus)
            sparse.add(corpus)
            deps = replace(deps, retriever=HybridRetriever(dense, sparse))
        case "snowflake":
            from retrieval import SnowflakeVectorRetriever
            from snowflake_backend import service_connection

            table = os.environ.get("PASSAGES_TABLE")
            if not table or not os.environ.get("SNOWFLAKE_CONNECTION_NAME"):
                raise RuntimeError("RETRIEVER_BACKEND=snowflake requires PASSAGES_TABLE and SNOWFLAKE_CONNECTION_NAME.")
            deps = replace(deps, retriever=SnowflakeVectorRetriever(service_connection, table))
        case other:
            raise RuntimeError(f"Unknown RETRIEVER_BACKEND '{other}' (use keyword, vector, hybrid or snowflake).")
    return deps
