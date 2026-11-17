"""runtime.py - Assemble the workflow's dependencies from the environment.

  CATALOG_PATH        JSON catalog exported by snowflake_backend.py (default: demo catalog)
  AGENT_BACKEND       simulated (default) | llm            needs AGENT_MODEL
  SYNTHESIS_MODE      extractive (default) | abstractive   see llm_agents.LLMSynthesizer
  WAREHOUSE_BACKEND   simulated (default) | snowflake      needs SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE
  RETRIEVER_BACKEND   keyword (default) | vector | snowflake
                      vector: CORPUS_PATH (JSONL); snowflake: PASSAGES_TABLE, SNOWFLAKE_CONNECTION_NAME

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

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            agent_backend=os.environ.get("AGENT_BACKEND", "simulated").lower(),
            warehouse_backend=os.environ.get("WAREHOUSE_BACKEND", "simulated").lower(),
            retriever_backend=os.environ.get("RETRIEVER_BACKEND", "keyword").lower(),
            catalog_source=os.environ.get("CATALOG_PATH") or "demo",
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


def load_catalog(config: RuntimeConfig) -> Catalog:
    if config.catalog_source == "demo":
        return DEFAULT_CATALOG
    from snowflake_backend import load_catalog_file

    return load_catalog_file(config.catalog_source)


def build_agent_deps(config: RuntimeConfig | None = None) -> AgentDeps:
    config = config or RuntimeConfig.from_env()
    catalog = load_catalog(config)

    match config.agent_backend:
        case "simulated":
            deps = AgentDeps(catalog=catalog)
        case "llm":
            from llm_agents import build_llm_deps

            mode = os.environ.get("SYNTHESIS_MODE", "extractive").lower()
            if mode not in ("extractive", "abstractive"):
                raise RuntimeError(f"Unknown SYNTHESIS_MODE '{mode}' (use extractive or abstractive).")
            deps = build_llm_deps(_lazy_anthropic_completion(), catalog, synthesis_mode=mode)
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

            path = os.environ.get("CORPUS_PATH")
            index = VectorIndex(HashingEmbedder())
            index.add(load_corpus(path) if path else list(KNOWLEDGE_BASE))
            deps = replace(deps, retriever=VectorRetriever(index))
        case "snowflake":
            from retrieval import SnowflakeVectorRetriever
            from snowflake_backend import service_connection

            table = os.environ.get("PASSAGES_TABLE")
            if not table or not os.environ.get("SNOWFLAKE_CONNECTION_NAME"):
                raise RuntimeError("RETRIEVER_BACKEND=snowflake requires PASSAGES_TABLE and SNOWFLAKE_CONNECTION_NAME.")
            deps = replace(deps, retriever=SnowflakeVectorRetriever(service_connection, table))
        case other:
            raise RuntimeError(f"Unknown RETRIEVER_BACKEND '{other}' (use keyword, vector or snowflake).")
    return deps
