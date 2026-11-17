"""retrieval.py - Vector retrieval for the RAG and diagnostic agents.

Retrievers (RETRIEVER_BACKEND):
  keyword    graph.keyword_retriever: term overlap over the sample corpus (dev default)
  vector     VectorRetriever: cosine search over an in-process index of a JSONL
             corpus (CORPUS_PATH), using any Embedder
  hybrid     HybridRetriever: dense plus BM25 keyword search, fused by reciprocal rank
  snowflake  SnowflakeVectorRetriever: VECTOR_COSINE_SIMILARITY over a passages
             table, with the query embedded by SNOWFLAKE.CORTEX.EMBED_TEXT_768

Scores feed the thin-set gate (average confidence >= 0.4), so the threshold
has to be calibrated for each embedder: cosine values from the offline
HashingEmbedder are lower than those from a trained embedding model.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable
from typing import Any, Protocol

from schemas import DataDomain, Passage

OVERFETCH = 5
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")
_STOPWORDS = frozenset(
    "a an the is are was were what which who how why when where do does did for of on in to and or "
    "with at by from this that it its be as can should would".split()
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic offline embedder: hashed unigrams and bigrams with
    sublinear term frequency, L2-normalised. No model download, no network.
    A lexical baseline, not a semantic model."""

    def __init__(self, dim: int = 1024) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in _STOPWORDS]
        stems = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]
        return stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:])]

    def _one(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            counts[index] = counts.get(index, 0.0) + sign
        vector = [0.0] * self.dim
        for index, value in counts.items():
            vector[index] = math.copysign(1.0 + math.log(abs(value)), value) if value else 0.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def chunk_text(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Split a document into passage-sized chunks on sentence boundaries."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_corpus(path: str) -> list[Passage]:
    """Read passages from JSONL: {"passage_id", "source", "text", "domain"?} per line."""
    passages: list[Passage] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                passage = Passage.model_validate({"score": 0.0} | json.loads(line))
            except ValueError as err:
                raise ValueError(f"{path}:{number}: invalid passage: {err}") from err
            if passage.domain is None:
                raise ValueError(f"{path}:{number}: passage has no domain, so entitlement cannot be checked")
            if passage.passage_id in seen:
                raise ValueError(f"{path}:{number}: duplicate passage_id {passage.passage_id!r}")
            seen.add(passage.passage_id)
            passages.append(passage)
    if not passages:
        raise ValueError(f"{path}: corpus is empty")
    return passages


def _clamp(score: float) -> float:
    return 0.0 if not math.isfinite(score) else round(min(1.0, max(0.0, score)), 4)


class VectorIndex:
    """Exact cosine search over an in-memory set of passages."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._passages: list[Passage] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._passages)

    @staticmethod
    def _normalise(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def add(self, passages: list[Passage]) -> None:
        vectors = self._embedder.embed([p.text for p in passages])
        if len(vectors) != len(passages):
            raise ValueError("Embedder returned a different number of vectors than passages.")
        self._passages.extend(passages)
        self._vectors.extend(self._normalise(v) for v in vectors)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[Passage]:
        if not query.strip() or not self._passages:
            return []
        q = self._normalise(self._embedder.embed([query])[0])
        scored = (
            (_clamp(sum(a * b for a, b in zip(q, vector))), passage)
            for vector, passage in zip(self._vectors, self._passages)
        )
        hits = sorted((item for item in scored if item[0] > min_score), key=lambda item: -item[0])[:top_k]
        return [passage.model_copy(update={"score": score}) for score, passage in hits]


class VectorRetriever:
    """`graph.Retriever` over a VectorIndex."""

    def __init__(self, index: VectorIndex, top_k: int = 4, min_score: float = 0.2) -> None:
        self._index, self._top_k, self._min_score = index, top_k, min_score

    def __call__(self, query: str) -> list[Passage]:
        return self._index.search(query, self._top_k, self._min_score)


class BM25Index:
    """Sparse keyword index. Exact identifiers (fault codes, part numbers,
    sensor ids) match here even when an embedding model blurs them."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._passages: list[Passage] = []
        self._terms: list[dict[str, int]] = []
        self._doc_freq: dict[str, int] = {}
        self._total_len = 0

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in _STOPWORDS]

    def add(self, passages: list[Passage]) -> None:
        for passage in passages:
            counts: dict[str, int] = {}
            for token in self._tokens(passage.text):
                counts[token] = counts.get(token, 0) + 1
            for token in counts:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._passages.append(passage)
            self._terms.append(counts)
            self._total_len += sum(counts.values())

    def search(self, query: str, top_k: int = 4) -> list[tuple[float, Passage]]:
        """(share of query terms matched, passage), best BM25 score first."""
        terms = list(dict.fromkeys(self._tokens(query)))
        if not terms or not self._passages:
            return []
        n, avg_len = len(self._passages), self._total_len / len(self._passages)
        scored: list[tuple[float, float, Passage]] = []
        for counts, passage in zip(self._terms, self._passages):
            length = sum(counts.values()) or 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    idf = math.log(1 + (n - self._doc_freq[term] + 0.5) / (self._doc_freq[term] + 0.5))
                    score += idf * tf * (self._k1 + 1) / (tf + self._k1 * (1 - self._b + self._b * length / avg_len))
            if score > 0:
                coverage = sum(1 for t in terms if counts.get(t)) / len(terms)
                scored.append((score, coverage, passage))
        scored.sort(key=lambda item: -item[0])
        return [(round(coverage, 4), passage) for _, coverage, passage in scored[:top_k]]


class HybridRetriever:
    """`graph.Retriever` that fuses dense and sparse results by reciprocal
    rank. A passage's confidence is the stronger of its cosine similarity and
    the share of query terms it matched, so an exact identifier match is not
    penalised for a weak embedding."""

    def __init__(self, dense: VectorIndex, sparse: BM25Index, top_k: int = 4, min_score: float = 0.2, rrf_k: int = 60) -> None:
        self._dense, self._sparse, self._top_k, self._min_score, self._rrf_k = dense, sparse, top_k, min_score, rrf_k

    def __call__(self, query: str) -> list[Passage]:
        pool = self._top_k * 3
        fused: dict[str, float] = {}
        best: dict[str, Passage] = {}
        ranked_lists = [
            [(p.score, p) for p in self._dense.search(query, pool, 0.0)],
            self._sparse.search(query, pool),
        ]
        for ranked in ranked_lists:
            for rank, (confidence, passage) in enumerate(ranked):
                fused[passage.passage_id] = fused.get(passage.passage_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                current = best.get(passage.passage_id)
                if current is None or confidence > current.score:
                    best[passage.passage_id] = passage.model_copy(update={"score": _clamp(confidence)})
        ordered = sorted(best.values(), key=lambda p: -fused[p.passage_id])
        return [p for p in ordered if p.score > self._min_score][: self._top_k]


class SnowflakeVectorRetriever:
    """`graph.Retriever` that ranks passages inside Snowflake.

    Expects a table with columns PASSAGE_ID, SOURCE, TEXT, DOMAIN and
    EMBEDDING VECTOR(FLOAT, 768) built with the same embedding model.
    Entitlement filtering by data domain happens in the graph, after retrieval,
    so this over-fetches and the graph keeps the best few the caller may see.
    Rows with no recognised domain are never returned.
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        table: str,
        model: str = "snowflake-arctic-embed-m-v1.5",
        top_k: int = 4,
    ) -> None:
        if not _TABLE_NAME.match(table):
            raise ValueError(f"Invalid passages table name: {table!r}")
        self._connect, self._table, self._model, self._top_k = connect, table, model, top_k

    def __call__(self, query: str) -> list[Passage]:
        if not query.strip():
            return []
        sql = (
            "SELECT passage_id, source, text, domain, "
            "VECTOR_COSINE_SIMILARITY(embedding, SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s)) AS score "
            f"FROM {self._table} WHERE domain IS NOT NULL ORDER BY score DESC LIMIT {int(self._top_k) * OVERFETCH}"
        )
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, (self._model, query))
                rows = cursor.fetchall()
            finally:
                cursor.close()
        finally:
            conn.close()
        passages: list[Passage] = []
        for passage_id, source, text, domain, score in rows:
            try:
                domain_value = DataDomain(str(domain or "").strip().upper())
            except ValueError:
                continue  # missing or unknown domain cannot be entitlement-checked: drop it
            if text:
                passages.append(
                    Passage(passage_id=str(passage_id), source=str(source), text=str(text),
                            score=_clamp(float(score)), domain=domain_value)
                )
        return passages
