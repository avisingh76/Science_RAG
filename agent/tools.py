"""
tools.py  —  Hybrid Search + Optional Reranking.

CrossEncoder is disabled when ENABLE_RERANKING=false in environment.
Set ENABLE_RERANKING=false on Railway free tier to save RAM (~800MB saved).
Set ENABLE_RERANKING=true on paid/local for best answer quality.
"""

import logging
import os
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
import sqlparse
from langchain_core.tools import StructuredTool

logger = logging.getLogger("science_rag.tools")

# ── Config ─────────────────────────────────────────────────────────────────
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
if ENABLE_RERANKING:
    logger.info("Reranking: ENABLED (CrossEncoder)")
else:
    logger.info("Reranking: DISABLED (FAISS + BM25 only) — RAM saving mode")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
FAISS_PATH = DATA_DIR / "faiss_index.bin"
META_PATH  = DATA_DIR / "chunks_meta.pkl"
BM25_PATH  = DATA_DIR / "bm25_index.pkl"
DB_PATH    = DATA_DIR / "metadata.db"

_ALLOWED_TABLES = {"chunks", "pages", "sources"}

# ── Thread-safe singletons ─────────────────────────────────────────────────
_lock          = threading.Lock()
_st_model      = None
_cross_encoder = None
_faiss_index   = None
_chunks_meta   = None
_bm25_data     = None


def _get_model():
    global _st_model
    if _st_model is None:
        with _lock:
            if _st_model is None:
                logger.info("Loading SentenceTransformer model...")
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _get_cross_encoder():
    global _cross_encoder
    if not ENABLE_RERANKING:
        return None
    if _cross_encoder is None:
        with _lock:
            if _cross_encoder is None:
                logger.info("Loading CrossEncoder model...")
                from sentence_transformers import CrossEncoder
                _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _get_faiss():
    global _faiss_index
    if _faiss_index is None:
        with _lock:
            if _faiss_index is None:
                logger.info("Loading FAISS index...")
                _faiss_index = faiss.read_index(str(FAISS_PATH))
    return _faiss_index


def _get_chunks():
    global _chunks_meta
    if _chunks_meta is None:
        with _lock:
            if _chunks_meta is None:
                logger.info("Loading chunk metadata...")
                with open(META_PATH, "rb") as f:
                    _chunks_meta = pickle.load(f)
    return _chunks_meta


def _get_bm25():
    global _bm25_data
    if _bm25_data is None:
        with _lock:
            if _bm25_data is None:
                logger.info("Loading BM25 index...")
                with open(BM25_PATH, "rb") as f:
                    _bm25_data = pickle.load(f)
    return _bm25_data


def reload_index():
    global _faiss_index, _chunks_meta, _bm25_data
    with _lock:
        logger.info("Reloading all indexes after new ingest...")
        _faiss_index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            _chunks_meta = pickle.load(f)
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    logger.info("Reload complete — %d chunks.", len(_chunks_meta))


def _embed_query(query: str) -> np.ndarray:
    arr = _get_model().encode([query], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(arr)
    return arr


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ── SQL safety ─────────────────────────────────────────────────────────────
_SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "like", "limit",
    "order", "by", "asc", "desc", "count", "distinct", "as", "join",
    "inner", "left", "right", "on", "group", "having", "null", "is",
    "between", "case", "when", "then", "else", "end", "true", "false",
    "chunk_id", "page", "preview", "text", "source", "filename",
    "num_chunks", "num_pages", "ingested_at", "id",
}


def _validate_sql(sql: str) -> str | None:
    statements = sqlparse.parse(sql.strip())
    if len(statements) != 1:
        return "Only a single SQL statement is allowed."
    stmt = statements[0]
    if stmt.get_type() != "SELECT":
        return "Only SELECT statements are allowed."
    referenced = {t.value.lower() for t in stmt.flatten()
                  if t.ttype is sqlparse.tokens.Name}
    disallowed = referenced - _ALLOWED_TABLES - _SQL_KEYWORDS
    if disallowed:
        logger.warning("SQL blocked — disallowed identifiers: %s", disallowed)
        return f"Access denied. Allowed tables: {', '.join(sorted(_ALLOWED_TABLES))}."
    return None


# ── Tool 1: RAG Search ─────────────────────────────────────────────────────
def rag_search(query: str, source: Optional[str] = None) -> str:
    """
    Search the science textbook(s) using hybrid search + optional reranking.

    Pipeline:
      1. FAISS semantic search (top 20)
      2. BM25 keyword search   (top 20)
      3. Reciprocal Rank Fusion
      4. CrossEncoder reranking (if ENABLE_RERANKING=true)
      5. Return top 3 passages

    Args:
        query:  What you want to find.
        source: Optional PDF filename to restrict search to one book.
    """
    try:
        logger.info("Hybrid RAG search | query='%.80s' | source=%s | reranking=%s",
                    query, source, ENABLE_RERANKING)

        chunks    = _get_chunks()
        bm25_data = _get_bm25()
        bm25      = bm25_data["bm25"]
        bm25_ids  = bm25_data["chunk_ids"]
        id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

        # FAISS search
        query_vec          = _embed_query(query)
        distances, indices = _get_faiss().search(query_vec, k=20)
        faiss_ranking      = [
            int(idx) for idx in indices[0]
            if idx < len(chunks)
            and (source is None or chunks[int(idx)].get("source") == source)
        ]

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores     = bm25.get_scores(tokenized_query)
        bm25_order      = np.argsort(bm25_scores)[::-1]
        bm25_ranking: list[int] = []
        for bm25_pos in bm25_order[:20]:
            chunk_id = bm25_ids[bm25_pos]
            idx      = id_to_idx.get(chunk_id)
            if idx is None:
                continue
            if source and chunks[idx].get("source") != source:
                continue
            bm25_ranking.append(idx)

        # RRF fusion
        fused  = _rrf([faiss_ranking, bm25_ranking])
        top_10 = sorted(fused, key=fused.get, reverse=True)[:10]

        if not top_10:
            return f"No relevant content found{' in ' + source if source else ''}."

        # Optional CrossEncoder reranking
        ce = _get_cross_encoder()
        if ce is not None:
            ce_inputs = [(query, chunks[i]["text"]) for i in top_10]
            ce_scores = ce.predict(ce_inputs)
            reranked  = sorted(zip(top_10, ce_scores), key=lambda x: x[1], reverse=True)
            final_idxs = [idx for idx, _ in reranked[:3]]
        else:
            final_idxs = top_10[:3]

        results = []
        for idx in final_idxs:
            c = chunks[idx]
            results.append(
                f"[Source: {c.get('source', 'unknown')} — Page {c['page']}]\n{c['text']}\n"
            )

        logger.info("Search returned %d results.", len(results))
        return "\n---\n".join(results)

    except Exception:
        logger.exception("RAG search failed.")
        return "RAG search encountered an internal error. Please try again."


# ── Tool 2: SQL Query ──────────────────────────────────────────────────────
def sql_query(sql: str) -> str:
    """
    Run a SQL SELECT query on the textbook metadata database.

    Available tables:
      chunks(chunk_id, source, page, preview, text)
      pages(id, source, page, preview)
      sources(filename, num_chunks, num_pages, ingested_at)
    """
    logger.info("SQL query | sql='%.120s'", sql)
    error = _validate_sql(sql)
    if error:
        return f"Query rejected: {error}"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(10)
        conn.close()
        if not rows:
            return "No results found."
        cols  = [d[0] for d in cur.description]
        lines = [" | ".join(cols), "-" * 60]
        for row in rows:
            lines.append(" | ".join(str(v)[:80] for v in row))
        return "\n".join(lines)
    except Exception:
        logger.exception("SQL execution failed.")
        return "SQL query encountered an internal error."


# ── Tool 3: Wikipedia Search ───────────────────────────────────────────────
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for a brief summary of any science topic.
    """
    logger.info("Wikipedia search | topic='%s'", topic)
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        f"{topic.replace(' ', '_')}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return f"Wikipedia — {data['title']}:\n{data['extract']}"
        return f"No Wikipedia article found for '{topic}'."
    except Exception:
        logger.exception("Wikipedia search failed.")
        return "Wikipedia search encountered an internal error."


ALL_TOOLS = [
    StructuredTool.from_function(rag_search),
    StructuredTool.from_function(sql_query),
    StructuredTool.from_function(wikipedia_search),
]