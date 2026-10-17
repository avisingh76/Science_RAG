"""
eval_retrieval.py  —  Benchmark the retrieval stage of Science RAG.

Compares four retrieval strategies on the labelled test set, using the SAME
indexes and fusion logic as agent/tools.py:

  faiss_only     semantic search only
  bm25_only      keyword search only
  hybrid_rrf     FAISS + BM25 fused with Reciprocal Rank Fusion (production, no reranker)
  hybrid_rerank  hybrid_rrf + CrossEncoder reranking (production with ENABLE_RERANKING=true)

Metrics (reported at chunk level AND page level):
  Hit@1/3/5/10   fraction of questions whose source was in the top-k
  MRR            mean reciprocal rank
  Latency        mean per-query time (ms)
  95% bootstrap CI for Hit@3 (page level)

Production returns the top 3 passages to the LLM, so Hit@3 is the headline metric.

Usage (from project root, with the venv that runs the app):
  python -m evaluation.eval_retrieval
  python -m evaluation.eval_retrieval --no-rerank          # skip CrossEncoder (faster, low RAM)
  python -m evaluation.eval_retrieval --dedupe             # collapse duplicate chunks
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np

from evaluation import common
from evaluation.metrics import bootstrap_ci, first_rank, mean, summarize_ranks

logger = logging.getLogger("science_rag.eval.retrieval")

METHODS = ["faiss_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
LABELS = {
    "faiss_only":    "FAISS only",
    "bm25_only":     "BM25 only",
    "hybrid_rrf":    "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # same model as agent/tools.py


def load_reranker():
    from sentence_transformers import CrossEncoder
    logger.info("Loading CrossEncoder %s ...", RERANKER_NAME)
    return CrossEncoder(RERANKER_NAME)


# ── Retriever (mirrors agent/tools.py rag_search) ──────────────────────────
class Retriever:
    def __init__(self, k_max: int, candidates: int, dedupe: bool, reranker=None):
        from agent import tools as T            # reuse the app's own loaders
        self.T = T
        self.k_max      = k_max
        self.candidates = candidates
        self.dedupe     = dedupe
        self.reranker   = reranker

        self.chunks    = T._get_chunks()
        self.faiss     = T._get_faiss()
        bm25_data      = T._get_bm25()
        self.bm25      = bm25_data["bm25"]
        self.bm25_ids  = bm25_data["chunk_ids"]
        self.id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        logger.info("Loaded %d chunks | FAISS vectors: %d", len(self.chunks), self.faiss.ntotal)

    def _dedupe(self, idxs: list[int]) -> list[int]:
        if not self.dedupe:
            return idxs
        seen, out = set(), []
        for i in idxs:
            key = self.chunks[i]["text"].strip()
            if key not in seen:
                seen.add(key)
                out.append(i)
        return out

    def faiss_ranking(self, query: str) -> list[int]:
        vec = self.T._embed_query(query)
        _, idxs = self.faiss.search(vec, self.candidates)
        return [int(i) for i in idxs[0] if 0 <= int(i) < len(self.chunks)]

    def bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        order  = np.argsort(scores)[::-1][: self.candidates]
        out = []
        for pos in order:
            idx = self.id_to_idx.get(self.bm25_ids[pos])
            if idx is not None:
                out.append(idx)
        return out

    def retrieve_all(self, query: str) -> tuple[dict[str, list[int]], dict[str, float]]:
        """Return {method: ranked chunk indices} and {method: latency in ms}."""
        K = self.k_max

        t = time.perf_counter()
        faiss_r = self._dedupe(self.faiss_ranking(query))
        t_faiss = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bm25_r = self._dedupe(self.bm25_ranking(query))
        t_bm25 = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fused  = self.T._rrf([faiss_r, bm25_r])
        hybrid = self._dedupe(sorted(fused, key=fused.get, reverse=True))[:K]
        t_rrf  = (time.perf_counter() - t) * 1000

        rankings = {
            "faiss_only": faiss_r[:K],
            "bm25_only":  bm25_r[:K],
            "hybrid_rrf": hybrid,
        }
        latency = {
            "faiss_only": t_faiss,
            "bm25_only":  t_bm25,
            "hybrid_rrf": t_faiss + t_bm25 + t_rrf,
        }

        if self.reranker is not None and hybrid:
            t = time.perf_counter()
            scores = self.reranker.predict([(query, self.chunks[i]["text"]) for i in hybrid])
            reranked = [i for i, _ in sorted(zip(hybrid, scores), key=lambda x: x[1], reverse=True)]
            rankings["hybrid_rerank"] = reranked
            latency["hybrid_rerank"]  = latency["hybrid_rrf"] + (time.perf_counter() - t) * 1000

        return rankings, latency


# ── Relevance definitions ──────────────────────────────────────────────────
def chunk_match(chunk: dict, gt: dict) -> bool:
    """Retrieved chunk has the same text as the ground-truth chunk."""
    return chunk["text"].strip() == gt["chunk_text"].strip()


def page_match(chunk: dict, gt: dict) -> bool:
    """Retrieved chunk is from the same page (same book, or legacy chunks with no source)."""
    if chunk["page"] != gt["page"]:
        return False
    src = chunk.get("source")
    return src is None or src == gt["source"]


# ── Reporting ──────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_table(title: str, summary: dict, level: str, ks: list[int]) -> None:
    print(f"\n{title}")
    header = f"{'Method':<18}" + "".join(f"{f'Hit@{k}':>9}" for k in ks) + f"{'MRR':>8}{'ms/query':>11}"
    print(header)
    print("-" * len(header))
    for m in METHODS:
        if m not in summary:
            continue
        s = summary[m][level]
        row = f"{LABELS[m]:<18}" + "".join(f"{_pct(s[f'hit@{k}']):>9}" for k in ks)
        row += f"{s['mrr']:>8.3f}{summary[m]['latency_ms']:>11.1f}"
        print(row)


def write_failures(path: Path, records: list[dict], method: str, limit: int = 25) -> int:
    def _missed(r: dict) -> bool:
        m = r["methods"].get(method)
        return m is not None and (m["page_rank"] is None or m["page_rank"] > 3)

    misses = [r for r in records if _missed(r)]
    lines = [f"# Retrieval failures — `{method}` (page not in top 3)\n",
             f"{len(misses)} of {len(records)} questions.\n"]
    for r in misses[:limit]:
        lines.append(f"## {r['id']}: {r['question']}")
        lines.append(f"- Expected: **{r['source']}**, page **{r['page']}**")
        lines.append(f"- Ground-truth passage: _{r['chunk_preview']}_")
        lines.append("- Retrieved top 3:")
        for p in r["methods"][method]["top3"]:
            lines.append(f"  - page {p['page']} ({p['source']}): {p['preview']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(misses)


# ── Main ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> dict:
    common.setup_logging()
    ap = argparse.ArgumentParser(description="Benchmark FAISS / BM25 / Hybrid / Rerank retrieval.")
    ap.add_argument("--testset", type=str, default=str(common.TESTSET_PATH))
    ap.add_argument("--n", type=int, default=None, help="Use only a random subset of N questions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--candidates", type=int, default=20,
                    help="Candidates per retriever before fusion (production uses 20).")
    ap.add_argument("--no-rerank", action="store_true", help="Skip the CrossEncoder stage.")
    ap.add_argument("--dedupe", action="store_true",
                    help="Collapse duplicate-text chunks in every ranking "
                         "(shows the effect of cleaning duplicates from the index).")
    ap.add_argument("--failure-method", type=str, default="hybrid_rrf", choices=METHODS)
    args = ap.parse_args(argv)

    testset = common.read_jsonl(Path(args.testset))
    if not testset:
        raise SystemExit(f"Empty/missing test set: {args.testset}. Run generate_testset first.")
    if args.n and args.n < len(testset):
        testset = random.Random(args.seed).sample(testset, args.n)
    ks = sorted(set(args.ks))

    reranker = None if args.no_rerank else load_reranker()
    retriever = Retriever(k_max=max(ks), candidates=args.candidates,
                          dedupe=args.dedupe, reranker=reranker)
    chunks = retriever.chunks
    methods = [m for m in METHODS if m != "hybrid_rerank" or reranker is not None]

    ranks   = {m: {"chunk": [], "page": []} for m in methods}
    lat     = {m: [] for m in methods}
    records = []

    logger.info("Evaluating %d questions | methods=%s | dedupe=%s", len(testset), methods, args.dedupe)
    for n, q in enumerate(testset, start=1):
        rankings, latency = retriever.retrieve_all(q["question"])
        rec = {"id": q["id"], "question": q["question"], "source": q["source"],
               "page": q["page"], "chunk_preview": q["chunk_text"][:160].replace("\n", " "),
               "methods": {}}
        for m in methods:
            idxs = rankings.get(m, [])
            c_rank = first_rank([chunk_match(chunks[i], q) for i in idxs])
            p_rank = first_rank([page_match(chunks[i], q) for i in idxs])
            ranks[m]["chunk"].append(c_rank)
            ranks[m]["page"].append(p_rank)
            lat[m].append(latency[m])
            rec["methods"][m] = {
                "chunk_rank": c_rank, "page_rank": p_rank,
                "top3": [{"page": chunks[i]["page"], "source": chunks[i].get("source"),
                          "preview": chunks[i]["text"][:100].replace("\n", " ")} for i in idxs[:3]],
            }
        records.append(rec)
        if n % 10 == 0 or n == len(testset):
            logger.info("  %d/%d done", n, len(testset))

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = {}
    for m in methods:
        page_hits_at3 = [1.0 if (r is not None and r <= 3) else 0.0 for r in ranks[m]["page"]]
        lo, hi = bootstrap_ci(page_hits_at3)
        summary[m] = {
            "chunk": summarize_ranks(ranks[m]["chunk"], ks),
            "page":  summarize_ranks(ranks[m]["page"], ks),
            "page_hit@3_ci95": [lo, hi],
            "latency_ms": mean(lat[m]),
        }

    print(f"\nQuestions evaluated: {len(testset)}  |  duplicates collapsed: {args.dedupe}")
    print_table("PAGE-LEVEL  (retrieved passage is from the correct book + page)", summary, "page", ks)
    print_table("CHUNK-LEVEL (retrieved passage is the exact source passage)", summary, "chunk", ks)

    # ── Headline for the resume ────────────────────────────────────────────
    n = len(testset)
    best = max(methods, key=lambda m: summary[m]["page"]["hit@3"])
    base = summary["faiss_only"]["page"]["hit@3"]
    b = summary[best]
    print("\nHEADLINE (page-level Hit@3):")
    print("  " + " | ".join(f"{LABELS[m]} {_pct(summary[m]['page']['hit@3']).strip()}" for m in methods))
    print(f"  Best: {LABELS[best]} {_pct(b['page']['hit@3']).strip()} "
          f"(95% CI {100*b['page_hit@3_ci95'][0]:.0f}-{100*b['page_hit@3_ci95'][1]:.0f}%), "
          f"{100*(b['page']['hit@3'] - base):+.1f} pts vs FAISS-only")
    print(f"  Resume line: \"Evaluated retrieval on {n} LLM-generated questions; "
          f"{LABELS[best].lower()} reached {100*b['page']['hit@3']:.0f}% Hit@3 / "
          f"{b['page']['mrr']:.2f} MRR vs {100*base:.0f}% Hit@3 for FAISS-only.\"")

    # ── Save ───────────────────────────────────────────────────────────────
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"config": {"n_questions": n, "ks": ks, "candidates": args.candidates,
                      "dedupe": args.dedupe, "reranker": None if reranker is None else RERANKER_NAME,
                      "index_chunks": len(chunks)},
           "summary": summary, "per_question": records}
    common.save_json(common.RESULTS_DIR / "retrieval_results.json", out)
    fail_method = args.failure_method if args.failure_method in methods else "hybrid_rrf"
    n_fail = write_failures(common.RESULTS_DIR / "retrieval_failures.md", records, fail_method)
    print(f"\nSaved: {common.RESULTS_DIR / 'retrieval_results.json'}")
    print(f"Saved: {common.RESULTS_DIR / 'retrieval_failures.md'}  ({n_fail} misses for {fail_method}) — read these for error analysis")
    return out


if __name__ == "__main__":
    main()
