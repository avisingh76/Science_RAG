"""
eval_answers.py  —  End-to-end evaluation of the Science RAG agent.

For each test question the real LangGraph agent is run (same graph, tools and
prompt as production). Each run is scored on:

  Heuristic (no LLM):
    rag_used            agent called rag_search
    context_page_hit    the ground-truth book+page was in the retrieved context
    citation_page_hit   the final answer cites the ground-truth page number
    citation_source_hit the final answer names the ground-truth PDF
    latency, errors

  LLM-as-judge (OpenRouter, 1-5 scale):
    correctness         agrees with the reference answer
    faithfulness        every claim is supported by retrieved context (hallucination check)
    relevance           answers the question that was asked

Results are appended per question, so a crashed/rate-limited run can be resumed.

Usage (from project root):
  python -m evaluation.eval_answers --n 30
  python -m evaluation.eval_answers --n 30 --resume          # continue a previous run
  python -m evaluation.eval_answers --no-judge               # heuristic metrics only
  EVAL_JUDGE_MODEL=<other-model> python -m evaluation.eval_answers   # recommended
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path

from evaluation import common
from evaluation.metrics import extract_cited_pages, mean, percentile

logger = logging.getLogger("science_rag.eval.answers")

RESULTS_PATH = common.RESULTS_DIR / "answer_eval.jsonl"
SUMMARY_PATH = common.RESULTS_DIR / "answer_eval_summary.json"

# tools.py formats each hit as "[Source: <file> — Page <n>]"
_CONTEXT_HEADER = re.compile(r"\[Source:\s*(.*?)\s*[—-]\s*Page\s*(\d+)\]")

JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented science tutor for students.

QUESTION:
{question}

REFERENCE ANSWER (ground truth taken from the textbook):
{reference}

RETRIEVED CONTEXT (everything the system saw from its tools):
{context}

SYSTEM ANSWER:
{answer}

Score each criterion from 1 to 5:
- correctness: Does the SYSTEM ANSWER agree with the REFERENCE ANSWER on the key facts?
  5 = fully correct and complete, 3 = partly correct or missing key points, 1 = wrong, contradicts it, or no real answer.
- faithfulness: Is every factual claim in the SYSTEM ANSWER supported by the RETRIEVED CONTEXT?
  Judge ONLY against the context, not your own knowledge.
  5 = all claims supported, 3 = some unsupported details, 1 = mostly unsupported or invented.
  If the context is empty, give 1 unless the answer clearly says the information could not be found.
- relevance: Does the answer directly address the QUESTION? 5 = directly, 1 = off-topic.

Return ONLY valid JSON, no markdown:
{{"reason": "<one or two sentences>", "correctness": <1-5>, "faithfulness": <1-5>, "relevance": <1-5>}}
"""


# ── Agent runner ───────────────────────────────────────────────────────────
_graph = None


def get_graph():
    """Build the production graph once (imported lazily: needs .env + app deps)."""
    global _graph
    if _graph is None:
        from agent.graph import build_graph
        _graph = build_graph()
    return _graph


def run_question(question: str) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    try:
        from agent.graph import RECURSION_LIMIT
    except Exception:
        RECURSION_LIMIT = 10

    t0 = time.perf_counter()
    try:
        result = get_graph().invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as exc:
        logger.warning("Agent failed: %s", exc)
        return {"error": str(exc), "latency_s": time.perf_counter() - t0,
                "answer": "", "tool_calls": [], "rag_context": "", "all_context": ""}

    msgs = result["messages"]
    tool_calls = [tc["name"] for m in msgs if isinstance(m, AIMessage)
                  for tc in (getattr(m, "tool_calls", None) or [])]
    rag_ctx = [common.as_text(m.content) for m in msgs
               if isinstance(m, ToolMessage) and getattr(m, "name", "") == "rag_search"]
    all_ctx = [common.as_text(m.content) for m in msgs if isinstance(m, ToolMessage)]
    return {
        "error": None,
        "latency_s": time.perf_counter() - t0,
        "answer": common.as_text(msgs[-1].content).strip(),
        "tool_calls": tool_calls,
        "rag_context": "\n---\n".join(rag_ctx),
        "all_context": "\n---\n".join(all_ctx),
    }


# ── Heuristic scoring ──────────────────────────────────────────────────────
def context_page_hit(rag_context: str, source: str, page: int) -> bool:
    for src, pg in _CONTEXT_HEADER.findall(rag_context or ""):
        if int(pg) == page and (src in ("None", "unknown") or src == source):
            return True
    return False


def citation_source_hit(answer: str, source: str) -> bool:
    low = (answer or "").lower()
    stem = Path(source).stem.lower()
    return source.lower() in low or stem in low or stem.replace("_", " ") in low


def heuristic_scores(q: dict, run: dict) -> dict:
    return {
        "rag_used":            "rag_search" in run["tool_calls"],
        "context_page_hit":    context_page_hit(run["rag_context"], q["source"], q["page"]),
        "citation_page_hit":   q["page"] in extract_cited_pages(run["answer"]),
        "citation_source_hit": citation_source_hit(run["answer"], q["source"]),
    }


# ── LLM judge ──────────────────────────────────────────────────────────────
def judge(q: dict, run: dict, model: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        reference=q["reference_answer"],
        context=(run["all_context"] or "(empty)")[:6000],
        answer=run["answer"] or "(no answer)",
    )
    result = common.llm_json(prompt, model=model, max_tokens=500)
    if not result:
        return {"judge_error": True}
    out = {"judge_error": False, "judge_reason": str(result.get("reason", ""))[:400]}
    for key in ("correctness", "faithfulness", "relevance"):
        try:
            out[key] = max(1, min(5, int(result[key])))
        except (KeyError, TypeError, ValueError):
            out["judge_error"] = True
    return out


# ── Aggregation ────────────────────────────────────────────────────────────
def summarize(rows: list[dict], judged: bool) -> dict:
    ok = [r for r in rows if not r.get("error")]
    n, n_ok = len(rows), len(ok)

    def rate(key, subset=ok):
        return mean([1.0 if r.get(key) else 0.0 for r in subset]) if subset else 0.0

    lat = [r["latency_s"] for r in ok]
    summary = {
        "n_questions": n,
        "n_completed": n_ok,
        "error_rate": (n - n_ok) / n if n else 0.0,
        "rag_used_rate": rate("rag_used"),
        "context_page_hit_rate": rate("context_page_hit"),
        "citation_page_hit_rate": rate("citation_page_hit"),
        "citation_source_hit_rate": rate("citation_source_hit"),
        "latency_s": {"mean": mean(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)},
    }
    if judged:
        scored = [r for r in ok if not r.get("judge_error") and "correctness" in r]
        summary["n_judged"] = len(scored)
        if scored:
            for key in ("correctness", "faithfulness", "relevance"):
                vals = [r[key] for r in scored]
                summary[f"{key}_mean"] = mean(vals)
                summary[f"{key}_pct_ge4"] = mean([1.0 if v >= 4 else 0.0 for v in vals])
            summary["hallucination_rate"] = mean([1.0 if r["faithfulness"] <= 2 else 0.0 for r in scored])
            # does better retrieval produce better answers?
            hit  = [r["correctness"] for r in scored if r.get("context_page_hit")]
            miss = [r["correctness"] for r in scored if not r.get("context_page_hit")]
            summary["correctness_mean_when_context_hit"]  = mean(hit)  if hit  else None
            summary["correctness_mean_when_context_miss"] = mean(miss) if miss else None
    return summary


def print_summary(s: dict, judged: bool) -> None:
    pct = lambda x: f"{100 * x:.1f}%"
    print("\n=== END-TO-END ANSWER EVALUATION ===")
    print(f"Questions: {s['n_questions']}  completed: {s['n_completed']}  error rate: {pct(s['error_rate'])}")
    print(f"Agent used rag_search:          {pct(s['rag_used_rate'])}")
    print(f"Correct page in context:        {pct(s['context_page_hit_rate'])}")
    print(f"Answer cites correct page:      {pct(s['citation_page_hit_rate'])}")
    print(f"Answer names correct PDF:       {pct(s['citation_source_hit_rate'])}")
    L = s["latency_s"]
    print(f"Latency (s): mean {L['mean']:.1f} | p50 {L['p50']:.1f} | p95 {L['p95']:.1f}")
    if judged and s.get("n_judged"):
        print(f"\nLLM judge (n={s['n_judged']}, 1-5 scale):")
        for key in ("correctness", "faithfulness", "relevance"):
            print(f"  {key:<13} mean {s[f'{key}_mean']:.2f} | score>=4: {pct(s[f'{key}_pct_ge4'])}")
        print(f"  hallucination rate (faithfulness<=2): {pct(s['hallucination_rate'])}")
        h, m = s["correctness_mean_when_context_hit"], s["correctness_mean_when_context_miss"]
        if h is not None and m is not None:
            print(f"  correctness when right page retrieved: {h:.2f}  vs  missed: {m:.2f}")


# ── Main ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> dict:
    common.setup_logging()
    ap = argparse.ArgumentParser(description="End-to-end evaluation of the Science RAG agent.")
    ap.add_argument("--testset", type=str, default=str(common.TESTSET_PATH))
    ap.add_argument("--n", type=int, default=30, help="Number of questions (each costs agent + judge calls).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to pause between questions (rate limits).")
    ap.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    ap.add_argument("--resume", action="store_true", help="Keep existing results and skip finished questions.")
    args = ap.parse_args(argv)

    testset = common.read_jsonl(Path(args.testset))
    if not testset:
        raise SystemExit(f"Empty/missing test set: {args.testset}. Run generate_testset first.")
    if args.n < len(testset):
        testset = random.Random(args.seed).sample(testset, args.n)

    judged = not args.no_judge
    jmodel = common.judge_model() if judged else None
    if judged:
        logger.info("Judge model: %s", jmodel)

    if not args.resume and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    done = {r["id"] for r in common.read_jsonl(RESULTS_PATH)}

    for i, q in enumerate(testset, start=1):
        if q["id"] in done:
            continue
        run = run_question(q["question"])
        row = {"id": q["id"], "question": q["question"], "reference_answer": q["reference_answer"],
               "source": q["source"], "page": q["page"], **run}
        if not run["error"]:
            row.update(heuristic_scores(q, run))
            if judged:
                row.update(judge(q, run, jmodel))
        common.append_jsonl(RESULTS_PATH, row)
        logger.info("[%d/%d] %s | tools=%s | ctx_hit=%s | %s",
                    i, len(testset), q["id"], row["tool_calls"], row.get("context_page_hit"),
                    {k: row[k] for k in ("correctness", "faithfulness") if k in row})
        if args.sleep:
            time.sleep(args.sleep)

    ids = {q["id"] for q in testset}
    rows = [r for r in common.read_jsonl(RESULTS_PATH) if r["id"] in ids]
    summary = summarize(rows, judged)
    summary["judge_model"] = jmodel
    common.save_json(SUMMARY_PATH, summary)
    print_summary(summary, judged)
    print(f"\nSaved: {RESULTS_PATH}\nSaved: {SUMMARY_PATH}")
    print("Tip: sort answer_eval.jsonl by correctness/faithfulness and read the worst 5 for error analysis.")
    return summary


if __name__ == "__main__":
    main()
