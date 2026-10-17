"""
generate_testset.py  —  Build a labelled evaluation set from YOUR indexed books.

For each sampled chunk, an OpenRouter LLM writes a student-style question that the
chunk answers, plus a short reference answer. The source chunk's (source, page,
chunk_id) is the ground truth used by the retrieval and answer evaluators.

Quality controls:
  • Only chunks that have a source filename and are not duplicates.
  • LLM may skip non-teaching text (copyright page, index, exercises...).
  • Questions that copy 4+ consecutive words from the chunk are rejected
    (otherwise BM25 gets an unfair advantage from lexical leakage).
  • Sampling is stratified across books and seeded for reproducibility.

Usage (from project root):
  python -m evaluation.generate_testset --n 60
"""

import argparse
import logging
from pathlib import Path
import math
import random

from evaluation import common
from evaluation.metrics import shares_ngram

logger = logging.getLogger("science_rag.eval.generate")

PROMPT = """You are building an evaluation set for a retrieval system over science textbooks.

Read the passage below and write ONE question that a student might ask, whose answer is contained in the passage.

Rules:
- The question must be self-contained. Never mention "the passage", "the text", "this page", or "the author".
- Paraphrase. Do NOT copy any run of 4 or more consecutive words from the passage.
- Be specific enough that this passage is the best place to find the answer. Avoid yes/no questions.
- Also write a concise reference answer (1-3 sentences) using ONLY information in the passage.
- If the passage has no teachable science content (copyright notice, table of contents, index, acknowledgements, bare lists of exercises), return {{"skip": true}}.

Return ONLY valid JSON, no markdown:
{{"question": "...", "answer": "..."}}

Passage (from {source}, page {page}):
\"\"\"{text}\"\"\"
"""

_BANNED = ("the passage", "the text", "this page", "the author", "the excerpt", "the chapter")


def candidate_chunks(chunks: list[dict], min_chars: int) -> dict[str, list[dict]]:
    """Group usable chunks by source, dropping duplicates and junk."""
    by_source: dict[str, list[dict]] = {}
    seen_text: set[str] = set()
    dropped_no_source = dropped_dup = dropped_short = 0

    for c in chunks:
        text = (c.get("text") or "").strip()
        source = c.get("source")
        if not source:
            dropped_no_source += 1
            continue
        if len(text) < min_chars:
            dropped_short += 1
            continue
        alpha_ratio = sum(ch.isalpha() or ch.isspace() for ch in text) / len(text)
        if alpha_ratio < 0.75:          # tables, formulas, index pages
            dropped_short += 1
            continue
        if text in seen_text:
            dropped_dup += 1
            continue
        seen_text.add(text)
        by_source.setdefault(source, []).append(c)

    logger.info("Chunks: %d total | usable %d | no-source %d | duplicates %d | short/junk %d",
                len(chunks), sum(len(v) for v in by_source.values()),
                dropped_no_source, dropped_dup, dropped_short)
    return by_source


def valid_question(question: str, chunk_text: str) -> bool:
    q = question.strip()
    if not (15 <= len(q) <= 300):
        return False
    if any(b in q.lower() for b in _BANNED):
        return False
    if shares_ngram(q, chunk_text, n=4):
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    common.setup_logging()
    ap = argparse.ArgumentParser(description="Generate an LLM-labelled retrieval test set.")
    ap.add_argument("--n", type=int, default=60, help="Target number of questions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=250,
                    help="Ignore chunks shorter than this.")
    ap.add_argument("--max-attempts-factor", type=int, default=4,
                    help="Try up to n*factor chunks (some get skipped/rejected).")
    ap.add_argument("--out", type=str, default=str(common.TESTSET_PATH))
    args = ap.parse_args(argv)

    chunks = common.load_chunks()
    by_source = candidate_chunks(chunks, args.min_chars)
    if not by_source:
        raise SystemExit("No usable chunks found. Did you run ingestion?")

    rng = random.Random(args.seed)
    model = common.generation_model()
    quota = math.ceil(args.n / len(by_source))
    logger.info("Generating ~%d questions | books=%d | quota/book=%d | model=%s",
                args.n, len(by_source), quota, model)

    testset: list[dict] = []
    for source, pool in sorted(by_source.items()):
        rng.shuffle(pool)
        accepted, attempts = 0, 0
        max_attempts = quota * args.max_attempts_factor
        for chunk in pool:
            if accepted >= quota or attempts >= max_attempts:
                break
            attempts += 1
            result = common.llm_json(
                PROMPT.format(source=source, page=chunk["page"], text=chunk["text"].strip()),
                model=model, temperature=0.4,
            )
            if not result or result.get("skip"):
                continue
            question = str(result.get("question", "")).strip()
            answer = str(result.get("answer", "")).strip()
            if not question or not answer:
                continue
            if not question.endswith("?"):
                question += "?"
            if not valid_question(question, chunk["text"]):
                continue
            accepted += 1
            testset.append({
                "id":               f"q{len(testset) + 1:03d}",
                "question":         question,
                "reference_answer": answer,
                "source":           source,
                "page":             chunk["page"],
                "chunk_id":         chunk["chunk_id"],
                "chunk_text":       chunk["text"].strip(),
            })
            if len(testset) % 10 == 0:
                logger.info("  %d questions so far...", len(testset))
        logger.info("%s: accepted %d / %d attempts", source, accepted, attempts)

    testset = testset[: args.n]
    common.write_jsonl(Path(args.out), testset)
    logger.info("Saved %d questions → %s", len(testset), args.out)
    logger.info("REVIEW a sample by hand before trusting the numbers "
                "(delete bad questions from the file; keep it fixed afterwards).")


if __name__ == "__main__":
    main()
