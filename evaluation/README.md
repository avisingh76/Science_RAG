# Science RAG — Evaluation Suite

Copy this `evaluation/` folder into the project root (next to `agent/`, `api/`, `ingestion/`).
No new dependencies beyond the app's own `requirements.txt`.

## Run (from the project root, in the app's virtual environment)

```bash
# 1. Generate a labelled test set from your indexed books (OpenRouter LLM)
python -m evaluation.generate_testset --n 60
#    -> open evaluation/testset.jsonl, delete weak questions, then freeze the file

# 2. Retrieval benchmark: FAISS vs BM25 vs Hybrid(RRF) vs Hybrid+Rerank
python -m evaluation.eval_retrieval
python -m evaluation.eval_retrieval --no-rerank      # low-RAM machines
python -m evaluation.eval_retrieval --dedupe         # collapse duplicate chunks

# 3. End-to-end: real agent + LLM-as-judge (correctness / faithfulness / relevance)
python -m evaluation.eval_answers --n 30
python -m evaluation.eval_answers --n 30 --resume    # continue after a rate-limit/crash
```

Optional `.env` settings (otherwise `OPENROUTER_MODEL` is used):

```
EVAL_GEN_MODEL=<model that writes questions>
EVAL_JUDGE_MODEL=<a DIFFERENT model than the app uses, to avoid self-preference bias>
```

## Outputs (`evaluation/results/`)

| File | Contents |
|---|---|
| `retrieval_results.json` | Hit@k, MRR, latency, 95% CI, per-question ranks |
| `retrieval_failures.md` | Questions where the right page was not in the top 3 (error analysis) |
| `answer_eval.jsonl` | Per-question answer, tools used, context, judge scores + reasons |
| `answer_eval_summary.json` | Aggregated end-to-end metrics |

## Metric definitions

- **Hit@k**: fraction of questions whose source passage is in the top-k results (production returns top 3, so Hit@3 is the headline).
- **Page-level vs chunk-level**: page-level counts any chunk from the correct book + page; chunk-level requires the exact source passage.
- **MRR**: mean of 1/rank of the first correct result.
- **Context page hit**: the correct page was in what the agent's `rag_search` returned.
- **Faithfulness / hallucination rate**: judge checks every claim against retrieved context only.

## Limitations to state honestly

- Questions are LLM-generated from single chunks, so some are answerable from several passages; page-level metrics are more forgiving.
- LLM judges are noisy. Use a different judge model than the app, read the lowest-scored answers, and report n.
- Re-generating the test set changes the numbers; keep `testset.jsonl` fixed once you are happy with it.
