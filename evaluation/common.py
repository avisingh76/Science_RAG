"""
common.py  —  Shared helpers for the Science RAG evaluation suite.

Uses the same OpenRouter credentials as the app (.env):
  OPENROUTER_API_KEY   required
  OPENROUTER_MODEL     default model (used by the app)
  EVAL_GEN_MODEL       optional: model that writes the test questions
  EVAL_JUDGE_MODEL     optional: model that grades answers
                       (use a DIFFERENT model from the one in the app
                        to avoid self-preference bias)
"""

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
EVAL_DIR     = BASE_DIR / "evaluation"
RESULTS_DIR  = EVAL_DIR / "results"
TESTSET_PATH = EVAL_DIR / "testset.jsonl"
META_PATH    = DATA_DIR / "chunks_meta.pkl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger("science_rag.eval")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Models ─────────────────────────────────────────────────────────────────
def generation_model() -> str:
    model = os.getenv("EVAL_GEN_MODEL") or os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("Set EVAL_GEN_MODEL or OPENROUTER_MODEL in .env")
    return model


def judge_model() -> str:
    model = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("Set EVAL_JUDGE_MODEL or OPENROUTER_MODEL in .env")
    return model


_client = None


def get_client():
    """Lazy OpenRouter client (OpenAI-compatible)."""
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
        _client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    return _client


# ── LLM → JSON ─────────────────────────────────────────────────────────────
def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply (tolerates ``` fences)."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def llm_json(prompt: str, model: str, max_tokens: int = 800,
             temperature: float = 0.0, retries: int = 3) -> Optional[dict]:
    """Call OpenRouter and parse a JSON reply. Returns None if all retries fail."""
    client = get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = extract_json(resp.choices[0].message.content or "")
            if parsed is not None:
                return parsed
            logger.warning("Reply was not valid JSON (attempt %d/%d).", attempt + 1, retries)
        except Exception as exc:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries, exc)
        time.sleep(2 ** attempt)
    return None


# ── Files ──────────────────────────────────────────────────────────────────
def load_chunks() -> list[dict]:
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def as_text(content) -> str:
    """LangChain message content can be a str or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")
