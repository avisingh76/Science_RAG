"""
metrics.py  —  Pure-Python metric helpers (no heavy dependencies).
"""

import random
import re
import statistics
from typing import Optional, Sequence

_WORD = re.compile(r"[a-z0-9]+")

# "Page 45", "page 45", "pages 45 and 46", "p. 45", "pp. 45, 46"
_CITED_PAGES = re.compile(
    r"\b(?:pages?|pp?\.)\s*(\d+(?:\s*(?:,|and|&|–|-)\s*\d+)*)", re.IGNORECASE
)


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def shares_ngram(a: str, b: str, n: int = 4) -> bool:
    """True if text `a` copies any run of n consecutive words from text `b`."""
    ta, tb = tokens(a), tokens(b)
    if len(ta) < n or len(tb) < n:
        return False
    grams_b = {tuple(tb[i:i + n]) for i in range(len(tb) - n + 1)}
    return any(tuple(ta[i:i + n]) in grams_b for i in range(len(ta) - n + 1))


def extract_cited_pages(answer: str) -> set[int]:
    pages: set[int] = set()
    for group in _CITED_PAGES.findall(answer or ""):
        pages.update(int(p) for p in re.findall(r"\d+", group))
    return pages


# ── Ranking metrics ────────────────────────────────────────────────────────
def first_rank(flags: Sequence[bool]) -> Optional[int]:
    """1-based rank of the first True in `flags`, or None."""
    for i, flag in enumerate(flags, start=1):
        if flag:
            return i
    return None


def hit_rate(ranks: Sequence[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]], k: Optional[int] = None) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and (k is None or r <= k):
            total += 1.0 / r
    return total / len(ranks)


def summarize_ranks(ranks: Sequence[Optional[int]], ks: Sequence[int]) -> dict:
    out = {f"hit@{k}": hit_rate(ranks, k) for k in ks}
    out["mrr"] = mrr(ranks, max(ks))
    return out


# ── Statistics ─────────────────────────────────────────────────────────────
def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
