"""Scoring for a set of (label, probability) predictions. Pure Python so the
numbers are easy to audit."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
import random
from dataclasses import asdict, dataclass

AI_THRESHOLD = 0.75
HUMAN_THRESHOLD = 0.25


def auc(labels: list[int], probs: list[float]) -> float | None:
    """Probability that a random AI video scores above a random real one
    (ties count half). None when one class is missing."""
    pos = [p for y, p in zip(labels, probs) if y == 1]
    neg = [p for y, p in zip(labels, probs) if y == 0]
    if not pos or not neg:
        return None
    neg = sorted(neg)
    # for each AI score: real videos below it count 1, ties count half
    wins = sum(bisect_left(neg, a) + (bisect_right(neg, a) - bisect_left(neg, a)) / 2 for a in pos)
    return wins / (len(pos) * len(neg))


def brier(labels: list[int], probs: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probs)) / len(labels)


def ece(labels: list[int], probs: list[float], bins: int = 5) -> float:
    """Expected calibration error: average gap between stated confidence and
    observed frequency, weighted by how many predictions fall in each bin."""
    total, n = 0.0, len(labels)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            freq = sum(labels[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(conf - freq)
    return total


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for k successes in n trials. Unlike the plain
    normal interval it stays honest at 0/n and n/n, which small sets hit often."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def auc_ci(labels: list[int], probs: list[float], reps: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile bootstrap for the AUC, resampling AI and real videos
    separately so every replicate keeps both classes. Seeded: same data, same CI."""
    pos = [p for y, p in zip(labels, probs) if y == 1]
    neg = [p for y, p in zip(labels, probs) if y == 0]
    if not pos or not neg:
        return None
    rng = random.Random(seed)
    stats = sorted(
        auc([1] * len(pos) + [0] * len(neg), rng.choices(pos, k=len(pos)) + rng.choices(neg, k=len(neg)))
        for _ in range(reps)
    )
    return round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)


def recall_at_zero_fpr(labels: list[int], probs: list[float]) -> float | None:
    """Share of AI videos scored above every real video: what the model would
    catch with its threshold set so that no real video in the set is accused."""
    pos = [p for y, p in zip(labels, probs) if y == 1]
    neg = [p for y, p in zip(labels, probs) if y == 0]
    if not pos or not neg:
        return None
    return sum(p > max(neg) for p in pos) / len(pos)


@dataclass
class Scores:
    n: int
    auc: float | None
    brier: float
    ece: float
    recall_ai: float  # AI videos flagged "likely AI"
    false_positive_rate: float  # real videos flagged "likely AI" (the costly error)
    missed_as_human: float  # AI videos called "likely not AI"
    inconclusive_rate: float
    accuracy_when_decided: float | None
    n_ai: int = 0
    n_real: int = 0
    false_positives: int = 0  # count behind false_positive_rate
    false_positive_rate_ci: tuple[float, float] | None = None  # 95% Wilson
    recall_ai_ci: tuple[float, float] | None = None  # 95% Wilson
    auc_ci: tuple[float, float] | None = None  # 95% bootstrap
    recall_at_zero_fpr: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def score(labels: list[int], probs: list[float]) -> Scores:
    n = len(labels)
    ai = [p for y, p in zip(labels, probs) if y == 1]
    real = [p for y, p in zip(labels, probs) if y == 0]
    decided = [(y, p) for y, p in zip(labels, probs) if p >= AI_THRESHOLD or p <= HUMAN_THRESHOLD]
    correct = sum(1 for y, p in decided if (p >= AI_THRESHOLD) == (y == 1))
    fp = sum(p >= AI_THRESHOLD for p in real)
    caught = sum(p >= AI_THRESHOLD for p in ai)
    return Scores(
        n=n,
        auc=auc(labels, probs),
        brier=brier(labels, probs),
        ece=ece(labels, probs),
        recall_ai=caught / len(ai) if ai else 0.0,
        false_positive_rate=fp / len(real) if real else 0.0,
        missed_as_human=sum(p <= HUMAN_THRESHOLD for p in ai) / len(ai) if ai else 0.0,
        inconclusive_rate=1 - len(decided) / n,
        accuracy_when_decided=correct / len(decided) if decided else None,
        n_ai=len(ai),
        n_real=len(real),
        false_positives=fp,
        false_positive_rate_ci=wilson(fp, len(real)),
        recall_ai_ci=wilson(caught, len(ai)),
        auc_ci=auc_ci(labels, probs),
        recall_at_zero_fpr=recall_at_zero_fpr(labels, probs),
    )


# --- paired comparisons (same videos, two conditions) ---------------------

VERDICT_RANK = {"likely_human": 0, "inconclusive": 1, "likely_ai": 2}


def verdict(p: float) -> str:
    return "likely_ai" if p >= AI_THRESHOLD else "likely_human" if p <= HUMAN_THRESHOLD else "inconclusive"


@dataclass
class Flips:
    n: int
    flipped: int  # verdict changed
    toward_ai: int  # e.g. inconclusive -> likely AI, likely human -> inconclusive
    toward_real: int

    @property
    def rate(self) -> float | None:
        return self.flipped / self.n if self.n else None


def flips(before: dict[str, float], after: dict[str, float]) -> Flips:
    """Verdict changes between two runs over the videos present in both."""
    ids = sorted(before.keys() & after.keys())
    up = down = 0
    for i in ids:
        d = VERDICT_RANK[verdict(after[i])] - VERDICT_RANK[verdict(before[i])]
        up += d > 0
        down += d < 0
    return Flips(n=len(ids), flipped=up + down, toward_ai=up, toward_real=down)


def mcnemar_exact(toward_ai: int, toward_real: int) -> float | None:
    """Two-sided exact McNemar p-value: are changes in one direction more
    common than chance would give if the condition pushed neither way?"""
    n = toward_ai + toward_real
    if n == 0:
        return None
    k = min(toward_ai, toward_real)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fpr_at(labels: list[int], probs: list[float], threshold: float = AI_THRESHOLD) -> float | None:
    real = [p for y, p in zip(labels, probs) if y == 0]
    return sum(p >= threshold for p in real) / len(real) if real else None


def paired_delta(labels: list[int], before: list[float], after: list[float], metric,
                 reps: int = 1000, seed: int = 0) -> tuple[float | None, tuple[float, float] | None]:
    """metric(after) - metric(before), with a 95% paired bootstrap interval:
    each replicate resamples videos (AI and real separately) and keeps every
    video's two scores together, so per-video difficulty cancels out."""
    point_a, point_b = metric(labels, after), metric(labels, before)
    if point_a is None or point_b is None:
        return None, None
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        idx = (rng.choices(pos, k=len(pos)) if pos else []) + (rng.choices(neg, k=len(neg)) if neg else [])
        ls = [labels[i] for i in idx]
        a, b = metric(ls, [after[i] for i in idx]), metric(ls, [before[i] for i in idx])
        if a is not None and b is not None:
            stats.append(a - b)
    if not stats:
        return round(point_a - point_b, 4), None
    stats.sort()
    k = len(stats)
    return round(point_a - point_b, 4), (round(stats[int(0.025 * k)], 4), round(stats[max(int(0.975 * k) - 1, 0)], 4))


def mean_shift(labels: list[int], before: list[float], after: list[float], cls: int) -> float | None:
    """Average change in score for one class (0 = real, 1 = AI)."""
    d = [a - b for y, a, b in zip(labels, after, before) if y == cls]
    return round(sum(d) / len(d), 4) if d else None
