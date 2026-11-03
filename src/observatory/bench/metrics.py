"""Scoring for a set of (label, probability) predictions. Pure Python so the
numbers are easy to audit."""

from __future__ import annotations

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
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
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

    def as_dict(self) -> dict:
        return asdict(self)


def score(labels: list[int], probs: list[float]) -> Scores:
    n = len(labels)
    ai = [p for y, p in zip(labels, probs) if y == 1]
    real = [p for y, p in zip(labels, probs) if y == 0]
    decided = [(y, p) for y, p in zip(labels, probs) if p >= AI_THRESHOLD or p <= HUMAN_THRESHOLD]
    correct = sum(1 for y, p in decided if (p >= AI_THRESHOLD) == (y == 1))
    return Scores(
        n=n,
        auc=auc(labels, probs),
        brier=brier(labels, probs),
        ece=ece(labels, probs),
        recall_ai=sum(p >= AI_THRESHOLD for p in ai) / len(ai) if ai else 0.0,
        false_positive_rate=sum(p >= AI_THRESHOLD for p in real) / len(real) if real else 0.0,
        missed_as_human=sum(p <= HUMAN_THRESHOLD for p in ai) / len(ai) if ai else 0.0,
        inconclusive_rate=1 - len(decided) / n,
        accuracy_when_decided=correct / len(decided) if decided else None,
    )
