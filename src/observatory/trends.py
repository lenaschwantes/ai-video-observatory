"""Emerging-trend detection on daily hashtag series.

Baseline method: robust z-score of today's value against the trailing window,
using median and MAD instead of mean and standard deviation so one viral day in
the history does not hide the next one. Change-point detection and a forecast
baseline are planned on top of this (see README roadmap)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date

# Scales MAD so the score is comparable to a z-score under normality.
_MAD_SCALE = 0.6745


@dataclass(frozen=True)
class Emerging:
    tag: str
    day: date
    value: float
    baseline: float
    score: float


def robust_zscore(history: list[float], value: float) -> float:
    median = statistics.median(history)
    mad = statistics.median(abs(x - median) for x in history)
    if mad == 0:
        return 0.0 if value <= median else float("inf")
    return _MAD_SCALE * (value - median) / mad


def detect_emerging(
    series: dict[str, list[tuple[date, float]]],
    window: int = 7,
    threshold: float = 3.5,
    min_history: int = 5,
) -> list[Emerging]:
    """Flag tags whose latest value is an outlier against their own recent past.

    `series` maps tag -> [(day, value), ...]. Tags with fewer than `min_history`
    earlier points are skipped: a brand-new tag has no baseline to beat."""
    flagged = []
    for tag, points in series.items():
        points = sorted(points)
        if len(points) < min_history + 1:
            continue
        day, value = points[-1]
        history = [v for _, v in points[-(window + 1) : -1]]
        score = robust_zscore(history, value)
        if score >= threshold:
            flagged.append(Emerging(tag, day, value, statistics.median(history), score))
    return sorted(flagged, key=lambda e: e.score, reverse=True)
