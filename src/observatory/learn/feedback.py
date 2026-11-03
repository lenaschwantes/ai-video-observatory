"""Visitor votes and curated labels, both append-only JSONL.

Votes are hints: they only decide which reports a curator looks at first.
Only curated labels (with a source) are used for training, so a wave of
false votes cannot teach the model to accuse real videos."""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LEARN_DIR = Path(os.environ.get("LEARN_DIR", "data/learning"))
# Voters are stored as a salted hash, never as an IP. Without a fixed salt the
# hash changes on restart, which only weakens duplicate detection.
_SALT = os.environ.get("FEEDBACK_SALT") or secrets.token_hex(16)
_lock = threading.Lock()


class Vote(BaseModel):
    slug: str
    claimed: Literal["ai", "real", "unsure"]
    evidence_url: str | None = Field(None, max_length=500)
    voter: str
    model_verdict: str
    model_probability: float
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evidence_url")
    @classmethod
    def _http_only(cls, url: str | None) -> str | None:
        url = (url or "").strip() or None
        if url and not url.startswith(("https://", "http://")):
            raise ValueError("evidence must be an http(s) link")
        return url


class Label(BaseModel):
    slug: str
    label: Literal["ai", "real"]
    source: str = Field(min_length=3, description="why we know: creator's statement, upload date, ...")
    curated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def voter_id(ip: str) -> str:
    return hashlib.sha256(f"{_SALT}:{ip}".encode()).hexdigest()[:16]


def _append(path: Path, row: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(row.model_dump_json() + "\n")


def _read(path: Path, model):
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def votes(directory: Path | None = None) -> list[Vote]:
    return _read((directory or LEARN_DIR) / "votes.jsonl", Vote)


def labels(directory: Path | None = None) -> dict[str, Label]:
    """Latest curated label per report (a curator can correct an earlier one)."""
    return {lab.slug: lab for lab in _read((directory or LEARN_DIR) / "labels.jsonl", Label)}


def add_vote(vote: Vote, directory: Path | None = None) -> bool:
    """Store a vote unless this voter already voted on this report."""
    directory = directory or LEARN_DIR
    if any(v.slug == vote.slug and v.voter == vote.voter for v in votes(directory)):
        return False
    _append(directory / "votes.jsonl", vote)
    return True


def add_label(label: Label, directory: Path | None = None) -> None:
    _append((directory or LEARN_DIR) / "labels.jsonl", label)


def queue(reports: dict[str, dict], directory: Path | None = None) -> list[dict]:
    """Reports waiting for a curator, most useful first: the model was unsure
    (active learning), or visitors disagree with it, or many people voted."""
    directory = directory or LEARN_DIR
    done = labels(directory)
    by_slug: dict[str, list[Vote]] = {}
    for v in votes(directory):
        by_slug.setdefault(v.slug, []).append(v)
    rows = []
    for slug, report in reports.items():
        if slug in done:
            continue
        vs = by_slug.get(slug, [])
        p = report["probability"]
        model_says = "ai" if p >= 0.5 else "real"
        disagree = sum(1 for v in vs if v.claimed in ("ai", "real") and v.claimed != model_says)
        uncertainty = 1 - abs(p - 0.5) * 2  # 1 at p=0.5, 0 at p=0 or 1
        rows.append({
            "slug": slug,
            "title": report.get("title", ""),
            "probability": p,
            "votes": len(vs),
            "disagree": disagree,
            "evidence": [v.evidence_url for v in vs if v.evidence_url],
            "priority": round(2 * uncertainty + disagree + 0.2 * len(vs), 3),
        })
    return sorted(rows, key=lambda r: -r["priority"])
