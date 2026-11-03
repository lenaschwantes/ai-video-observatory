"""Versioned signal weights. The score of a video is

    p = sigmoid(bias + sum(weight[signal] * llr[signal]))

v0 is the hand-set starting point (every weight 1, bias 0). Later versions
are fitted on curated labels and stored as weights/vN.json, with
weights/current.json naming the one in use and weights/history.jsonl keeping
every attempt."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", ROOT / "weights"))
SIGNALS = ("platform_label", "provenance", "text", "visual")


class Weights(BaseModel):
    version: str
    bias: float = 0.0
    per_signal: dict[str, float] = Field(default_factory=lambda: {s: 1.0 for s in SIGNALS})
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trained_on: int = 0
    metrics: dict | None = None

    def weight(self, signal: str) -> float:
        return self.per_signal.get(signal, 1.0)


DEFAULT = Weights(version="v0-heuristic", created_at=datetime(2026, 11, 3, tzinfo=timezone.utc))


def load_current(directory: Path | None = None) -> Weights:
    directory = directory or WEIGHTS_DIR
    pointer = directory / "current.json"
    if not pointer.exists():
        return DEFAULT
    name = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    return Weights.model_validate_json((directory / f"{name}.json").read_text(encoding="utf-8"))


def next_version(directory: Path | None = None) -> str:
    directory = directory or WEIGHTS_DIR
    taken = [int(p.stem[1:]) for p in directory.glob("v*.json") if p.stem[1:].isdigit()]
    return f"v{max(taken, default=0) + 1}"


def adopt(weights: Weights, directory: Path | None = None) -> None:
    directory = directory or WEIGHTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{weights.version}.json").write_text(weights.model_dump_json(indent=2), encoding="utf-8")
    (directory / "current.json").write_text(json.dumps({"version": weights.version}), encoding="utf-8")


def record(entry: dict, directory: Path | None = None) -> None:
    """Append one fitting attempt (adopted or rejected) to the public history."""
    directory = directory or WEIGHTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def history(directory: Path | None = None) -> list[dict]:
    path = (directory or WEIGHTS_DIR) / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
