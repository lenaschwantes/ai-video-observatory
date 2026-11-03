"""Bronze layer: land raw payloads untouched, next to their normalized form.

Layout: data/raw/dt=YYYY-MM-DD/<source>.jsonl, one envelope per line.
Rows that break the contract go to dt=YYYY-MM-DD/_rejected/<source>.jsonl
instead of silently disappearing. Writing the same day twice replaces the file,
so backfills and retries are safe."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from observatory.sources.base import VideoSource


class IngestError(RuntimeError):
    def __init__(self, captured_on: date, failures: list[tuple[str, str]]) -> None:
        self.failures = failures
        detail = "; ".join(f"{name}: {err}" for name, err in failures)
        super().__init__(f"every source failed for {captured_on}: {detail}")


@dataclass
class IngestResult:
    captured_on: date
    source: str
    accepted: int
    rejected: int
    path: Path
    failed_sources: list[tuple[str, str]] = field(default_factory=list)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def ingest(sources: list[VideoSource], captured_on: date, raw_dir: Path, limit: int = 100) -> IngestResult:
    """Try sources in order and land the first one that delivers."""
    failures: list[tuple[str, str]] = []
    for source in sources:
        try:
            payloads = source.fetch(captured_on, limit)
            if not payloads:
                raise RuntimeError("returned no rows")
        except Exception as exc:
            failures.append((source.name, str(exc)))
            continue
        break
    else:
        raise IngestError(captured_on, failures)

    ingested_at = datetime.now(timezone.utc).isoformat()
    accepted, rejected = [], []
    for payload in payloads:
        envelope = {"source": source.name, "captured_on": captured_on.isoformat(), "ingested_at": ingested_at}
        try:
            snapshot = source.normalize(payload, captured_on)
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            rejected.append({**envelope, "error": str(exc), "payload": payload})
            continue
        accepted.append({**envelope, "normalized": snapshot.model_dump(mode="json"), "payload": payload})

    day_dir = raw_dir / f"dt={captured_on.isoformat()}"
    path = day_dir / f"{source.name}.jsonl"
    _write_jsonl(path, accepted)
    rejected_path = day_dir / "_rejected" / f"{source.name}.jsonl"
    if rejected:
        _write_jsonl(rejected_path, rejected)
    elif rejected_path.exists():
        rejected_path.unlink()

    return IngestResult(captured_on, source.name, len(accepted), len(rejected), path, failures)
