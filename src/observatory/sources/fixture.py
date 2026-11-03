"""Replays recorded (synthetic) payloads, one file per day. Lets the whole
pipeline, the tests and the demo run without touching TikTok."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from observatory.models import VideoSnapshot
from observatory.sources.base import SourceUnavailable, normalize_tiktok_item

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "tiktok_trending"


class FixtureSource:
    name = "fixture"

    def __init__(self, directory: Path | str = DEFAULT_DIR) -> None:
        self.directory = Path(directory)

    def fetch(self, captured_on: date, limit: int) -> list[dict[str, Any]]:
        path = self.directory / f"{captured_on.isoformat()}.json"
        if not path.exists():
            raise SourceUnavailable(f"no fixture for {captured_on}")
        return json.loads(path.read_text(encoding="utf-8"))[:limit]

    def normalize(self, payload: dict[str, Any], captured_on: date) -> VideoSnapshot:
        return normalize_tiktok_item(payload, captured_on, self.name)
