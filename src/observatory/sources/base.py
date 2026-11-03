"""Source interface. Every collector (official API, scraper, paid API, replayed
fixtures) implements the same two methods, so a broken collector can be swapped
or used as a fallback without touching the rest of the pipeline."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Protocol

from observatory.models import VideoSnapshot


class SourceUnavailable(RuntimeError):
    """The source could not deliver data (blocked, rate limited, empty, offline)."""


class VideoSource(Protocol):
    name: str

    def fetch(self, captured_on: date, limit: int) -> list[dict[str, Any]]:
        """Return raw payloads exactly as the upstream delivered them."""
        ...

    def normalize(self, payload: dict[str, Any], captured_on: date) -> VideoSnapshot:
        """Map one raw payload onto the shared contract."""
        ...


_HASHTAG = re.compile(r"#(\w+)")


def normalize_tiktok_item(payload: dict[str, Any], captured_on: date, source: str) -> VideoSnapshot:
    """Map TikTok's item shape (as returned by the web API) onto VideoSnapshot."""
    stats = payload.get("stats") or {}
    desc = payload.get("desc") or ""
    tags = [c.get("title", "") for c in payload.get("challenges") or []]
    if not tags:
        tags = _HASHTAG.findall(desc)
    created = payload.get("createTime")
    return VideoSnapshot(
        video_id=str(payload["id"]),
        captured_on=captured_on,
        source=source,
        author=(payload.get("author") or {}).get("uniqueId"),
        description=desc,
        hashtags=tags,
        music_id=str((payload.get("music") or {}).get("id")) if payload.get("music") else None,
        created_at=datetime.fromtimestamp(int(created), tz=timezone.utc) if created else None,
        views=int(stats.get("playCount", 0)),
        likes=int(stats.get("diggCount", 0)),
        comments=int(stats.get("commentCount", 0)),
        shares=int(stats.get("shareCount", 0)),
    )
