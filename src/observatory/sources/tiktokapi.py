"""Live collector built on the unofficial TikTok-Api wrapper
(github.com/davidteather/TikTok-Api). Install with `pip install -e .[tiktok]`
and `python -m playwright install`. Set MS_TOKEN to a ms_token cookie from
tiktok.com for more reliable sessions."""

from __future__ import annotations

import asyncio
import os
from datetime import date
from typing import Any

from observatory.models import VideoSnapshot
from observatory.sources.base import SourceUnavailable, normalize_tiktok_item


class TikTokApiSource:
    name = "tiktokapi"

    def fetch(self, captured_on: date, limit: int) -> list[dict[str, Any]]:
        if captured_on != date.today():
            raise SourceUnavailable("live source only serves today's trending feed")
        try:
            return asyncio.run(self._fetch(limit))
        except ImportError as exc:
            raise SourceUnavailable("TikTokApi not installed: pip install -e .[tiktok]") from exc
        except Exception as exc:  # upstream breaks in many shapes; the caller falls back
            raise SourceUnavailable(f"TikTokApi failed: {exc!r}") from exc

    async def _fetch(self, limit: int) -> list[dict[str, Any]]:
        from TikTokApi import TikTokApi

        ms_token = os.environ.get("MS_TOKEN")
        async with TikTokApi() as api:
            await api.create_sessions(
                ms_tokens=[ms_token] if ms_token else None,
                num_sessions=1,
                sleep_after=3,
                headless=True,
            )
            return [video.as_dict async for video in api.trending.videos(count=limit)]

    def normalize(self, payload: dict[str, Any], captured_on: date) -> VideoSnapshot:
        return normalize_tiktok_item(payload, captured_on, self.name)
