"""Smoke test for the live collector: fetch a few videos from one hashtag and
save the raw payloads for inspection. Not part of CI (needs network and a
browser). Run: python scripts/smoke_live.py [hashtag] [out.json]"""

import asyncio
import json
import os
import sys

from TikTokApi import TikTokApi


async def main(tag: str, out: str) -> None:
    ms_token = os.environ.get("MS_TOKEN")
    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token] if ms_token else None,
            num_sessions=1,
            sleep_after=3,
            headless=os.environ.get("HEADLESS", "1") != "0",
            browser="chromium",
        )
        items = [v.as_dict async for v in api.hashtag(name=tag).videos(count=10)]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    print(f"{len(items)} videos from #{tag} -> {out}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "aivideo", sys.argv[2] if len(sys.argv) > 2 else "data/smoke.json"))
