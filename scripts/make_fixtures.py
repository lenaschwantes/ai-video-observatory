"""Generate 14 days of synthetic "trending" payloads in TikTok's item shape.

Everything here is fake (ids, creators, captions). The data hides one story
for the pipeline to find: #aiasmr is a minor tag until 2026-10-13, then breaks
out. Day 2026-10-06 carries one malformed row to exercise the quarantine.

Run: python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "tiktok_trending"
START = date(2026, 10, 1)
DAYS = 14
NEW_PER_DAY = 15
TRENDING_SIZE = 40
BREAKOUT_TAG = "aiasmr"
BREAKOUT_DAY = 12  # index of 2026-10-13

TAGS = {
    "aivideo": 5.0,
    "aiart": 3.0,
    "sora": 2.0,
    "veo": 2.0,
    "kling": 1.5,
    "runway": 1.0,
    "aianimation": 2.0,
    "aicomedy": 2.5,
    "aifilter": 1.5,
    BREAKOUT_TAG: 0.6,
}


def main() -> None:
    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    videos: list[dict] = []
    for d in range(DAYS):
        day = START + timedelta(days=d)
        weights = dict(TAGS)
        if d >= BREAKOUT_DAY:
            weights[BREAKOUT_TAG] = 8.0
        for _ in range(NEW_PER_DAY):
            n = len(videos)
            tags = set(rng.choices(list(weights), weights=list(weights.values()), k=rng.choice([1, 2])))
            rate = rng.lognormvariate(10.8, 0.8)
            if BREAKOUT_TAG in tags and d >= BREAKOUT_DAY:
                rate *= 4
            created = datetime.combine(day, time(rng.randrange(24), rng.randrange(60)), tzinfo=timezone.utc)
            videos.append(
                {
                    "id": f"synthetic-{n:05d}",
                    "desc": f"synthetic clip {n} " + " ".join(f"#{t}" for t in sorted(tags)),
                    "createTime": int(created.timestamp()),
                    "author": {"uniqueId": f"synthetic_creator_{rng.randrange(60):02d}"},
                    "music": {"id": f"synthetic-music-{rng.randrange(30):03d}"},
                    "challenges": [{"title": t} for t in sorted(tags)],
                    "stats": {"playCount": 0, "diggCount": 0, "commentCount": 0, "shareCount": 0},
                    "_born": d,
                    "_rate": rate,
                }
            )

        gains = []
        for v in videos:
            age = d - v["_born"]
            if age > 3:
                continue
            gain = int(v["_rate"] * (0.55**age) * rng.uniform(0.7, 1.3))
            s = v["stats"]
            s["playCount"] += gain
            s["diggCount"] += int(gain * rng.uniform(0.04, 0.12))
            s["commentCount"] += int(gain * rng.uniform(0.002, 0.01))
            s["shareCount"] += int(gain * rng.uniform(0.003, 0.02))
            gains.append((gain, v))

        trending = [v for _, v in sorted(gains, key=lambda g: g[0], reverse=True)[:TRENDING_SIZE]]
        payloads = [{k: val for k, val in v.items() if not k.startswith("_")} for v in trending]
        payloads = json.loads(json.dumps(payloads))  # freeze today's cumulative stats
        if day == date(2026, 10, 6):
            payloads[-1]["stats"]["playCount"] = -1  # contract violation on purpose
        (OUT / f"{day.isoformat()}.json").write_text(json.dumps(payloads, indent=1), encoding="utf-8")
    print(f"wrote {DAYS} days to {OUT}")


if __name__ == "__main__":
    main()
