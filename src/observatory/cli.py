"""Command line entry point: `observatory ingest` and `observatory trends`."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from observatory.ingest import IngestError, ingest
from observatory.sources import build_source
from observatory.trends import detect_emerging


def _days(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def cmd_ingest(args: argparse.Namespace) -> int:
    sources = [build_source(name) for name in args.sources.split(",")]
    start = date.fromisoformat(args.start) if args.start else date.today()
    end = date.fromisoformat(args.end) if args.end else start
    status = 0
    for day in _days(start, end):
        try:
            result = ingest(sources, day, Path(args.raw_dir), args.limit)
        except IngestError as exc:
            print(f"[fail] {exc}", file=sys.stderr)
            status = 1
            continue
        fallback = f" (fallback after: {', '.join(n for n, _ in result.failed_sources)})" if result.failed_sources else ""
        print(f"[ok] {day} {result.source}: {result.accepted} rows, {result.rejected} rejected{fallback}")
    return status


def cmd_trends(args: argparse.Namespace) -> int:
    import duckdb

    con = duckdb.connect(args.db, read_only=True)
    rows = con.execute("select tag, captured_on, views_gained from mart_hashtag_daily order by captured_on").fetchall()
    series: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for tag, day, value in rows:
        series[tag].append((day, float(value)))
    flagged = detect_emerging(series, window=args.window, threshold=args.threshold)
    if not flagged:
        print("no emerging tags on the latest day")
        return 0
    print(f"{'tag':<20}{'day':<12}{'views gained':>14}{'baseline':>12}{'score':>8}")
    for e in flagged:
        print(f"#{e.tag:<19}{e.day!s:<12}{e.value:>14,.0f}{e.baseline:>12,.0f}{e.score:>8.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="observatory")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="land raw payloads for one day or a range")
    p.add_argument("--sources", default="tiktokapi,fixture", help="comma separated, tried in order")
    p.add_argument("--start", help="YYYY-MM-DD, default today")
    p.add_argument("--end", help="YYYY-MM-DD, default = start")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--raw-dir", default="data/raw")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("trends", help="flag hashtags breaking out on the latest day")
    p.add_argument("--db", default="data/observatory.duckdb")
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--threshold", type=float, default=3.5)
    p.set_defaults(func=cmd_trends)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
