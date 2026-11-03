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


def cmd_analyze(args: argparse.Namespace) -> int:
    from observatory.detect import analyze
    from observatory.detect.fetch import FetchError
    from observatory.detect.pipeline import analyze_file

    try:
        if Path(args.url).is_file():
            report = analyze_file(Path(args.url), Path(args.reports_dir), refresh=args.refresh)
        else:
            report = analyze(args.url, Path(args.reports_dir), refresh=args.refresh)
    except FetchError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1
    print(f"{report.video.title or report.slug}")
    print(f"{report.probability:.0%} -> {report.verdict.value}")
    for s in report.signals:
        state = s.direction.value if s.available else "n/a"
        print(f"  [{state:>7}] {s.label}: {s.note or ''}")
        for e in s.evidence:
            when = f" @{e.timestamp:.1f}s" if e.timestamp is not None else ""
            print(f"            - ({e.direction.value}){when} {e.text}")
    print(f"report: {Path(args.reports_dir) / report.slug / 'report.json'}")
    return 0


FREE_PROVIDERS = ("groq", "gemini", "sightengine", "hive")


def cmd_bench(args: argparse.Namespace) -> int:
    import observatory.detect  # noqa: F401  (loads .env)
    from observatory.bench import runner
    from observatory.detect.vision import PROVIDERS, has_key, model_for

    rows = runner.load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    from observatory.bench.robustness import REF, VARIANTS

    if args.step == "prepare":
        if args.variants:
            from observatory.bench.robustness import prepare_variants

            names = list(VARIANTS) if args.variants == "all" else [v.strip() for v in args.variants.split(",")]
            unknown = [v for v in names if v not in VARIANTS]
            if unknown:
                print(f"unknown variant(s): {', '.join(unknown)}; choose from {', '.join(VARIANTS)}", file=sys.stderr)
                return 2
            runner.prepare(rows, pause=args.pause)  # instants come from the normal frames
            prepare_variants(rows, runner.CACHE, names, pause=args.pause)
        else:
            runner.prepare(rows, pause=args.pause)
    elif args.step == "run":
        from observatory.bench import detectors

        if args.variant == "all":  # ref twice (noise floor), then every re-encode
            plan = [(REF, 1), (REF, 2), *[(v, 1) for v in VARIANTS if v != REF]]
        elif args.variant:
            if args.variant not in VARIANTS:
                print(f"unknown variant {args.variant}; choose from {', '.join(VARIANTS)} or all", file=sys.stderr)
                return 2
            plan = [(args.variant, args.repeat)]
        else:
            plan = [(None, 1)]

        if args.provider == "all":
            names = [*PROVIDERS, *detectors.DETECTORS]
        elif args.provider == "free":  # providers with a free tier; Claude and OpenAI are paid
            names = [*FREE_PROVIDERS]
        else:
            names = [args.provider]
        stopped = []
        for name in names:
            if name in detectors.DETECTORS:
                if not detectors.has_keys(name):
                    print(f"[skip] {name}: needs {' and '.join(detectors.DETECTORS[name])}")
                    continue
                model = {"sightengine": "genai", "hive": "ai-generated-image-detection"}[name]
                for variant, repeat in plan:
                    print(f"== {name} / {model}" + (f" @ {variant}" if variant else "") + (f" (run {repeat})" if repeat > 1 else ""))
                    try:
                        runner.run(name, model, rows, retry_errors=args.retry_errors, assess_with=detectors.assess_with,
                                   variant=variant, repeat=repeat, pace=args.pace)
                    except runner.QuotaReached as exc:
                        print(f"[stop] {exc}")
                        stopped.append(name)
                        break
                continue
            model = args.model or model_for(name)
            if not has_key(name) or not model:
                print(f"[skip] {name}: needs {' or '.join(PROVIDERS[name].key_env)}"
                      + ("" if model else f" and {PROVIDERS[name].model_env}"))
                continue
            for variant, repeat in plan:
                print(f"== {name} / {model}" + (f" @ {variant}" if variant else "") + (f" (run {repeat})" if repeat > 1 else ""))
                try:
                    runner.run(name, model, rows, retry_errors=args.retry_errors, variant=variant, repeat=repeat,
                               pace=args.pace)
                except runner.QuotaReached as exc:
                    print(f"[stop] {exc}")
                    stopped.append(name)
                    break  # the next versions would hit the same cap
        if stopped:
            print(f"stopped early: {', '.join(stopped)}. Run the same command later (e.g. tomorrow) to resume.")
    elif args.step == "status":
        print(f"{'results file':<48}{'scored':>7}{'failed':>7}{'left':>6}{'tokens/call':>13}")
        for s in runner.status(rows):
            print(f"{s['file']:<48}{s['scored']:>7}{s['failed']:>7}{s['left']:>6}{s['tokens_per_call'] or '':>13}")
    else:
        for m in runner.report(rows):
            s = m["scores"] or {}
            print(f"{m['provider']}/{m['model']}: AUC={s.get('auc')} FPR={s.get('false_positive_rate')} "
                  f"recall={s.get('recall_ai')} failures={m['failures']}/{m['attempted']}")
        for r in runner.robustness_summary():
            print(f"{r['provider']}/{r['model']} @ {r['variant']}: flips={r['flip_rate']} (noise {r['noise_flip_rate']}) "
                  f"FPR {r['fpr_ref']} -> {r['fpr']}")
        print("wrote docs/benchmarks.md and docs/benchmarks.json")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("observatory.detect.web:app", host=args.host, port=args.port)
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

    p = sub.add_parser("analyze", help="estimate whether one video is AI-generated, with evidence")
    p.add_argument("url", help="video link, or the path of a local video file")
    p.add_argument("--reports-dir", default="data/reports")
    p.add_argument("--refresh", action="store_true", help="re-run even if a report exists")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bench", help="benchmark the visual signal across vision models")
    p.add_argument("step", choices=["prepare", "run", "report", "status"])
    p.add_argument("--provider", default="all",
                   help="groq, anthropic, openai, gemini, sightengine, hive, all, or free (groq, gemini, sightengine, hive)")
    p.add_argument("--model", help="override the provider's model")
    p.add_argument("--limit", type=int, help="only the first N dataset rows (smoke test)")
    p.add_argument("--retry-errors", action="store_true", help="re-run rows that failed before")
    p.add_argument("--pause", type=float, default=20.0, help="prepare: seconds between downloads")
    p.add_argument("--variants", help="prepare: re-encoded versions to build (comma list, or all)")
    p.add_argument("--variant", help="run: score a re-encoded version (ref, crf28, 720p-crf35, or all)")
    p.add_argument("--repeat", type=int, default=1, help="run: run number, for the noise floor (ref twice)")
    p.add_argument("--pace", type=float, default=0.0,
                   help="run: seconds between calls, for free tiers' per-minute limits (e.g. 3)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("serve", help="run the web front end")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
