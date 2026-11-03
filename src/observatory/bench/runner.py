"""prepare -> run -> report.

prepare  downloads each dataset video once, keeps only its keyframes
         (eval/cache, not committed) and deletes the video.
run      sends the cached frames to one provider/model; results are appended
         to eval/results/<provider>__<model>.jsonl, so an interrupted run
         resumes where it stopped.
report   scores every results file and writes docs/benchmarks.{md,json}."""

from __future__ import annotations

import csv
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from observatory.bench.metrics import AI_THRESHOLD, HUMAN_THRESHOLD, score
from observatory.detect.models import Frame

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "eval" / "dataset.csv"
CACHE = ROOT / "eval" / "cache"
RESULTS = ROOT / "eval" / "results"
PRICES = ROOT / "eval" / "prices.json"
DOCS = ROOT / "docs"
LOW_RES = 480


def load_dataset(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frames_file(vid: str, cache: Path) -> Path:
    return cache / vid / "frames.json"


def load_frames(vid: str, cache: Path = CACHE) -> list[Frame] | None:
    path = _frames_file(vid, cache)
    if not path.exists():
        return None
    return [Frame(**f) for f in json.loads(path.read_text(encoding="utf-8"))["frames"]]


def prepare(rows: list[dict], cache: Path = CACHE, fetcher=None, pause: float = 20.0) -> None:
    """Downloads are spaced out (`pause` seconds): YouTube starts asking for
    sign-in after a burst of requests from one address."""
    import time

    from observatory.detect.fetch import fetch
    from observatory.detect.frames import extract_frames

    fetcher = fetcher or fetch
    first = True
    for row in rows:
        frames_file = _frames_file(row["id"], cache)
        if frames_file.exists() and all((cache / row["id"] / f.file).exists() for f in load_frames(row["id"], cache) or []):
            continue
        if not first:
            time.sleep(pause)
        first = False
        out = cache / row["id"]
        try:
            with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
                _, video = fetcher(row["url"], Path(tmp))
                frames = extract_frames(video, out)
        except Exception as exc:
            print(f"[skip] {row['id']}: {exc}")
            continue
        _frames_file(row["id"], cache).write_text(
            json.dumps({"frames": [f.model_dump() for f in frames]}, indent=1), encoding="utf-8"
        )
        print(f"[ok] {row['id']}: {len(frames)} frames")


def results_path(provider: str, model: str, results: Path = RESULTS) -> Path:
    return results / f"{provider}__{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(provider: str, model: str, rows: list[dict], cache: Path = CACHE, results: Path = RESULTS,
        retry_errors: bool = False, assess_with=None) -> Path:
    if assess_with is None:
        from observatory.detect.vision import assess_with

    path = results_path(provider, model, results)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _read_jsonl(path)
    done = {r["id"] for r in previous if retry_errors is False or not r.get("error")}
    if retry_errors:
        previous = [r for r in previous if not r.get("error")]
        path.write_text("".join(json.dumps(r) + "\n" for r in previous), encoding="utf-8")

    for row in rows:
        if row["id"] in done:
            continue
        frames = load_frames(row["id"], cache)
        if not frames:
            continue
        record = {"id": row["id"], "label": row["label"], "provider": provider, "model": model,
                  "run_at": datetime.now(timezone.utc).isoformat()}
        try:
            result, usage = assess_with(provider, model, frames, cache / row["id"])
            record |= {"p": result.ai_likelihood, "findings": len(result.findings), "summary": result.summary,
                       "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                       "calls": usage.calls, "seconds": round(usage.seconds, 2), "error": None}
            print(f"[{row['label']:>4}] {row['id']} p={result.ai_likelihood:.2f}")
        except Exception as exc:
            record |= {"p": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"[fail] {row['id']}: {record['error']}")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return path


def _subset_scores(labels, probs, ok, keep) -> dict | None:
    sub = [(l, p) for l, p, r in zip(labels, probs, ok) if keep(r["id"])]
    if sub and len({l for l, _ in sub}) > 1:
        return score([l for l, _ in sub], [p for _, p in sub]).as_dict()
    return None


def _summarize(records: list[dict], heights: dict[str, int], prices: dict, regions: dict[str, str] | None = None) -> dict:
    ok = [r for r in records if r.get("p") is not None]
    labels = [1 if r["label"] == "ai" else 0 for r in ok]
    probs = [min(max(float(r["p"]), 0.0), 1.0) for r in ok]
    first = records[0]
    key = f"{first['provider']}/{first['model']}"
    summary = {
        "provider": first["provider"],
        "model": first["model"],
        "attempted": len(records),
        "failures": len(records) - len(ok),
        "scores": score(labels, probs).as_dict() if ok else None,
        "seconds_per_video": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None,
        "tokens_per_video": round(sum(r["input_tokens"] + r["output_tokens"] for r in ok) / len(ok)) if ok else None,
        "cost_per_100_videos_usd": None,
    }
    if ok and key in prices:
        price = prices[key]
        if isinstance(price, dict):  # detectors bill per image
            cost = sum(r["calls"] for r in ok) * price["per_call"] / len(ok)
        else:
            price_in, price_out = price
            cost = sum(r["input_tokens"] * price_in + r["output_tokens"] * price_out for r in ok) / 1e6 / len(ok)
        summary["cost_per_100_videos_usd"] = round(cost * 100, 4)
    regions = regions or {}
    by_res = {
        "low_res": _subset_scores(labels, probs, ok, lambda i: heights.get(i, 0) <= LOW_RES),
        "high_res": _subset_scores(labels, probs, ok, lambda i: heights.get(i, 0) > LOW_RES),
    }
    by_region = {
        name: _subset_scores(labels, probs, ok, lambda i, name=name: regions.get(i, "global") == name)
        for name in ("global", "br")
    }
    summary["by_resolution"] = {k: v for k, v in by_res.items() if v}
    summary["by_region"] = {k: v for k, v in by_region.items() if v}
    return summary


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ci(ci, fmt) -> str:
    return f" [{fmt(ci[0])}–{fmt(ci[1])}]" if ci else ""


def _count(s: dict, k: str, n: str) -> str:
    return f" ({s[k]}/{s[n]})" if s.get(n) else ""


def report(rows: list[dict], results: Path = RESULTS, docs: Path = DOCS, prices_file: Path = PRICES) -> list[dict]:
    heights = {r["id"]: int(float(r["height"] or 0)) for r in rows}
    regions = {r["id"]: r.get("region") or "global" for r in rows}
    prices = json.loads(prices_file.read_text(encoding="utf-8")) if prices_file.exists() else {}
    prices = {k: v for k, v in prices.items() if not k.startswith("_")}
    summaries = [_summarize(recs, heights, prices, regions) for p in sorted(results.glob("*.jsonl")) if (recs := _read_jsonl(p))]
    summaries.sort(key=lambda s: -((s["scores"] or {}).get("auc") or 0))

    n_ai = sum(r["label"] == "ai" for r in rows)
    n_br = sum(regions[r["id"]] == "br" for r in rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"videos": len(rows), "ai": n_ai, "real": len(rows) - n_ai, "br": n_br},
        "thresholds": {"likely_ai": AI_THRESHOLD, "likely_human": HUMAN_THRESHOLD},
        "models": summaries,
    }
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "benchmarks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (docs / "benchmarks.md").write_text(_markdown(payload), encoding="utf-8")
    return summaries


def _markdown(payload: dict) -> str:
    d = payload["dataset"]
    lines = [
        "# Benchmarks: visual signal by model",
        "",
        "_Generated by `observatory bench report`. Do not edit by hand._",
        "",
        f"Dataset: **{d['videos']} videos** ({d['ai']} AI-generated, {d['real']} real). "
        "Every model sees the same keyframes (up to 8 per video, one per scene) and nothing else: no title, caption or tags.",
        "",
        "Brackets are 95% intervals (Wilson for rates, bootstrap for AUC): with a set this small they are wide, and two models whose intervals overlap are not shown to differ.",
        "",
        "| Model | AUC | Real flagged as AI | AI caught | AI caught, no real flagged | AI called real | Inconclusive | Brier | Calibration error | s / video | Cost / 100 videos | Failures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in payload["models"]:
        s = m["scores"] or {}
        cost = f"${m['cost_per_100_videos_usd']:.2f}" if m["cost_per_100_videos_usd"] is not None else "n/a"
        lines.append(
            f"| {m['provider']} / `{m['model']}` | {_num(s.get('auc'))}{_ci(s.get('auc_ci'), _num)} | "
            f"{_pct(s.get('false_positive_rate'))}{_count(s, 'false_positives', 'n_real')}{_ci(s.get('false_positive_rate_ci'), _pct)} | "
            f"{_pct(s.get('recall_ai'))}{_ci(s.get('recall_ai_ci'), _pct)} | {_pct(s.get('recall_at_zero_fpr'))} | "
            f"{_pct(s.get('missed_as_human'))} | {_pct(s.get('inconclusive_rate'))} | "
            f"{_num(s.get('brier'))} | {_num(s.get('ece'))} | {_num(m['seconds_per_video'], 1)} | {cost} | "
            f"{m['failures']}/{m['attempted']} |"
        )
    lines += [
        "",
        "## By resolution",
        "",
        "| Model | Low res (≤480p): real flagged as AI | Low res: AUC | High res: real flagged as AI | High res: AUC |",
        "|---|---|---|---|---|",
    ]
    for m in payload["models"]:
        lo, hi = m["by_resolution"].get("low_res", {}), m["by_resolution"].get("high_res", {})
        lines.append(
            f"| {m['provider']} / `{m['model']}` | {_pct(lo.get('false_positive_rate'))} | {_num(lo.get('auc'))} | "
            f"{_pct(hi.get('false_positive_rate'))} | {_num(hi.get('auc'))} |"
        )
    lines += [
        "",
        "## Brazilian content vs the rest",
        "",
        "Do detectors trained mostly on foreign content do worse on Brazilian videos?",
        "",
        "| Model | Brazil: real flagged as AI | Brazil: AUC | Global: real flagged as AI | Global: AUC |",
        "|---|---|---|---|---|",
    ]
    for m in payload["models"]:
        br, gl = m["by_region"].get("br", {}), m["by_region"].get("global", {})
        lines.append(
            f"| {m['provider']} / `{m['model']}` | {_pct(br.get('false_positive_rate'))} | {_num(br.get('auc'))} | "
            f"{_pct(gl.get('false_positive_rate'))} | {_num(gl.get('auc'))} |"
        )
    t = payload["thresholds"]
    lines += [
        "",
        "## How to read it",
        "",
        "- **AUC**: chance that a random AI video gets a higher score than a random real one. 0.5 is a coin flip, 1.0 is perfect.",
        f"- **Real flagged as AI**: share of real videos scored ≥ {t['likely_ai']:.0%}. The most harmful error, so it is listed first.",
        f"- **AI caught / AI called real**: share of AI videos scored ≥ {t['likely_ai']:.0%} / ≤ {t['likely_human']:.0%}.",
        "- **Inconclusive**: share of videos left between the two thresholds.",
        "- **Brier** (lower is better) and **calibration error** (lower is better): whether a stated 80% is right about 80% of the time.",
        "- **Cost**: from token counts and `eval/prices.json`; n/a when no price is set for that model.",
        "",
        "## Method and caveats",
        "",
        "- AI videos: the uploader names the generator (Sora, Veo, Kling, Runway...). Tutorials, reactions and animation excluded.",
        "- Real videos: uploaded before 2022, before public models could generate photorealistic video. Car ads and VFX-heavy clips excluded.",
        "- The set is small. Differences of a few points between models are noise; read the table as a first ranking, not a verdict.",
        "- Real videos skew older and lower resolution than AI ones, so resolution itself can leak the label; the resolution table shows how much.",
        "- Reproduce: `observatory bench prepare`, `observatory bench run --provider <name>`, `observatory bench report`.",
        "",
    ]
    return "\n".join(lines)
