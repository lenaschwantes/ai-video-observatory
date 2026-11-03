"""prepare -> run -> report.

prepare  downloads each dataset video once, keeps only its keyframes
         (eval/cache, not committed) and deletes the video.
run      sends the cached frames to one provider/model; results are appended
         to eval/results/<provider>__<model>.jsonl, so an interrupted run
         resumes where it stopped.
report   scores every results file and writes docs/benchmarks.{md,json}.

Robustness (see robustness.py): `prepare --variants` builds re-encoded
versions in eval/cache/<id>@<variant>; `run --variant` writes
<provider>__<model>@<variant>.jsonl (".r2" for a repeat run); `report` adds a
re-compression section comparing each version with `ref`."""

from __future__ import annotations

import csv
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from observatory.bench.metrics import (
    AI_THRESHOLD,
    HUMAN_THRESHOLD,
    auc,
    flips,
    fpr_at,
    mcnemar_exact,
    mean_shift,
    paired_delta,
    score,
)
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


def results_path(provider: str, model: str, results: Path = RESULTS, variant: str | None = None,
                 repeat: int = 1) -> Path:
    suffix = (f"@{variant}" if variant else "") + (f".r{repeat}" if repeat > 1 else "")
    return results / f"{provider}__{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}{suffix}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


QUOTA_HINTS = ("per day", "daily", "tpd", "rpd", "perday", "quota", "resource_exhausted", "resource exhausted",
               "rate limit", "rate_limit", "usage_limit", "too many requests")


class QuotaReached(RuntimeError):
    """The provider refused for rate or quota after its own retries. The run
    stops without recording the row, so the next run picks it up."""


def is_quota_error(exc: BaseException) -> bool:
    """429s and quota messages, across SDKs: Groq/OpenAI (status_code),
    google-genai (code / RESOURCE_EXHAUSTED), plain HTTP errors from httpx."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    text = str(exc).lower()
    return status == 429 or re.search(r"\b429\b", text) is not None or any(h in text for h in QUOTA_HINTS)


def run(provider: str, model: str, rows: list[dict], cache: Path = CACHE, results: Path = RESULTS,
        retry_errors: bool = False, assess_with=None, variant: str | None = None, repeat: int = 1,
        pace: float = 0.0, sleep=None) -> Path:
    """`variant` reads frames from eval/cache/<id>@<variant>; `repeat` > 1 runs
    the same frames again into a separate file (the noise floor).

    Free tiers: `pace` waits that many seconds between calls (per-minute
    limits). When the provider still answers 429 or "quota" after its own
    retries (usually a daily cap), the run stops cleanly and raises
    QuotaReached; nothing is written for that row, so running the same
    command later resumes exactly there."""
    import time

    sleep = sleep or time.sleep
    if assess_with is None:
        from observatory.detect.vision import assess_with

    path = results_path(provider, model, results, variant, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _read_jsonl(path)
    done = {r["id"] for r in previous if retry_errors is False or not r.get("error")}
    if retry_errors:
        previous = [r for r in previous if not r.get("error")]
        path.write_text("".join(json.dumps(r) + "\n" for r in previous), encoding="utf-8")

    todo = [r for r in rows if r["id"] not in done]
    first = True
    for n, row in enumerate(todo, 1):
        key = f"{row['id']}@{variant}" if variant else row["id"]
        frames = load_frames(key, cache)
        if not frames:
            continue
        record = {"id": row["id"], "label": row["label"], "provider": provider, "model": model,
                  "variant": variant, "repeat": repeat, "run_at": datetime.now(timezone.utc).isoformat()}
        if pace and not first:
            sleep(pace)
        first = False
        try:
            result, usage = assess_with(provider, model, frames, cache / key)
            record |= {"p": result.ai_likelihood, "findings": len(result.findings), "summary": result.summary,
                       "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                       "calls": usage.calls, "seconds": round(usage.seconds, 2), "error": None}
            print(f"[{row['label']:>4}] {row['id']} p={result.ai_likelihood:.2f}  ({n}/{len(todo)})")
        except Exception as exc:
            if is_quota_error(exc):
                left = len(todo) - n + 1
                raise QuotaReached(f"{provider}: rate limit or quota reached with {left} video(s) left; "
                                   f"progress is saved, run the same command later to continue") from exc
            record |= {"p": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"[fail] {row['id']}: {record['error']}")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    return path


def status(rows: list[dict], results: Path = RESULTS) -> list[dict]:
    """Progress of every results file against the dataset: scored, failed,
    still to run, and average tokens per call (to plan a daily free quota)."""
    ids = {r["id"] for r in rows}
    out = []
    for p in sorted(results.glob("*.jsonl")):
        recs = [r for r in _read_jsonl(p) if r["id"] in ids]
        ok = [r for r in recs if r.get("p") is not None]
        tokens = [r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in ok]
        out.append({
            "file": p.stem,
            "scored": len(ok),
            "failed": len(recs) - len(ok),
            "left": len(ids) - len({r["id"] for r in recs}),
            "tokens_per_call": round(sum(tokens) / len(tokens)) if tokens and any(tokens) else None,
        })
    return out


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


def _signed(x: float | None, fmt) -> str:
    if x is None:
        return "n/a"
    return ("+" if x > 0 else "−" if x < 0 else "±") + fmt(abs(x))


def _robustness_markdown(rob: list[dict], leak: dict) -> list[str]:
    lines = [
        "",
        "## Re-compression",
        "",
        "Platforms re-encode every upload. Does that change the verdict? Each video is re-encoded "
        "(`ref` is the file as downloaded) and keyframes are taken at the same instants from every version.",
        "",
    ]
    if not rob:
        lines += ["_Not run yet: `observatory bench prepare --variants all`, then `observatory bench run --variant all`._", ""]
    else:
        lines += [
            "| Model | Version | SSIM vs ref | Bitrate vs ref | Real flagged as AI: ref → version | Δ [95%] | AUC Δ [95%] "
            "| Verdicts flipped | Noise floor (ref vs ref) | Flips toward AI / real | McNemar p | Score shift, real videos |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rob:
            lines.append(
                f"| {r['provider']} / `{r['model']}` | {r['variant']} | {_num(r['ssim'], 3)} | "
                f"{_pct(r['bitrate_ratio'])} | {_pct(r['fpr_ref'])} → {_pct(r['fpr'])} | "
                f"{_signed(r['fpr_delta'], _pct)}{_ci(r['fpr_delta_ci'], lambda x: _signed(x, _pct))} | "
                f"{_signed(r['auc_delta'], _num)}{_ci(r['auc_delta_ci'], lambda x: _signed(x, _num))} | "
                f"{_pct(r['flip_rate'])} (n={r['n']}) | {_pct(r['noise_flip_rate'])} | "
                f"{r['toward_ai']} / {r['toward_real']} | {_num(r['mcnemar_p'], 3)} | {_signed(r['shift_real'], _num)} |"
            )
        lines += [
            "",
            "- **Verdicts flipped** only means something above the **noise floor**: the share of verdicts the same model "
            "changes when shown the very same frames twice. Run `--variant ref --repeat 2` to measure it.",
            "- **Δ** intervals are paired bootstraps (each video keeps its two scores together); an interval that "
            "includes zero means no change is shown.",
            "- **McNemar p** asks whether flips lean one way; small p (< 0.05) with more flips toward AI means "
            "compression makes real footage look generated.",
            "- Videos whose re-encoded frames no longer matched `ref` (SSIM below 0.5) are left out of that version.",
            "",
        ]
    h, b = leak.get("height"), leak.get("bitrate")
    if h or b:
        lines += [
            "### Can the label be guessed from the file alone?",
            "",
            "AUC of a single file property used as the score. 0.5 means it says nothing about the label; "
            "far from 0.5 means a detector could score well by learning the file instead of the generator.",
            "",
            "| Property | AUC | Videos |",
            "|---|---|---|",
        ]
        if h:
            lines.append(f"| Resolution (height) | {_num(h['auc'])} | {h['n']} |")
        if b:
            lines.append(f"| Source bitrate | {_num(b['auc'])} | {b['n']} |")
        lines.append("")
    return lines


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ci(ci, fmt) -> str:
    return f" [{fmt(ci[0])}–{fmt(ci[1])}]" if ci else ""


def _count(s: dict, k: str, n: str) -> str:
    return f" ({s[k]}/{s[n]})" if s.get(n) else ""


def _ok(records: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in records if r.get("p") is not None}


def _variant_files(results: Path) -> dict[str, dict[tuple[str, int], Path]]:
    """{"provider__model": {(variant, repeat): path}} for every <..>@<variant>[.rN].jsonl."""
    out: dict[str, dict[tuple[str, int], Path]] = {}
    for p in sorted(results.glob("*@*.jsonl")):
        base, _, tail = p.stem.partition("@")
        m = re.fullmatch(r"(.+?)(?:\.r(\d+))?", tail)
        out.setdefault(base, {})[(m[1], int(m[2] or 1))] = p
    return out


def robustness_summary(results: Path = RESULTS, cache: Path = CACHE) -> list[dict]:
    """Compare each re-encoded version with `ref`, per model, on the videos
    scored in both whose frames stayed aligned. The ref-vs-ref repeat gives
    the noise floor: verdict flips the model makes with no change at all."""
    from observatory.bench.robustness import REF, load_manifest

    rows = []
    for base, files in _variant_files(results).items():
        if (REF, 1) not in files:
            continue
        ref = _ok(_read_jsonl(files[(REF, 1)]))
        if not ref:
            continue
        first = next(iter(ref.values()))
        noise = None
        if (REF, 2) in files:
            again = _ok(_read_jsonl(files[(REF, 2)]))
            noise = flips({i: r["p"] for i, r in ref.items()}, {i: r["p"] for i, r in again.items()})
        for (variant, repeat), path in sorted(files.items()):
            if variant == REF or repeat != 1:
                continue
            var = _ok(_read_jsonl(path))
            manifests = {i: load_manifest(cache, i, variant) for i in ref.keys() & var.keys()}
            ids = sorted(i for i, m in manifests.items() if m is None or m.get("aligned", True))
            if not ids:
                continue
            labels = [1 if ref[i]["label"] == "ai" else 0 for i in ids]
            before = [min(max(float(ref[i]["p"]), 0.0), 1.0) for i in ids]
            after = [min(max(float(var[i]["p"]), 0.0), 1.0) for i in ids]
            fl = flips(dict(zip(ids, before)), dict(zip(ids, after)))
            d_fpr, d_fpr_ci = paired_delta(labels, before, after, fpr_at)
            d_auc, d_auc_ci = paired_delta(labels, before, after, auc)
            known = [m for m in manifests.values() if m]
            ssims = [m["ssim_mean"] for m in known if m.get("ssim_mean") is not None]
            ratios = sorted(m["kbps"] / m["source_kbps"] for m in known if m.get("kbps") and m.get("source_kbps"))
            rows.append({
                "provider": first["provider"],
                "model": first["model"],
                "variant": variant,
                "n": len(ids),
                "misaligned": len(manifests) - len(ids),
                "fpr_ref": fpr_at(labels, before),
                "fpr": fpr_at(labels, after),
                "fpr_delta": d_fpr,
                "fpr_delta_ci": d_fpr_ci,
                "auc_ref": auc(labels, before),
                "auc": auc(labels, after),
                "auc_delta": d_auc,
                "auc_delta_ci": d_auc_ci,
                "flip_rate": fl.rate,
                "toward_ai": fl.toward_ai,
                "toward_real": fl.toward_real,
                "mcnemar_p": mcnemar_exact(fl.toward_ai, fl.toward_real),
                "noise_flip_rate": noise.rate if noise else None,
                "noise_n": noise.n if noise else None,
                "shift_real": mean_shift(labels, before, after, 0),
                "shift_ai": mean_shift(labels, before, after, 1),
                "ssim": round(sum(ssims) / len(ssims), 4) if ssims else None,
                "bitrate_ratio": round(ratios[len(ratios) // 2], 3) if ratios else None,
            })
    return rows


def leakage(rows: list[dict], cache: Path = CACHE) -> dict:
    """Can the label be guessed without looking at the content? AUC of
    resolution alone (dataset.csv) and of source bitrate alone (ref manifests).
    Near 0.5 is good; far from it means a detector can score well by learning
    the file, not the generator."""
    from observatory.bench.robustness import REF, load_manifest

    def side(key):
        pairs = [(1 if r["label"] == "ai" else 0, v) for r in rows if (v := key(r)) is not None]
        if len({y for y, _ in pairs}) < 2:
            return None
        return {"auc": round(auc([y for y, _ in pairs], [v for _, v in pairs]), 3), "n": len(pairs)}

    def kbps(r):
        m = load_manifest(cache, r["id"], REF)
        return m.get("source_kbps") if m else None

    return {
        "height": side(lambda r: float(r["height"]) if r.get("height") else None),
        "bitrate": side(kbps),
    }


def report(rows: list[dict], results: Path = RESULTS, docs: Path = DOCS, prices_file: Path = PRICES,
           cache: Path = CACHE) -> list[dict]:
    heights = {r["id"]: int(float(r["height"] or 0)) for r in rows}
    regions = {r["id"]: r.get("region") or "global" for r in rows}
    prices = json.loads(prices_file.read_text(encoding="utf-8")) if prices_file.exists() else {}
    prices = {k: v for k, v in prices.items() if not k.startswith("_")}
    main_files = [p for p in sorted(results.glob("*.jsonl")) if "@" not in p.name]
    summaries = [_summarize(recs, heights, prices, regions) for p in main_files if (recs := _read_jsonl(p))]
    summaries.sort(key=lambda s: -((s["scores"] or {}).get("auc") or 0))

    n_ai = sum(r["label"] == "ai" for r in rows)
    n_br = sum(regions[r["id"]] == "br" for r in rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"videos": len(rows), "ai": n_ai, "real": len(rows) - n_ai, "br": n_br},
        "thresholds": {"likely_ai": AI_THRESHOLD, "likely_human": HUMAN_THRESHOLD},
        "models": summaries,
        "robustness": robustness_summary(results, cache),
        "leakage": leakage(rows, cache),
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
    lines += _robustness_markdown(payload.get("robustness") or [], payload.get("leakage") or {})
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
