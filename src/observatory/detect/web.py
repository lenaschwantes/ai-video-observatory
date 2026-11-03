"""Web front end: the detector ranking, the single-video analyzer and the
method page, in Portuguese (default) or English.

Public deployments pay for every model call, so live analysis is rate limited
per visitor and per day, and can be switched off entirely (ALLOW_LIVE=0) to
serve only the bundled example reports."""

from __future__ import annotations

import csv
import html
import json
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from observatory.detect import i18n
from observatory.detect.fetch import FetchError, check_url
from observatory.detect.models import Report
from observatory.detect.pipeline import AI_THRESHOLD, HUMAN_THRESHOLD, analyze, load_report
from observatory.detect.vision import PROVIDERS, assess_with

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "data/reports"))
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", ROOT / "examples" / "reports"))
BENCHMARKS_FILE = Path(os.environ.get("BENCHMARKS_FILE", ROOT / "docs" / "benchmarks.json"))
DATASET_FILE = Path(os.environ.get("DATASET_FILE", ROOT / "eval" / "dataset.csv"))
ALLOW_LIVE = os.environ.get("ALLOW_LIVE", "1") != "0"
PER_IP_DAILY = int(os.environ.get("LIVE_LIMIT_PER_IP", "3"))
GLOBAL_DAILY = int(os.environ.get("LIVE_LIMIT_PER_DAY", "30"))
ALLOW_BYOK = os.environ.get("ALLOW_BYOK", "1") != "0"
BYOK_PER_IP_DAILY = int(os.environ.get("BYOK_LIMIT_PER_IP", "20"))
REPO_URL = "https://github.com/lenaschwantes/ai-video-observatory"

DISPLAY = {
    "groq": ("Qwen (Groq)", "vision"),
    "anthropic": ("Claude", "vision"),
    "openai": ("GPT", "vision"),
    "gemini": ("Gemini", "vision"),
    "sightengine": ("Sightengine", "detector"),
    "hive": ("Hive", "detector"),
}

app = FastAPI(title="AI Video Observatory")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.globals |= {
    "repo_url": REPO_URL,
    "ai_threshold": AI_THRESHOLD,
    "human_threshold": HUMAN_THRESHOLD,
}

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
if EXAMPLES_DIR.exists():
    for example in EXAMPLES_DIR.iterdir():
        if example.is_dir() and not (REPORTS_DIR / example.name).exists():
            shutil.copytree(example, REPORTS_DIR / example.name)
app.mount("/media", StaticFiles(directory=REPORTS_DIR), name="media")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

_usage: dict[str, list[float]] = defaultdict(list)


# --- helpers -----------------------------------------------------------------


def _lang(request: Request) -> str:
    return i18n.pick_lang(request.query_params.get("lang"), request.cookies.get("lang"), request.headers.get("accept-language"))


def render(request: Request, name: str, context: dict, status_code: int = 200) -> HTMLResponse:
    lang = _lang(request)
    t = i18n.translator(lang)
    response = templates.TemplateResponse(
        request,
        name,
        {"lang": lang, "t": t, "note": lambda text: i18n.note(text, lang), "path": request.url.path, **context},
        status_code=status_code,
    )
    if request.query_params.get("lang") in i18n.LANGS:
        response.set_cookie("lang", lang, max_age=365 * 86400, samesite="lax")
    return response


def _within_limits(bucket: str, per_bucket: int = PER_IP_DAILY, count_global: bool = True) -> bool:
    cutoff = time.time() - 86400
    for key in (bucket, "*"):
        _usage[key] = [t for t in _usage[key] if t > cutoff]
    return len(_usage[bucket]) < per_bucket and (not count_global or len(_usage["*"]) < GLOBAL_DAILY)


def _recent(limit: int = 6) -> list[Report]:
    reports = [r for d in REPORTS_DIR.iterdir() if d.is_dir() and (r := load_report(REPORTS_DIR, d.name))]
    return sorted(reports, key=lambda r: r.analyzed_at, reverse=True)[:limit]


def _dataset_counts() -> dict:
    if not DATASET_FILE.exists():
        return {"videos": 0, "ai": 0, "real": 0, "br": 0}
    with DATASET_FILE.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "videos": len(rows),
        "ai": sum(r["label"] == "ai" for r in rows),
        "real": sum(r["label"] == "real" for r in rows),
        "br": sum((r.get("region") or "global") == "br" for r in rows),
    }


def benchmark_view() -> dict:
    """Shape docs/benchmarks.json for the charts: one row per tool with the
    numbers each chart needs, plus the hero (worst and best false positives)."""
    data = json.loads(BENCHMARKS_FILE.read_text(encoding="utf-8")) if BENCHMARKS_FILE.exists() else None
    if not data or not data.get("models"):
        return {"ready": False, "dataset": _dataset_counts()}
    rows = []
    for m in data["models"]:
        s = m.get("scores") or {}
        if s.get("false_positive_rate") is None:
            continue
        name, kind = DISPLAY.get(m["provider"], (m["provider"], "vision"))
        regions = m.get("by_region") or {}
        rows.append({
            "name": name,
            "kind": kind,
            "model": m["model"],
            "fpr": s["false_positive_rate"],
            "auc": s.get("auc"),
            "recall": s.get("recall_ai"),
            "missed": s.get("missed_as_human"),
            "inconclusive": s.get("inconclusive_rate"),
            "ece": s.get("ece"),
            "seconds": m.get("seconds_per_video"),
            "cost": m.get("cost_per_100_videos_usd"),
            "failures": m.get("failures"),
            "attempted": m.get("attempted"),
            "fpr_br": (regions.get("br") or {}).get("false_positive_rate"),
            "fpr_global": (regions.get("global") or {}).get("false_positive_rate"),
        })
    if not rows:
        return {"ready": False, "dataset": _dataset_counts()}
    by_fpr = sorted(rows, key=lambda r: (-r["fpr"], r["name"]))
    return {
        "ready": True,
        "dataset": {**_dataset_counts(), **data.get("dataset", {})},
        "generated_at": data.get("generated_at", "")[:10],
        "rows": by_fpr,
        "by_auc": sorted([r for r in rows if r["auc"] is not None], key=lambda r: -r["auc"]),
        "worst": by_fpr[0],
        "best": by_fpr[-1],
        "has_regions": any(r["fpr_br"] is not None and r["fpr_global"] is not None for r in rows),
    }


def _home(request: Request, error: str | None = None, status_code: int = 200) -> HTMLResponse:
    return render(
        request,
        "index.html",
        {"b": benchmark_view(), "reports": _recent(), "allow_live": ALLOW_LIVE, "allow_byok": ALLOW_BYOK,
         "error": error, "limit": PER_IP_DAILY, "providers": PROVIDERS},
        status_code,
    )


# --- routes ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _home(request)


STAGES = ["download", "frames", "signals", "visual"]
_jobs: dict[str, dict] = {}


class BYOK:
    """A visitor's own vision key. Lives only in this object for one analysis:
    never written to disk, logged, or echoed back (errors are scrubbed)."""

    def __init__(self, provider: str, model: str, key: str):
        self.provider, self.model, self._key = provider, model, key

    def assessor(self):
        def run(frames, frame_dir):
            result, _ = assess_with(self.provider, self.model, frames, frame_dir, api_key=self._key)
            result.summary = f"{result.summary} (model: {self.model}, visitor key)"
            return result
        return run

    def scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text


def _parse_byok(provider: str, model: str, key: str, pt: bool) -> BYOK | None:
    key = (key or "").strip()
    if not key:
        return None
    if provider not in PROVIDERS:
        raise ValueError("Escolha um provedor." if pt else "Pick a provider.")
    if not 10 <= len(key) <= 300 or any(c.isspace() for c in key):
        raise ValueError("Essa chave não parece válida." if pt else "That key does not look valid.")
    model = (model or "").strip() or PROVIDERS[provider].default_model or ""
    if not model:
        raise ValueError("Informe o modelo para esse provedor." if pt else "Enter the model for this provider.")
    return BYOK(provider, model, key)


def _start_job(url: str, byok: BYOK | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = _jobs[job_id] = {"stage": "download", "slug": None, "error": None}
    kwargs = {"assessor": byok.assessor()} if byok else {}

    def work():
        try:
            report = analyze(url, REPORTS_DIR, progress=lambda s: job.update(stage=s), **kwargs)
            job.update(stage="done", slug=report.slug)
        except Exception as exc:  # shown to the visitor, so keep it short
            msg = str(exc)
            job.update(stage="error", error=(byok.scrub(msg) if byok else msg)[:200])

    threading.Thread(target=work, daemon=True).start()
    return job_id


def _progress(request: Request, job_id: str) -> HTMLResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["stage"] == "done":
        response = HTMLResponse("")
        response.headers["HX-Redirect"] = f"/report/{job['slug']}"
        return response
    return render(request, "_progress.html", {"job_id": job_id, "job": job, "stages": STAGES})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    return _progress(request, job_id)


@app.post("/analyze")
def run_analysis(
    request: Request,
    url: str = Form(...),
    key_provider: str = Form(""),
    key_model: str = Form(""),
    api_key: str = Form(""),
):
    pt = _lang(request) == "pt"
    htmx = request.headers.get("HX-Request") == "true"
    t = i18n.translator(_lang(request))

    def fail(msg: str, status: int):
        # htmx swaps fragments into the form's status area; plain posts get the page.
        if htmx:
            return HTMLResponse(f'<div class="error" role="alert">{html.escape(msg)}</div>')
        return _home(request, msg, status)

    try:
        byok = _parse_byok(key_provider, key_model, api_key, pt)
    except ValueError as exc:
        return fail(str(exc), 400)
    # With the visitor's own key the model cost is theirs, so live analysis is
    # allowed even when the host switched it off, under a looser limit.
    if not ALLOW_LIVE and not (byok and ALLOW_BYOK):
        return fail(t("live_off"), 403)
    try:
        url = check_url(url)
    except FetchError:
        return fail("Só links públicos do YouTube, TikTok ou Instagram." if pt else "Only public YouTube, TikTok or Instagram links.", 400)
    ip = request.client.host if request.client else "unknown"
    bucket = f"byok:{ip}" if byok else ip
    if not _within_limits(bucket, BYOK_PER_IP_DAILY if byok else PER_IP_DAILY, count_global=not byok):
        return fail("Limite diário atingido. Tente amanhã ou rode localmente." if pt else "Daily limit reached. Try tomorrow or run it locally.", 429)
    _usage[bucket].append(time.time())
    if not byok:
        _usage["*"].append(time.time())

    if htmx:
        return _progress(request, _start_job(url, byok))
    try:
        report = analyze(url, REPORTS_DIR, **({"assessor": byok.assessor()} if byok else {}))
    except FetchError as exc:
        msg = str(exc)
        return fail(("Não foi possível baixar o vídeo: " if pt else "Could not download the video: ") + (byok.scrub(msg) if byok else msg)[:200], 422)
    return RedirectResponse(f"/report/{report.slug}", status_code=303)


@app.get("/benchmarks", response_class=HTMLResponse)
def benchmarks(request: Request):
    return render(request, "benchmarks.html", {"b": benchmark_view()})


@app.get("/metodo", response_class=HTMLResponse)
@app.get("/method", response_class=HTMLResponse)
def method(request: Request):
    return render(request, "method.html", {"b": benchmark_view()})


@app.get("/report/{slug}", response_class=HTMLResponse)
def show_report(request: Request, slug: str):
    report = load_report(REPORTS_DIR, slug)
    if report is None:
        raise HTTPException(404, "report not found")
    duration = report.video.duration or max([e.timestamp or 0 for s in report.signals for e in s.evidence] + [1])
    marks = [
        {"t": e.timestamp, "pos": min(e.timestamp / duration, 1.0), "dir": e.direction.value, "text": e.text}
        for s in report.signals for e in s.evidence if e.timestamp is not None
    ]
    return render(request, "report.html", {"r": report, "marks": marks, "duration": duration})


@app.get("/api/report/{slug}")
def report_json(slug: str):
    report = load_report(REPORTS_DIR, slug)
    if report is None:
        raise HTTPException(404, "report not found")
    return JSONResponse(report.model_dump(mode="json"))


@app.get("/api/benchmarks")
def benchmarks_json():
    if not BENCHMARKS_FILE.exists():
        raise HTTPException(404, "no benchmark yet")
    return JSONResponse(json.loads(BENCHMARKS_FILE.read_text(encoding="utf-8")))
