"""Web front end: paste a link, read the report.

Public deployments pay for every model call, so live analysis is rate limited
per visitor and per day, and can be switched off entirely (ALLOW_LIVE=0) to
serve only the bundled example reports."""

from __future__ import annotations

import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from observatory.detect.fetch import FetchError, check_url
from observatory.detect.models import Report
from observatory.detect.pipeline import analyze, load_report

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "data/reports"))
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", Path(__file__).resolve().parents[3] / "examples" / "reports"))
ALLOW_LIVE = os.environ.get("ALLOW_LIVE", "1") != "0"
PER_IP_DAILY = int(os.environ.get("LIVE_LIMIT_PER_IP", "3"))
GLOBAL_DAILY = int(os.environ.get("LIVE_LIMIT_PER_DAY", "30"))

VERDICT_TEXT = {
    "likely_ai": "Likely AI-generated",
    "inconclusive": "Inconclusive",
    "likely_human": "Likely not AI-generated",
}

app = FastAPI(title="AI Video Observatory")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.globals["verdict_text"] = VERDICT_TEXT

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
if EXAMPLES_DIR.exists():
    for example in EXAMPLES_DIR.iterdir():
        if example.is_dir() and not (REPORTS_DIR / example.name).exists():
            shutil.copytree(example, REPORTS_DIR / example.name)
app.mount("/media", StaticFiles(directory=REPORTS_DIR), name="media")

_usage: dict[str, list[float]] = defaultdict(list)


def _within_limits(ip: str) -> bool:
    cutoff = time.time() - 86400
    for key in (ip, "*"):
        _usage[key] = [t for t in _usage[key] if t > cutoff]
    return len(_usage[ip]) < PER_IP_DAILY and len(_usage["*"]) < GLOBAL_DAILY


def _recent(limit: int = 12) -> list[Report]:
    reports = [r for d in REPORTS_DIR.iterdir() if d.is_dir() and (r := load_report(REPORTS_DIR, d.name))]
    return sorted(reports, key=lambda r: r.analyzed_at, reverse=True)[:limit]


def _index(request: Request, error: str | None = None, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"reports": _recent(), "allow_live": ALLOW_LIVE, "error": error, "limit": PER_IP_DAILY},
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _index(request)


@app.post("/analyze")
def run_analysis(request: Request, url: str = Form(...)):
    if not ALLOW_LIVE:
        return _index(request, "Live analysis is switched off on this deployment. Browse the examples below.", 403)
    try:
        url = check_url(url)
    except FetchError as exc:
        return _index(request, str(exc), 400)
    ip = request.client.host if request.client else "unknown"
    if not _within_limits(ip):
        return _index(request, "Daily analysis limit reached. Try again tomorrow or run it locally.", 429)
    _usage[ip].append(time.time())
    _usage["*"].append(time.time())
    try:
        report = analyze(url, REPORTS_DIR)
    except FetchError as exc:
        return _index(request, f"Could not download the video: {exc}", 422)
    return RedirectResponse(f"/report/{report.slug}", status_code=303)


@app.get("/report/{slug}", response_class=HTMLResponse)
def show_report(request: Request, slug: str):
    report = load_report(REPORTS_DIR, slug)
    if report is None:
        raise HTTPException(404, "report not found")
    return templates.TemplateResponse(request, "report.html", {"r": report})


@app.get("/api/report/{slug}")
def report_json(slug: str):
    report = load_report(REPORTS_DIR, slug)
    if report is None:
        raise HTTPException(404, "report not found")
    return JSONResponse(report.model_dump(mode="json"))
