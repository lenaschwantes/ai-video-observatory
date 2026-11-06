"""Web front end: the detector ranking, the single-video analyzer and the
method page, in Portuguese (default) or English.

Public deployments pay for every model call, so live analysis is rate limited
per visitor and per day, and can be switched off entirely (ALLOW_LIVE=0) to
serve only the bundled example reports."""

from __future__ import annotations

import csv
from functools import partial
import html
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from observatory.detect import docs, i18n
from observatory.detect.fetch import MAX_BYTES, VIDEO_SUFFIXES, FetchError, check_url
from observatory.detect.models import Report
from observatory.detect.pipeline import AI_THRESHOLD, HUMAN_THRESHOLD, analyze, analyze_file, load_report
from observatory.detect.vision import PROVIDERS, assess, assess_with, list_vision_models, provider as host_provider

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "data/reports"))
EXAMPLES_DIR = Path(os.environ.get("EXAMPLES_DIR", ROOT / "examples" / "reports"))
BENCHMARKS_FILE = Path(os.environ.get("BENCHMARKS_FILE", ROOT / "docs" / "benchmarks.json"))
DATASET_FILE = Path(os.environ.get("DATASET_FILE", ROOT / "eval" / "dataset.csv"))
ALLOW_LIVE = os.environ.get("ALLOW_LIVE", "1") != "0"
PER_IP_DAILY = int(os.environ.get("LIVE_LIMIT_PER_IP", "3"))
GLOBAL_DAILY = int(os.environ.get("LIVE_LIMIT_PER_DAY", "30"))
ALLOW_BYOK = os.environ.get("ALLOW_BYOK", "1") != "0"
# Uploads put arbitrary third-party video on the server: on by default for local
# use, off in the Docker image (ALLOW_UPLOAD=0) so a public site does not host them.
ALLOW_UPLOAD = os.environ.get("ALLOW_UPLOAD", "1") != "0"
# "Open in Kaggle": Kaggle imports a notebook straight from a public GitHub URL,
# so the button works before anyone publishes it there. Override once published.
KAGGLE_URL = os.environ.get(
    "KAGGLE_NOTEBOOK_URL",
    "https://www.kaggle.com/kernels/welcome?src=https://github.com/lenaschwantes/ai-video-observatory/blob/main/notebooks/analyze_on_kaggle.ipynb",
)
# The hero shows what a result looks like before anyone pastes a link: a real
# analysis (a Veo 3 clip, 84%), summarized. The thumbnail is YouTube's own.
HERO_EXAMPLE = {
    # Taken from the stored report of this video (data/reports/youtube-weizh2jfsnw);
    # the clues paraphrase what the vision model actually wrote for those frames.
    "slug": "youtube-weizh2jfsnw",
    "url": "https://www.youtube.com/watch?v=WeIzH2jFsNw",
    "thumb": "https://i.ytimg.com/vi/WeIzH2jFsNw/hqdefault.jpg",
    "frame": "frame_06.jpg",
    "probability": 0.84,
    "verdict": "likely_ai",
    "title": {"pt": "Clipe feito com o Veo 3", "en": "Clip made with Veo 3"},
    "clues": [
        ("ai", 21.5, {"pt": "O texto no relógio sai levemente amassado e os ponteiros parecem de plástico",
                      "en": "The text on the clock is slightly squashed and the hands look like plastic"}),
        ("ai", 25.0, {"pt": "O celular tem um brilho plástico, não de tinta de verdade",
                      "en": "The phone has a plastic sheen rather than real paint"}),
    ],
}
# Where the static site (GitHub Pages) sends people to check a video: a hosted
# copy of this app with its own key (e.g. a Hugging Face Space). Empty means
# no hosted analyzer yet, and the static site offers Kaggle instead.
ANALYZER_URL = os.environ.get("ANALYZER_URL", "")
# One-click examples: one generated clip and one real recording from before 2022.
EXAMPLES = [
    {"url": "https://www.youtube.com/watch?v=WeIzH2jFsNw", "pt": "Clipe feito com Veo 3", "en": "Clip made with Veo 3"},
    {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "pt": "Primeiro vídeo do YouTube (2005)", "en": "First YouTube video (2005)"},
]
# Set by scripts/export_static.py: the page is a read-only copy (GitHub Pages).
STATIC_SITE = os.environ.get("STATIC_SITE") == "1"
BYOK_PER_IP_DAILY = int(os.environ.get("BYOK_LIMIT_PER_IP", "20"))
REPO_URL = "https://github.com/lenaschwantes/ai-video-observatory"
# Absolute URL for link previews (og:url, og:image); falls back to the request host.
PUBLIC_URL = os.environ.get("PUBLIC_URL", "")

DISPLAY = {
    "groq": ("Qwen (Groq)", "vision"),
    "anthropic": ("Claude", "vision"),
    "openai": ("GPT", "vision"),
    "gemini": ("Gemini", "vision"),
    "sightengine": ("Sightengine", "detector"),
    "hive": ("Hive", "detector"),
}

app = FastAPI(title="AI Video Observatory", docs_url="/api/docs", redoc_url=None)  # /docs is the project documentation
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
        {"lang": lang, "t": t, "note": lambda text: i18n.note(text, lang),
         "ev": lambda text: i18n.evidence(text, lang), "path": request.url.path,
         "base_url": (PUBLIC_URL or str(request.base_url)).rstrip("/"), **context},
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
    """Public list on the home page. Uploaded files are private to whoever
    holds the link, so they are never listed."""
    reports = [r for d in REPORTS_DIR.iterdir() if d.is_dir() and (r := load_report(REPORTS_DIR, d.name))]
    reports = [r for r in reports if r.video.platform != "upload"]
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
            "fpr_ci": s.get("false_positive_rate_ci"),
            # .get: a benchmarks.json written by an older version lacks the counts
            "fpr_count": (s["false_positives"], s["n_real"]) if s.get("n_real") and "false_positives" in s else None,
            "real_flagged": s.get("false_positives"),
            "n_real": s.get("n_real"),
            "auc": s.get("auc"),
            "auc_ci": s.get("auc_ci"),
            "recall_zero_fp": s.get("recall_at_zero_fpr"),
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
        "robustness": [
            {**r, "name": DISPLAY.get(r["provider"], (r["provider"], "vision"))[0]}
            for r in data.get("robustness") or []
        ],
        "leakage": data.get("leakage") or {},
    }


def _home(request: Request, error: str | None = None, status_code: int = 200) -> HTMLResponse:
    return render(
        request,
        "index.html",
        {"b": benchmark_view(), "reports": _recent(), "allow_live": ALLOW_LIVE, "allow_byok": ALLOW_BYOK, "allow_upload": ALLOW_UPLOAD, "static_site": STATIC_SITE,
         "host_vision": bool(host_provider()) and ALLOW_LIVE, "kaggle_url": KAGGLE_URL, "examples": EXAMPLES,
         "analyzer_url": ANALYZER_URL, "hero_example": HERO_EXAMPLE,
         "hero_example_local": (REPORTS_DIR / HERO_EXAMPLE["slug"] / "report.json").exists(),
         "prefill_url": _prefill(request),
         "error": error, "limit": PER_IP_DAILY, "providers": PROVIDERS},
        status_code,
    )


def _prefill(request: Request) -> str:
    """A link handed over by the static site (?url=...), so the visitor never retypes it."""
    url = (request.query_params.get("url") or "").strip()[:500]
    return url if url.startswith(("https://", "http://")) else ""


# --- routes ------------------------------------------------------------------


MAX_REQUEST_BYTES = MAX_BYTES + 1024 * 1024  # largest upload plus form fields


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Refuse oversized bodies from the header, before Starlette spools them to
    disk: the per-chunk check in _save_upload only runs after the whole upload
    has already been received."""
    length = request.headers.get("content-length")
    if request.method == "POST" and length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def not_found(request: Request, exc: StarletteHTTPException):
    """Friendly 404 page for people; JSON stays JSON for the API routes."""
    if exc.status_code == 404 and not request.url.path.startswith(("/api/", "/jobs/", "/media/", "/static/")):
        return render(request, "404.html", {}, status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _home(request)


STAGES = ["download", "frames", "signals", "visual"]
_jobs: dict[str, dict] = {}
_jobs_guard = threading.Lock()
# A video analysis holds a download, OpenCV and a model call: cap how many run
# at once so a burst of visitors cannot exhaust CPU, disk or memory.
MAX_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
_job_slots = threading.BoundedSemaphore(MAX_JOBS)
JOB_TTL = 3600  # finished jobs are forgotten after an hour


def _prune_jobs(now: float) -> None:
    with _jobs_guard:
        for job_id in [j for j, job in _jobs.items() if job.get("ended") and now - job["ended"] > JOB_TTL]:
            del _jobs[job_id]


class BYOK:
    """A visitor's own vision key. Lives only in this object for one analysis:
    never written to disk, logged, or echoed back (errors are scrubbed)."""

    def __init__(self, provider: str, model: str, key: str):
        self.provider, self.model, self._key = provider, model, key

    def assessor(self, language: str = "en"):
        def run(frames, frame_dir):
            result, _ = assess_with(self.provider, self.model, frames, frame_dir, api_key=self._key, language=language)
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


def _save_upload(upload: UploadFile) -> Path:
    """Stream an upload to a private temp dir, enforcing type and size."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise FetchError(f"unsupported file type; use {', '.join(VIDEO_SUFFIXES)}")
    target = Path(tempfile.mkdtemp(prefix="observatory-upload-")) / f"upload{suffix}"
    size = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                out.close()
                shutil.rmtree(target.parent, ignore_errors=True)
                raise FetchError(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
            out.write(chunk)
    if size == 0:
        shutil.rmtree(target.parent, ignore_errors=True)
        raise FetchError("empty file")
    return target


def _run(url: str, upload: Path | None, **kwargs) -> Report:
    """Analyze a link or a saved upload; an upload is deleted afterwards."""
    if upload is None:
        return analyze(url, REPORTS_DIR, **kwargs)
    try:
        return analyze_file(upload, REPORTS_DIR, **kwargs)
    finally:
        shutil.rmtree(upload.parent, ignore_errors=True)


def _assessor_for(byok: BYOK | None, language: str):
    """The visual check in the visitor's language, on their key if they gave one."""
    return byok.assessor(language) if byok else partial(assess, language=language)


def _start_job(url: str, byok: BYOK | None = None, upload: Path | None = None, language: str = "en") -> str:
    _prune_jobs(time.time())
    job_id = uuid.uuid4().hex[:12]
    job = {"stage": "download", "slug": None, "error": None, "ended": None}
    with _jobs_guard:
        _jobs[job_id] = job
    kwargs = {"assessor": _assessor_for(byok, language)}

    def work():
        try:
            report = _run(url, upload, progress=lambda s: job.update(stage=s), **kwargs)
            job.update(stage="done", slug=report.slug)
        except Exception as exc:  # shown to the visitor, so keep it short
            msg = str(exc)
            job.update(stage="error", error=(byok.scrub(msg) if byok else msg)[:200])
        finally:
            job["ended"] = time.time()
            _job_slots.release()

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
    url: str = Form(""),
    file: UploadFile | None = File(None),
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
    has_file = file is not None and bool(file.filename)
    if has_file and not ALLOW_UPLOAD:
        return fail("O envio de arquivos está desligado neste site; use um link." if pt
                    else "File uploads are off on this site; use a link.", 403)
    if not has_file:
        try:
            url = check_url(url)
        except FetchError:
            return fail("Só links públicos do YouTube, TikTok, Instagram ou X, ou um arquivo de vídeo." if pt
                        else "Only public YouTube, TikTok, Instagram or X links, or a video file.", 400)
    ip = request.client.host if request.client else "unknown"
    bucket = f"byok:{ip}" if byok else ip
    if not _within_limits(bucket, BYOK_PER_IP_DAILY if byok else PER_IP_DAILY, count_global=not byok):
        return fail("Limite diário atingido. Tente amanhã ou rode localmente." if pt else "Daily limit reached. Try tomorrow or run it locally.", 429)
    _usage[bucket].append(time.time())
    if not byok:
        _usage["*"].append(time.time())

    upload = None
    if has_file:
        try:
            upload = _save_upload(file)
        except FetchError as exc:
            return fail(("Arquivo recusado: " if pt else "File rejected: ") + str(exc), 400)
    if not _job_slots.acquire(blocking=False):
        if upload is not None:
            shutil.rmtree(upload.parent, ignore_errors=True)
        return fail("O servidor está ocupado com outras análises. Tente de novo em um minuto." if pt
                    else "The server is busy with other analyses. Try again in a minute.", 503)
    if htmx:
        return _progress(request, _start_job(url, byok, upload, _lang(request)))
    try:
        report = _run(url, upload, assessor=_assessor_for(byok, _lang(request)))
    except FetchError as exc:
        msg = str(exc)
        return fail(("Não foi possível baixar o vídeo: " if pt else "Could not download the video: ") + (byok.scrub(msg) if byok else msg)[:200], 422)
    finally:
        _job_slots.release()
    return RedirectResponse(f"/report/{report.slug}", status_code=303)


KEYCHECK_PER_IP_DAILY = 30


@app.post("/api/keys/check")
def check_key(request: Request, provider: str = Form(""), api_key: str = Form("")):
    """Validate a visitor's key before any video is downloaded and list the
    vision models it can use. The key is used for this one call and never
    returned, logged or stored."""
    pt = _lang(request) == "pt"
    key = (api_key or "").strip()
    if provider not in PROVIDERS:
        return JSONResponse({"ok": False, "error": "Escolha um provedor." if pt else "Pick a provider."}, status_code=400)
    if not 10 <= len(key) <= 300 or any(c.isspace() for c in key):
        return JSONResponse({"ok": False, "error": "Essa chave não parece válida." if pt else "That key does not look valid."},
                            status_code=400)
    ip = request.client.host if request.client else "unknown"
    bucket = f"keycheck:{ip}"
    if not _within_limits(bucket, KEYCHECK_PER_IP_DAILY, count_global=False):
        return JSONResponse({"ok": False, "error": "Limite de testes atingido hoje." if pt else "Too many key checks today."},
                            status_code=429)
    _usage[bucket].append(time.time())
    try:
        models = list_vision_models(provider, key)
    except Exception as exc:  # each SDK raises its own types; the message is scrubbed below
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        refused = status in (401, 403) or any(w in str(exc).lower() for w in ("auth", "invalid api key", "api key not valid", "permission"))
        msg = ("O provedor recusou a chave." if pt else "The provider refused the key.") if refused else (
            ("Não foi possível falar com o provedor: " if pt else "Could not reach the provider: ") + str(exc).replace(key, "***")[:160])
        return JSONResponse({"ok": False, "error": msg}, status_code=200)
    if not models:
        return JSONResponse({"ok": False, "error": ("A chave funciona, mas não achamos modelos de visão nela." if pt
                                                    else "The key works, but no vision models were found for it.")})
    return JSONResponse({"ok": True, "models": models, "default": models[0]})


@app.get("/benchmarks", response_class=HTMLResponse)
def benchmarks(request: Request):
    return render(request, "benchmarks.html", {"b": benchmark_view()})


@app.get("/metodo", response_class=HTMLResponse)
@app.get("/method", response_class=HTMLResponse)
def method(request: Request):
    return render(request, "method.html", {"b": benchmark_view()})


@app.get("/docs", response_class=HTMLResponse)
def docs_index():
    return RedirectResponse("/docs/projeto", status_code=307)


@app.get("/docs/{slug}", response_class=HTMLResponse)
def docs_page(request: Request, slug: str):
    page = docs.PAGES.get(slug)
    if page is None:
        raise HTTPException(404, "doc not found")
    pt = _lang(request) == "pt"
    rendered = docs.render(page)
    html_body, toc = rendered if rendered else (None, [])
    return render(request, "docs.html", {
        "pages": list(docs.PAGES.values()),
        "current": slug,
        "page_title": page.title_pt if pt else page.title_en,
        "html": html_body,
        "toc": toc,
        "docs_lang_note": "A documentação técnica é escrita em inglês, como o código.",
        "missing": ("Este relatório é gerado na primeira rodada do benchmark e ainda não existe." if pt
                    else "This report is generated by the first benchmark run and does not exist yet."),
    })


@app.get("/report/{slug}", response_class=HTMLResponse)
def show_report(request: Request, slug: str):
    report = load_report(REPORTS_DIR, slug)
    if report is None:
        raise HTTPException(404, "report not found")
    duration = report.video.duration or max([e.timestamp or 0 for s in report.signals for e in s.evidence] + [1])
    lang = _lang(request)
    # One mark per moment: clues at the same timestamp are grouped so dots never stack.
    grouped: dict[float, list] = defaultdict(list)
    for s in report.signals:
        for e in s.evidence:
            if e.timestamp is not None:
                grouped[round(e.timestamp, 1)].append(e)
    marks = []
    for ts, items in sorted(grouped.items()):
        dirs = {e.direction.value for e in items}
        marks.append({
            "t": ts,
            "pos": min(ts / duration, 1.0),
            "dir": "ai" if dirs == {"ai"} else "human" if dirs == {"human"} else ("mixed" if len(dirs) > 1 else dirs.pop()),
            "n": len(items),
            "texts": [i18n.evidence(e.text, lang) for e in items],
        })
    return render(request, "report.html", {"r": report, "marks": marks, "duration": duration,
                                           "embed_id": youtube_embed_id(report.video)})


_YT_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def youtube_embed_id(video) -> str | None:
    """The id to embed the original video next to the verdict, YouTube only.

    Validated so a stored report can never inject anything into the iframe URL."""
    vid = getattr(video, "video_id", "") or ""
    if getattr(video, "platform", "") == "youtube" and _YT_ID.fullmatch(vid):
        return vid
    return None


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
