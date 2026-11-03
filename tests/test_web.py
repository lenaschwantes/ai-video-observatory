import json
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from observatory.detect import web  # noqa: E402
from observatory.detect.models import Evidence, Frame, Report, Signal, Verdict, VideoInfo  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(web, "BENCHMARKS_FILE", tmp_path / "missing.json")
    web._usage.clear()
    return TestClient(web.app)


def _report(tmp_path) -> Report:
    r = Report(
        slug="youtube-x",
        video=VideoInfo(url="https://youtu.be/x", platform="youtube", video_id="x", title="Robot", duration=10),
        analyzed_at=datetime.now(timezone.utc),
        probability=0.9,
        verdict=Verdict.LIKELY_AI,
        signals=[Signal(name="visual", label="v", available=True, llr=2.0,
                        evidence=[Evidence(text="hand merges", direction="ai", timestamp=4.0, frame="frame_01.jpg")])],
        frames=[Frame(index=1, timestamp=4.0, file="frame_01.jpg")],
        method_version="v0",
    )
    (tmp_path / r.slug).mkdir()
    (tmp_path / r.slug / "report.json").write_text(r.model_dump_json())
    return r


def test_home_shows_pending_state_without_benchmark(client):
    page = client.get("/?lang=pt").text
    assert "Benchmark em andamento" in page
    assert "Helena Schwantes" in page


def test_language_switch_sets_cookie(client):
    resp = client.get("/?lang=en")
    assert "Do AI-video detectors accuse real videos?" in resp.text
    assert resp.cookies.get("lang") == "en"


def test_ranking_renders_hero_from_benchmark(client, tmp_path, monkeypatch):
    bench = tmp_path / "b.json"
    bench.write_text(json.dumps({"generated_at": "2026-11-03", "dataset": {"videos": 2}, "models": [
        {"provider": "sightengine", "model": "genai", "attempted": 2, "failures": 0,
         "scores": {"false_positive_rate": 0.5, "auc": 0.7, "recall_ai": 1.0}, "by_region": {}},
        {"provider": "anthropic", "model": "claude-opus-5-5", "attempted": 2, "failures": 0,
         "scores": {"false_positive_rate": 0.0, "auc": 1.0, "recall_ai": 1.0}, "by_region": {}},
    ]}))
    monkeypatch.setattr(web, "BENCHMARKS_FILE", bench)
    page = client.get("/?lang=en").text
    assert "of real videos were called AI by Sightengine" in page
    assert "Fewest errors: Claude, at 0%." in page
    assert client.get("/benchmarks").status_code == 200
    assert client.get("/metodo").status_code == 200


def test_report_page_shows_evidence_and_timeline(client, tmp_path):
    _report(tmp_path)
    page = client.get("/report/youtube-x?lang=en").text
    assert "Likely AI" in page and "hand merges" in page
    assert 'class="mark ai"' in page and "left:40.0%" in page


def test_htmx_rejects_unsupported_link(client):
    resp = client.post("/analyze", data={"url": "https://evil.example/v"}, headers={"HX-Request": "true"})
    assert 'class="error"' in resp.text


def test_htmx_job_reports_progress_then_redirects(client, tmp_path, monkeypatch):
    import threading

    report = _report(tmp_path)
    release = threading.Event()

    def fake_analyze(url, reports_dir, progress):
        progress("frames")
        release.wait(5)
        progress("visual")
        return report

    monkeypatch.setattr(web, "analyze", fake_analyze)
    resp = client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x"}, headers={"HX-Request": "true"})
    assert "hx-get=\"/jobs/" in resp.text
    job_id = resp.text.split('hx-get="/jobs/')[1].split('"')[0]
    assert "✓" in client.get(f"/jobs/{job_id}").text  # download step shown as done
    release.set()
    for _ in range(50):
        done = client.get(f"/jobs/{job_id}")
        if done.headers.get("HX-Redirect"):
            break
        time.sleep(0.02)
    assert done.headers["HX-Redirect"] == "/report/youtube-x"


def _run_job(client, data):
    resp = client.post("/analyze", data=data, headers={"HX-Request": "true"})
    if 'hx-get="/jobs/' not in resp.text:
        return resp, resp  # fast job: the first response is already final
    job_id = resp.text.split('hx-get="/jobs/')[1].split('"')[0]
    for _ in range(100):
        r = client.get(f"/jobs/{job_id}")
        if r.headers.get("HX-Redirect") or "stage_error" in r.text or 'class="error"' in r.text:
            return resp, r
        time.sleep(0.02)
    return resp, r


SECRET = "gsk_test_visitor_key_1234567890"


def test_visitor_key_reaches_the_provider_and_is_never_echoed(client, tmp_path, monkeypatch):
    seen = {}

    def fake_analyze(url, reports_dir, progress, assessor=None):
        try:
            assessor([], tmp_path)
        except RuntimeError as exc:
            raise RuntimeError(f"provider rejected key {SECRET}: {exc}")

    def fake_assess_with(provider, model, frames, frame_dir, api_key=None):
        seen.update(provider=provider, model=model, key=api_key)
        raise RuntimeError("401")

    monkeypatch.setattr(web, "analyze", fake_analyze)
    monkeypatch.setattr(web, "assess_with", fake_assess_with)
    _, status = _run_job(client, {"url": "https://www.youtube.com/watch?v=x", "key_provider": "groq",
                                  "key_model": "", "api_key": SECRET})
    assert seen == {"provider": "groq", "model": "qwen/qwen3.8-27b", "key": SECRET}
    assert SECRET not in status.text and "***" in status.text
    assert all(SECRET not in str(job) for job in web._jobs.values())


def test_visitor_key_allows_analysis_when_live_is_off(client, monkeypatch, tmp_path):
    report = _report(tmp_path)
    monkeypatch.setattr(web, "ALLOW_LIVE", False)
    monkeypatch.setattr(web, "analyze", lambda url, reports_dir, progress, assessor=None: report)
    blocked = client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x"}, headers={"HX-Request": "true"})
    assert 'class="error"' in blocked.text
    _, done = _run_job(client, {"url": "https://www.youtube.com/watch?v=x", "key_provider": "groq", "api_key": SECRET})
    assert done.headers["HX-Redirect"] == "/report/youtube-x"


@pytest.mark.parametrize("data", [
    {"key_provider": "nope", "api_key": SECRET},
    {"key_provider": "groq", "api_key": "short"},
    {"key_provider": "openai", "api_key": SECRET},  # OpenAI has no default model
])
def test_invalid_visitor_key_input_is_rejected(client, data):
    resp = client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x", **data}, headers={"HX-Request": "true"})
    assert 'class="error"' in resp.text and SECRET not in resp.text


def test_upload_runs_analysis_and_deletes_the_file(client, tmp_path, monkeypatch):
    report = _report(tmp_path)
    seen = {}

    def fake_analyze_file(path, reports_dir, **kwargs):
        seen["path"] = path
        assert path.read_bytes() == b"fake video"
        return report

    monkeypatch.setattr(web, "analyze_file", fake_analyze_file)
    resp = client.post("/analyze", files={"file": ("clip.mp4", b"fake video", "video/mp4")}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/report/youtube-x"
    assert not seen["path"].exists()  # upload removed after analysis


def test_upload_rejects_other_file_types(client):
    resp = client.post("/analyze", files={"file": ("notes.txt", b"hi", "text/plain")},
                       headers={"HX-Request": "true"})
    assert 'class="error"' in resp.text


def test_no_link_and_no_file_is_rejected(client):
    resp = client.post("/analyze", data={"url": ""}, headers={"HX-Request": "true"})
    assert 'class="error"' in resp.text


def test_benchmarks_page_shows_recompression_table(client, tmp_path, monkeypatch):
    from observatory.bench.metrics import score

    s = score([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]).as_dict()
    rob = {"provider": "groq", "model": "m", "variant": "720p-crf35", "n": 4, "ssim": 0.91, "bitrate_ratio": 0.38,
           "fpr_ref": 0.0, "fpr": 0.5, "fpr_delta_ci": [0.0, 1.0], "flip_rate": 0.25, "noise_flip_rate": 0.0,
           "toward_ai": 1, "toward_real": 0, "mcnemar_p": 1.0}
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"models": [{"provider": "groq", "model": "m", "scores": s, "attempted": 4, "failures": 0}],
                                "robustness": [rob], "leakage": {"height": {"auc": 0.81, "n": 4}}}))
    monkeypatch.setattr(web, "BENCHMARKS_FILE", path)
    page = client.get("/benchmarks?lang=pt").text
    assert "E depois que a plataforma recomprime?" in page
    assert "720p-crf35" in page and "SSIM 0.91" in page and "0% → 50%" in page
    assert "AUC 0.81" in page


def test_byok_form_explains_free_keys_and_that_the_model_matters(client):
    page = client.get("/?lang=pt").text
    assert "https://console.groq.com/keys" in page and "https://aistudio.google.com/apikey" in page
    assert "O modelo muda o resultado" in page and 'href="/benchmarks"' in page
    assert "The model changes the result" in client.get("/?lang=en").text
