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
