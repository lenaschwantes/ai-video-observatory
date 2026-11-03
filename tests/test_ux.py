"""The analyzer's run modes, the key check and the Kaggle notebook."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from observatory.detect import web
from observatory.detect.models import Evidence, Frame, Report, Signal, Verdict, VideoInfo

ROOT = Path(__file__).resolve().parents[1]
KEY = "gsk_visitor_key_for_tests_0123456789"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(web, "BENCHMARKS_FILE", tmp_path / "none.json")
    web._usage.clear()
    return TestClient(web.app)


def test_key_check_lists_models_and_never_echoes_the_key(client, monkeypatch):
    seen = {}

    def fake(provider, key):
        seen.update(provider=provider, key=key)
        return ["qwen/qwen3.8-27b"]

    monkeypatch.setattr(web, "list_vision_models", fake)
    r = client.post("/api/keys/check", data={"provider": "groq", "api_key": KEY})
    assert r.json() == {"ok": True, "models": ["qwen/qwen3.8-27b"], "default": "qwen/qwen3.8-27b"}
    assert seen == {"provider": "groq", "key": KEY} and KEY not in r.text


def test_refused_key_gets_a_plain_message_without_the_key(client, monkeypatch):
    class Refused(Exception):
        status_code = 401

    monkeypatch.setattr(web, "list_vision_models", lambda p, k: (_ for _ in ()).throw(Refused(f"bad key {k}")))
    r = client.post("/api/keys/check?lang=pt", data={"provider": "groq", "api_key": KEY})
    assert r.json()["ok"] is False and "recusou" in r.json()["error"] and KEY not in r.text


def test_key_check_validates_input_and_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(web, "list_vision_models", lambda p, k: ["m"])
    assert client.post("/api/keys/check", data={"provider": "nope", "api_key": KEY}).status_code == 400
    assert client.post("/api/keys/check", data={"provider": "groq", "api_key": "short"}).status_code == 400
    monkeypatch.setattr(web, "KEYCHECK_PER_IP_DAILY", 1)
    assert client.post("/api/keys/check", data={"provider": "groq", "api_key": KEY}).json()["ok"]
    assert client.post("/api/keys/check", data={"provider": "groq", "api_key": KEY}).status_code == 429


def test_analyzer_asks_how_to_run_first(client, monkeypatch):
    monkeypatch.setattr(web, "host_provider", lambda: None)
    page = client.get("/?lang=pt").text
    assert "Como você quer rodar?" in page and 'data-mode="key"' in page and 'data-mode="kaggle"' in page
    assert 'data-mode="demo"' not in page  # no host key: no demo option
    assert 'id="key-check"' in page and "kernels/welcome?src=" in page
    assert "Primeiro vídeo do YouTube (2005)" in page  # one-click examples


def test_demo_mode_appears_only_with_a_host_key(client, monkeypatch):
    monkeypatch.setattr(web, "host_provider", lambda: "groq")
    page = client.get("/?lang=en").text
    assert 'data-mode="demo"' in page and "Site demo" in page


def test_kaggle_notebook_installs_from_the_repo_and_reads_secrets():
    nb = json.loads((ROOT / "notebooks" / "analyze_on_kaggle.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "git+https://github.com/lenaschwantes/ai-video-observatory" in src
    assert "UserSecretsClient" in src and "print(os.environ" not in src  # keys are never printed
    assert nb["nbformat"] == 4


def test_kaggle_notebook_report_cell_renders_a_report(tmp_path):
    nb = json.loads((ROOT / "notebooks" / "analyze_on_kaggle.ipynb").read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][-1]["source"])
    report = Report(
        slug="youtube-x", video=VideoInfo(url="u", platform="youtube", video_id="x", title="<b>t</b>"),
        analyzed_at=datetime.now(timezone.utc), probability=0.9, verdict=Verdict.LIKELY_AI,
        signals=[Signal(name="visual", label="Visual", available=True, llr=2,
                        evidence=[Evidence(text="hand <merges>", direction="ai", timestamp=1.0, frame="f.jpg")])],
        frames=[Frame(index=0, timestamp=1.0, file="f.jpg")], method_version="v0",
    )
    shown = []
    ns = {"report": report, "REPORTS": tmp_path}
    import IPython.display as disp

    orig = disp.display
    disp.display = lambda obj: shown.append(obj.data)
    try:
        exec(cell, ns)
    finally:
        disp.display = orig
    html = shown[0]
    assert "90%" in html and "&lt;merges&gt;" in html and "&lt;b&gt;t&lt;/b&gt;" in html  # escaped
