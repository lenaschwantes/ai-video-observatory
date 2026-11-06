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


def test_home_puts_the_check_first_in_plain_language(client, monkeypatch):
    monkeypatch.setattr(web, "host_provider", lambda: "groq")
    page = client.get("/?lang=pt").text
    assert page.index("Esse vídeo foi feito com IA?") < page.index('id="video-url"') < page.index('class="example"') < page.index("<h2>Como funciona")
    # the hero shows what a result looks like, with real clues, before anyone types
    assert "Exemplo de resultado" in page and "84%" in page and "texto no relógio" in page
    assert page.count("<svg") >= 3
    assert "Verificar vídeo" in page and "Primeiro vídeo do YouTube (2005)" in page
    # technical paths stay available, folded away
    assert '<details class="advanced"' in page and 'id="key-check"' in page and "kernels/welcome?src=" in page
    for jargon in ("AUC", "C2PA", "log-odds", "Como você quer rodar?"):
        assert jargon not in page.split("Opções avançadas")[0]


def test_without_a_host_key_the_home_says_so_and_offers_a_free_key(client, monkeypatch):
    monkeypatch.setattr(web, "host_provider", lambda: None)
    page = client.get("/?lang=en").text
    assert 'id="partial-note"' in page and "Connect a free key" in page
    monkeypatch.setattr(web, "host_provider", lambda: "groq")
    assert 'id="partial-note"' not in client.get("/?lang=en").text


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
    # The notebook runs in Jupyter; here a stand-in IPython.display captures what it shows.
    import sys
    import types

    shown = []
    fake = types.ModuleType("IPython.display")
    fake.HTML = lambda data: data
    fake.display = shown.append
    ipython = types.ModuleType("IPython")
    ipython.display = fake
    saved = {k: sys.modules.get(k) for k in ("IPython", "IPython.display")}
    sys.modules.update({"IPython": ipython, "IPython.display": fake})
    try:
        exec(cell, {"report": report, "REPORTS": tmp_path})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    html = shown[0]
    assert "90%" in html and "&lt;merges&gt;" in html and "&lt;b&gt;t&lt;/b&gt;" in html  # escaped


def test_report_embeds_only_valid_youtube_ids():
    from types import SimpleNamespace as NS
    assert web.youtube_embed_id(NS(platform="youtube", video_id="WeIzH2jFsNw")) == "WeIzH2jFsNw"
    assert web.youtube_embed_id(NS(platform="tiktok", video_id="WeIzH2jFsNw")) is None
    for bad in ('x" onload="alert(1)', "short", "WeIzH2jFsNw?autoplay=1", ""):
        assert web.youtube_embed_id(NS(platform="youtube", video_id=bad)) is None


def test_static_site_hands_the_link_to_the_hosted_analyzer(client, monkeypatch):
    monkeypatch.setattr(web, "STATIC_SITE", True)
    monkeypatch.setattr(web, "ANALYZER_URL", "https://someone-ai-video-observatory.hf.space")
    page = client.get("/?lang=pt").text
    assert 'action="https://someone-ai-video-observatory.hf.space/"' in page and 'method="get"' in page
    assert 'name="url"' in page and 'id="video-url"' in page


def test_hosted_analyzer_prefills_and_starts_with_a_handed_link(client, monkeypatch):
    monkeypatch.setattr(web, "host_provider", lambda: "groq")
    page = client.get("/?url=https://www.youtube.com/watch?v=WeIzH2jFsNw").text
    assert 'value="https://www.youtube.com/watch?v=WeIzH2jFsNw" data-autostart' in page
    assert "data-autostart" not in client.get("/?url=javascript:alert(1)").text.split("<script")[0]
