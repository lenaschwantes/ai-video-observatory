"""Nothing a visitor sends can reach the front page, another host, or the disk
outside the reports folder."""

import shutil

import pytest
from fastapi.testclient import TestClient

from observatory.detect import fetch, web


@pytest.fixture
def client(tmp_path, monkeypatch):
    reports, examples = tmp_path / "reports", tmp_path / "examples"
    src = web.ROOT / "examples" / "reports" / "youtube-weizh2jfsnw"
    for d in (reports, examples):
        shutil.copytree(src, d / src.name)
    # a visitor's analysis of some other video
    shutil.copytree(src, reports / "youtube-visitor0001")
    monkeypatch.setattr(web, "REPORTS_DIR", reports)
    monkeypatch.setattr(web, "EXAMPLES_DIR", examples)
    return TestClient(web.app)


def test_home_lists_only_curated_examples(client):
    page = client.get("/").text
    assert "/report/youtube-weizh2jfsnw" in page
    assert "youtube-visitor0001" not in page


def test_visitor_reports_are_reachable_but_not_indexed(client):
    visitor = client.get("/report/youtube-visitor0001")
    assert visitor.status_code == 200 and 'name="robots" content="noindex"' in visitor.text
    assert 'content="noindex"' not in client.get("/report/youtube-weizh2jfsnw").text


@pytest.mark.parametrize("slug", ["..", "..%5Creports", "YouTube-X", "a" * 101, "x.y"])
def test_report_slugs_cannot_leave_the_reports_folder(client, slug):
    assert client.get(f"/api/report/{slug}").status_code == 404


def test_security_headers_allow_hugging_face_to_frame_the_app(client):
    h = client.get("/").headers
    assert h["x-content-type-options"] == "nosniff"
    csp = h["content-security-policy"]
    assert "frame-ancestors 'self' https://huggingface.co https://*.hf.space" in csp
    assert "object-src 'none'" in csp and "frame-src https://www.youtube-nocookie.com" in csp


def test_client_ip_trusts_only_the_proxy_hop(monkeypatch):
    from types import SimpleNamespace as NS

    req = NS(headers={"x-forwarded-for": "6.6.6.6, 203.0.113.9"}, client=NS(host="10.0.0.1"))
    monkeypatch.setattr(web, "TRUST_PROXY", False)
    assert web.client_ip(req) == "10.0.0.1"  # header ignored unless behind a proxy
    monkeypatch.setattr(web, "TRUST_PROXY", True)
    assert web.client_ip(req) == "203.0.113.9"  # the spoofable leftmost entry is ignored
    monkeypatch.setattr(web, "PROXY_HOPS", 2)  # two proxies of ours appended an entry each
    assert web.client_ip(req) == "6.6.6.6"
    monkeypatch.setattr(web, "PROXY_HOPS", 3)  # fewer entries than proxies: fall back to the socket
    assert web.client_ip(req) == "10.0.0.1"


def test_downloads_use_only_the_platform_extractors():
    import yt_dlp

    ydl = yt_dlp.YoutubeDL({"allowed_extractors": fetch.EXTRACTORS, "quiet": True})
    names = {ie.IE_NAME for ie in ydl._ies.values()}
    assert "generic" not in names and "twitter:card" not in names
    for url in ("https://www.youtube.com/shorts/abcdefghijk", "https://youtu.be/abcdefghijk",
                "https://x.com/a/status/1", "https://www.tiktok.com/@a/video/1", "https://www.instagram.com/reel/abc/"):
        assert any(ie.suitable(url) for ie in ydl._ies.values()), url


def test_failed_downloads_read_as_one_plain_sentence():
    assert web.friendly_error("ERROR: [youtube] abc: Private video. Sign in", True).startswith("A plataforma pediu login")
    assert "não está disponível" in web.friendly_error("ERROR: [youtube] abc: Video unavailable", True)
    assert "20 minutes" in web.friendly_error("video too long or live (limit 20 min)", False)
    odd = web.friendly_error("Traceback /app/src/observatory/x.py line 3", False)
    assert "/app" not in odd and odd.startswith("We could not analyze")
