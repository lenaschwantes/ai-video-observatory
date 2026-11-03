import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_export_rewrites_links_for_the_pages_subpath(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_static

    from observatory.detect import web

    # the exporter switches the app to read-only; restore it for the other tests
    for name in ("ALLOW_LIVE", "ALLOW_BYOK", "ALLOW_UPLOAD", "STATIC_SITE", "REPORTS_DIR", "PUBLIC_URL"):
        monkeypatch.setattr(web, name, getattr(web, name))

    out = tmp_path / "site"
    monkeypatch.setattr(sys, "argv", ["export_static.py", "--base", "/repo", "--out", str(out)])
    export_static.main()
    home = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="/repo/benchmarks/"' in home and 'href="/repo/en/"' in home
    assert 'href="/repo/static/favicon.svg"' in home
    assert 'type="file"' not in home and 'name="api_key"' not in home  # no analyzer on the static copy
    assert (out / "en" / "docs" / "projeto" / "index.html").exists()
    assert (out / "404.html").exists() and (out / ".nojekyll").exists()
