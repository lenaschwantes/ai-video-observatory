"""Export the read-only part of the site (ranking, method, docs, example
reports) as static HTML for GitHub Pages.

The pages are rendered by the real app, with live analysis, visitor keys and
uploads switched off, so the static copy never drifts from the server. Links
are rewritten for the Pages sub-path (e.g. /ai-video-observatory/) and for the
language: Portuguese at the root, English under /en/.

Run: python scripts/export_static.py --base /ai-video-observatory \
         --public-url https://lenaschwantes.github.io/ai-video-observatory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["/", "/benchmarks", "/metodo", "/docs/projeto", "/docs/benchmarks"]
LANGS = {"pt": "", "en": "/en"}
ATTR = re.compile(r'(href|src)="(/[^"/][^"]*|/)"')


def page_dir(path: str) -> str:
    """'/benchmarks' -> 'benchmarks/', '/' -> ''."""
    return path.strip("/") + "/" if path.strip("/") else ""


def rewrite(html: str, base: str, lang_prefix: str, current: str) -> str:
    def fix(m: re.Match) -> str:
        attr, url = m.group(1), m.group(2)
        path, _, anchor = url.partition("#")
        if path.startswith(("/static/", "/media/")):
            target = base + path
        elif path.startswith("/api/report/") or path == "/api/benchmarks":
            target = base + path + ".json"
        else:
            target = base + lang_prefix + "/" + page_dir(path.split("?")[0])
        return f'{attr}="{target}{"#" + anchor if anchor else ""}"'

    html = ATTR.sub(fix, html)
    # the language chips link to the same page in the other language
    for lang, prefix in LANGS.items():
        html = html.replace(f'href="?lang={lang}"', f'href="{base}{prefix}/{page_dir(current)}"')
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="URL sub-path the site is served from, e.g. /ai-video-observatory")
    ap.add_argument("--public-url", default="", help="absolute URL of the site, for link previews")
    ap.add_argument("--out", default=str(ROOT / "site"))
    args = ap.parse_args()
    base = args.base.rstrip("/")
    out = Path(args.out)

    # Read-only app: no live analysis, no visitor keys, no uploads, and only the
    # bundled example reports (never the local data/reports).
    reports = Path(tempfile.mkdtemp(prefix="static-reports-"))
    os.environ.update({"ALLOW_LIVE": "0", "ALLOW_BYOK": "0", "ALLOW_UPLOAD": "0", "STATIC_SITE": "1",
                       "REPORTS_DIR": str(reports), "PUBLIC_URL": args.public_url})
    sys.path.insert(0, str(ROOT / "src"))
    from fastapi.testclient import TestClient

    from observatory.detect import web

    # The module may already be imported (e.g. in tests), so set the switches
    # on it as well as in the environment.
    for name, value in {"ALLOW_LIVE": False, "ALLOW_BYOK": False, "ALLOW_UPLOAD": False,
                        "STATIC_SITE": True, "REPORTS_DIR": reports, "PUBLIC_URL": args.public_url}.items():
        setattr(web, name, value)
    client = TestClient(web.app)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    slugs = [d.name for d in reports.iterdir() if (d / "report.json").exists()]
    pages = PAGES + [f"/report/{s}" for s in slugs]
    for lang, prefix in LANGS.items():
        for path in pages:
            resp = client.get(path, params={"lang": lang})
            resp.raise_for_status()
            target = out / prefix.strip("/") / page_dir(path) / "index.html"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rewrite(resp.text, base, prefix, path), encoding="utf-8")
            print("wrote", target.relative_to(out))
    not_found = client.get("/__missing__", params={"lang": "pt"})
    (out / "404.html").write_text(rewrite(not_found.text, base, "", "/"), encoding="utf-8")

    shutil.copytree(ROOT / "src" / "observatory" / "detect" / "static", out / "static")
    if slugs:
        shutil.copytree(reports, out / "media", ignore=shutil.ignore_patterns("report.json"))
        for s in slugs:
            api = out / "api" / "report" / f"{s}.json"
            api.parent.mkdir(parents=True, exist_ok=True)
            api.write_text(client.get(f"/api/report/{s}").text, encoding="utf-8")
    bench = client.get("/api/benchmarks")
    if bench.status_code == 200:
        (out / "api").mkdir(exist_ok=True)
        (out / "api" / "benchmarks.json").write_text(json.dumps(bench.json()), encoding="utf-8")
    (out / ".nojekyll").write_text("")  # serve folders starting with _ as-is
    print(f"static site in {out} ({len(pages) * len(LANGS)} pages)")


if __name__ == "__main__":
    main()
