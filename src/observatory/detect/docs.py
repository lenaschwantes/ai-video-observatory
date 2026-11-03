"""Project documentation rendered inside the site, from the Markdown files in
the repository, so readers never have to leave for GitHub.

Only the files listed in PAGES can be shown (no path from the URL ever
reaches the filesystem), and raw HTML inside the Markdown is not rendered."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(os.environ.get("DOCS_ROOT", Path(__file__).resolve().parents[3]))
REPO_BLOB = "https://github.com/lenaschwantes/ai-video-observatory/blob/main/"


@dataclass(frozen=True)
class DocPage:
    slug: str
    path: str  # relative to ROOT
    title_pt: str
    title_en: str


PAGES = {
    "projeto": DocPage("projeto", "README.md", "Sobre o projeto", "About the project"),
    "benchmarks": DocPage("benchmarks", "docs/benchmarks.md", "Relatório do benchmark", "Benchmark report"),
}
# Repository files linked from the docs that have a page of their own here.
INTERNAL = {page.path: f"/docs/{page.slug}" for page in PAGES.values()}

_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable("table")


def _slug(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text).strip("-")


def _rewrite(href: str) -> str:
    """Links between docs stay on the site; other repository files open on GitHub."""
    if href.startswith(("http://", "https://", "#", "/", "mailto:")):
        return href
    path, _, anchor = href.partition("#")
    if path in INTERNAL:
        return INTERNAL[path] + (f"#{anchor}" if anchor else "")
    return REPO_BLOB + href


def render(page: DocPage) -> tuple[str, list[dict]] | None:
    """HTML for the page and its table of contents, or None if the file
    does not exist yet (e.g. the benchmark report before the first run)."""
    source = ROOT / page.path
    if not source.exists():
        return None
    tokens = _md.parse(source.read_text(encoding="utf-8"))
    toc, seen = [], set()
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3"):
            text = tokens[i + 1].content
            anchor = base = _slug(text) or "section"
            n = 2
            while anchor in seen:
                anchor, n = f"{base}-{n}", n + 1
            seen.add(anchor)
            tok.attrSet("id", anchor)
            toc.append({"level": int(tok.tag[1]), "text": text, "id": anchor})
        if tok.type == "inline" and tok.children:
            for child in tok.children:
                if child.type == "link_open":
                    child.attrSet("href", _rewrite(child.attrGet("href") or ""))
                    if (child.attrGet("href") or "").startswith("http"):
                        child.attrSet("rel", "noopener noreferrer")
    html = _md.renderer.render(tokens, _md.options, {})
    # wide tables scroll inside their own box on small screens
    html = html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return html, toc
