"""Publish the web analyzer to a Hugging Face Space (Docker SDK).

The Space runs the same Dockerfile as everywhere else, so anyone can paste a
link and get a verdict without a key of their own. The host key is NOT sent by
this script: add it yourself in the Space settings (Settings → Variables and
secrets → New secret, name GROQ_API_KEY). Keys never leave your machine here.

    pip install huggingface_hub
    hf auth login                       # once, with a write token
    python scripts/deploy_space.py --space <your-user>/ai-video-observatory

Then set the repository variable ANALYZER_URL to the Space's public URL
(https://<your-user>-ai-video-observatory.hf.space) so the GitHub Pages site
sends visitors straight to it.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What the image needs, and nothing else (no local data, no .env, no venv).
ALLOW = [
    "Dockerfile", "pyproject.toml", "LICENSE",
    "src/**", "examples/**", "docs/**", "eval/dataset.csv", "README*.md",
]
IGNORE = ["**/__pycache__/**", "**/*.pyc", ".env", "**/.env", "data/**", ".venv/**"]

SPACE_CARD = """---
title: AI Video Observatory
emoji: 🔎
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Was this video made with AI? Paste a link and see why.
---

Live copy of [ai-video-observatory](https://github.com/lenaschwantes/ai-video-observatory)
by Helena Schwantes. Paste a public video link and get a probability, a verdict and the
frames that led to it. The video is deleted right after the analysis.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<user>/<name> of the Space")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    files = sorted({p for pat in ALLOW for p in ROOT.glob(pat) if p.is_file()})
    files = [p for p in files if not any(p.match(i) for i in IGNORE) and "__pycache__" not in p.parts]
    if args.dry_run:
        for p in files:
            print(p.relative_to(ROOT).as_posix())
        print(f"{len(files)} files")
        return

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(SPACE_CARD, encoding="utf-8")
        # the repo README stays on GitHub; the Space needs its own card with the frontmatter
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(ROOT),
                          allow_patterns=ALLOW, ignore_patterns=IGNORE + ["README.md"],
                          commit_message="Deploy analyzer")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=args.space, repo_type="space", commit_message="Space card")
    user, name = args.space.split("/", 1)
    print(f"Pushed. Building at https://huggingface.co/spaces/{args.space}")
    print(f"Public URL once it is up: https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")
    print("Remember: add GROQ_API_KEY as a secret in the Space settings.")


if __name__ == "__main__":
    main()
