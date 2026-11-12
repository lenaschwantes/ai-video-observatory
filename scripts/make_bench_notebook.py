"""Write notebooks/benchmark_on_kaggle.ipynb.

The notebook is generated so its cells stay plain, testable Python (no shell
magics). Run: python scripts/make_bench_notebook.py"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Benchmark on a free GPU · Benchmark numa GPU grátis

Runs the [AI Video Observatory](https://github.com/lenaschwantes/ai-video-observatory) benchmark with an **open vision model on Kaggle's free GPU**, through Ollama. No API key, no daily quota, no cost.
Roda o benchmark com um **modelo aberto na GPU grátis do Kaggle**, via Ollama. Sem chave de API, sem cota diária, sem custo.

1. **Copy & Edit** this notebook. / Clique em **Copy & Edit**.
2. **Settings → Accelerator: GPU T4** and **Settings → Internet: on** (needs a phone-verified Kaggle account). / Ative a GPU T4 e a internet.
3. Adjust the settings cell below, then **Run all**. / Ajuste a célula de configuração e clique em **Run all**.
4. Download `benchmark_results.zip` from the **Output** panel, unzip it into your clone and run `observatory bench report`. / Baixe o zip, descompacte no seu clone e rode `observatory bench report`.

**If YouTube blocks downloads from Kaggle** (it often distrusts cloud machines): on your own computer run `observatory bench prepare --variants all`, zip the `eval/cache` folder, upload it with **Add data → Upload** as a dataset, and put its path in `CACHE_DATASET`. Only the frames go up; the heavy part still runs here.
**Se o YouTube bloquear os downloads:** rode o `prepare` no seu computador, suba a pasta `eval/cache` como dataset e informe o caminho em `CACHE_DATASET`.

A session lasts about 9 to 12 hours. If it ends early, run the notebook again: finished videos are skipped.
Uma sessão dura umas 9 a 12 horas. Se cair, rode de novo: os vídeos prontos são pulados."""

SETTINGS = '''# 1. Settings / Configuração
MODEL = "qwen2.5vl:7b"  # any Ollama vision model that fits the GPU (T4: 16 GB); see ollama.com/search?c=vision
LIMIT = 10              # videos from the dataset; None for all of them. Start small.
ROBUSTNESS = True       # also score the re-encoded versions (ref twice, crf28, 720p-crf35)
CACHE_DATASET = ""      # e.g. "/kaggle/input/my-bench-cache" when YouTube blocks downloads here
REPO = "https://github.com/lenaschwantes/ai-video-observatory"'''

HELPERS = '''# 2. Helpers
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CODE = WORK / "ai-video-observatory"


def sh(*cmd, cwd=None, check=True):
    """Run a command and stream its output into the notebook."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}")
    return proc.returncode


def limit_args():
    return ["--limit", str(LIMIT)] if LIMIT else []'''

INSTALL = '''# 3. Get the code and install it (a couple of minutes)
if not CODE.exists():
    sh("git", "clone", "--depth", "1", REPO, CODE)
else:
    sh("git", "pull", "--ff-only", cwd=CODE)
sh(sys.executable, "-m", "pip", "install", "-q", "-e", f"{CODE}[detect,bench,local]")'''

OLLAMA = '''# 4. Start Ollama on the GPU and download the model (about 5 minutes the first time)
if shutil.which("ollama") is None:
    sh("bash", "-c", "apt-get install -y -qq zstd >/dev/null 2>&1 || true; curl -fsSL https://ollama.com/install.sh | sh")

os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
os.environ["OLLAMA_VISION_MODEL"] = MODEL


def ollama_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
        return True
    except Exception:
        return False


if not ollama_up():
    log = open(WORK / "ollama.log", "a")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=os.environ.copy())
    for _ in range(60):
        if ollama_up():
            break
        time.sleep(1)
    else:
        raise SystemExit("Ollama did not start; see ollama.log in the Output panel")
sh("ollama", "pull", MODEL)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"], check=False)'''

PREPARE = '''# 5. Keyframes: from your uploaded cache, or downloaded here
cache = CODE / "eval" / "cache"
if CACHE_DATASET:
    src = Path(CACHE_DATASET)
    # the upload may hold eval/cache, cache/, or the video folders directly
    hits = sorted(src.rglob("frames.json"))
    if not hits:
        raise SystemExit(f"no frames.json under {src}: upload the eval/cache folder from your computer")
    root = hits[0].parent.parent
    shutil.copytree(root, cache, dirs_exist_ok=True)
    print(f"copied {len({p.parent for p in hits})} video folders from {root}")
else:
    args = ["--variants", "all"] if ROBUSTNESS else []
    sh(sys.executable, "-m", "observatory.cli", "bench", "prepare", *args, *limit_args(), "--pause", "5", cwd=CODE)

ready = sum(1 for p in cache.glob("*/frames.json") if "@" not in p.parent.name)
print(f"videos with keyframes: {ready}")
if not ready:
    raise SystemExit("no keyframes: downloads were blocked. Use CACHE_DATASET (see the top of the notebook).")'''

RUN = '''# 6. Score: the main benchmark, then the re-encoded versions
base = [sys.executable, "-m", "observatory.cli", "bench", "run", "--provider", "ollama", "--model", MODEL, *limit_args()]
sh(*base, cwd=CODE, check=False)
if ROBUSTNESS:
    sh(*base, "--variant", "all", cwd=CODE, check=False)
sh(sys.executable, "-m", "observatory.cli", "bench", "status", *limit_args(), cwd=CODE, check=False)'''

REPORT = '''# 7. Report, and a zip to download / Relatório e um zip para baixar
sh(sys.executable, "-m", "observatory.cli", "bench", "report", cwd=CODE, check=False)
bundle = WORK / "benchmark_results"
if bundle.exists():
    shutil.rmtree(bundle)
shutil.copytree(CODE / "eval" / "results", bundle / "eval" / "results")
docs = CODE / "docs"
for name in ("benchmarks.md", "benchmarks.json"):
    if (docs / name).exists():
        (bundle / "docs").mkdir(parents=True, exist_ok=True)
        shutil.copy(docs / name, bundle / "docs" / name)
archive = shutil.make_archive(str(bundle), "zip", bundle)
print(f"download {Path(archive).name} from the Output panel and unzip it into your clone")
if (docs / "benchmarks.md").exists():
    print((docs / "benchmarks.md").read_text(encoding="utf-8")[:3000])'''


def _cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    return cell


def build() -> dict:
    return {
        "cells": [_cell("markdown", INTRO)] + [_cell("code", c) for c in (SETTINGS, HELPERS, INSTALL, OLLAMA, PREPARE, RUN, REPORT)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks" / "benchmark_on_kaggle.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
