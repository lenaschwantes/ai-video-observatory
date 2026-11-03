# AI Video Observatory

**An open test of AI-video detectors, commercial and open, with Brazilian content and one priority: never accuse a real video.**

Commercial detectors advertise 98%+ accuracy, but the numbers come from the vendors themselves, and independent tests have caught them flagging real footage as fake. This project measures them in the open:

1. **[Benchmarks](docs/benchmarks.md)**: commercial detectors (Sightengine, Hive) and general vision models (Groq, Claude, GPT, Gemini) scored on the same labelled videos, global and Brazilian, with real videos wrongly flagged as AI as the headline number.
2. **Analyzer**: paste a link and get an estimate with the evidence behind it, combining several independent signals and saying *inconclusive* instead of guessing.

> Status: **working, early.** The labelled set is small and the analyzer's weights are not yet fitted to it; see the roadmap. Treat results as a triage aid, never as proof.

## What it checks

| Signal | How | Strength |
|---|---|---|
| **Platform AI label** | YouTube's `containsSyntheticMedia` declaration via the Data API | Strong when present, often absent |
| **Content credentials** | C2PA manifest in the file (`c2pa-python`): AI vs camera digital source type | Very strong, usually stripped on re-encode |
| **Caption and tags** | Explicit statements ("made with AI", `#aigenerated`) and generator names | Weak to medium, cheap |
| **Visual analysis** | Keyframes (one per scene, PySceneDetect) inspected by a multimodal model that must cite frame-level evidence. Runs on Groq (open-weight vision model, free tier; frames sent in batches of 3) or Claude | Medium, explains itself |

Each signal contributes a log-likelihood ratio. They are summed into a probability, and the verdict is *likely AI* above 75%, *likely not AI* below 25%, and *inconclusive* in between or when nothing informative was found.

Example output:

```
Dancing robot #aiart
94% -> likely_ai
  [    n/a] Content credentials (C2PA): No content credentials in the file. Common: platforms strip them on re-encode.
  [     ai] Title, caption and tags
            - (ai) Caption/tags say "#aiart"
  [     ai] Visual analysis of keyframes: Typical generator artifacts.
            - (ai) @1.5s Robot's hand merges into the wall
```

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate   # source .venv/bin/activate on macOS/Linux
pip install -e ".[detect,dev]"
cp .env.example .env                              # add GROQ_API_KEY (or ANTHROPIC_API_KEY) for the visual signal

observatory analyze "https://www.youtube.com/shorts/<id>"   # command line (YouTube, TikTok, Instagram, X)
observatory analyze ./clip.mp4                               # or a local file
observatory serve                                            # web UI at http://127.0.0.1:8000
```

Without a key the visual signal is skipped and the other three still run.

## Benchmarks

How well can each vision model tell AI video from real footage, looking only at the frames? [docs/benchmarks.md](docs/benchmarks.md) compares providers on a labelled set ([eval/dataset.csv](eval/dataset.csv)): AI clips whose uploader names the generator, and real videos uploaded before 2022. Every model sees the same cached keyframes and nothing else, and is scored on AUC, real videos wrongly flagged as AI, AI videos caught, calibration, speed and cost. The same table is on the site at `/benchmarks`.

```bash
pip install -e ".[detect,bench]"
observatory bench prepare                  # download once, keep keyframes, delete videos
observatory bench run --provider all       # every provider with a key (and a model) in .env
observatory bench report                   # writes docs/benchmarks.md and .json
```

Supported:

- **Vision models**: Groq, Anthropic, OpenAI and Gemini. OpenAI and Gemini need the model name in `OPENAI_VISION_MODEL` / `GEMINI_VISION_MODEL`.
- **Commercial detectors**: Sightengine (`SIGHTENGINE_API_USER`, `SIGHTENGINE_API_SECRET`, free tier) and Hive, through NVIDIA's hosted endpoint (`NVIDIA_API_KEY`). Detectors score each keyframe and the video score is their mean in log-odds.

The dataset has a Brazilian subset (`region=br`): AI clips by Brazilian creators (fake TV reports, church sketches, ads) paired with real local news, church services and vlogs, to test whether detectors do worse on content they were not built around. Runs resume after interruptions; `--retry-errors` re-runs failed rows.

## Deploy

The `Dockerfile` runs the web UI on port 7860 and works unchanged on Hugging Face Spaces (Docker SDK), Render or Fly. Set `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) as a secret. Every public analysis costs a model call, so live analysis is rate limited (`LIVE_LIMIT_PER_IP`, `LIVE_LIMIT_PER_DAY`) and can be switched off with `ALLOW_LIVE=0`, in which case the site serves the reports in `examples/reports/`.

Visitors can also **bring their own key** (Groq, Anthropic, OpenAI or Gemini) in the analyzer form. The visual check then runs on their account, so live analysis stays available even with `ALLOW_LIVE=0`, under its own limit (`BYOK_LIMIT_PER_IP`, default 20/day; `ALLOW_BYOK=0` turns it off). The key is used for that one analysis and kept only in memory: it is never written to disk or logs, never returned in a response, and scrubbed from error messages (tested in `tests/test_web.py`). Serve the site over HTTPS.

**Instagram and X** links work, but both often demand a login even for public Reels, and blocks server IPs more than home ones. When it does, the analyzer says so plainly. To get past it, set `YTDLP_COOKIES` to a `cookies.txt` exported from a *secondary* account (accounts used this way can be blocked; `cookies*.txt` is git-ignored).

**Local files** (mp4, mov, m4v, webm, mkv, up to 200 MB) can be uploaded in the web form or passed to the CLI. There is no caption or platform label to read, so the estimate rests on the frames and on content credentials, which an original file is far more likely to still carry than a re-encoded post. Uploads are deleted right after analysis; the report is keyed by a hash of the file.

Downloads from cloud servers are sometimes blocked by the platforms. When that happens the site says so; running locally always works.

## Design decisions

| Decision | Why |
|---|---|
| **Several weak signals, not one model** | Each signal fails differently. Combining them, and showing each one, beats a single black-box score. |
| **"Inconclusive" is a first-class answer** | A false "AI" verdict harms a real creator. The thresholds keep the tool quiet when evidence is thin. |
| **Evidence must point at a frame** | The visual check returns findings tied to a frame index and timestamp, and those frames are kept in the report. |
| **The model's confidence is shrunk** | Its log-odds are multiplied by 0.8, so an uncalibrated judge cannot dominate the stronger signals. |
| **Videos are deleted after analysis** | Only the report and the frames cited as evidence are kept. |
| **Reports are cached by video** | Re-analyzing the same link costs nothing and returns the same answer. |

## Also in this repository: the trend pipeline

The analyzer grew out of a daily pipeline that tracks AI video hashtags and flags breakouts. It is kept because the analyzer will run on it at scale (see roadmap).

- Swappable collectors with fallback, raw payloads landed next to normalized rows, contract violations quarantined (`src/observatory/sources`, `ingest.py`).
- dbt + DuckDB warehouse with data tests for duplicates, decreasing counters and volume collapse (`transform/`).
- Robust z-score (median/MAD) detection of emerging hashtags (`trends.py`).
- 14 days of synthetic fixtures with one hidden breakout; CI fails if the pipeline does not find it.

```bash
observatory ingest --sources fixture --start 2026-10-01 --end 2026-10-14
dbt build --project-dir transform --profiles-dir transform
observatory trends
```

Live TikTok collection through the unofficial client is currently blocked by bot detection, and TikTok's Research API only admits Brazilian institutions for youth-safety studies, so the pipeline runs on fixtures for now.

## Roadmap

- [x] **v0, analyzer**: four signals, combination with an inconclusive band, CLI, web UI, Docker.
- [ ] **v1, trustworthy**: hand-labelled evaluation set (Label Studio) plus a public benchmark (GenVidBench); fit the signal weights; calibration curve; publish accuracy and where it fails.
- [ ] **v1, honest uncertainty**: conformal prediction (MAPIE) so "inconclusive" comes with a guaranteed error rate.
- [ ] **v2, robustness**: run the evaluation set through public watermark and metadata removers and report how much accuracy drops; add a specialised detector model (e.g. Veritas) as a fifth signal.
- [ ] **v2, labelling audit**: run the analyzer over many videos and measure how much AI content reaches viewers without a label.

## Project layout

```
src/observatory/detect/   analyzer: fetch, frames, signals, vision, pipeline, web
src/observatory/          trend pipeline: sources, ingest, trends, cli
transform/                dbt project
tests/                    pytest (no network, no API calls)
Dockerfile                web deployment
```

## Ethics

Only public videos are analyzed, on request, one at a time. Nothing is redistributed, the video file is deleted after analysis, and every report carries a disclaimer. The tool is meant to prioritise human review, not to accuse anyone.

## Author

Made by **Helena Schwantes**: [LinkedIn](https://www.linkedin.com/in/helenags) · [GitHub](https://github.com/lenaschwantes).

## License

MIT
