# AI Video Observatory

**Paste a video link, get an estimate of whether it was AI-generated, with the evidence behind it.**

Detecting AI video is unreliable when it rests on one signal: platforms strip metadata, watermark removers are a click away, and detection models fall behind every new generator. This project combines several independent signals, shows what each one found and says *inconclusive* when they disagree, instead of guessing.

> Status: **v0, working.** The weights are hand-set and not yet calibrated; see the roadmap. Treat results as a triage aid, never as proof.

## What it checks

| Signal | How | Strength |
|---|---|---|
| **Platform AI label** | YouTube's `containsSyntheticMedia` declaration via the Data API | Strong when present, often absent |
| **Content credentials** | C2PA manifest in the file (`c2pa-python`): AI vs camera digital source type | Very strong, usually stripped on re-encode |
| **Caption and tags** | Explicit statements ("made with AI", `#aigenerated`) and generator names | Weak to medium, cheap |
| **Visual analysis** | Keyframes (one per scene, PySceneDetect) inspected by a multimodal model that must cite frame-level evidence | Medium, explains itself |

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
cp .env.example .env                              # add ANTHROPIC_API_KEY for the visual signal

observatory analyze "https://www.youtube.com/shorts/<id>"   # command line
observatory serve                                            # web UI at http://127.0.0.1:8000
```

Without an API key the visual signal is skipped and the other three still run.

## Deploy

The `Dockerfile` runs the web UI on port 7860 and works unchanged on Hugging Face Spaces (Docker SDK), Render or Fly. Set `ANTHROPIC_API_KEY` as a secret. Every public analysis costs a model call, so live analysis is rate limited (`LIVE_LIMIT_PER_IP`, `LIVE_LIMIT_PER_DAY`) and can be switched off with `ALLOW_LIVE=0`, in which case the site serves the reports in `examples/reports/`.

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

## License

MIT
