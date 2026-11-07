# AI Video Observatory

**An open test of AI-video detectors, commercial and open, with Brazilian content and one priority: never accuse a real video.**

Commercial detectors advertise 98%+ accuracy, but the numbers come from the vendors themselves, and independent tests have caught them flagging real footage as fake. This project measures them in the open:

1. **[Benchmarks](docs/benchmarks.md)**: commercial detectors (Sightengine, Hive) and general vision models (Groq, Claude, GPT, Gemini) scored on the same labelled videos, global and Brazilian, with real videos wrongly flagged as AI as the headline number.
2. **Analyzer**: paste a link and get an estimate with the evidence behind it, combining several independent signals and saying *inconclusive* instead of guessing.

> Status: **working, early.** The labelled set is small and the analyzer's weights are not yet fitted to it; see the roadmap. Treat results as a triage aid, never as proof.

The website (Portuguese by default, English with `?lang=en`) has the ranking, the analyzer, the method and this documentation at `/docs`, read straight from the files in this repository. A static copy without the analyzer is published on GitHub Pages (see [Documentation site](#documentation-site)).

## What it checks

| Signal | How | Strength |
|---|---|---|
| **Platform AI label** | YouTube's `containsSyntheticMedia` declaration via the Data API | Strong when present, often absent |
| **Content credentials** | C2PA manifest in the file (`c2pa-python`): AI vs camera digital source type | Very strong, usually stripped on re-encode |
| **Caption and tags** | Explicit statements ("made with AI", `#aigenerated`) and generator names | Weak to medium, cheap |
| **Visual analysis** | Keyframes (one per scene, PySceneDetect) inspected by a multimodal model that must cite frame-level evidence, written in the reader's language. Runs on Groq (open-weight, free tier; frames in batches of 3), Claude, GPT or Gemini, on the host's key or the visitor's own | Medium, explains itself |

Each signal contributes a log-likelihood ratio. The score is `sigmoid(bias + Σ weight × llr)`, with versioned weights (`weights/`; v0 is the hand-set start, every weight 1), and the verdict is *likely AI* at 75% or more, *likely not AI* at 25% or less, and *inconclusive* in between or when nothing informative was found. Each report records the weights version that produced it.

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

### Three ways to run it

The analyzer asks **how do you want to run it?** before asking for a video:

| Mode | For whom | What happens |
|---|---|---|
| **My own key** (free) | Anyone with a Groq or Gemini key | The key is checked first (`POST /api/keys/check`), which lists the vision models it can use, so a wrong key fails in a second, not after a download. Used for that analysis only, never stored. |
| **Kaggle** (no install) | Anyone with a Kaggle account | [`notebooks/analyze_on_kaggle.ipynb`](notebooks/analyze_on_kaggle.ipynb) opens on Kaggle straight from GitHub: Copy & Edit, add a free key under *Add-ons → Secrets*, paste the link, Run all. The report renders in the notebook. |
| **Site demo** | Visitors of a deployment that has its own key | Uses the host's key under the daily limit. Only shown when the server has one. |

Before submitting, the form says which signals will run, so a missing key never looks like a broken tool. The static GitHub Pages copy offers Kaggle and running locally.

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

The dataset has a Brazilian subset (`region=br`): AI clips by Brazilian creators (fake TV reports, church sketches, ads) paired with real local news, church services and vlogs, to test whether detectors do worse on content they were not built around. Runs resume after interruptions; `--retry-errors` re-runs failed rows. Every rate comes with a 95% interval (Wilson for rates, a seeded bootstrap for AUC): with a set this small, overlapping intervals mean two models are not shown to differ.

### Running it for free

Everything above runs on free tiers; it only takes longer. `--provider free` uses the providers that have one: Groq and Gemini (vision models), Sightengine and Hive via NVIDIA (detectors). Claude and OpenAI are paid and are skipped.

```bash
observatory bench run --provider free --pace 3        # 3 s between calls for per-minute limits
observatory bench status                              # scored / failed / left per results file, tokens per call
```

When a provider's daily cap runs out, the run stops cleanly instead of filling the results with errors: nothing is written for the video it was on, and running the same command the next day continues from there. `bench status` shows tokens per call, so you can see how many videos fit in a day's quota. Gemini needs a model name in `GEMINI_VISION_MODEL`; pick one that AI Studio lists with a free tier.

Suggested order, so each day adds a usable result: the main benchmark on Groq first, then the robustness run with `--limit 10` (`--variant all`), then the rest of the dataset.

### Robustness to re-compression

Platforms re-encode every upload, and vendors never say what that does to their detectors. The benchmark re-encodes each video and scores every version on frames taken at the **same instants** (ffmpeg exact seek), so compression is the only difference:

| Version | ffmpeg | Stands for |
|---|---|---|
| `ref` | none | the file as downloaded (already platform-compressed) |
| `crf28` | H.264, CRF 28 | one more pass, like a repost |
| `720p-crf35` | short side ≤ 720, H.264, CRF 35 | a clip forwarded a few times |

```bash
observatory bench prepare --variants all                 # needs the normal prepare; downloads again
observatory bench run --provider groq --variant all      # ref twice (noise floor), then each version
observatory bench report
```

What makes the numbers hold up:

- **Noise floor.** `ref` is scored twice; the share of verdicts the model changes on identical frames is the bar a re-compression effect has to clear.
- **Paired statistics.** Changes are paired bootstraps over videos, and a two-sided exact McNemar test says whether flips lean toward "AI" or toward "real".
- **Alignment check.** Every frame's SSIM and PSNR against `ref` are stored; a video whose frames stop matching (SSIM < 0.5) is dropped from that version instead of compared on different pictures.
- **Traceability.** Each version writes a `manifest.json`: ffmpeg version, exact arguments, source and output bitrate, and the SHA-256 of every frame. Encoding uses a fixed thread count, so the same input gives the same bytes.
- **Leakage check.** The report shows how well resolution or bitrate *alone* predict the label. If they do, a detector can look good by learning the file rather than the generator.

ffmpeg comes from the system, or from the `imageio-ffmpeg` wheel in the `bench` extra (handy on Windows; it bundles a GPL ffmpeg binary). The variant levels follow the light/heavy compression split used by [SynthForensics](https://deepfake-demo.aisec.fraunhofer.de/related_work/2602.04939), so the numbers can be read against an external benchmark. Scoring all versions is about 3× the calls of a normal run (≈ $12–18 for 40 videos on Claude Opus; free tier on Groq); start with `--limit 10`.

## The website

| Page | What it shows |
|---|---|
| `/` | The question, the headline number with its counts ("9 of 33 real videos"), the interactive ranking and the analyzer |
| `/benchmarks` | Ranking with three views (real videos accused, AUC, Brazil vs global), the full table with 95% intervals and the re-compression table |
| `/report/<id>` | Probability bar with the three zones, a timeline of the clues, zoomable evidence frames and every signal |
| `/metodo` | How the test works, in plain language |
| `/docs` | This documentation and the benchmark report, rendered from the repository's Markdown |
| `/api/report/<id>`, `/api/benchmarks` | The same data as JSON (`/api/docs` lists the API) |

Built with server-rendered templates, d3 for the charts, htmx for live analysis progress and bigger-picture for the evidence lightbox. It works without JavaScript (the charts fall back to tables), is keyboard and screen-reader accessible, has light and dark themes, and shares well: each page has Open Graph previews in both languages.

## Learning from use

The analyzer is designed to improve from the cases people test, without letting anyone teach it to accuse real videos:

1. **Votes are hints.** A report can be marked "AI / real / not sure", optionally with a source. Voters are stored as a salted hash, one vote per report (`src/observatory/learn/feedback.py`).
2. **Only curated labels train.** Votes rank a curation queue (uncertain and disputed cases first: active learning); a label needs a source, such as the creator's own statement or an upload date before 2022.
3. **Champion / challenger.** Refitted weights replace the current ones only if they do not accuse more real videos on the fixed benchmark. Every attempt, adopted or not, is kept in `weights/history.jsonl`.
4. **The benchmark stays an exam.** Cases used for training never enter the test set; new test sets are cut by date as versioned releases.

Status: versioned weights and the vote/label store are in place; the refit, the vote button and the public version history are next.

## Documentation site

The site's `/docs` pages render `README.md` and `docs/benchmarks.md` from the repository (raw HTML in the Markdown is not rendered), so documentation never needs a trip to GitHub. `scripts/export_static.py` writes a static copy of the ranking, method and documentation pages to `site/`, and `.github/workflows/pages.yml` publishes it to GitHub Pages on every push to `main`. The static copy has no analyzer: that needs a server (run locally, or the Docker image).

## Deploy

The `Dockerfile` runs the web UI on port 7860 and works unchanged on Hugging Face Spaces (Docker SDK), Render or Fly. Set `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) as a secret. Every public analysis costs a model call, so live analysis is rate limited (`LIVE_LIMIT_PER_IP`, `LIVE_LIMIT_PER_DAY`) and can be switched off with `ALLOW_LIVE=0`, in which case the site serves the reports in `examples/reports/`.

Visitors can also **bring their own key** (Groq, Anthropic, OpenAI or Gemini) in the analyzer form. The visual check then runs on their account, so live analysis stays available even with `ALLOW_LIVE=0`, under its own limit (`BYOK_LIMIT_PER_IP`, default 20/day; `ALLOW_BYOK=0` turns it off). The key is used for that one analysis and kept only in memory: it is never written to disk or logs, never returned in a response, and scrubbed from error messages (tested in `tests/test_web.py`). Serve the site over HTTPS.

**Instagram and X** links work, but both often demand a login even for public Reels, and blocks server IPs more than home ones. When it does, the analyzer says so plainly. To get past it, set `YTDLP_COOKIES` to a `cookies.txt` exported from a *secondary* account (accounts used this way can be blocked; `cookies*.txt` is git-ignored).

**Local files** (mp4, mov, m4v, webm, mkv, up to 200 MB) can be uploaded in the web form or passed to the CLI. There is no caption or platform label to read, so the estimate rests on the frames and on content credentials, which an original file is far more likely to still carry than a re-encoded post. Uploads are deleted right after analysis; the report is keyed by a hash of the file, is never listed on the home page and asks search engines not to index it. Uploads are **off in the Docker image** (`ALLOW_UPLOAD=0`) so a public deployment does not host third-party video; set `ALLOW_UPLOAD=1` to turn them on.

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
| **Uploads stay private** | Reports of uploaded files are never listed and are not indexed; uploads are off in the public image. |
| **Visitor keys never persist** | A key pasted in the form lives in memory for one analysis and is scrubbed from errors. |
| **Weights are versioned** | Every report says which weights produced it, and a new version must not accuse more real videos. |

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
- [x] **v0.2, three more signals**: file fingerprint (generator/camera tags, standard clip lengths), generator watermarks read by local OCR (RapidOCR), and motion consistency from optical flow (experimental, capped at ±0.6 until calibrated).
- [x] **Open benchmark**: commercial detectors and vision models on the same frames, Brazilian subset, 95% intervals, recall with no real video accused.
- [x] **Robustness to re-compression**: noise floor, paired bootstrap, McNemar, alignment check, leakage check.
- [x] **Website**: Portuguese/English, accessible charts, live progress, visitor keys, private uploads, docs inside the site.
- [ ] **Learning from use**: refit with champion/challenger, vote button, public version history (groundwork done).
- [ ] **v1, trustworthy**: hand-labelled evaluation set (Label Studio) plus a public benchmark (GenVidBench); fit the signal weights; calibration curve; publish accuracy and where it fails.
- [ ] **v1, honest uncertainty**: conformal prediction (MAPIE) so "inconclusive" comes with a guaranteed error rate.
- [ ] **v2, robustness**: run the evaluation set through public watermark and metadata removers and report how much accuracy drops; add a specialised detector model (e.g. Veritas) as a fifth signal.
- [ ] **v2, labelling audit**: run the analyzer over many videos and measure how much AI content reaches viewers without a label.

## Project layout

```
src/observatory/detect/   analyzer: fetch, frames, signals, vision, pipeline, web, docs, i18n
src/observatory/bench/    benchmark: runner, metrics, robustness, commercial detectors
src/observatory/learn/    learning from use: versioned weights, votes and curated labels
src/observatory/          trend pipeline: sources, ingest, trends, cli
eval/                     labelled dataset, prices; keyframe cache (not committed)
transform/                dbt project
tests/                    pytest (no network, no API calls)
Dockerfile                web deployment
```

## Ethics

Only public videos are analyzed, on request, one at a time. Nothing is redistributed, the video file is deleted after analysis, and every report carries a disclaimer. The tool is meant to prioritise human review, not to accuse anyone.

## Author

Made by **Helena Schwantes**: [LinkedIn](https://www.linkedin.com/in/helenags) · [GitHub](https://github.com/lenaschwantes).

Running the public demo costs API credits. If it helped you, you can [sponsor it](https://github.com/sponsors/lenaschwantes).

## License

MIT
