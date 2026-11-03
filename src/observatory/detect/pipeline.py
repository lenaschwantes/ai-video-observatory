"""Link in, report out. The downloaded video is deleted after analysis; only
the report and the frames cited as evidence are kept."""

from __future__ import annotations

import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from observatory.detect.fetch import fetch, fetch_local, slugify
from observatory.detect.frames import extract_frames
from observatory.detect.models import Report, Signal, Verdict
from observatory.detect.signals import platform_label_signal, provenance_signal, text_signal
from observatory.detect.vision import assess, visual_signal
from observatory.learn.weights import DEFAULT, Weights, load_current

AI_THRESHOLD = 0.75
HUMAN_THRESHOLD = 0.25


def combine(signals: list[Signal], weights: Weights = DEFAULT) -> tuple[float, Verdict]:
    """Weighted sum of the log-likelihood ratios on even prior odds; the weights
    are learned from curated feedback (see observatory.learn). With no
    informative signal the answer is "inconclusive", never a guess."""
    informative = [s for s in signals if s.available and abs(s.llr) >= 0.2]
    if not informative:
        return 0.5, Verdict.INCONCLUSIVE
    z = weights.bias + sum(weights.weight(s.name) * s.llr for s in informative)
    p = 1 / (1 + math.exp(-z))
    if p >= AI_THRESHOLD:
        return p, Verdict.LIKELY_AI
    if p <= HUMAN_THRESHOLD:
        return p, Verdict.LIKELY_HUMAN
    return p, Verdict.INCONCLUSIVE


def load_report(reports_dir: Path, slug: str) -> Report | None:
    path = reports_dir / slug / "report.json"
    return Report.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def analyze(url: str, reports_dir: Path, *, refresh: bool = False, fetcher=fetch, assessor=assess,
            progress=lambda stage: None) -> Report:
    """`progress` is called with each stage name (download, frames, signals,
    visual) so a UI can show where the analysis is."""
    with tempfile.TemporaryDirectory(prefix="observatory-") as tmp:
        progress("download")
        video, video_path = fetcher(url, Path(tmp))
        slug = slugify(video.platform, video.video_id)
        if not refresh and (cached := load_report(reports_dir, slug)):
            return cached

        out_dir = reports_dir / slug
        if out_dir.exists():
            shutil.rmtree(out_dir)
        progress("frames")
        frames = extract_frames(video_path, out_dir)
        progress("signals")
        signals = [platform_label_signal(video), provenance_signal(video_path), text_signal(video)]
        progress("visual")
        signals.append(visual_signal(frames, out_dir, assessor=assessor))
    # leaving the with-block deletes the downloaded video

    weights = load_current()
    probability, verdict = combine(signals, weights)
    cited = {e.frame for s in signals for e in s.evidence if e.frame}
    for f in frames:
        if f.file not in cited:
            (out_dir / f.file).unlink(missing_ok=True)
    report = Report(
        slug=slug,
        video=video,
        analyzed_at=datetime.now(timezone.utc),
        probability=round(probability, 3),
        verdict=verdict,
        signals=signals,
        frames=[f for f in frames if f.file in cited],
        method_version=weights.version,
    )
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def analyze_file(path: Path, reports_dir: Path, **kwargs) -> Report:
    """Same as `analyze`, for a local video file instead of a link. The file
    itself is left where it is."""
    return analyze(str(path), reports_dir, fetcher=lambda _url, _tmp: fetch_local(Path(path)), **kwargs)
