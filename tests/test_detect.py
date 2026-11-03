from pathlib import Path

import pytest

from observatory.detect.fetch import FetchError, check_url
from observatory.detect.models import Signal, Verdict, VideoInfo
from observatory.detect.pipeline import analyze, combine
from observatory.detect.signals import text_signal
from observatory.detect.vision import Finding, VisualAssessment

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scenedetect")


def info(**kw):
    return VideoInfo(url="https://www.youtube.com/shorts/x", platform="youtube", video_id="x", **kw)


def make_video(path: Path, colors=((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)), seconds_each=1, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for color in colors:
        for _ in range(seconds_each * fps):
            frame = __import__("numpy").full((120, 160, 3), color, dtype="uint8")
            writer.write(frame)
    writer.release()
    return path


# --- combine -------------------------------------------------------------


def sig(llr, available=True):
    return Signal(name="s", label="s", available=available, llr=llr)


def test_no_informative_signal_is_inconclusive():
    assert combine([sig(0.0), sig(3.0, available=False)]) == (0.5, Verdict.INCONCLUSIVE)


def test_strong_ai_evidence_wins():
    p, verdict = combine([sig(4.0), sig(0.5)])
    assert verdict is Verdict.LIKELY_AI and p > 0.95


def test_conflicting_evidence_stays_inconclusive():
    assert combine([sig(1.5), sig(-1.2)])[1] is Verdict.INCONCLUSIVE


# --- text ----------------------------------------------------------------


def test_text_signal_reads_explicit_statements_and_tools():
    s = text_signal(info(title="Cat pilots a plane", description="made with AI using Sora", tags=["aivideo"]))
    assert s.llr == 3.0  # capped
    assert any("sora" in e.text for e in s.evidence)


def test_text_signal_ignores_ordinary_words():
    s = text_signal(info(title="Paris fashion week runway highlights"))
    assert s.llr == 0 and not s.evidence


# --- url guard -----------------------------------------------------------


@pytest.mark.parametrize("url", ["https://evil.example/video", "file:///etc/passwd", "https://youtube.com.evil.io/x"])
def test_rejects_other_hosts(url):
    with pytest.raises(FetchError):
        check_url(url)


def test_accepts_supported_hosts():
    assert check_url(" https://www.tiktok.com/@a/video/1 ") == "https://www.tiktok.com/@a/video/1"


# --- end to end with fakes ----------------------------------------------


def test_pipeline_writes_report_keeps_only_cited_frames(tmp_path):
    def fake_fetch(url, workdir):
        return info(title="Dancing robot #aiart"), make_video(workdir / "x.mp4")

    def fake_assess(frames, frame_dir):
        return VisualAssessment(
            ai_likelihood=0.9,
            findings=[Finding(frame_index=1, observation="Robot's hand merges into the wall", direction="ai")],
            summary="Typical generator artifacts.",
        )

    report = analyze("https://www.youtube.com/shorts/x", tmp_path, fetcher=fake_fetch, assessor=fake_assess)
    out = tmp_path / report.slug
    assert report.verdict is Verdict.LIKELY_AI
    assert (out / "report.json").exists()
    assert sorted(p.name for p in out.glob("*.jpg")) == [report.frames[0].file]
    visual = next(s for s in report.signals if s.name == "visual")
    assert visual.evidence[0].frame == report.frames[0].file


def test_cached_report_is_reused(tmp_path):
    calls = []

    def fake_fetch(url, workdir):
        return info(), make_video(workdir / "x.mp4")

    def fake_assess(frames, frame_dir):
        calls.append(1)
        return VisualAssessment(ai_likelihood=0.5, findings=[], summary="")

    analyze("u", tmp_path, fetcher=fake_fetch, assessor=fake_assess)
    analyze("u", tmp_path, fetcher=fake_fetch, assessor=fake_assess)
    assert len(calls) == 1
