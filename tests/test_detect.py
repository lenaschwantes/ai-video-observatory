from pathlib import Path

import pytest

from observatory.detect.fetch import LOGIN_MESSAGE, FetchError, check_url, cookies_file, friendly_error
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


@pytest.mark.parametrize("url", ["https://www.instagram.com/reel/C0abc123/", "https://instagram.com/p/C0abc123/"])
def test_accepts_instagram(url):
    assert check_url(url) == url


def test_login_errors_become_clear_message():
    raw = "ERROR: [Instagram] C0abc: Requested content is not available, rate-limit reached or login required"
    assert friendly_error(raw) == LOGIN_MESSAGE
    assert friendly_error("ERROR: Video unavailable") == "ERROR: Video unavailable"


def test_cookies_file_only_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    assert cookies_file() is None
    monkeypatch.setenv("YTDLP_COOKIES", str(tmp_path / "missing.txt"))
    assert cookies_file() is None
    jar = tmp_path / "cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("YTDLP_COOKIES", str(jar))
    assert cookies_file() == str(jar)


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


def test_frames_are_written_under_non_ascii_paths(tmp_path):
    from observatory.detect.frames import extract_frames

    out = tmp_path / "Área de Trabalho"
    frames = extract_frames(make_video(tmp_path / "v.mp4"), out)
    assert frames and all((out / f.file).stat().st_size > 0 for f in frames)


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


# --- vision providers ----------------------------------------------------


def test_merge_averages_batches_in_log_odds_and_keeps_all_findings():
    from observatory.detect.vision import merge

    a = VisualAssessment(ai_likelihood=0.9, findings=[Finding(frame_index=0, observation="x", direction="ai")], summary="a")
    b = VisualAssessment(ai_likelihood=0.5, findings=[Finding(frame_index=3, observation="y", direction="neutral")], summary="b")
    merged = merge([a, b])
    assert 0.7 < merged.ai_likelihood < 0.9
    assert [f.frame_index for f in merged.findings] == [0, 3]


def test_provider_prefers_groq_then_anthropic(monkeypatch):
    from observatory.detect.vision import provider

    for var in ("VISION_PROVIDER", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert provider() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert provider() == "anthropic"
    monkeypatch.setenv("GROQ_API_KEY", "y")
    assert provider() == "groq"
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    assert provider() == "anthropic"
