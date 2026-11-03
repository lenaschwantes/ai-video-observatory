import json
import shutil
import subprocess

import pytest

from observatory.bench import robustness, runner
from observatory.bench.metrics import auc, flips, fpr_at, mcnemar_exact, mean_shift, paired_delta
from observatory.detect.models import Frame
from observatory.detect.vision import Usage, VisualAssessment

# --- paired metrics --------------------------------------------------------


def test_flips_count_direction_on_shared_videos_only():
    before = {"a": 0.9, "b": 0.5, "c": 0.1, "d": 0.5, "only_before": 0.9}
    after = {"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.2, "only_after": 0.1}
    f = flips(before, after)
    assert (f.n, f.flipped, f.toward_ai, f.toward_real) == (4, 3, 2, 1)
    assert f.rate == 0.75


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)  # all 6 flips one way
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) is None


def test_paired_delta_is_zero_with_no_change_and_catches_a_shift():
    labels = [1, 1, 1, 0, 0, 0, 0]
    before = [0.9, 0.8, 0.3, 0.1, 0.2, 0.3, 0.6]
    delta, ci = paired_delta(labels, before, before, fpr_at)
    assert delta == 0 and ci == (0, 0)
    after = [p if y else 0.8 for y, p in zip(labels, before)]  # every real video now accused
    delta, ci = paired_delta(labels, before, after, fpr_at)
    assert delta == pytest.approx(1.0) and ci[0] > 0.5
    assert paired_delta(labels, before, after, auc)[0] < 0


def test_mean_shift_per_class():
    assert mean_shift([0, 0, 1], [0.1, 0.2, 0.9], [0.3, 0.4, 0.9], cls=0) == pytest.approx(0.2)
    assert mean_shift([0, 0, 1], [0.1, 0.2, 0.9], [0.3, 0.4, 0.9], cls=1) == 0


# --- results files and the report -------------------------------------------


def test_results_path_names_variant_and_repeat(tmp_path):
    assert runner.results_path("groq", "m/x", tmp_path).name == "groq__m-x.jsonl"
    assert runner.results_path("groq", "m/x", tmp_path, "crf28").name == "groq__m-x@crf28.jsonl"
    assert runner.results_path("groq", "m/x", tmp_path, "ref", 2).name == "groq__m-x@ref.r2.jsonl"


def _write(path, scores, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"id": i, "label": labels[i], "provider": "groq", "model": "m", "p": p}) + "\n"
        for i, p in scores.items()))


def test_robustness_summary_compares_with_ref_and_reports_noise_floor(tmp_path):
    labels = {"a1": "ai", "a2": "ai", "r1": "real", "r2": "real", "r3": "real", "r4": "real"}
    ref = {"a1": 0.9, "a2": 0.8, "r1": 0.1, "r2": 0.2, "r3": 0.1, "r4": 0.5}
    again = {**ref, "r4": 0.8}  # the model alone flips one of six
    worse = {**ref, "r1": 0.8, "r2": 0.9, "r3": 0.5}  # compression pushes real videos up
    res = tmp_path / "results"
    _write(res / "groq__m@ref.jsonl", ref, labels)
    _write(res / "groq__m@ref.r2.jsonl", again, labels)
    _write(res / "groq__m@crf28.jsonl", worse, labels)
    _write(res / "groq__m.jsonl", ref, labels)  # main benchmark file: not a variant

    (row,) = runner.robustness_summary(res, tmp_path / "cache")
    assert row["variant"] == "crf28" and row["n"] == 6
    assert row["noise_flip_rate"] == pytest.approx(1 / 6)
    assert row["flip_rate"] == pytest.approx(3 / 6)
    assert (row["toward_ai"], row["toward_real"]) == (3, 0)
    assert row["fpr_ref"] == 0 and row["fpr"] == 0.5 and row["fpr_delta"] == 0.5
    assert row["shift_real"] > 0.3


def test_misaligned_videos_are_left_out(tmp_path):
    labels = {"a1": "ai", "r1": "real", "r2": "real"}
    ref = {"a1": 0.9, "r1": 0.1, "r2": 0.1}
    res, cache = tmp_path / "results", tmp_path / "cache"
    _write(res / "groq__m@ref.jsonl", ref, labels)
    _write(res / "groq__m@crf28.jsonl", {**ref, "r2": 0.9}, labels)
    d = robustness.variant_dir(cache, "r2", "crf28")
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"aligned": False, "ssim_mean": 0.2}))
    (row,) = runner.robustness_summary(res, cache)
    assert row["n"] == 2 and row["misaligned"] == 1 and row["flip_rate"] == 0


def test_leakage_flags_resolution_that_gives_away_the_label():
    rows = [{"id": "a", "label": "ai", "height": "2160"}, {"id": "b", "label": "ai", "height": "1080"},
            {"id": "c", "label": "real", "height": "360"}, {"id": "d", "label": "real", "height": "480"}]
    leak = runner.leakage(rows)
    assert leak["height"] == {"auc": 1.0, "n": 4}
    assert leak["bitrate"] is None  # no manifests


def test_run_reads_variant_frames_and_report_writes_section(tmp_path):
    rows = [{"id": "a1", "label": "ai", "height": "1080"}, {"id": "r1", "label": "real", "height": "1080"}]
    cache = tmp_path / "cache"
    for r in rows:
        for v in ("ref", "crf28"):
            d = cache / f"{r['id']}@{v}"
            d.mkdir(parents=True)
            (d / "frames.json").write_text(json.dumps({"frames": [Frame(index=0, timestamp=1.0, file="f.jpg").model_dump()]}))
    seen = []

    def fake(provider, model, frames, frame_dir):
        seen.append(frame_dir.name)
        p = 0.9 if frame_dir.name.startswith("a1") or frame_dir.name == "r1@crf28" else 0.1
        return VisualAssessment(ai_likelihood=p, findings=[], summary=""), Usage(10, 5, 1, 0.1)

    kw = dict(cache=cache, results=tmp_path / "results", assess_with=fake)
    runner.run("groq", "m", rows, variant="ref", **kw)
    runner.run("groq", "m", rows, variant="ref", repeat=2, **kw)
    runner.run("groq", "m", rows, variant="crf28", **kw)
    assert seen == ["a1@ref", "r1@ref", "a1@ref", "r1@ref", "a1@crf28", "r1@crf28"]

    runner.report(rows, results=tmp_path / "results", docs=tmp_path / "docs", prices_file=tmp_path / "none.json",
                  cache=cache)
    data = json.loads((tmp_path / "docs" / "benchmarks.json").read_text())
    assert data["models"] == []  # variant files stay out of the main table
    (row,) = data["robustness"]
    assert row["fpr_ref"] == 0 and row["fpr"] == 1.0 and row["noise_flip_rate"] == 0
    md = (tmp_path / "docs" / "benchmarks.md").read_text()
    assert "## Re-compression" in md and "crf28" in md


# --- ffmpeg ------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1080x1920:rate=30:duration=2", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@needs_ffmpeg
def test_build_variants_aligned_frames_and_manifests(tmp_path, clip):
    cache, work = tmp_path / "cache", tmp_path / "work"
    work.mkdir()
    made = robustness.build_variants(clip, [0.5, 1.5], cache, "v1", ["crf28", "720p-crf35"], work)
    assert set(made) == {"ref", "crf28", "720p-crf35"}
    for variant, m in made.items():
        frames = runner.load_frames(f"v1@{variant}", cache)
        assert [f.timestamp for f in frames] == [0.5, 1.5]  # same instants in every version
        assert m["aligned"] and m["ffmpeg"].startswith("ffmpeg")
        assert all(len(f["sha256"]) == 64 for f in m["frames"])
    assert made["ref"]["ssim_mean"] is None
    assert made["crf28"]["ssim_mean"] > made["720p-crf35"]["ssim_mean"] > robustness.ALIGN_MIN
    assert made["720p-crf35"]["height"] == 1280  # short side capped at 720 on a vertical video
    assert made["720p-crf35"]["kbps"] < made["crf28"]["kbps"] < made["ref"]["kbps"]
    assert not list(work.glob("*.mp4"))  # re-encoded files are deleted


@needs_ffmpeg
def test_encoding_is_reproducible(tmp_path, clip):
    a = robustness.encode(clip, tmp_path / "a.mp4", "crf28")
    b = robustness.encode(clip, tmp_path / "b.mp4", "crf28")
    assert a.read_bytes() == b.read_bytes()


@needs_ffmpeg
def test_compare_detects_a_different_picture(tmp_path, clip):
    same = robustness.grab_frame(clip, 1.0, tmp_path / "x.jpg")
    assert robustness.compare(same, same)[0] == pytest.approx(1.0)
    black = tmp_path / "black.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=black:size=768x1366", "-frames:v", "1", str(black)], check=True)
    assert robustness.compare(same, black)[0] < robustness.ALIGN_MIN


def test_prepare_variants_needs_normal_prepare_first(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(robustness, "ffmpeg_exe", lambda: "ffmpeg")
    robustness.prepare_variants([{"id": "x", "url": "u"}], tmp_path, ["crf28"], fetcher=lambda *a: None, pause=0)
    assert "run `observatory bench prepare` first" in capsys.readouterr().out
