"""File fingerprint, text/watermark in frames and motion consistency."""

from pathlib import Path

import numpy as np
import pytest

from observatory.detect.fingerprint import fingerprint_signal
from observatory.detect.frametext import frame_text_signal
from observatory.detect.models import Direction, Frame, VideoInfo
from observatory.detect.motion import motion_signal

cv2 = pytest.importorskip("cv2")


def _video(path: Path, frame_fn, seconds=4.0, fps=24, size=(320, 240)) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        out.write(frame_fn(i))
    out.release()
    return path


def _textured(seed=0, size=(240, 320)):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (*size, 3), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)


def _info(**kw):
    return VideoInfo(url="", platform="upload", video_id="x", **kw)


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_reads_generator_and_camera_tags(tmp_path):
    clip = _video(tmp_path / "a.mp4", lambda i: _textured(i))
    plain = fingerprint_signal(_info(), clip)
    assert plain.available and plain.llr == 0 and "No generator or camera tags" in plain.note

    tagged = tmp_path / "b.mp4"
    tagged.write_bytes(clip.read_bytes() + b"\x00Made with RunwayML\x00")
    s = fingerprint_signal(_info(), tagged)
    assert s.llr >= 2 and s.evidence[0].direction == Direction.AI and "Runway" in s.evidence[0].text

    phone = tmp_path / "c.mp4"
    phone.write_bytes(b"com.apple.quicktime.make Apple" + clip.read_bytes())
    assert fingerprint_signal(_info(), phone).llr < 0


def test_fingerprint_flags_standard_generator_lengths_only_when_exact(tmp_path):
    clip = _video(tmp_path / "a.mp4", lambda i: _textured(i))
    assert fingerprint_signal(_info(duration=8.0), clip).llr == pytest.approx(0.35)
    assert fingerprint_signal(_info(duration=8.4), clip).llr == 0
    assert fingerprint_signal(_info(duration=40.0), clip).llr == 0


# --- text and watermarks in frames ---------------------------------------------

def _frames(n=3):
    return [Frame(index=i, timestamp=float(i), file=f"frame_{i:02d}.jpg") for i in range(n)]


def test_generator_name_in_a_corner_is_a_watermark(tmp_path):
    corner = lambda _p: [("Veo", 0.95, (0.88, 0.90, 0.97, 0.96))]
    s = frame_text_signal(_frames(), tmp_path, reader=corner)
    assert s.llr == pytest.approx(2.5) and s.evidence[0].direction == Direction.AI
    assert s.evidence[0].frame == "frame_00.jpg"


def test_generator_name_in_a_caption_is_not_a_watermark(tmp_path):
    caption = lambda _p: [("O Google Veo 3 foi lançado", 0.97, (0.2, 0.85, 0.8, 0.93))]
    s = frame_text_signal(_frames(), tmp_path, reader=caption)
    assert s.llr == 0 and not s.evidence


def test_unreadable_text_is_shown_but_not_scored(tmp_path):
    garbled = lambda _p: [("UNOTHES", 0.55, (0.3, 0.4, 0.6, 0.5))]
    s = frame_text_signal(_frames(), tmp_path, reader=garbled)
    assert s.llr == 0 and s.evidence and all(e.direction == Direction.NEUTRAL for e in s.evidence)


def test_ocr_failure_leaves_the_signal_unavailable(tmp_path):
    def broken(_p):
        raise RuntimeError("model file damaged")
    s = frame_text_signal(_frames(), tmp_path, reader=broken)
    assert not s.available and "OCR failed" in s.note


# --- motion consistency ----------------------------------------------------------

def test_motion_explained_by_movement_points_to_a_camera(tmp_path):
    scene = _textured(1, (240, 480))
    pan = _video(tmp_path / "pan.mp4", lambda i: np.ascontiguousarray(scene[:, i % 160: i % 160 + 320]))
    s, pair = motion_signal(pan, tmp_path)
    assert s.available and s.llr < 0 and pair is None


def test_texture_that_changes_without_moving_points_to_ai(tmp_path):
    # the picture "boils": every frame is new texture, nothing moves to explain it
    boil = _video(tmp_path / "boil.mp4", lambda i: cv2.addWeighted(_textured(0), 0.5, _textured(i + 10), 0.5, 0))
    s, pair = motion_signal(boil, tmp_path)
    assert s.available and s.llr > 0.2
    assert pair is not None and (tmp_path / pair.file).exists() and s.evidence[0].frame == pair.file


def test_static_video_cannot_be_judged(tmp_path):
    still = _video(tmp_path / "still.mp4", lambda i: _textured(3))
    s, pair = motion_signal(still, tmp_path)
    assert not s.available and pair is None
