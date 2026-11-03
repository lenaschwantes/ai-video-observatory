"""Pick a handful of representative frames: one per detected scene, falling
back to even sampling for single-shot videos."""

from __future__ import annotations

from pathlib import Path

from observatory.detect.models import Frame

MAX_FRAMES = 8
MAX_WIDTH = 768


def _scene_midpoints(video: Path) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector())
    return [(start.get_seconds() + end.get_seconds()) / 2 for start, end in scenes]


def _even(duration: float, n: int) -> list[float]:
    return [duration * (i + 0.5) / n for i in range(n)]


def extract_frames(video: Path, out_dir: Path, max_frames: int = MAX_FRAMES) -> list[Frame]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    times = _scene_midpoints(video)
    if len(times) < 3:
        times = _even(duration, min(6, max_frames))
    if len(times) > max_frames:
        step = len(times) / max_frames
        times = [times[int(i * step)] for i in range(max_frames)]

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        if w > MAX_WIDTH:
            img = cv2.resize(img, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        name = f"frame_{i:02d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(Frame(index=len(frames), timestamp=round(t, 2), file=name))
    cap.release()
    return frames
