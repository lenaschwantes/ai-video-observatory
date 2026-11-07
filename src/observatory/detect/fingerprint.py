"""File fingerprint: what the video file itself says about where it came from.

Two kinds of trace:
- names that generators and cameras write into the container (encoder,
  software and make/model tags). They survive in files sent directly
  (uploads, the Kaggle notebook); platforms usually rewrite them on re-encode.
- the clip length. Generators produce fixed lengths (Veo 8 s; Sora, Kling,
  Runway, Hailuo 5, 6 or 10 s) and a single unedited clip keeps it to the
  frame. Weak on its own: plenty of real clips are 10 s too.
"""

from __future__ import annotations

from pathlib import Path

from observatory.detect.models import Direction, Evidence, Signal, VideoInfo

SCAN_BYTES = 4 * 1024 * 1024  # container metadata sits at the start or the end

# Long, specific tokens only: short words occur by chance in compressed data.
AI_MARKERS = {
    b"trainedalgorithmicmedia": "IPTC tag for AI-generated media",
    b"openai sora": "Sora", b"sora.com": "Sora", b"runwayml": "Runway", b"klingai": "Kling",
    b"kling ai": "Kling", b"hailuoai": "Hailuo", b"pika labs": "Pika", b"lumalabs": "Luma",
    b"google veo": "Veo", b"synthid": "SynthID", b"midjourney": "Midjourney",
}
# Only long QuickTime/Android keys: short brand names ("DJI", "GoPro") turned up
# by chance inside the compressed video of a 2005 recording during calibration.
CAMERA_MARKERS = {
    b"com.apple.quicktime.make": "an Apple camera", b"com.apple.quicktime.model": "an Apple camera",
    b"com.android.capture.fps": "an Android camera", b"com.android.manufacturer": "an Android camera",
}
GENERATOR_LENGTHS = (5.0, 6.0, 8.0, 10.0)
LENGTH_TOLERANCE = 0.08  # seconds; about two frames


def _scan(path: Path) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(SCAN_BYTES)
        if size > 2 * SCAN_BYTES:
            fh.seek(size - SCAN_BYTES)
            return (head + fh.read()).lower()
        return (head + fh.read()).lower()


def _probe(path: Path) -> tuple[float | None, float | None, int | None, int | None]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, (count / fps if fps and count else None), w or None, h or None


def fingerprint_signal(video: VideoInfo, video_path: Path) -> Signal:
    label = "File fingerprint"
    try:
        blob = _scan(video_path)
        fps, duration, w, h = _probe(video_path)
    except OSError as exc:
        return Signal(name="fingerprint", label=label, available=False, note=f"Could not read the file: {exc}")
    duration = video.duration or duration
    evidence, llr = [], 0.0

    ai = sorted({name for token, name in AI_MARKERS.items() if token in blob})
    cam = sorted({name for token, name in CAMERA_MARKERS.items() if token in blob})
    if ai:
        evidence.append(Evidence(text=f"File metadata names {', '.join(ai)}", direction=Direction.AI))
        llr += 2.0
    if cam:
        evidence.append(Evidence(text=f"File metadata was written by {', '.join(cam)}", direction=Direction.HUMAN))
        llr -= 1.0
    if duration and duration <= 10.5:
        match = next((g for g in GENERATOR_LENGTHS if abs(duration - g) <= LENGTH_TOLERANCE), None)
        if match:
            evidence.append(Evidence(text=f"Standard generator clip length: {match:.0f} s", direction=Direction.AI))
            llr += 0.35

    facts = ", ".join(x for x in (
        f"{duration:.2f} s" if duration else "",
        f"{fps:.2f} fps" if fps else "",
        f"{w}x{h}" if w and h else "",
    ) if x)
    facts = f" File facts: {facts}." if facts else ""
    note = ("No generator or camera tags found; platforms usually strip them." + facts) if not evidence else (facts.strip() or None)
    return Signal(name="fingerprint", label=label, available=True, llr=round(llr, 2), evidence=evidence, note=note)
