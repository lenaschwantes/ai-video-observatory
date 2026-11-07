"""Text in the frames, read by a local OCR model (RapidOCR, ONNX, runs on CPU).

- A generator's name in a corner of the picture is its visible watermark
  (Sora, Veo, Kling, Hailuo...). Strong evidence when it sits at the edge.
- Unreadable text is shown as an observation but does not move the score:
  in calibration real street scenes produced as much unreadable OCR as AI
  clips, and crisp text is no sign of a camera either, since AI compilations
  add captions in editing.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from observatory.detect.models import Direction, Evidence, Frame, Signal

GENERATORS = ("sora", "veo", "kling", "hailuo", "minimax", "pika", "runway", "vidu", "pixverse",
              "luma", "seedance", "higgsfield", "invideo", "hedra", "synthesia")
EDGE = 0.18         # watermark must sit in the outer 18% of the frame
GARBLED_BELOW = 0.70  # region found, letters not confidently read
_LATIN = re.compile(r"[A-Za-zÀ-ÿ0-9 .,:;!?'&@#%/()-]+")

_engine = None
_engine_lock = threading.Lock()


def _ocr():
    """One shared engine; loading it takes a few seconds the first time."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            _engine = RapidOCR(params={"Rec.lang_type": "pt", "Global.log_level": "error"})
        return _engine


def _read(path: Path) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """(text, confidence, box as fractions x0, y0, x1, y1) for each region."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    h, w = img.shape[:2]
    r = _ocr()(img)
    if r.boxes is None or r.txts is None:
        return []
    out = []
    for box, text, score in zip(r.boxes, r.txts, r.scores):
        text = text.strip()
        # the recognizer also knows CJK glyphs and invents them on texture
        if not text or not _LATIN.fullmatch(text):
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append((text, float(score), (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)))
    return out


def _at_edge(box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return (y1 - y0) < 0.12 and (x0 > 1 - EDGE * 2 or x1 < EDGE * 2) and (y0 > 1 - EDGE or y1 < EDGE)


def frame_text_signal(frames: list[Frame], frame_dir: Path, reader=_read) -> Signal:
    label = "Text and watermarks in the frames"
    if not frames:
        return Signal(name="frame_text", label=label, available=False, note="No frames could be extracted.")
    try:
        readings = [(f, reader(frame_dir / f.file)) for f in frames]
    except ImportError:
        return Signal(name="frame_text", label=label, available=False, note="OCR is not installed (pip install rapidocr onnxruntime).")
    except Exception as exc:  # noqa: BLE001 - a broken model file must not stop the analysis
        return Signal(name="frame_text", label=label, available=False, note=f"OCR failed: {exc}")

    evidence, llr = [], 0.0
    marks, garbled = [], []
    for f, regions in readings:
        for text, score, box in regions:
            words = set(re.findall(r"[a-z]+", text.lower()))
            gen = next((g for g in GENERATORS if g in words), None)
            if gen and _at_edge(box):
                marks.append((f, gen))
            elif len(re.sub(r"[^A-Za-z]", "", text)) >= 4 and score < GARBLED_BELOW:
                garbled.append((f, text))

    if marks:
        f, gen = marks[0]
        evidence.append(Evidence(text=f'Generator watermark in a corner of the picture: "{gen.capitalize()}"',
                                 direction=Direction.AI, timestamp=f.timestamp, frame=f.file))
        llr += 2.5
    for f, text in garbled[:2]:  # shown, not scored (see module docstring)
        evidence.append(Evidence(text=f'Text the OCR could not read cleanly: "{text[:30]}"',
                                 direction=Direction.NEUTRAL, timestamp=f.timestamp))

    note = None if marks else "No generator watermark in the frames."
    return Signal(name="frame_text", label=label, available=True, llr=round(llr, 2), evidence=evidence, note=note)


def warmup() -> None:
    """Load (and download, the first time) both OCR models by reading one line of text."""
    import cv2
    import numpy as np

    img = np.full((80, 360, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Observatory", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    print("OCR ready:", _ocr()(img).txts)


if __name__ == "__main__":
    import sys

    if "--warmup" in sys.argv:
        warmup()
