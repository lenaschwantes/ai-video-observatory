"""Motion consistency between nearby frames (OpenCV optical flow).

For pairs of frames ~0.07 s apart, estimate how every pixel moved and warp
the second frame back onto the first. In a camera recording, motion explains
almost all of the change. Generated video often has regions that change
without moving (objects that morph, textures that boil), so part of the
change is left over even after following the motion.

The score is the leftover in the worst regions (90th percentile of 32 px
blocks), median over the video. Experimental: on 14 labelled videos it ranked
an AI clip above a real one in about 80% of pairs, but scene cuts, fast
motion and heavy compression also leave a residue. It therefore moves the
score by at most 0.6 until the benchmark calibrates it.
"""

from __future__ import annotations

from pathlib import Path

from observatory.detect.models import Direction, Evidence, Frame, Signal

STEP = 0.5        # seconds between sampled pairs
GAP = 2           # frames between the two images of a pair
WIDTH = 320
BLOCK = 32
MIN_CHANGE = 6.0  # a block must actually change to be judged
MAX_PAIRS = 40
CENTER = 0.72     # residue around which real and AI clips split in calibration
SLOPE = 4.0
CAP = 0.6


def _pair_residue(a, b):
    """90th-percentile leftover/raw change over blocks, or None if not judgeable."""
    import cv2
    import numpy as np

    h = int(a.shape[0] * WIDTH / a.shape[1])
    g1 = cv2.cvtColor(cv2.resize(a, (WIDTH, h)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(b, (WIDTH, h)), cv2.COLOR_BGR2GRAY)
    same_scene = cv2.compareHist(cv2.calcHist([g1], [0], None, [32], [0, 256]),
                                 cv2.calcHist([g2], [0], None, [32], [0, 256]), cv2.HISTCMP_CORREL)
    if same_scene < 0.6:  # a cut, not motion
        return None
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    gx, gy = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(h, dtype=np.float32))
    f1, f2 = g1.astype(np.float32), g2.astype(np.float32)
    left = np.abs(cv2.remap(f2, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR) - f1)
    raw = np.abs(f2 - f1)
    ratios = []
    for y in range(8, h - BLOCK - 8, BLOCK):
        for x in range(8, WIDTH - BLOCK - 8, BLOCK):
            r = raw[y:y + BLOCK, x:x + BLOCK].mean()
            if r >= MIN_CHANGE:
                ratios.append(left[y:y + BLOCK, x:x + BLOCK].mean() / r)
    return float(np.percentile(ratios, 90)) if len(ratios) >= 4 else None


def _save_pair(a, b, path: Path) -> None:
    import cv2
    import numpy as np

    h = int(a.shape[0] * 384 / a.shape[1])
    left, right = cv2.resize(a, (384, h)), cv2.resize(b, (384, h))
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", np.hstack([left, gap, right]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        path.write_bytes(buf.tobytes())  # imencode + write_bytes: non-ASCII paths on Windows


def motion_signal(video_path: Path, out_dir: Path) -> tuple[Signal, Frame | None]:
    """The signal and, when a pair stands out, the side-by-side image cited as proof."""
    import cv2
    import numpy as np

    label = "Motion consistency between nearby frames"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scores, worst = [], None
    t = 0.25
    while t * fps + GAP < total and len(scores) < MAX_PAIRS:
        idx = int(t * fps)
        t += STEP
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, a = cap.read()
        for _ in range(GAP - 1):
            cap.read()
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        r = _pair_residue(a, b)
        if r is None:
            continue
        scores.append(r)
        if worst is None or r > worst[0]:
            worst = (r, idx / fps, a, b)
    cap.release()

    if len(scores) < 3:
        return Signal(name="motion", label=label, available=False,
                      note="Too little steady motion to judge (static shots or constant cuts)."), None
    median = float(np.median(scores))
    llr = max(-CAP, min(CAP, SLOPE * (median - CENTER)))
    evidence, frame = [], None
    if llr > 0.2 and worst is not None:
        name = "pair_00.jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_pair(worst[2], worst[3], out_dir / name)
        frame = Frame(index=900, timestamp=round(worst[1], 2), file=name)
        evidence.append(Evidence(text="Parts of the picture change between two frames 0.1 s apart without moving there",
                                 direction=Direction.AI, timestamp=frame.timestamp, frame=name))
    elif llr < -0.2:
        evidence.append(Evidence(text="Changes between nearby frames are explained by motion, as in a camera recording",
                                 direction=Direction.HUMAN))
    note = f"Experimental. Unexplained change in the worst regions: {median:.2f} (calibration split near {CENTER})."
    return Signal(name="motion", label=label, available=True, llr=round(llr, 2), evidence=evidence, note=note), frame
