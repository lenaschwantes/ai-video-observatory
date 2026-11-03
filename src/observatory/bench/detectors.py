"""Commercial AI-content detectors, called on the same keyframes the vision
models see. Each frame gets a score; the video score is the mean in
log-odds space, the same rule used to merge vision batches.

- Sightengine: public docs, `genai` model on /1.0/check.json.
- Hive: through NVIDIA's hosted endpoint (Hive's own API is sales-only).
  The response schema is not documented, so the score is located by name
  and an unrecognised payload fails loudly with a sample of the JSON."""

from __future__ import annotations

import base64
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from observatory.detect.models import Frame
from observatory.detect.vision import Finding, Usage, VisualAssessment

DETECTORS = {
    "sightengine": ("SIGHTENGINE_API_USER", "SIGHTENGINE_API_SECRET"),
    "hive": ("NVIDIA_API_KEY",),
}
HIVE_URL = "https://ai.api.nvidia.com/v1/cv/hive/ai-generated-image-detection"
HIVE_MAX_B64 = 180_000  # inline limit; larger frames are re-encoded smaller


def has_keys(name: str) -> bool:
    return all(os.environ.get(k) for k in DETECTORS[name])


def _sightengine(path: Path) -> float:
    import httpx

    with path.open("rb") as fh:
        resp = httpx.post(
            "https://api.sightengine.com/1.0/check.json",
            data={"models": "genai", "api_user": os.environ["SIGHTENGINE_API_USER"],
                  "api_secret": os.environ["SIGHTENGINE_API_SECRET"]},
            files={"media": fh},
            timeout=60,
        )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(f"sightengine: {body.get('error') or body}")
    return float(body["type"]["ai_generated"])


def _small_jpeg_b64(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    if len(data) < HIVE_MAX_B64:
        return data
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    for width, quality in ((640, 80), (512, 75), (384, 70)):
        h, w = img.shape[:2]
        small = cv2.resize(img, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        data = base64.b64encode(buf.tobytes()).decode()
        if len(data) < HIVE_MAX_B64:
            return data
    raise RuntimeError("frame too large for Hive's inline limit")


def find_ai_score(payload: Any) -> float | None:
    """Find the AI-generated probability in Hive's output, whatever the nesting:
    either {"class": "ai_generated", "score": x} entries or an "ai_generated" key."""
    if isinstance(payload, dict):
        cls = str(payload.get("class") or payload.get("label") or "").lower()
        if cls in ("ai_generated", "ai-generated", "ai") and isinstance(payload.get("score"), (int, float)):
            return float(payload["score"])
        for key in ("ai_generated", "is_ai_generated"):
            if isinstance(payload.get(key), (int, float)) and not isinstance(payload.get(key), bool):
                return float(payload[key])
        values = payload.values()
    elif isinstance(payload, list):
        values = payload
    else:
        return None
    for value in values:
        found = find_ai_score(value)
        if found is not None:
            return found
    return None


def _hive(path: Path) -> float:
    import httpx

    resp = httpx.post(
        HIVE_URL,
        headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}", "Accept": "application/json"},
        json={"input": [f"data:image/jpeg;base64,{_small_jpeg_b64(path)}"]},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    score = find_ai_score(body)
    if score is None:
        raise RuntimeError(f"hive: unrecognised response {str(body)[:300]}")
    return score


SCORERS = {"sightengine": _sightengine, "hive": _hive}


def assess_with(name: str, model: str, frames: list[Frame], frame_dir: Path) -> tuple[VisualAssessment, Usage]:
    scorer = SCORERS[name]
    start = time.perf_counter()
    scores = [scorer(frame_dir / f.file) for f in frames]
    clipped = [min(max(s, 0.02), 0.98) for s in scores]
    mean = statistics.fmean(math.log(s / (1 - s)) for s in clipped)
    findings = [
        Finding(frame_index=f.index, observation=f"{name} score {s:.2f}",
                direction="ai" if s >= 0.75 else "human" if s <= 0.25 else "neutral")
        for f, s in zip(frames, scores)
    ]
    assessment = VisualAssessment(ai_likelihood=1 / (1 + math.exp(-mean)), findings=findings,
                                  summary=f"per-frame scores: {', '.join(f'{s:.2f}' for s in scores)}")
    return assessment, Usage(calls=len(frames), seconds=time.perf_counter() - start)
