"""Cheap, explainable signals. Each returns a Signal whose log-likelihood
ratio (llr) pushes the score towards AI (>0) or human (<0).

The llr values are hand-set priors for now. Phase v1 replaces them with
values fitted on a labelled set (see README)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from observatory.detect.models import Direction, Evidence, Signal, VideoInfo

# --- text ---------------------------------------------------------------

# Explicit statements that the video is synthetic.
_STRONG = [
    r"\bai[- ]generated\b", r"\bgenerated (?:with|by) ai\b", r"\bmade (?:with|by|using) ai\b",
    r"\bgerado (?:por|com) ia\b", r"\bfeito com ia\b", r"#aigenerated\b", r"#aivideo\b",
    r"#aiart\b", r"#aianimation\b", r"#iagenerativa\b",
]
# Generator names: suggestive, but some are ordinary words ("runway", "pika").
_TOOLS = [
    "sora", "veo", "kling", "hailuo", "minimax", "seedance", "midjourney", "runway gen",
    "runwayml", "pika labs", "luma dream machine", "dream machine", "stable diffusion",
    "comfyui", "wan 2", "hunyuan video", "higgsfield",
]
_PROMPT = r"\bprompt\b"


def text_signal(video: VideoInfo) -> Signal:
    haystack = " ".join([video.title, video.description, " ".join("#" + t for t in video.tags)]).lower()
    evidence, llr = [], 0.0
    for pattern in _STRONG:
        if m := re.search(pattern, haystack):
            evidence.append(Evidence(text=f'Caption/tags say "{m.group(0)}"', direction=Direction.AI))
            llr += 1.5
    for tool in _TOOLS:
        if re.search(rf"\b{re.escape(tool)}\b", haystack):
            evidence.append(Evidence(text=f'Mentions the generator "{tool}"', direction=Direction.AI))
            llr += 0.8
    if re.search(_PROMPT, haystack):
        evidence.append(Evidence(text='Mentions a "prompt"', direction=Direction.AI))
        llr += 0.5
    return Signal(
        name="text",
        label="Title, caption and tags",
        available=True,
        llr=min(llr, 3.0),
        evidence=evidence,
        note=None if evidence else "No AI-related wording found. Absence is weak evidence: most AI videos say nothing.",
    )


# --- provenance (C2PA content credentials) -------------------------------

# IPTC digital source types written by generators vs cameras.
_AI_SOURCES = ("trainedalgorithmicmedia", "compositesynthetic", "algorithmicmedia")
_CAPTURE_SOURCES = ("digitalcapture",)


def provenance_signal(video_path: Path) -> Signal:
    try:
        from c2pa import Reader

        reader = Reader.try_create(str(video_path))
    except Exception as exc:  # unsupported container, corrupt manifest...
        return Signal(name="provenance", label="Content credentials (C2PA)", available=False, note=f"Could not read: {exc}")
    if reader is None:
        return Signal(
            name="provenance",
            label="Content credentials (C2PA)",
            available=False,
            note="No content credentials in the file. Common: platforms strip them on re-encode.",
        )
    with reader:
        manifest = reader.json()
    blob = json.dumps(json.loads(manifest)).lower()
    if any(s in blob for s in _AI_SOURCES):
        return Signal(
            name="provenance",
            label="Content credentials (C2PA)",
            available=True,
            llr=4.0,
            evidence=[Evidence(text="C2PA manifest declares an AI/algorithmic digital source type", direction=Direction.AI)],
        )
    if any(s in blob for s in _CAPTURE_SOURCES):
        return Signal(
            name="provenance",
            label="Content credentials (C2PA)",
            available=True,
            llr=-2.0,
            evidence=[Evidence(text="C2PA manifest declares a camera capture", direction=Direction.HUMAN)],
        )
    return Signal(
        name="provenance",
        label="Content credentials (C2PA)",
        available=True,
        evidence=[Evidence(text="C2PA manifest present but does not state the source type")],
    )


# --- platform label -------------------------------------------------------


def platform_label_signal(video: VideoInfo) -> Signal:
    """YouTube's `status.containsSyntheticMedia`, when the API exposes it."""
    label = "Platform AI label"
    if video.platform != "youtube":
        return Signal(name="platform_label", label=label, available=False, note=f"Not exposed for {video.platform}.")
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return Signal(name="platform_label", label=label, available=False, note="Set YOUTUBE_API_KEY to read it.")
    import httpx

    try:
        resp = httpx.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "status", "id": video.video_id, "key": key},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items") or []
    except httpx.HTTPError as exc:
        return Signal(name="platform_label", label=label, available=False, note=f"YouTube API error: {exc}")
    value = (items[0].get("status") or {}).get("containsSyntheticMedia") if items else None
    if value is True:
        return Signal(
            name="platform_label",
            label=label,
            available=True,
            llr=3.0,
            evidence=[Evidence(text="The creator declared altered or synthetic content", direction=Direction.AI)],
        )
    if value is False:
        return Signal(
            name="platform_label",
            label=label,
            available=True,
            llr=-0.3,
            evidence=[Evidence(text="No synthetic-content declaration (weak: many creators never declare)")],
        )
    return Signal(name="platform_label", label=label, available=False, note="The API did not return the field for this video.")
