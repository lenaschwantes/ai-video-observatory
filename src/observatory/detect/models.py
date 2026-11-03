"""Contracts for the detector: signals, evidence and the final report."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    AI = "ai"
    HUMAN = "human"
    NEUTRAL = "neutral"


class Verdict(str, Enum):
    LIKELY_AI = "likely_ai"
    INCONCLUSIVE = "inconclusive"
    LIKELY_HUMAN = "likely_human"


class Evidence(BaseModel):
    text: str
    direction: Direction = Direction.NEUTRAL
    timestamp: float | None = Field(None, description="seconds into the video")
    frame: str | None = Field(None, description="file name of the frame shown as proof")


class Signal(BaseModel):
    name: str
    label: str
    available: bool = Field(description="False when the signal could not be measured for this video")
    llr: float = Field(0.0, description="log-likelihood ratio added to the score; >0 points to AI")
    evidence: list[Evidence] = Field(default_factory=list)
    note: str | None = None

    @property
    def direction(self) -> Direction:
        if not self.available or abs(self.llr) < 0.2:
            return Direction.NEUTRAL
        return Direction.AI if self.llr > 0 else Direction.HUMAN


class VideoInfo(BaseModel):
    url: str
    platform: str
    video_id: str
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    uploader: str | None = None
    duration: float | None = None


class Frame(BaseModel):
    index: int
    timestamp: float
    file: str


class Report(BaseModel):
    slug: str
    video: VideoInfo
    analyzed_at: datetime
    probability: float = Field(ge=0, le=1, description="estimated probability the video is AI-generated")
    verdict: Verdict
    signals: list[Signal]
    frames: list[Frame]
    method_version: str
    disclaimer: str = (
        "Automated estimate, not proof. Weights are provisional until calibrated on a labelled set; "
        "use it to prioritise human review, never to accuse a creator."
    )
