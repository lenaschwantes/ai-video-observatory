"""Data contracts shared by every stage of the pipeline."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class VideoSnapshot(BaseModel):
    """One video as seen on one day. Metrics are cumulative at capture time."""

    video_id: str = Field(min_length=1)
    captured_on: date
    source: str
    author: str | None = None
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    music_id: str | None = None
    created_at: datetime | None = None
    views: int = Field(ge=0)
    likes: int = Field(ge=0)
    comments: int = Field(ge=0)
    shares: int = Field(ge=0)

    @field_validator("hashtags")
    @classmethod
    def _normalize_tags(cls, tags: list[str]) -> list[str]:
        return sorted({t.strip().lstrip("#").lower() for t in tags if t and t.strip()})


class VideoFormat(str, Enum):
    TALKING_HEAD = "talking_head"
    SKIT = "skit"
    ASMR = "asmr"
    MUSIC_VIDEO = "music_video"
    TUTORIAL = "tutorial"
    SLIDESHOW = "slideshow"
    OTHER = "other"


class VideoLabels(BaseModel):
    """Output contract for the classifier (phase 3, not implemented yet)."""

    video_id: str
    is_ai_generated: bool
    generator_tool: str | None = Field(None, description="e.g. the model or app that made it, if identifiable")
    format: VideoFormat
    hook: str = Field(description="What grabs attention in the first 3 seconds, in one sentence")
    niche: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(description="Why the classifier decided this; kept for auditing")
