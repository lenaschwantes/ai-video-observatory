from observatory.sources.base import SourceUnavailable, VideoSource
from observatory.sources.fixture import FixtureSource


def build_source(name: str) -> VideoSource:
    """Instantiate a source by name. Live sources are imported lazily so the
    pipeline runs without their optional dependencies."""
    if name == "fixture":
        return FixtureSource()
    if name == "tiktokapi":
        from observatory.sources.tiktokapi import TikTokApiSource

        return TikTokApiSource()
    raise ValueError(f"unknown source: {name}")


__all__ = ["SourceUnavailable", "VideoSource", "FixtureSource", "build_source"]
