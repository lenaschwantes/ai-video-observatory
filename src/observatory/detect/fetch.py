"""Download one public video and its metadata with yt-dlp."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from observatory.detect.models import VideoInfo

ALLOWED_HOSTS = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")
MAX_BYTES = 200 * 1024 * 1024


class FetchError(RuntimeError):
    pass


def check_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise FetchError(f"only public links from {', '.join(ALLOWED_HOSTS)} are supported")
    return url.strip()


def slugify(platform: str, video_id: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", f"{platform}-{video_id}".lower()).strip("-")


def fetch(url: str, workdir: Path) -> tuple[VideoInfo, Path]:
    """Return metadata and the path of a downloaded MP4 (at most 720p)."""
    import yt_dlp

    opts = {
        # Video only: audio is not analyzed, and YouTube often serves no muxed file.
        # H.264 first because OpenCV decodes it everywhere (AV1 often not).
        "format": "bestvideo[vcodec^=avc1][height<=720]/best[vcodec^=avc1][height<=720]/bestvideo[height<=720]/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_BYTES,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(check_url(url), download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(str(exc)) from exc
    if not path.exists():
        raise FetchError("video could not be downloaded")
    meta = VideoInfo(
        url=info.get("webpage_url") or url,
        platform=(info.get("extractor_key") or "unknown").lower(),
        video_id=str(info.get("id")),
        title=info.get("title") or "",
        description=info.get("description") or "",
        tags=list(info.get("tags") or []),
        uploader=info.get("uploader") or info.get("channel"),
        duration=info.get("duration"),
    )
    return meta, path
