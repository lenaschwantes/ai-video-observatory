"""Download one public video and its metadata with yt-dlp."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from observatory.detect.models import VideoInfo

ALLOWED_HOSTS = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com", "x.com", "twitter.com")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "1200"))
# Only the extractors of the platforms above. Without this, a post whose card
# points elsewhere could make yt-dlp's generic extractor fetch any address.
EXTRACTORS = ["youtube", "YoutubeYtBe", "TikTok", "vm.tiktok", "Instagram", "twitter"]


LOGIN_HINTS = ("login required", "log in", "rate-limit reached", "cookies")
LOGIN_MESSAGE = (
    "the platform required a login for this video (common on Instagram and X). "
    "Set YTDLP_COOKIES to a cookies.txt from a secondary account, or try another link"
)


class FetchError(RuntimeError):
    pass


_cookie_copy: Path | None = None


def cookies_file() -> str | None:
    """A private, writable copy of the file in YTDLP_COOKIES, if it exists.

    yt-dlp writes refreshed cookies back when a download ends, and on Cloud Run
    the secret is mounted read-only; working on a copy keeps both happy and
    never touches the original."""
    global _cookie_copy
    path = os.environ.get("YTDLP_COOKIES", "").strip()
    if not path or not Path(path).is_file():
        return None
    if _cookie_copy is None or not _cookie_copy.exists():
        fd, tmp = tempfile.mkstemp(prefix="ytcookies-", suffix=".txt")
        os.close(fd)
        shutil.copyfile(path, tmp)
        os.chmod(tmp, 0o600)
        _cookie_copy = Path(tmp)
    return str(_cookie_copy)


def friendly_error(message: str) -> str:
    """Turn yt-dlp's login/rate-limit errors into one clear sentence. YouTube's
    bot check also mentions cookies, but it is about our server, not the video,
    so it keeps its own wording for the web layer to explain."""
    low = message.lower()
    if "not a bot" in low:
        return "youtube blocked the download as a bot (not a bot check); cookies of a secondary account fix it"
    return LOGIN_MESSAGE if any(h in low for h in LOGIN_HINTS) else message


def check_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise FetchError(f"only public links from {', '.join(ALLOWED_HOSTS)} are supported")
    return url.strip()


def slugify(platform: str, video_id: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", f"{platform}-{video_id}".lower()).strip("-")


def fetch_local(path: Path) -> tuple[VideoInfo, Path]:
    """Metadata for a video file on disk. The id is a content hash, so the
    same file uploaded twice hits the cached report."""
    path = Path(path)
    if not path.is_file():
        raise FetchError(f"file not found: {path.name}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise FetchError(f"unsupported file type; use {', '.join(VIDEO_SUFFIXES)}")
    if path.stat().st_size > MAX_BYTES:
        raise FetchError(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    meta = VideoInfo(url="", platform="upload", video_id=digest.hexdigest()[:16], title=path.name)
    return meta, path


def fetch(url: str, workdir: Path) -> tuple[VideoInfo, Path]:
    """Return metadata and the path of a downloaded MP4 (at most 720p)."""
    import yt_dlp

    opts = {
        # Video only: audio is not analyzed, and YouTube often serves no muxed file.
        # H.264 first because OpenCV decodes it everywhere (AV1 often not).
        "format": "bestvideo[vcodec^=avc1][height<=720]/best[vcodec^=avc1][height<=720]/bestvideo[height<=720]/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        # a stalled connection must not hold an analysis slot forever
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": MAX_BYTES,
        "allowed_extractors": EXTRACTORS,
        # live streams have no size to check, and long videos tie up a slot
        "match_filter": yt_dlp.utils.match_filter_func(f"!is_live & duration <=? {MAX_SECONDS}"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    if cookies := cookies_file():
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(check_url(url), download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(friendly_error(str(exc))) from exc
    if not path.exists():
        if info.get("is_live") or (info.get("duration") or 0) > MAX_SECONDS:
            raise FetchError(f"video too long or live (limit {MAX_SECONDS // 60} min)")
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
