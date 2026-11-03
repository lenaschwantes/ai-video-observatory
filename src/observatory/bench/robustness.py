"""Does platform re-compression change what detectors say?

Each dataset video is re-encoded at fixed settings and keyframes are taken at
the *same instants* from every version, with ffmpeg's exact seek, so the only
thing that differs between versions is the compression:

    ref          the downloaded file, no re-encode (baseline)
    crf28        one more H.264 pass, like a repost
    720p-crf35   capped at 720p and heavily compressed, like a forwarded clip

Every version gets a manifest.json (ffmpeg version, exact arguments, bitrate,
per-frame SHA-256, SSIM and PSNR against the ref frame), so a result can be
traced back to the bytes the model saw. A frame whose SSIM against ref falls
below ALIGN_MIN is taken as misaligned, and the video is left out of that
version rather than compared on different pictures."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from observatory.detect.models import Frame

REF = "ref"
VARIANTS: dict[str, list[str]] = {
    REF: [],
    "crf28": ["-c:v", "libx264", "-preset", "medium", "-crf", "28"],
    # short side capped at 720 (a vertical 1080x1920 becomes 720x1280, as platforms serve it)
    "720p-crf35": ["-vf", "scale='if(gt(iw,ih),-2,min(720,iw))':'if(gt(iw,ih),min(720,ih),-2)'",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "35"],
}
# Fixed thread count (x264 is deterministic for a given count), fixed pixel
# format, no audio: same input, same bytes out on any machine.
COMMON = ["-an", "-pix_fmt", "yuv420p", "-threads", "4", "-map_metadata", "-1"]
MAX_WIDTH = 768
ALIGN_MIN = 0.5


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    """System ffmpeg first; the imageio-ffmpeg wheel (bench extra) as fallback,
    so Windows works without a separate install."""
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # not installed, or no binary for this platform
        raise FFmpegMissing("ffmpeg not found: install it or `pip install -e .[bench]`") from exc


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-300:]}")
    return proc


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def probe(video: Path) -> dict:
    """Duration (s), height and bitrate (kbps) from ffmpeg's banner. ffprobe is
    not shipped with imageio-ffmpeg, so parse `ffmpeg -i` instead."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    err = proc.stderr
    info: dict = {"duration": None, "height": None, "kbps": None}
    if m := re.search(r"Duration: (\d+):(\d+):([\d.]+)", err):
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    if m := re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", err):
        info["height"] = int(m[2])
    if info["duration"]:
        info["kbps"] = round(video.stat().st_size * 8 / info["duration"] / 1000, 1)
    return info


def encode(src: Path, dst: Path, variant: str) -> Path:
    if variant == REF:
        return src
    _ffmpeg("-y", "-i", str(src), *VARIANTS[variant], *COMMON, str(dst))
    return dst


def grab_frame(video: Path, t: float, dst: Path, size: tuple[int, int] | None = None) -> Path:
    """One frame at `t` seconds. -ss before -i seeks to the keyframe and then
    decodes forward to t, which ffmpeg does exactly when the output is
    re-encoded (it is: a JPEG). `size` forces the output size; otherwise the
    width is capped at MAX_WIDTH, as in the analyzer."""
    scale = f"scale={size[0]}:{size[1]}" if size else f"scale='min({MAX_WIDTH},iw)':-2"
    _ffmpeg("-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", scale, "-q:v", "3", str(dst))
    if not dst.exists():
        raise RuntimeError(f"no frame at {t:.2f}s")
    return dst


def compare(ref: Path, other: Path) -> tuple[float | None, float | None]:
    """SSIM and PSNR of `other` against `ref` (other is scaled to ref's size)."""
    w, h = image_size(ref)
    proc = _ffmpeg("-i", str(ref), "-i", str(other), "-lavfi",
                   f"[1:v]scale={w}:{h},split[b0][b1];[0:v]split[a0][a1];[a0][b0]ssim;[a1][b1]psnr",
                   "-f", "null", "-")
    ssim = re.search(r"SSIM .*?All:([\d.]+)", proc.stderr)
    psnr = re.search(r"PSNR .*?average:([\d.]+|inf)", proc.stderr)
    return (float(ssim[1]) if ssim else None,
            (99.0 if psnr and psnr[1] == "inf" else float(psnr[1])) if psnr else None)


def image_size(path: Path) -> tuple[int, int]:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Video: .*?, (\d+)x(\d+)", proc.stderr)
    if not m:
        raise RuntimeError(f"cannot read size of {path.name}")
    return int(m[1]), int(m[2])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_dir(cache: Path, vid: str, variant: str) -> Path:
    return cache / f"{vid}@{variant}"


def build_variants(video: Path, times: list[float], cache: Path, vid: str,
                   variants: list[str], workdir: Path) -> dict[str, dict]:
    """Write frames.json + manifest.json for `ref` and each variant of one video."""
    version = ffmpeg_version()
    src_info = probe(video)
    out: dict[str, dict] = {}
    ref_dir = variant_dir(cache, vid, REF)
    for variant in [REF, *[v for v in variants if v != REF]]:
        target = variant_dir(cache, vid, variant)
        target.mkdir(parents=True, exist_ok=True)
        encoded = encode(video, workdir / f"{variant}.mp4", variant)
        info = src_info if variant == REF else probe(encoded)
        frames, rows = [], []
        for i, t in enumerate(times):
            name = f"frame_{i:02d}.jpg"
            try:
                grab_frame(encoded, t, target / name)
            except RuntimeError:
                continue
            ssim = psnr = None
            if variant != REF and (ref_dir / name).exists():
                ssim, psnr = compare(ref_dir / name, target / name)
            frames.append(Frame(index=len(frames), timestamp=round(t, 2), file=name))
            rows.append({"file": name, "t": round(t, 2), "sha256": _sha256(target / name), "ssim": ssim, "psnr": psnr})
        ssims = [r["ssim"] for r in rows if r["ssim"] is not None]
        manifest = {
            "video": vid,
            "variant": variant,
            "ffmpeg": version,
            "args": [*VARIANTS[variant], *COMMON] if variant != REF else [],
            "source_kbps": src_info["kbps"],
            "source_height": src_info["height"],
            "kbps": info["kbps"],
            "height": info["height"],
            "ssim_mean": round(sum(ssims) / len(ssims), 4) if ssims else None,
            "aligned": all(s >= ALIGN_MIN for s in ssims) if ssims else variant == REF,
            "frames": rows,
        }
        (target / "frames.json").write_text(json.dumps({"frames": [f.model_dump() for f in frames]}, indent=1),
                                            encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if encoded != video:
            encoded.unlink(missing_ok=True)
        out[variant] = manifest
    return out


def load_manifest(cache: Path, vid: str, variant: str) -> dict | None:
    path = variant_dir(cache, vid, variant) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def prepare_variants(rows: list[dict], cache: Path, variants: list[str], fetcher=None,
                     pause: float = 20.0, times_for=None) -> None:
    """Download each video again (prepare deleted it), then build every version.
    Frame instants come from the normal prepare (eval/cache/<id>/frames.json),
    so run `bench prepare` first."""
    from observatory.bench.runner import load_frames
    from observatory.detect.fetch import fetch

    ffmpeg_exe()  # fail early, before any download
    fetcher = fetcher or fetch
    times_for = times_for or (lambda vid: [f.timestamp for f in load_frames(vid, cache) or []])
    first = True
    for row in rows:
        vid = row["id"]
        if all(load_manifest(cache, vid, v) for v in [REF, *variants]):
            continue
        times = times_for(vid)
        if not times:
            print(f"[skip] {vid}: run `observatory bench prepare` first")
            continue
        if not first:
            time.sleep(pause)
        first = False
        try:
            with tempfile.TemporaryDirectory(prefix="bench-var-") as tmp:
                _, video = fetcher(row["url"], Path(tmp))
                made = build_variants(video, times, cache, vid, variants, Path(tmp))
        except Exception as exc:
            print(f"[skip] {vid}: {exc}")
            continue
        desc = ", ".join(f"{v} ssim={m['ssim_mean']}" for v, m in made.items() if v != REF)
        print(f"[ok] {vid}: {desc}")
