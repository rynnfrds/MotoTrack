"""Video I/O: probe metadata, stream frames, and write a browser-playable file.

OpenCV reads and writes what it likes, which on Windows is usually mp4v. Browsers
will not play that, so the annotated intermediate is always re-encoded through
ffmpeg to H.264 / yuv420p with the moov atom up front.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


class VideoError(RuntimeError):
    """Raised when a file cannot be read as video."""


def _ffprobe_json(path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise VideoError("ffprobe not found on PATH; install ffmpeg")
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:format=duration",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed: {proc.stderr.strip()[:300]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoError(f"ffprobe returned unreadable output: {exc}") from None


def _parse_rate(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str) -> VideoMeta:
    """Read stream metadata, falling back gracefully on VFR and odd containers."""
    if not os.path.exists(path):
        raise VideoError(f"file not found: {path}")
    if os.path.getsize(path) == 0:
        raise VideoError(f"file is empty: {path}")

    data = _ffprobe_json(path)
    streams = data.get("streams") or []
    if not streams:
        raise VideoError("no video stream found; is this a video file?")
    stream = streams[0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoError("video stream reports no usable dimensions")

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
        stream.get("r_frame_rate")
    )

    duration = 0.0
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    frame_count = 0
    try:
        frame_count = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV cannot open {path}")
        if fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if fps <= 0 or fps != fps:
        fps = 25.0  # last resort so downstream timing maths never divides by zero
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))

    return VideoMeta(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration,
        codec=str(stream.get("codec_name") or "unknown"),
    )


def iter_frames(path: str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) until the stream ends."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"OpenCV cannot open {path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


class VideoWriter:
    """Write BGR frames to an intermediate file, then transcode to H.264."""

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self.final_path = path
        self.raw_path = os.path.splitext(path)[0] + ".raw.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.raw_path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise VideoError(f"cannot open video writer for {self.raw_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> str:
        """Finish the intermediate and produce the H.264 file browsers can play."""
        self._writer.release()
        self._transcode()
        return self.final_path

    def _transcode(self) -> None:
        if shutil.which("ffmpeg") is None:
            # Not fatal: keep the mp4v file rather than losing the whole run.
            os.replace(self.raw_path, self.final_path)
            return
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", self.raw_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                self.final_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.exists(self.final_path):
            os.replace(self.raw_path, self.final_path)
            return
        try:
            os.remove(self.raw_path)
        except OSError:
            pass

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        if not self._writer.isOpened():
            return
        try:
            self.close()
        except Exception:
            pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def is_probably_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in config.ALLOWED_UPLOAD_EXTENSIONS
