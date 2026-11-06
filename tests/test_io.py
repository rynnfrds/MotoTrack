"""Video I/O tests: metadata parsing and the H.264 transcode contract."""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from mototrack import config
from mototrack.io import VideoError, VideoWriter, iter_frames, probe


def _write_clip(path: str, frames: int = 8, width: int = 160, height: int = 120, fps: float = 10.0) -> str:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_clip(tmp_path):
    return _write_clip(str(tmp_path / "tiny.mp4"))


def test_probe_reads_dimensions_and_rate(tiny_clip):
    meta = probe(tiny_clip)
    assert meta.width == 160
    assert meta.height == 120
    assert meta.fps == pytest.approx(10.0, abs=0.5)
    assert meta.frame_count == 8


def test_probe_rejects_missing_file():
    with pytest.raises(VideoError):
        probe("does-not-exist.mp4")


def test_probe_rejects_non_video(tmp_path):
    path = tmp_path / "notvideo.mp4"
    path.write_bytes(b"definitely not a video" * 50)
    with pytest.raises(VideoError):
        probe(str(path))


def test_iter_frames_yields_every_frame(tiny_clip):
    frames = list(iter_frames(tiny_clip))
    assert len(frames) == 8
    assert frames[0][0] == 0
    assert frames[-1][1].shape == (120, 160, 3)


def test_writer_output_is_h264_for_browsers(tmp_path):
    out = str(tmp_path / "annotated.mp4")
    writer = VideoWriter(out, 160, 120, 10.0)
    for i in range(8):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    path = writer.close()

    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "annotated.raw.mp4"))

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt", "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True,
    )
    assert "codec_name=h264" in result.stdout
    assert "pix_fmt=yuv420p" in result.stdout


def test_upload_extension_allowlist():
    assert ".mp4" in config.ALLOWED_UPLOAD_EXTENSIONS
    assert ".mov" in config.ALLOWED_UPLOAD_EXTENSIONS
    assert ".exe" not in config.ALLOWED_UPLOAD_EXTENSIONS
