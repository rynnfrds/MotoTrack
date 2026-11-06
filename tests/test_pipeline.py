"""Regression test: analysed frames must never be written raw.

A 60 fps source analysed at 10 fps means 5 of every 6 output frames carry no
detection. Writing those bare made boxes and the HUD strobe, which looked like
a jittery model but was a pipeline defect.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mototrack import config
from mototrack.pipeline import analyse


class _StubDetector:
    name = "mixed"

    def __init__(self):
        self.calls = 0

    def infer(self, frame, persist=True):
        from mototrack.detectors import Detection

        self.calls += 1
        h, w = frame.shape[:2]
        # One steady box, well inside the frame.
        return [
            Detection(
                track_id=1,
                cls_name="2-wheeler",
                cls_id=1,
                conf=0.9,
                x1=0.35 * w,
                y1=0.45 * h,
                x2=0.55 * w,
                y2=0.70 * h,
            )
        ]


def _write_video(path: str, frames: int, width: int, height: int, fps: float) -> str:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 40 + (i % 5), dtype=np.uint8)
        # A bright block that the overlay never draws, so we can tell them apart.
        frame[0:10, 0:10] = (200, 200, 200)
        writer.write(frame)
    writer.release()
    return path


def _annotated_frames(video: str) -> list:
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_skipped_frames_are_annotated_not_raw(tmp_path):
    """Every output frame must carry the overlay, not just the analysed ones."""
    src = _write_video(str(tmp_path / "src.mp4"), 30, 320, 240, 30.0)
    detector = _StubDetector()

    result = analyse(
        src,
        out_dir=str(tmp_path / "out"),
        analysis_fps=10.0,
        detector=detector,
    )

    frames = _annotated_frames(result.annotated_video)
    assert len(frames) == 30
    # 30 fps analysed at 10 fps => step 3 => 2 of every 3 frames used to be raw.
    assert detector.calls == 10

    # The stub box spans x=112..176, y=108..168 at 320x240. The overlay draws a
    # corridor with green tint; a raw frame has none of that. Check a pixel band
    # the corridor covers, and require it differs from the plain source frame.
    tinted = 0
    for frame in frames:
        band = frame[200:235, 20:300]
        # source is near-uniform grey; annotation adds coloured lines and text
        if band.std() > 8.0:
            tinted += 1
    assert tinted == len(frames), (
        f"only {tinted}/{len(frames)} output frames were annotated; "
        "skipped frames are being written raw"
    )


def test_annotation_is_continuous_across_stride_boundary(tmp_path):
    """Two adjacent output frames must not jump between raw and annotated."""
    src = _write_video(str(tmp_path / "src.mp4"), 24, 320, 240, 24.0)
    result = analyse(
        src,
        out_dir=str(tmp_path / "out2"),
        analysis_fps=8.0,
        detector=_StubDetector(),
    )
    frames = _annotated_frames(result.annotated_video)
    scores = [float(frame[200:235, 20:300].std()) for frame in frames]
    # Neighbouring frames should look alike; a raw frame between annotated ones
    # produces a large drop.
    worst = max(abs(a - b) for a, b in zip(scores, scores[1:]))
    assert worst < 10.0, f"adjacent output frames differ by {worst:.1f}; overlay is strobing"
