"""Regression tests for overlay smoothness on high-frame-rate sources.

A 60 fps source analysed at 10 fps means 5 of every 6 output frames sit between
analysis ticks. Writing them raw made the overlay strobe; holding the last state
made it freeze 67% of the time. They must be interpolated.
"""

from __future__ import annotations

import cv2
import numpy as np

from mototrack.detectors import Detection
from mototrack.pipeline import _interpolate, analyse


def det(track_id, x1, y1, x2, y2):
    return Detection(
        track_id=track_id,
        cls_name="2-wheeler",
        cls_id=1,
        conf=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def test_interpolate_midpoint():
    a = [det(1, 0, 0, 100, 100)]
    b = [det(1, 100, 100, 200, 200)]
    out = _interpolate(a, b, 0.5)
    assert len(out) == 1
    assert out[0].x1 == 50.0
    assert out[0].y2 == 150.0


def test_interpolate_at_zero_returns_previous():
    a = [det(1, 0, 0, 100, 100)]
    b = [det(1, 100, 100, 200, 200)]
    out = _interpolate(a, b, 0.0)
    assert out[0].x1 == 0.0


def test_interpolate_at_one_returns_next():
    a = [det(1, 0, 0, 100, 100)]
    b = [det(1, 100, 100, 200, 200)]
    out = _interpolate(a, b, 1.0)
    assert out[0].x1 == 100.0


def test_new_track_is_not_teleported_across_screen():
    # Track 2 appears only in the later state; it must not slide in from nowhere.
    a = [det(1, 0, 0, 100, 100)]
    b = [det(1, 20, 0, 120, 100), det(2, 900, 900, 1000, 1000)]
    out = _interpolate(a, b, 0.5)
    ids = {d.track_id for d in out}
    assert 2 not in ids, "an appearing track must not be interpolated from another position"
    assert 1 in ids


def test_vanished_track_is_held_not_dropped_mid_step():
    a = [det(1, 0, 0, 100, 100), det(2, 500, 500, 600, 600)]
    b = [det(1, 20, 0, 120, 100)]
    out = _interpolate(a, b, 0.5)
    ids = {d.track_id for d in out}
    assert 2 in ids, "a track present before but not after should be held, not vanish"


def test_weight_is_clamped():
    a = [det(1, 0, 0, 100, 100)]
    b = [det(1, 100, 0, 200, 100)]
    assert _interpolate(a, b, 5.0)[0].x1 == 100.0
    assert _interpolate(a, b, -2.0)[0].x1 == 0.0


def _write_video(path, frames, width, height, fps):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        frame[0:12, 0:12] = (210, 210, 210)
        writer.write(frame)
    writer.release()
    return path


class _MovingDetector:
    name = "mixed"

    def __init__(self):
        self.calls = 0

    def infer(self, frame, persist=True):
        self.calls += 1
        h, w = frame.shape[:2]
        # Slide steadily rightwards so interpolation has real motion to blend.
        x = 20 + (self.calls - 1) * 12
        return [det(1, x, 40, x + 40, 120)]


def _frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_overlay_moves_on_most_output_frames(tmp_path):
    """The bug: 67% of output frames had a byte-identical overlay."""
    src = _write_video(str(tmp_path / "src.mp4"), 60, 320, 240, 60.0)
    result = analyse(
        src,
        out_dir=str(tmp_path / "out"),
        analysis_fps=10.0,
        detector=_MovingDetector(),
    )
    frames = _frames(result.annotated_video)
    assert len(frames) == 60

    changed = 0
    for a, b in zip(frames, frames[1:]):
        if int((cv2.absdiff(a, b).max(axis=2) > 30).sum()) >= 50:
            changed += 1
    ratio = changed / (len(frames) - 1)
    assert ratio > 0.6, (
        f"only {ratio:.0%} of frame pairs showed overlay movement; "
        "frames between analysis ticks are not being interpolated"
    )


def test_every_output_frame_is_annotated(tmp_path):
    src = _write_video(str(tmp_path / "src2.mp4"), 40, 320, 240, 40.0)
    result = analyse(
        src,
        out_dir=str(tmp_path / "out2"),
        analysis_fps=10.0,
        detector=_MovingDetector(),
    )
    frames = _frames(result.annotated_video)
    assert len(frames) == 40
    for i, frame in enumerate(frames):
        band = frame[200:235, 20:300]
        assert band.std() > 8.0 or band.max() > 70, f"frame {i} has no overlay"
