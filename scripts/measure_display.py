"""Re-measure the three display defects on the current clip.

The numbers in docs/METRICS.md were measured on a 60 fps source. The default
clip is now 29.9 fps, so the same measurements have to be re-taken rather than
carried over.

Reports, for the shipped pipeline:
  * share of output frame pairs whose HUD region is byte-identical (frozen
    overlay defect),
  * mean per-frame box-corner jump with smoothing on and off (jitter defect).

Run:  .venv/Scripts/python.exe scripts/measure_display.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import cv2  # noqa: E402

from mototrack import config  # noqa: E402
from mototrack.detectors import build_detector  # noqa: E402
from mototrack.geometry import compute_gap  # noqa: E402
from mototrack.io import iter_frames, probe  # noqa: E402
from mototrack.pipeline import analyse  # noqa: E402
from mototrack.smoothing import DetectionSmoother  # noqa: E402

CLIP = os.path.join(config.DATA_DIR, "raw", "clip_001.mp4")
OUT = os.path.join(config.OUT_DIR, "display_check")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def static_overlay_ratio(video_path: str) -> tuple[int, int, float]:
    frames = read_frames(video_path)
    if len(frames) < 2:
        return len(frames), 0, 0.0
    static = 0
    for a, b in zip(frames, frames[1:]):
        # HUD band only: the top strip, where the verdict text is drawn.
        band_a = a[0:110, 0:520]
        band_b = b[0:110, 0:520]
        if int(cv2.absdiff(band_a, band_b).max()) == 0:
            static += 1
    return len(frames), static, static / (len(frames) - 1)


def box_jitter(alpha: float) -> float:
    """Mean corner movement per analysed frame, with the given EMA alpha."""
    meta = probe(CLIP)
    detector = build_detector(config.DEFAULT_DETECTOR)
    smoother = DetectionSmoother(alpha=alpha, hold_seconds=0.0)
    step = max(1, round(meta.fps / config.ANALYSIS_FPS))
    previous = {}
    total = 0.0
    samples = 0
    for index, frame in iter_frames(CLIP):
        if index % step != 0:
            continue
        t = index / meta.fps
        dets = smoother.update(detector.infer(frame, persist=True), t)
        for det in dets:
            if det.track_id is None:
                continue
            prev = previous.get(det.track_id)
            if prev is not None:
                total += (
                    abs(det.x1 - prev[0])
                    + abs(det.y1 - prev[1])
                    + abs(det.x2 - prev[2])
                    + abs(det.y2 - prev[3])
                ) / 4.0
                samples += 1
            previous[det.track_id] = (det.x1, det.y1, det.x2, det.y2)
    return total / samples if samples else 0.0


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    result = analyse(CLIP, out_dir=OUT, detector_kind=config.DEFAULT_DETECTOR)
    total, static, ratio = static_overlay_ratio(result.annotated_video)

    print("clip            :", CLIP)
    print("detector        :", config.DEFAULT_DETECTOR)
    print("output frames   :", total)
    print(f"static HUD pairs: {static} ({ratio:.1%})")

    raw = box_jitter(1.0)
    smooth = box_jitter(config.BOX_SMOOTHING_ALPHA)
    print(f"box jitter, no smoothing : {raw:.2f} px/frame")
    print(f"box jitter, alpha={config.BOX_SMOOTHING_ALPHA}   : {smooth:.2f} px/frame")
    if raw > 0:
        print(f"reduction        : {100.0 * (1.0 - smooth / raw):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
