"""Single source of truth for tunables.

Every number that affects a verdict lives here so it can be changed without
touching logic. Values are starting guesses documented in docs/MODEL_CARD.md
and are expected to be re-tuned against the gold set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")
JOBS_DIR = os.path.join(PROJECT_ROOT, "app", "jobs")

# --- Web / serving ----------------------------------------------------------
# Default 8770: verified free on the development machine. Port 8000 is used by
# an unrelated local application and must not be touched.

DEFAULT_PORT = 8770
DEFAULT_HOST = "127.0.0.1"
PORT_ENV_VAR = "MOTOTRACK_PORT"


def resolve_port(cli_port: int | None = None) -> int:
    """Port precedence: explicit CLI flag > MOTOTRACK_PORT env > DEFAULT_PORT."""
    if cli_port is not None:
        return cli_port
    raw = os.environ.get(PORT_ENV_VAR)
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(
                f"{PORT_ENV_VAR} must be an integer, got {raw!r}"
            ) from None
    return DEFAULT_PORT


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
MAX_WEB_VIDEO_SECONDS = 90.0

# --- Detection --------------------------------------------------------------

DETECTOR_COCO = "coco"
DETECTOR_MIXED = "mixed"
DEFAULT_DETECTOR = DETECTOR_COCO

MODEL_FILES = {
    DETECTOR_COCO: {
        "filename": "yolo26n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "approx_bytes": 5_544_453,
    },
    DETECTOR_MIXED: {
        "filename": "visaitech_mixed_traffic_yolo26n.pt",
        "url": (
            "https://huggingface.co/visaitech/vehicle-mixed-traffic-yolo26n"
            "/resolve/main/best.pt"
        ),
        "approx_bytes": 5_384_254,
    },
}

DEPTH_MODEL = {
    "filename": "yolo26n-depth.pt",
    "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt",
    "approx_bytes": 12_998_295,
}

# COCO class ids treated as obstacles. We never branch on class name; these are
# only used to filter obvious non-obstacles out of the detection stream.
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 6 train, 7 truck
COCO_OBSTACLE_CLASS_IDS = (0, 1, 2, 3, 5, 6, 7)

IMGSZ = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
DEFAULT_TRACKER = "bytetrack.yaml"
ANALYSIS_FPS = 10.0  # frames analysed per second of source video

# --- Temporal stabilisation -------------------------------------------------
# Frame-by-frame detection jitters and occasionally drops a track for a sample
# or two. At display rate that reads as the whole overlay flickering. These
# control EMA box smoothing and a short coast window for missed tracks.
#
# alpha 0.45 keeps a visible but quick response: at 10 analysed fps a track
# reaches ~90% of a step change in about four samples (~0.4 s).
BOX_SMOOTHING_ALPHA = 0.45
# How long a track survives a detector dropout. 0.4 s covers the observed
# one-to-two-sample gaps (~0.1-0.3 s at 10 fps) without hiding a departed object.
TRACK_HOLD_SECONDS = 0.4
# Confidence multiplier per missed sample, so a coasted box fades out.
TRACK_HOLD_CONF_DECAY = 0.75

# --- Monocular distance (pinhole similar triangles) -------------------------
# distance_m = (real_height_m * focal_px) / bbox_height_px
# focal_px = (image_width_px / 2) / tan(hfov_rad / 2)
#
# These are approximations, not calibration. Results are reported as buckets,
# never as authoritative metre readings.

ASSUMED_HFOV_DEG = 78.0

ASSUMED_HEIGHT_M = {
    # COCO names
    "person": 1.65,
    "bicycle": 1.30,
    "motorcycle": 1.30,
    "car": 1.60,
    "bus": 2.90,
    "truck": 2.90,
    "train": 2.90,
    # Visaitech mixed-traffic names
    "pedestrian": 1.65,
    "2-wheeler": 1.30,
    "3-wheeler": 1.90,
    "4-wheeler": 1.60,
}
DEFAULT_ASSUMED_HEIGHT_M = 1.60

NEAR_BUCKET_MAX_M = 3.0
MID_BUCKET_MAX_M = 8.0

# --- Ego corridor (relative frame coordinates) ------------------------------
# Trapezoid from a wide bottom edge to a narrow top edge at the horizon row.
# Defined relatively so portrait 9:16 reels work without a separate code path.

CORRIDOR_BOTTOM_LEFT_X = 0.30
CORRIDOR_BOTTOM_RIGHT_X = 0.70
CORRIDOR_TOP_LEFT_X = 0.455
CORRIDOR_TOP_RIGHT_X = 0.545
CORRIDOR_HORIZON_Y = 0.45

# Ego bodywork (bonnet, mirror stalk, fender) is only discounted when it is BOTH
# anchored at the bottom of the frame AND spanning most of its width. A vehicle
# two metres ahead also touches the bottom edge, and must never be masked away.
EGO_MASK_BOTTOM_Y = 0.95
EGO_MASK_MIN_WIDTH_FRAC = 0.75

# --- Verdict thresholds -----------------------------------------------------
# gap_ratio is free-gap width divided by median obstacle width at the same
# depth band, so it is scale invariant and needs no camera calibration.
#
# These are fitted, not guessed: scripts/calibrate.py swept threshold pairs
# against hand-labelled frames and 1.8 / 0.4 maximised agreement while still
# never calling a frame SAFE that a human called BLOCKED. Re-run that script
# whenever the gold set grows; the numbers below are only as good as it is.
#
# On the current 30-frame gold set (portrait ego-POV footage) the sweep found a
# plateau: every pair with safe_ratio in 1.0-1.8 scored the same, so 1.8 was kept
# as the more conservative end of that plateau rather than being re-tuned to a
# value the data cannot distinguish.

VERDICT_SAFE_RATIO = 1.8
VERDICT_TIGHT_RATIO = 0.4
# Closing fast enough that any gap will be gone soon: drop a severity level.
TTC_DOWNGRADE_SECONDS = 1.5
HISTORY_LEN = 5
CLOSING_EPSILON_M_PER_FRAME = 0.02

# Temporal stabilisation
MEDIAN_FILTER_SECONDS = 2.0
HYSTERESIS_SECONDS = 0.5
MIN_CONFIDENT_DETECTIONS = 1

VERDICT_SAFE = "SAFE"
VERDICT_TIGHT = "TIGHT"
VERDICT_BLOCKED = "BLOCKED"
VERDICT_UNKNOWN = "UNKNOWN"
VERDICT_ORDER = (VERDICT_BLOCKED, VERDICT_TIGHT, VERDICT_SAFE)
