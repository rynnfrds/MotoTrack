"""Monocular geometry for the ego-corridor overtaking decision.

Everything here is a pure function of pixel coordinates: no camera calibration,
no video I/O, no model. That keeps the assumptions testable in isolation.

The corridor is a trapezoid in *relative* frame coordinates, so portrait 9:16
reels and 16:9 clips share one code path.

What this is NOT: a depth sensor. Distance comes from assumed real-world object
heights and an assumed horizontal field of view, combined through the pinhole
similar-triangles relation. Results are reported as bands, never as authoritative
metre readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from . import config
from .detectors import Detection

BAND_NEAR = "near"
BAND_MID = "mid"
BAND_FAR = "far"
BAND_ORDER = {BAND_NEAR: 0, BAND_MID: 1, BAND_FAR: 2}


# --- Pinhole distance -------------------------------------------------------


def focal_length_px(
    image_width_px: float,
    hfov_deg: float = config.ASSUMED_HFOV_DEG,
) -> float:
    """Focal length in pixels from an assumed horizontal field of view."""
    if image_width_px <= 0:
        raise ValueError("image_width_px must be positive")
    if not 0 < hfov_deg < 180:
        raise ValueError("hfov_deg must be between 0 and 180")
    return (image_width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def estimate_distance_m(
    cls_name: str,
    bbox_height_px: float,
    image_width_px: float,
    hfov_deg: float = config.ASSUMED_HFOV_DEG,
) -> float:
    """Rough range to an object from its pixel height.

    Returns math.inf when the box is degenerate. Callers must treat the result
    as a ranking signal inside a band, not as a measurement.
    """
    if bbox_height_px <= 1e-6:
        return math.inf
    real_height_m = config.ASSUMED_HEIGHT_M.get(
        cls_name, config.DEFAULT_ASSUMED_HEIGHT_M
    )
    return (real_height_m * focal_length_px(image_width_px, hfov_deg)) / bbox_height_px


def distance_band(distance_m: float) -> str:
    if distance_m < config.NEAR_BUCKET_MAX_M:
        return BAND_NEAR
    if distance_m < config.MID_BUCKET_MAX_M:
        return BAND_MID
    return BAND_FAR


# --- Ego corridor -----------------------------------------------------------


def corridor_x_bounds(
    y_frac: float,
    horizon_y: float = config.CORRIDOR_HORIZON_Y,
    bottom_left: float = config.CORRIDOR_BOTTOM_LEFT_X,
    bottom_right: float = config.CORRIDOR_BOTTOM_RIGHT_X,
    top_left: float = config.CORRIDOR_TOP_LEFT_X,
    top_right: float = config.CORRIDOR_TOP_RIGHT_X,
) -> tuple[float, float] | None:
    """Corridor edges as width fractions at vertical position y_frac.

    Returns None above the horizon: there is no drivable corridor in the sky.
    """
    if not 0.0 <= y_frac <= 1.0:
        return None
    if y_frac < horizon_y:
        return None
    span = 1.0 - horizon_y
    t = 1.0 if span <= 0 else (y_frac - horizon_y) / span
    t = min(max(t, 0.0), 1.0)
    left = top_left + t * (bottom_left - top_left)
    right = top_right + t * (bottom_right - top_right)
    return (left, right)


def corridor_polygon_relative(
    horizon_y: float = config.CORRIDOR_HORIZON_Y,
    bottom_left: float = config.CORRIDOR_BOTTOM_LEFT_X,
    bottom_right: float = config.CORRIDOR_BOTTOM_RIGHT_X,
    top_left: float = config.CORRIDOR_TOP_LEFT_X,
    top_right: float = config.CORRIDOR_TOP_RIGHT_X,
) -> tuple[tuple[float, float], ...]:
    """Four relative-coordinate points, clockwise from the bottom-left corner."""
    return (
        (bottom_left, 1.0),
        (bottom_right, 1.0),
        (top_right, horizon_y),
        (top_left, horizon_y),
    )


def point_in_corridor(
    x_frac: float,
    y_frac: float,
    horizon_y: float = config.CORRIDOR_HORIZON_Y,
) -> bool:
    bounds = corridor_x_bounds(y_frac, horizon_y=horizon_y)
    if bounds is None:
        return False
    left, right = bounds
    return left <= x_frac <= right


# --- Per-detection tests ----------------------------------------------------


def is_ego_bodywork(
    det: Detection,
    image_height_px: float,
    image_width_px: float | None = None,
) -> bool:
    """True only for frame-filling bodywork, not for a vehicle close ahead.

    A genuine bonnet, mirror stalk or fender is anchored to the bottom edge AND
    stretches across most of the frame width. A car two metres ahead also
    touches the bottom edge, so bottom position alone cannot be the test: that
    would hide exactly the obstacle that matters most.
    """
    if image_height_px <= 0:
        return False
    if (det.bottom_y / image_height_px) <= config.EGO_MASK_BOTTOM_Y:
        return False
    if image_width_px is None or image_width_px <= 0:
        return False
    return det.width >= config.EGO_MASK_MIN_WIDTH_FRAC * image_width_px


def is_blocking_candidate(det: Detection, image_width_px: float, image_height_px: float) -> bool:
    """Ground-contact test: does this object sit on the road ahead of me?

    Uses the bottom-centre of the box, not the centre, because the centre of a
    far-away box drifts upward while its wheels stay on the road.
    """
    if image_width_px <= 0 or image_height_px <= 0:
        return False
    if det.width <= 0 or det.height <= 0:
        return False
    x_frac = det.bottom_center_x / image_width_px
    y_frac = det.bottom_y / image_height_px
    if is_ego_bodywork(det, image_height_px, image_width_px):
        return False
    return point_in_corridor(x_frac, y_frac)


# --- Free gap ---------------------------------------------------------------


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or touching 1-D intervals, sorted by start."""
    cleaned = [(min(a, b), max(a, b)) for a, b in intervals if max(a, b) > min(a, b)]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def free_gaps(
    blocked: list[tuple[float, float]],
    span_start: float,
    span_end: float,
) -> list[tuple[float, float]]:
    """Complement of the blocked intervals inside [span_start, span_end]."""
    if span_end <= span_start:
        return []
    gaps: list[tuple[float, float]] = []
    cursor = span_start
    for start, end in merge_intervals(blocked):
        if end <= span_start or start >= span_end:
            continue
        if start > cursor:
            gaps.append((cursor, min(start, span_end)))
        cursor = max(cursor, end)
        if cursor >= span_end:
            break
    if cursor < span_end:
        gaps.append((cursor, span_end))
    return [(a, b) for a, b in gaps if b > a]


@dataclass(frozen=True)
class CorridorGap:
    """Free lateral space in the ego corridor at the nearest obstacle band."""

    gap_px: float
    corridor_width_px: float
    reference_width_px: float
    scanline_y_px: float
    band: str | None
    near_distance_m: float | None
    blocking_ids: tuple[int | None, ...]
    n_candidates: int
    gap_m: float | None

    @property
    def gap_ratio(self) -> float:
        """Free gap measured in median-vehicle-widths. Scale invariant."""
        if self.reference_width_px <= 0:
            return math.inf if self.gap_px > 0 else 0.0
        return self.gap_px / self.reference_width_px


def compute_gap(
    detections: list[Detection],
    image_width_px: float,
    image_height_px: float,
) -> CorridorGap:
    """Measure free lateral space in the corridor at the nearest band present.

    With nothing in the corridor the whole corridor counts as free, but the
    ratio is only meaningful if some object exists in frame to set the scale.
    """
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("frame dimensions must be positive")

    bottom_bounds = corridor_x_bounds(1.0) or (
        config.CORRIDOR_BOTTOM_LEFT_X,
        config.CORRIDOR_BOTTOM_RIGHT_X,
    )
    widest_corridor_px = (bottom_bounds[1] - bottom_bounds[0]) * image_width_px

    all_widths = [d.width for d in detections if d.width > 0]

    candidates = [
        d for d in detections if is_blocking_candidate(d, image_width_px, image_height_px)
    ]
    if not candidates:
        reference_width_px = median(all_widths) if all_widths else 0.0
        return CorridorGap(
            gap_px=widest_corridor_px,
            corridor_width_px=widest_corridor_px,
            reference_width_px=reference_width_px,
            scanline_y_px=float(image_height_px),
            band=None,
            near_distance_m=None,
            blocking_ids=(),
            n_candidates=0,
            gap_m=None,
        )

    distances = {
        id(d): estimate_distance_m(d.cls_name, d.height, image_width_px) for d in candidates
    }
    bands = {id(d): distance_band(distances[id(d)]) for d in candidates}
    nearest_band = min((bands[id(d)] for d in candidates), key=lambda b: BAND_ORDER[b])
    band_members = [d for d in candidates if bands[id(d)] == nearest_band]

    scanline_y_px = median([d.bottom_y for d in band_members])
    y_frac = scanline_y_px / image_height_px
    bounds = corridor_x_bounds(y_frac)
    if bounds is None:
        # Occupants crowded above the horizon; treat the corridor bottom as the
        # reference scanline rather than inventing one off-frame.
        y_frac = 1.0
        scanline_y_px = float(image_height_px)
        bounds = bottom_bounds

    span_start = bounds[0] * image_width_px
    span_end = bounds[1] * image_width_px
    blocked = [(d.x1, d.x2) for d in band_members]
    gaps = free_gaps(blocked, span_start, span_end)
    gap_px = max((end - start for start, end in gaps), default=0.0)

    # Scale reference must come from the same depth band as the gap. Using every
    # detection in frame lets a distant 30 px box shrink the divisor and inflate
    # the ratio, which reads as a wider gap than actually exists down the road.
    band_widths = [d.width for d in band_members if d.width > 0]
    reference_width_px = median(band_widths) if band_widths else (
        median(all_widths) if all_widths else 0.0
    )

    near_distance = median([distances[id(d)] for d in band_members])
    focal = focal_length_px(image_width_px)
    gap_m = (gap_px * near_distance / focal) if focal > 0 and math.isfinite(near_distance) else None

    return CorridorGap(
        gap_px=gap_px,
        corridor_width_px=span_end - span_start,
        reference_width_px=reference_width_px,
        scanline_y_px=scanline_y_px,
        band=nearest_band,
        near_distance_m=near_distance if math.isfinite(near_distance) else None,
        blocking_ids=tuple(d.track_id for d in band_members),
        n_candidates=len(candidates),
        gap_m=gap_m,
    )


def corridor_center_covered(
    detections: list[Detection],
    image_width_px: float,
    image_height_px: float,
) -> bool:
    """True when a near-band object sits on the corridor centre line.

    A vehicle immediately ahead closes the line regardless of how much space
    opens on either side of it: riding through that gap is a different, riskier
    manoeuvre than overtaking a single slow vehicle.
    """
    for det in detections:
        if not is_blocking_candidate(det, image_width_px, image_height_px):
            continue
        distance = estimate_distance_m(det.cls_name, det.height, image_width_px)
        if distance_band(distance) != BAND_NEAR:
            continue
        y_frac = det.bottom_y / image_height_px
        bounds = corridor_x_bounds(y_frac)
        if bounds is None:
            continue
        center_frac = (bounds[0] + bounds[1]) / 2.0
        center_px = center_frac * image_width_px
        if det.x1 <= center_px <= det.x2:
            return True
    return False
