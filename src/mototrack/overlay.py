"""Draw the corridor, obstacle boxes and verdict HUD onto a frame."""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import config
from .detectors import Detection
from .geometry import CorridorGap, compute_gap, corridor_polygon_relative, is_blocking_candidate

VERDICT_COLORS = {
    config.VERDICT_SAFE: (80, 190, 80),
    config.VERDICT_TIGHT: (40, 180, 230),
    config.VERDICT_BLOCKED: (60, 60, 220),
    config.VERDICT_UNKNOWN: (170, 170, 170),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(frame, text, org, scale, color, thickness=2):
    """Legible text: a dark halo behind the coloured glyphs.

    Only for text drawn straight onto footage, where the background is whatever
    the camera saw. HUD text sits on the dark panel and must not use this: the
    halo pass and the glyph pass land on different pixels at small scales, which
    prints every line twice with a visible ghost one glyph to the right.
    """
    halo = thickness + 1 if scale < 0.7 else thickness + 2
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), halo, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _panel_text(frame, text, org, scale, color, thickness=1):
    """Text on the HUD panel. One pass; the panel already supplies contrast."""
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_corridor(frame: np.ndarray, verdict: str, alpha: float = 0.18) -> None:
    """Tint the ego corridor with the current verdict colour."""
    height, width = frame.shape[:2]
    polygon = np.array(
        [[(int(x * width), int(y * height)) for x, y in corridor_polygon_relative()]],
        dtype=np.int32,
    )
    color = VERDICT_COLORS.get(verdict, VERDICT_COLORS[config.VERDICT_UNKNOWN])
    shade = frame.copy()
    cv2.fillPoly(shade, polygon, color)
    cv2.addWeighted(shade, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
    cv2.polylines(frame, polygon, True, color, 2, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    blocking_ids: tuple[int | None, ...] = (),
) -> None:
    height, width = frame.shape[:2]
    for det in detections:
        blocking = det.track_id in blocking_ids
        color = (60, 60, 220) if blocking else (120, 190, 120)
        thickness = 3 if blocking else 2
        pt1 = (int(det.x1), int(det.y1))
        pt2 = (int(det.x2), int(det.y2))
        if getattr(det, "predicted", False):
            # Coasted through a detector dropout: draw dashed and dimmer so a
            # viewer can tell an inferred box from an observed one.
            _dashed_rect(frame, pt1, pt2, color, thickness)
            label = "~" + (det.cls_name if det.track_id is None else f"{det.cls_name} #{det.track_id}")
        else:
            cv2.rectangle(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
            label = det.cls_name if det.track_id is None else f"{det.cls_name} #{det.track_id}"
        _text(frame, label, (int(det.x1), max(14, int(det.y1) - 6)), 0.45, color, 1)
        _ = height, width


def _dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 10,
) -> None:
    """Rectangle drawn as dashes, used for boxes the detector did not see."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def _fit_scale(
    text: str,
    max_width_px: float,
    base_scale: float,
    thickness: int = 1,
) -> float:
    """Largest font scale at or below base_scale whose text fits max_width_px.

    Portrait frames are only ~720 px wide, so a panel sized as a fraction of
    width leaves less room than landscape footage did. Scaling down per line is
    cheaper than truncating the evidence string, and it keeps the HUD readable
    instead of letting it run off the panel and collide with the box labels.
    """
    if not text or max_width_px <= 0:
        return base_scale
    scale = base_scale
    for _ in range(24):
        (width_px, _height_px), _baseline = cv2.getTextSize(
            text, FONT, scale, thickness
        )
        if width_px <= max_width_px or scale <= 0.22:
            break
        scale *= 0.92
    return scale


def draw_hud(
    frame: np.ndarray,
    verdict: str,
    gap: CorridorGap,
    reason: str,
    t_seconds: float | None = None,
    raw_verdict: str | None = None,
) -> None:
    """Top-left panel: the smoothed verdict, then this frame's own evidence.

    verdict is what the stabiliser decided to show; raw_verdict is what this
    single frame saw. They disagree near transitions by design, so the evidence
    line is labelled instead of being presented as the verdict.
    """
    height, width = frame.shape[:2]
    scale = max(0.5, height / 900.0)
    color = VERDICT_COLORS.get(verdict, VERDICT_COLORS[config.VERDICT_UNKNOWN])

    # Panel spans the frame width on narrow (portrait) footage: 62% of 720 px is
    # not enough for the evidence line at any readable size.
    panel_w = min(int(width * (0.98 if height > width else 0.62)), 560)
    pad = int(10 * scale)
    inner = panel_w - 2 * pad

    lines: list[tuple[str, float, int, tuple]] = [
        (verdict, 0.95 * scale, 2, color),
    ]
    if gap.n_candidates == 0:
        detail = "this frame: corridor empty"
    elif not math.isfinite(gap.gap_ratio):
        detail = "this frame: gap unbounded"
    else:
        detail = f"this frame: gap {gap.gap_ratio:.2f} vehicle-widths"
    lines.append((detail, 0.5 * scale, 1, (240, 240, 240)))

    if gap.gap_m is not None:
        caveat = f"~{gap.gap_m:.1f} m estimated, not a sensor"
    else:
        caveat = "range estimate unavailable"
    lines.append((caveat, 0.42 * scale, 1, (185, 185, 185)))

    if raw_verdict and raw_verdict != verdict:
        stamp = f"holding {verdict} (frame reads {raw_verdict})"
    else:
        stamp = reason or ""
    if t_seconds is not None:
        stamp = f"t={t_seconds:5.1f}s  {stamp}".rstrip()
    lines.append((stamp, 0.38 * scale, 1, (150, 150, 150)))

    # Measure with the fitted scale so the panel height follows the real layout
    # instead of a fixed 96 px that overlapped lines on portrait frames.
    laid_out = []
    y = 0
    for text, base, thickness, colour in lines:
        if not text:
            continue
        fitted = _fit_scale(text, inner, base, thickness)
        (_tw, th), _bl = cv2.getTextSize(text, FONT, fitted, thickness)
        # 1.75x the glyph height leaves a clear gap between baselines; the old
        # fixed 96 px panel held four lines only at landscape sizes.
        y += int(th * 1.75) + int(6 * scale)
        laid_out.append((text, fitted, thickness, colour, y))

    panel_h = y + int(10 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (18, 18, 18), -1)
    # Near-opaque: on portrait footage the panel spans most of the frame width,
    # and at 0.62 alpha the box labels drawn earlier bled through behind the HUD
    # text, which read as extra stray glyphs rather than as background.
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0.0, dst=frame)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), color, 2)

    for text, fitted, thickness, colour, y in laid_out:
        _panel_text(frame, text, (pad, y), fitted, colour, thickness)


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    verdict: str,
    reason: str = "",
    gap: CorridorGap | None = None,
    t_seconds: float | None = None,
    raw_verdict: str | None = None,
) -> np.ndarray:
    """Return a new annotated copy; the input frame is left untouched."""
    out = frame.copy()
    height, width = out.shape[:2]
    if gap is None:
        gap = compute_gap(detections, width, height)
    draw_corridor(out, verdict)
    draw_detections(out, detections, gap.blocking_ids)
    draw_hud(out, verdict, gap, reason, t_seconds, raw_verdict=raw_verdict)
    return out


def draw_timeline(
    segments: list,
    total_seconds: float,
    out_path: str,
    width: int = 1200,
    height: int = 220,
) -> str:
    """Render verdict-over-time as a coloured strip with a time axis."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    margin_l, margin_r = 60, 40
    bar_top, bar_h = 60, 60
    plot_w = width - margin_l - margin_r

    cv2.putText(canvas, "Overtaking verdict over time", (margin_l, 34), FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    if total_seconds <= 0:
        total_seconds = 1.0
    for seg in segments:
        x0 = margin_l + int(plot_w * (seg.t0 / total_seconds))
        x1 = margin_l + int(plot_w * (seg.t1 / total_seconds))
        color = VERDICT_COLORS.get(seg.verdict, VERDICT_COLORS[config.VERDICT_UNKNOWN])
        cv2.rectangle(canvas, (x0, bar_top), (max(x0 + 1, x1), bar_top + bar_h), color, -1)

    cv2.rectangle(
        canvas,
        (margin_l, bar_top),
        (margin_l + plot_w, bar_top + bar_h),
        (120, 120, 120),
        1,
    )

    for tick in range(0, int(total_seconds) + 1, max(1, int(total_seconds // 10) or 1)):
        x = margin_l + int(plot_w * (tick / total_seconds))
        cv2.line(canvas, (x, bar_top + bar_h), (x, bar_top + bar_h + 6), (120, 120, 120), 1)
        cv2.putText(canvas, f"{tick}s", (x - 10, bar_top + bar_h + 24), FONT, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

    legend_y = bar_top + bar_h + 60
    x_cursor = margin_l
    for verdict in (config.VERDICT_SAFE, config.VERDICT_TIGHT, config.VERDICT_BLOCKED, config.VERDICT_UNKNOWN):
        color = VERDICT_COLORS[verdict]
        cv2.rectangle(canvas, (x_cursor, legend_y - 12), (x_cursor + 18, legend_y + 2), color, -1)
        cv2.putText(canvas, verdict, (x_cursor + 26, legend_y + 2), FONT, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
        x_cursor += 150

    cv2.imwrite(out_path, canvas)
    return out_path