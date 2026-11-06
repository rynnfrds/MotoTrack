"""Verdict tests: thresholds, asymmetric caution, and temporal stability."""

from __future__ import annotations

import pytest

from mototrack import config
from mototrack.geometry import CorridorGap
from mototrack.verdict import (
    ClosingRateTracker,
    VerdictStabilizer,
    classify_ratio,
    decide,
    downgrade,
    upgrade,
)


def gap(ratio, n_candidates=1, distance=6.0, band="mid", gap_m=2.0):
    px = ratio * 100.0
    return CorridorGap(
        gap_px=px,
        corridor_width_px=px,
        reference_width_px=100.0,
        scanline_y_px=600.0,
        band=band,
        near_distance_m=distance,
        blocking_ids=(1,) if n_candidates else (),
        n_candidates=n_candidates,
        gap_m=gap_m,
    )


# --- thresholds ------------------------------------------------------------


def test_ratio_at_safe_boundary_is_safe():
    assert classify_ratio(config.VERDICT_SAFE_RATIO) == config.VERDICT_SAFE


def test_ratio_just_under_safe_boundary_is_tight():
    assert classify_ratio(config.VERDICT_SAFE_RATIO - 0.01) == config.VERDICT_TIGHT


def test_ratio_at_tight_boundary_is_tight():
    assert classify_ratio(config.VERDICT_TIGHT_RATIO) == config.VERDICT_TIGHT


def test_shipped_thresholds_were_calibrated_not_defaulted():
    # Guard against silently reverting to the old un-fitted guesses.
    assert config.VERDICT_SAFE_RATIO == 1.8
    assert config.VERDICT_TIGHT_RATIO == 0.4
    assert config.VERDICT_TIGHT_RATIO < config.VERDICT_SAFE_RATIO


def test_ratio_under_tight_boundary_is_blocked():
    assert classify_ratio(config.VERDICT_TIGHT_RATIO - 0.01) == config.VERDICT_BLOCKED


def test_wide_open_gap_is_safe():
    assert classify_ratio(4.0) == config.VERDICT_SAFE


# --- severity ordering -----------------------------------------------------


def test_downgrade_walks_toward_blocked():
    assert downgrade(config.VERDICT_SAFE) == config.VERDICT_TIGHT
    assert downgrade(config.VERDICT_TIGHT) == config.VERDICT_BLOCKED
    assert downgrade(config.VERDICT_BLOCKED) == config.VERDICT_BLOCKED
    assert downgrade(config.VERDICT_UNKNOWN) == config.VERDICT_UNKNOWN


def test_upgrade_walks_toward_safe():
    assert upgrade(config.VERDICT_BLOCKED) == config.VERDICT_TIGHT
    assert upgrade(config.VERDICT_TIGHT) == config.VERDICT_SAFE
    assert upgrade(config.VERDICT_SAFE) == config.VERDICT_SAFE


# --- decision composition --------------------------------------------------


def test_empty_frame_with_no_detections_is_unknown_not_safe():
    v = decide(
        gap(0.0, n_candidates=0),
        detection_count=0,
        closing_fast=False,
        center_covered=False,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_UNKNOWN
    assert v.reason == "low_detections"


def test_unconfirmed_empty_corridor_is_unknown():
    v = decide(
        gap(0.0, n_candidates=0),
        detection_count=4,
        closing_fast=False,
        center_covered=False,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_UNKNOWN
    assert v.reason == "no_obstacle_unconfirmed"


def test_confirmed_empty_corridor_is_safe():
    v = decide(
        gap(0.0, n_candidates=0),
        detection_count=4,
        closing_fast=False,
        center_covered=False,
        empty_corridor_confirmed=True,
    )
    assert v.verdict == config.VERDICT_SAFE
    assert v.reason == "corridor_clear"


def test_wide_gap_far_away_is_safe():
    v = decide(
        gap(2.5, distance=12.0, band="far"),
        detection_count=3,
        closing_fast=False,
        center_covered=False,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_SAFE


def test_near_band_obstacle_never_reports_safe():
    v = decide(
        gap(3.0, distance=2.0, band="near"),
        detection_count=3,
        closing_fast=False,
        center_covered=False,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_TIGHT
    assert "obstacle_in_near_band" in v.reason


def test_center_covered_downgrades_safe():
    v = decide(
        gap(2.5, distance=9.0),
        detection_count=3,
        closing_fast=False,
        center_covered=True,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_TIGHT
    assert "center_covered" in v.reason


def test_closing_fast_downgrades_one_step():
    v = decide(
        gap(2.5, distance=9.0),
        detection_count=3,
        closing_fast=True,
        center_covered=False,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_TIGHT
    assert "closing_fast" in v.reason


def test_blocked_gap_stays_blocked():
    v = decide(
        gap(0.4),
        detection_count=3,
        closing_fast=True,
        center_covered=True,
        empty_corridor_confirmed=False,
    )
    assert v.verdict == config.VERDICT_BLOCKED


# --- closing rate ----------------------------------------------------------


def test_closing_speed_is_positive_when_approaching():
    t = ClosingRateTracker()
    for d in (10.0, 9.5, 9.0, 8.5):
        t.update(1, d)
    assert t.closing_speed(1) == pytest.approx(0.5)


def test_closing_speed_is_negative_when_receding():
    t = ClosingRateTracker()
    for d in (8.0, 9.0, 10.0):
        t.update(1, d)
    assert t.closing_speed(1) < 0


def test_single_sample_has_no_closing_speed():
    t = ClosingRateTracker()
    t.update(1, 5.0)
    assert t.closing_speed(1) is None


def test_rapid_approach_triggers_fast_closing():
    t = ClosingRateTracker()
    for d in (9.0, 6.0, 3.0):
        t.update(1, d)
    assert t.is_closing_fast(1, fps=10.0)


def test_gentle_approach_does_not_trigger_fast_closing():
    t = ClosingRateTracker()
    for d in (9.0, 8.95, 8.9, 8.85):
        t.update(1, d)
    assert not t.is_closing_fast(1, fps=10.0)


def test_tracker_ignores_none_and_infinite_distance():
    t = ClosingRateTracker()
    t.update(None, 5.0)
    t.update(2, float("inf"))
    assert t.closing_speed(2) is None


def test_prune_forgets_stale_tracks():
    t = ClosingRateTracker()
    for d in (5.0, 4.0):
        t.update(1, d)
        t.update(2, d)
    t.prune({1})
    assert t.closing_speed(1) is not None
    assert t.closing_speed(2) is None


# --- stabiliser ------------------------------------------------------------


def test_stabiliser_ignores_single_frame_glitch():
    s = VerdictStabilizer(window_seconds=1.0, hysteresis_seconds=0.2)
    for _ in range(10):
        out = s.update(config.VERDICT_SAFE, fps=10.0)
    assert out == config.VERDICT_SAFE
    # one blocked frame cannot flip a one-second window of SAFE
    out = s.update(config.VERDICT_BLOCKED, fps=10.0)
    assert out == config.VERDICT_SAFE


def test_stabiliser_follows_a_sustained_change():
    s = VerdictStabilizer(window_seconds=0.5, hysteresis_seconds=0.1)
    for _ in range(5):
        s.update(config.VERDICT_SAFE, fps=10.0)
    result = config.VERDICT_SAFE
    for _ in range(30):
        result = s.update(config.VERDICT_BLOCKED, fps=10.0)
    assert result == config.VERDICT_BLOCKED


def test_stabiliser_starts_unknown():
    s = VerdictStabilizer()
    assert s.current == config.VERDICT_UNKNOWN


def test_stabiliser_reset_clears_state():
    s = VerdictStabilizer(window_seconds=0.5, hysteresis_seconds=0.1)
    for _ in range(20):
        s.update(config.VERDICT_BLOCKED, fps=10.0)
    s.reset()
    assert s.current == config.VERDICT_UNKNOWN


def test_stabiliser_never_reports_safe_on_alternating_tight_blocked():
    s = VerdictStabilizer(window_seconds=1.0, hysteresis_seconds=0.1)
    for i in range(60):
        raw = config.VERDICT_TIGHT if i % 2 else config.VERDICT_BLOCKED
        out = s.update(raw, fps=10.0)
        assert out != config.VERDICT_SAFE
def test_unknown_does_not_freeze_a_sustained_safe():
    s = VerdictStabilizer(window_seconds=2.0, hysteresis_seconds=0.5)
    # a burst of UNKNOWN while the detector loses track ...
    for _ in range(10):
        s.update(config.VERDICT_UNKNOWN, fps=10.0)
    # ... must not delay a genuinely clear corridor for a whole window
    out = config.VERDICT_UNKNOWN
    for _ in range(15):
        out = s.update(config.VERDICT_SAFE, fps=10.0)
    assert out == config.VERDICT_SAFE


def test_stabiliser_reports_unknown_when_it_has_no_opinion():
    s = VerdictStabilizer(window_seconds=0.5, hysteresis_seconds=0.1)
    for _ in range(20):
        out = s.update(config.VERDICT_UNKNOWN, fps=10.0)
    assert out == config.VERDICT_UNKNOWN