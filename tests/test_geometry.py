"""Geometry tests: every assertion is against hand-computed pixels."""

from __future__ import annotations

import math

import pytest

from mototrack import config
from mototrack.detectors import Detection
from mototrack.geometry import (
    BAND_FAR,
    BAND_MID,
    BAND_NEAR,
    compute_gap,
    corridor_center_covered,
    corridor_polygon_relative,
    corridor_x_bounds,
    distance_band,
    estimate_distance_m,
    focal_length_px,
    free_gaps,
    is_blocking_candidate,
    is_ego_bodywork,
    merge_intervals,
    point_in_corridor,
)

W, H = 1280, 720


def det(x1, y1, x2, y2, cls="car", track_id=1, conf=0.9):
    return Detection(
        track_id=track_id,
        cls_name=cls,
        cls_id=2,
        conf=conf,
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
    )


# --- pinhole ---------------------------------------------------------------


def test_focal_length_matches_half_width_over_tan():
    expected = (W / 2.0) / math.tan(math.radians(78.0) / 2.0)
    assert focal_length_px(W) == pytest.approx(expected)


def test_focal_length_rejects_degenerate_input():
    with pytest.raises(ValueError):
        focal_length_px(0)
    with pytest.raises(ValueError):
        focal_length_px(W, hfov_deg=0)


def test_distance_is_inverse_in_box_height():
    near = estimate_distance_m("car", 200, W)
    far = estimate_distance_m("car", 100, W)
    assert far == pytest.approx(near * 2)


def test_distance_uses_per_class_height():
    bus = estimate_distance_m("bus", 100, W)
    car = estimate_distance_m("car", 100, W)
    assert bus > car


def test_distance_of_degenerate_box_is_infinite():
    assert estimate_distance_m("car", 0, W) == math.inf


def test_distance_bands_split_at_configured_thresholds():
    assert distance_band(2.9) == BAND_NEAR
    assert distance_band(3.0) == BAND_MID
    assert distance_band(7.9) == BAND_MID
    assert distance_band(8.0) == BAND_FAR


# --- corridor --------------------------------------------------------------


def test_corridor_is_widest_at_bottom():
    bottom = corridor_x_bounds(1.0)
    top = corridor_x_bounds(config.CORRIDOR_HORIZON_Y)
    assert bottom is not None and top is not None
    assert (bottom[1] - bottom[0]) > (top[1] - top[0])


def test_corridor_has_no_width_above_horizon():
    assert corridor_x_bounds(0.0) is None
    assert corridor_x_bounds(config.CORRIDOR_HORIZON_Y - 0.01) is None


def test_corridor_bottom_matches_configured_fractions():
    left, right = corridor_x_bounds(1.0)
    assert left == pytest.approx(config.CORRIDOR_BOTTOM_LEFT_X)
    assert right == pytest.approx(config.CORRIDOR_BOTTOM_RIGHT_X)


def test_corridor_polygon_has_four_points_clockwise():
    poly = corridor_polygon_relative()
    assert len(poly) == 4
    assert poly[0][1] == 1.0 and poly[2][1] == config.CORRIDOR_HORIZON_Y


def test_centre_is_inside_corridor_near_bottom():
    assert point_in_corridor(0.5, 0.99)


def test_far_left_edge_is_outside_corridor():
    assert not point_in_corridor(0.05, 0.99)


# --- blocking candidates ---------------------------------------------------


def test_object_on_road_ahead_blocks():
    # bottom-centre at x=0.5W, y=0.8H: inside the corridor, on the road
    assert is_blocking_candidate(det(600, 500, 680, 576), W, H)


def test_object_parked_at_frame_edge_does_not_block():
    assert not is_blocking_candidate(det(10, 500, 90, 576), W, H)


def test_ego_bodywork_is_ignored():
    # spans 90% of the frame width and is anchored at the bottom edge
    ego_box = det(64, 640, 1216, 706)
    assert is_ego_bodywork(ego_box, H, W)
    assert not is_blocking_candidate(ego_box, W, H)


def test_close_vehicle_at_bottom_edge_is_not_mistaken_for_bodywork():
    # This was a real bug: a car ~2 m ahead touches the bottom edge but only
    # spans 8.6% of frame width, so it must stay a blocking candidate.
    close_car = det(590, 130, 700, 710)
    assert not is_ego_bodywork(close_car, H, W)
    assert is_blocking_candidate(close_car, W, H)


def test_bodywork_test_needs_width_to_conclude():
    # Without a width the function must not claim a box is bodywork.
    assert not is_ego_bodywork(det(64, 640, 1216, 706), H)


def test_degenerate_box_does_not_block():
    assert not is_blocking_candidate(det(600, 500, 600, 576), W, H)


# --- interval algebra ------------------------------------------------------


def test_merge_intervals_collapses_overlap():
    assert merge_intervals([(0, 10), (5, 20)]) == [(0, 20)]


def test_merge_intervals_keeps_disjoint_separate():
    assert merge_intervals([(30, 40), (0, 10)]) == [(0, 10), (30, 40)]


def test_merge_intervals_drops_empty():
    assert merge_intervals([(5, 5)]) == []


def test_free_gaps_returns_whole_span_when_nothing_blocks():
    assert free_gaps([], 0, 100) == [(0, 100)]


def test_free_gaps_splits_around_one_block():
    assert free_gaps([(40, 60)], 0, 100) == [(0, 40), (60, 100)]


def test_free_gaps_returns_none_when_fully_covered():
    assert free_gaps([(0, 100)], 0, 100) == []


def test_free_gaps_clips_block_extending_past_span():
    assert free_gaps([(-50, 20)], 0, 100) == [(20, 100)]


# --- whole-corridor gap ----------------------------------------------------


def test_empty_corridor_gap_equals_full_corridor_width():
    gap = compute_gap([], W, H)
    expected = (config.CORRIDOR_BOTTOM_RIGHT_X - config.CORRIDOR_BOTTOM_LEFT_X) * W
    assert gap.gap_px == pytest.approx(expected)
    assert gap.n_candidates == 0
    assert gap.band is None


def test_centred_obstacle_leaves_two_symmetric_slivers():
    obstacle = det(590, 500, 690, 600)
    gap = compute_gap([obstacle], W, H)
    assert gap.n_candidates == 1
    left, right = corridor_x_bounds(obstacle.bottom_y / H)
    left_px, right_px = left * W, right * W
    assert gap.gap_px == pytest.approx(max(590 - left_px, right_px - 690))


def test_obstacle_on_shoulder_outside_corridor_is_not_counted():
    gap = compute_gap([det(20, 500, 120, 600)], W, H)
    assert gap.n_candidates == 0


def test_nearest_band_wins_over_far_objects():
    # A 500 px tall car box at 78 deg HFOV on 1280 px works out to ~2.5 m,
    # while a 20 px tall one works out to ~63 m. Both sit in the corridor.
    near = det(600, 180, 700, 680, track_id=1)
    far = det(600, 380, 640, 400, track_id=2)
    gap = compute_gap([near, far], W, H)
    assert gap.band == BAND_NEAR
    assert 1 in gap.blocking_ids


def test_gap_ratio_is_reference_width_normalised():
    obstacle = det(600, 500, 700, 600)
    gap = compute_gap([obstacle], W, H)
    assert gap.reference_width_px == pytest.approx(100)
    assert gap.gap_ratio == pytest.approx(gap.gap_px / 100)


def test_gap_ratio_scales_with_frame_resolution():
    half = det(300, 250, 350, 300)
    full = det(600, 500, 700, 600)
    ratio_half = compute_gap([half], W / 2, H / 2).gap_ratio
    ratio_full = compute_gap([full], W, H).gap_ratio
    assert ratio_half == pytest.approx(ratio_full, rel=1e-6)


def test_gap_metres_are_reported_but_flagged_optional():
    gap = compute_gap([det(600, 500, 700, 600)], W, H)
    assert gap.gap_m is not None
    assert gap.gap_m > 0


def test_center_covered_detects_vehicle_directly_ahead():
    # near band (<3 m) is ~421 px tall for a car at this resolution
    assert corridor_center_covered([det(520, 180, 760, 680)], W, H)


def test_center_not_covered_when_vehicle_is_off_to_the_side():
    assert not corridor_center_covered([det(320, 500, 420, 620)], W, H)


def test_compute_gap_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        compute_gap([], 0, H)
def test_reference_width_ignores_detections_outside_the_band():
    # Two near boxes 100 px wide, plus one far 20 px box that must not shrink
    # the divisor used to normalise the near-band gap.
    near_a = det(200, 200, 300, 680, track_id=1)
    near_b = det(500, 200, 600, 680, track_id=2)
    tiny_far = det(620, 300, 640, 340, track_id=3)
    gap = compute_gap([near_a, near_b, tiny_far], W, H)
    assert gap.reference_width_px == pytest.approx(100)
