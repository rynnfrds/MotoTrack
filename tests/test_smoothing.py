"""Tests for temporal box smoothing and track coasting."""

from __future__ import annotations

import pytest

from mototrack import config
from mototrack.detectors import Detection
from mototrack.smoothing import DetectionSmoother


def box(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=300.0, conf=0.9):
    return Detection(
        track_id=track_id,
        cls_name="2-wheeler",
        cls_id=1,
        conf=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def test_first_observation_is_passed_through_unchanged():
    smoother = DetectionSmoother()
    out = smoother.update([box()], 0.0)
    assert len(out) == 1
    assert out[0].x1 == 100.0
    assert out[0].predicted is False


def test_jitter_is_reduced_but_not_frozen():
    smoother = DetectionSmoother(alpha=0.5)
    smoother.update([box(x1=100)], 0.0)
    out = smoother.update([box(x1=200)], 0.1)
    # Halfway between the two observations, not stuck at the old value.
    assert 100.0 < out[0].x1 < 200.0
    assert out[0].x1 == pytest.approx(150.0)


def test_missed_frame_coasts_instead_of_vanishing():
    smoother = DetectionSmoother(hold_seconds=0.5)
    smoother.update([box()], 0.0)
    out = smoother.update([], 0.1)
    assert len(out) == 1, "a one-sample dropout must not erase the box"
    assert out[0].predicted is True
    assert out[0].track_id == 1


def test_coasted_box_fades_then_expires():
    smoother = DetectionSmoother(hold_seconds=0.3, conf_decay=0.5)
    smoother.update([box(conf=0.9)], 0.0)
    first = smoother.update([], 0.1)[0]
    second = smoother.update([], 0.2)[0]
    assert second.conf < first.conf, "coasted confidence should decay"
    assert smoother.update([], 0.35) == [], "past the hold window it must expire"


def test_returning_track_clears_predicted_flag():
    smoother = DetectionSmoother(hold_seconds=0.5)
    smoother.update([box()], 0.0)
    assert smoother.update([], 0.1)[0].predicted is True
    back = smoother.update([box()], 0.2)[0]
    assert back.predicted is False


def test_different_tracks_do_not_mix():
    smoother = DetectionSmoother()
    smoother.update([box(track_id=1, x1=100), box(track_id=2, x1=500)], 0.0)
    out = smoother.update([box(track_id=2, x1=520)], 0.1)
    by_id = {d.track_id: d for d in out}
    assert by_id[1].predicted is True, "track 1 was missed and should coast"
    assert by_id[2].predicted is False, "track 2 was observed"


def test_untracked_detections_are_never_coasted():
    smoother = DetectionSmoother()
    untracked = Detection(None, "car", 2, 0.8, 10, 10, 50, 50)
    out = smoother.update([untracked], 0.0)
    assert len(out) == 1
    assert out[0].track_id is None


def test_hold_zero_disables_coasting():
    smoother = DetectionSmoother(hold_seconds=0.0)
    smoother.update([box()], 0.0)
    assert smoother.update([], 0.1) == []


def test_reset_forgets_everything():
    smoother = DetectionSmoother(hold_seconds=5.0)
    smoother.update([box()], 0.0)
    smoother.reset()
    assert smoother.update([], 0.1) == []


def test_defaults_come_from_config():
    smoother = DetectionSmoother()
    assert smoother.alpha == config.BOX_SMOOTHING_ALPHA
    assert smoother.hold_seconds == config.TRACK_HOLD_SECONDS


def test_alpha_is_clamped_to_a_usable_range():
    assert DetectionSmoother(alpha=0.0).alpha > 0.0
    assert DetectionSmoother(alpha=5.0).alpha <= 1.0
