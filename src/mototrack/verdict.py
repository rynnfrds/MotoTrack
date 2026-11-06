"""Turn per-frame geometry into a stable overtaking verdict.

Severity is asymmetric on purpose: a false SAFE lets a rider commit to a lane
change they cannot abort, while a false BLOCKED only costs time. Every smoothing
rule below therefore fails toward the more cautious answer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from . import config
from .geometry import CorridorGap

SEVERITY = {
    config.VERDICT_SAFE: 0,
    config.VERDICT_TIGHT: 1,
    config.VERDICT_BLOCKED: 2,
    config.VERDICT_UNKNOWN: 3,
}


def classify_ratio(
    gap_ratio: float,
    safe_ratio: float = config.VERDICT_SAFE_RATIO,
    tight_ratio: float = config.VERDICT_TIGHT_RATIO,
) -> str:
    """Map a calibraton-free gap ratio onto a severity class."""
    if gap_ratio >= safe_ratio:
        return config.VERDICT_SAFE
    if gap_ratio >= tight_ratio:
        return config.VERDICT_TIGHT
    return config.VERDICT_BLOCKED


def downgrade(verdict: str) -> str:
    """One step more cautious. UNKNOWN stays UNKNOWN: it is already the floor."""
    if verdict == config.VERDICT_SAFE:
        return config.VERDICT_TIGHT
    if verdict == config.VERDICT_TIGHT:
        return config.VERDICT_BLOCKED
    return verdict


def upgrade(verdict: str) -> str:
    """One step less cautious, never past SAFE."""
    if verdict == config.VERDICT_BLOCKED:
        return config.VERDICT_TIGHT
    if verdict == config.VERDICT_TIGHT:
        return config.VERDICT_SAFE
    return verdict


@dataclass(frozen=True)
class FrameVerdict:
    """Raw, per-frame decision before temporal stabilisation."""

    verdict: str
    gap_ratio: float | None
    gap_px: float | None
    gap_m: float | None
    band: str | None
    blocking_ids: tuple[int | None, ...]
    reason: str


def decide(
    gap: CorridorGap,
    *,
    detection_count: int,
    closing_fast: bool,
    center_covered: bool,
    empty_corridor_confirmed: bool,
    min_detections: int = config.MIN_CONFIDENT_DETECTIONS,
) -> FrameVerdict:
    """Combine geometry, motion and detection confidence into one verdict.

    ``empty_corridor_confirmed`` says whether the corridor has stayed genuinely
    empty for long enough to trust; an empty corridor on a frame where the
    detector simply found nothing is UNKNOWN, not SAFE.
    """
    if detection_count < min_detections and gap.n_candidates == 0:
        return FrameVerdict(
            verdict=config.VERDICT_UNKNOWN,
            gap_ratio=None,
            gap_px=None,
            gap_m=None,
            band=None,
            blocking_ids=(),
            reason="low_detections",
        )

    if gap.n_candidates == 0:
        if not empty_corridor_confirmed:
            return FrameVerdict(
                verdict=config.VERDICT_UNKNOWN,
                gap_ratio=None,
                gap_px=gap.gap_px,
                gap_m=None,
                band=None,
                blocking_ids=(),
                reason="no_obstacle_unconfirmed",
            )
        return FrameVerdict(
            verdict=config.VERDICT_SAFE,
            gap_ratio=None,
            gap_px=gap.gap_px,
            gap_m=None,
            band=None,
            blocking_ids=(),
            reason="corridor_clear",
        )

    ratio = gap.gap_ratio
    verdict = classify_ratio(ratio)
    reasons = ["gap_ratio"]

    # A vehicle directly ahead closes the line no matter how much room opens
    # beside it: squeezing past is a different manoeuvre than overtaking.
    if center_covered and verdict == config.VERDICT_SAFE:
        verdict = config.VERDICT_TIGHT
        reasons.append("center_covered")

    if gap.near_distance_m is not None and gap.near_distance_m < config.NEAR_BUCKET_MAX_M:
        if verdict == config.VERDICT_SAFE:
            verdict = config.VERDICT_TIGHT
            reasons.append("obstacle_in_near_band")

    if closing_fast:
        new = downgrade(verdict)
        if new != verdict:
            verdict = new
            reasons.append("closing_fast")

    return FrameVerdict(
        verdict=verdict,
        gap_ratio=ratio,
        gap_px=gap.gap_px,
        gap_m=gap.gap_m,
        band=gap.band,
        blocking_ids=gap.blocking_ids,
        reason="+".join(reasons),
    )


@dataclass
class ClosingRateTracker:
    """Per-track distance history, used to detect rapid approach."""

    history_len: int = config.HISTORY_LEN
    epsilon: float = config.CLOSING_EPSILON_M_PER_FRAME
    _history: dict = field(default_factory=dict)

    def update(self, track_id: int | None, distance_m: float) -> None:
        if track_id is None or distance_m != distance_m:  # NaN check
            return
        if distance_m == float("inf"):
            return
        buf = self._history.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.history_len)
            self._history[track_id] = buf
        buf.append(distance_m)

    def closing_speed(self, track_id: int | None) -> float | None:
        """Metres per frame, positive when approaching."""
        if track_id is None:
            return None
        buf = self._history.get(track_id)
        if buf is None or len(buf) < 2:
            return None
        return (buf[0] - buf[-1]) / (len(buf) - 1)

    def is_closing_fast(
        self,
        track_id: int | None,
        fps: float,
        ttc_seconds: float = config.TTC_DOWNGRADE_SECONDS,
    ) -> bool:
        speed = self.closing_speed(track_id)
        if speed is None or speed <= self.epsilon:
            return False
        buf = self._history.get(track_id)
        if buf is None or not buf:
            return False
        distance = buf[-1]
        if distance == float("inf"):
            return False
        if fps <= 0:
            return False
        ttc = distance / (speed * fps)
        return ttc < ttc_seconds

    def prune(self, active_ids: set[int]) -> None:
        for track_id in list(self._history):
            if track_id not in active_ids:
                del self._history[track_id]


@dataclass
class VerdictStabilizer:
    """Median filter plus hysteresis over the per-frame verdict stream.

    The median removes single-frame detection glitches. Hysteresis stops the
    shown label flapping between neighbouring severities at a threshold.
    """

    window_seconds: float = config.MEDIAN_FILTER_SECONDS
    hysteresis_seconds: float = config.HYSTERESIS_SECONDS
    # None marks an UNKNOWN sample. UNKNOWN is "no opinion", not "worst case",
    # so it is excluded from the median: letting it vote would freeze the label
    # for a whole window after every moment the detector briefly lost track.
    _severity_window: deque = field(default_factory=deque)
    _window_seconds: deque = field(default_factory=deque)
    _current: str = config.VERDICT_UNKNOWN
    _candidate: str | None = None
    _candidate_frames: int = 0
    _current_frames: int = 0

    def reset(self) -> None:
        self._severity_window.clear()
        self._window_seconds.clear()
        self._current = config.VERDICT_UNKNOWN
        self._candidate = None
        self._candidate_frames = 0
        self._current_frames = 0

    def update(self, raw_verdict: str, fps: float) -> str:
        frame_duration = 1.0 / fps if fps > 0 else 0.05
        raw_severity = SEVERITY[raw_verdict]
        self._severity_window.append(
            None if raw_verdict == config.VERDICT_UNKNOWN else raw_severity
        )
        self._window_seconds.append(frame_duration)

        # Trim oldest samples until the window spans at most window_seconds.
        total = sum(self._window_seconds)
        while len(self._window_seconds) > 1 and total > self.window_seconds:
            total -= self._window_seconds.popleft()
            self._severity_window.popleft()

        known = [s for s in self._severity_window if s is not None]
        if not known:
            self._current = config.VERDICT_UNKNOWN
            self._candidate = None
            self._candidate_frames = 0
            return self._current

        smoothed = _severity_to_verdict(median(known))

        if smoothed == self._current:
            self._candidate = None
            self._candidate_frames = 0
            self._current_frames += 1
            return self._current

        if smoothed == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = smoothed
            self._candidate_frames = 1

        hold_frames = max(
            1,
            int(round(self.hysteresis_seconds / frame_duration)) if frame_duration > 0 else 1,
        )
        if self._candidate_frames >= hold_frames:
            self._current = self._candidate
            self._candidate = None
            self._candidate_frames = 0
            self._current_frames = 0
        return self._current

    @property
    def current(self) -> str:
        return self._current


def _severity_to_verdict(value: float) -> str:
    """Nearest severity at or above the median value (ties fail cautious)."""
    if value <= SEVERITY[config.VERDICT_SAFE]:
        return config.VERDICT_SAFE
    if value <= SEVERITY[config.VERDICT_TIGHT]:
        return config.VERDICT_TIGHT
    if value <= SEVERITY[config.VERDICT_BLOCKED]:
        return config.VERDICT_BLOCKED
    return config.VERDICT_UNKNOWN
