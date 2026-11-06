"""Temporal stabilisation for per-frame detections.

A detector run frame-by-frame on real footage produces boxes that jitter in
position and, worse, briefly vanish: a track present in 87% of frames blinks out
for one or two samples at a time. Rendered at the source frame rate that reads as
the whole overlay flickering, which looks like a model problem but is a display
problem.

Two fixes, both local to display and geometry:

  * smooth each track box with an exponential moving average, so its corners stop
    twitching frame to frame;
  * coast a track for a short grace period when the detector misses it, so a
    one-sample dropout does not erase the box and its verdict contribution.

Coasted boxes are marked `predicted=True` so the overlay can draw them dashed and
the rest of the pipeline can decide whether to trust them. They expire quickly and
their confidence decays, so they cannot mask a genuinely departed object for long.

Deliberate asymmetry: a coasted box still counts as an obstacle. Forgetting a real
vehicle is worse than briefly over-reporting one, so a dropout fails toward
BLOCKED rather than toward SAFE.
"""

from __future__ import annotations

from dataclasses import replace

from . import config
from .detectors import Detection


class DetectionSmoother:
    """Per-track box smoothing plus short-lived track coasting."""

    def __init__(
        self,
        alpha: float = config.BOX_SMOOTHING_ALPHA,
        hold_seconds: float = config.TRACK_HOLD_SECONDS,
        conf_decay: float = config.TRACK_HOLD_CONF_DECAY,
    ) -> None:
        self.alpha = max(0.05, min(1.0, alpha))
        self.hold_seconds = max(0.0, hold_seconds)
        self.conf_decay = max(0.05, min(1.0, conf_decay))
        self._state: dict[int, Detection] = {}
        self._seen_at: dict[int, float] = {}
        self._misses: dict[int, int] = {}

    def reset(self) -> None:
        self._state.clear()
        self._seen_at.clear()
        self._misses.clear()

    def update(self, detections: list[Detection], t_seconds: float) -> list[Detection]:
        """Blend this frame's boxes with track history and coast missing ones."""
        out: list[Detection] = []
        current_ids: set[int] = set()

        for det in detections:
            if det.track_id is None:
                out.append(det)
                continue
            current_ids.add(det.track_id)
            previous = self._state.get(det.track_id)
            if previous is None:
                self._state[det.track_id] = det
            else:
                self._state[det.track_id] = self._blend(previous, det)
            self._seen_at[det.track_id] = t_seconds
            self._misses[det.track_id] = 0
            out.append(self._state[det.track_id])

        # Coast tracks the detector missed this frame.
        for track_id, last in list(self._state.items()):
            if track_id in current_ids:
                continue
            last_seen = self._seen_at.get(track_id)
            if last_seen is None:
                continue
            age = t_seconds - last_seen
            if age > self.hold_seconds or self.hold_seconds <= 0.0:
                del self._state[track_id]
                self._seen_at.pop(track_id, None)
                self._misses.pop(track_id, None)
                continue
            misses = self._misses.get(track_id, 0) + 1
            self._misses[track_id] = misses
            decay = self.conf_decay ** misses
            out.append(replace(last, conf=last.conf * decay, predicted=True))

        return out

    def _blend(self, previous: Detection, current: Detection) -> Detection:
        a = self.alpha
        b = 1.0 - a
        return replace(
            current,
            x1=a * current.x1 + b * previous.x1,
            y1=a * current.y1 + b * previous.y1,
            x2=a * current.x2 + b * previous.x2,
            y2=a * current.y2 + b * previous.y2,
            predicted=False,
        )