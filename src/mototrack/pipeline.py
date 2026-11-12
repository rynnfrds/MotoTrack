"""End-to-end analysis: video in, annotated video + events + summary out."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, replace

from . import config
from .detectors import Detection, build_detector
from .geometry import (
    compute_gap,
    corridor_center_covered,
    estimate_distance_m,
    is_blocking_candidate,
)
from .io import VideoMeta, VideoWriter, ensure_dir, iter_frames, probe
from .overlay import annotate, draw_timeline
from .smoothing import DetectionSmoother
from .verdict import ClosingRateTracker, FrameVerdict, VerdictStabilizer, decide

EMPTY_CONFIRM_SECONDS = 1.0


@dataclass
class VerdictSegment:
    """A run of consecutive frames that shared one verdict."""

    t0: float
    t1: float
    verdict: str
    min_ratio: float | None
    frame_count: int
    track_ids: list = field(default_factory=list)
    reason: str = ""


@dataclass
class _HeldState:
    """Last analysed frame's state, reused to annotate the frames in between.

    Without this, frames skipped by the analysis stride were written to the
    output video un-annotated, so boxes and the HUD strobed at the source frame
    rate (5 of every 6 frames at 60 fps).
    """

    detections: list
    verdict: str
    reason: str
    gap: object
    raw_verdict: str


def _interpolate(prev: list, nxt: list, weight: float) -> list:
    """Blend box corners between two states, matching tracks by id.

    Non-matching tracks are dropped rather than teleported: a box that appears or
    disappears between ticks should not slide across the screen from an unrelated
    object's position.
    """
    weight = min(1.0, max(0.0, weight))
    if not prev or weight <= 0.0:
        return prev
    if not nxt:
        return prev
    before = {d.track_id: d for d in prev if d.track_id is not None}
    after = {d.track_id: d for d in nxt if d.track_id is not None}
    blended = []
    for track_id, later in after.items():
        earlier = before.get(track_id)
        if earlier is None:
            continue
        blended.append(
            replace(
                later,
                x1=earlier.x1 + (later.x1 - earlier.x1) * weight,
                y1=earlier.y1 + (later.y1 - earlier.y1) * weight,
                x2=earlier.x2 + (later.x2 - earlier.x2) * weight,
                y2=earlier.y2 + (later.y2 - earlier.y2) * weight,
            )
        )
    for det in prev:
        if det.track_id is None or det.track_id not in after:
            blended.append(det)
    return blended


@dataclass
class AnalysisResult:
    source: str
    meta: dict
    detector: str
    out_dir: str
    annotated_video: str
    timeline_image: str
    events_path: str
    summary_path: str
    segments: list
    summary: dict


def _frame_times(meta: VideoMeta, analysis_fps: float):
    """Source frames we analyse, and the timestamp we report for each."""
    if analysis_fps <= 0 or analysis_fps >= meta.fps:
        step = 1
    else:
        step = max(1, int(round(meta.fps / analysis_fps)))
    analysed_fps = meta.fps / step
    return step, analysed_fps


def analyse(
    video_path: str,
    out_dir: str | None = None,
    detector_kind: str = config.DEFAULT_DETECTOR,
    analysis_fps: float = config.ANALYSIS_FPS,
    write_video: bool = True,
    progress_cb=None,
    detector=None,
    max_seconds: float | None = None,
) -> AnalysisResult:
    """Run the full pipeline on one clip.

    ``detector`` may be injected for tests; otherwise one is built from
    ``detector_kind``.
    """
    meta = probe(video_path)
    if max_seconds is not None and meta.duration_s > max_seconds:
        # Trim by frame budget rather than re-encoding: cheap and deterministic.
        trimmed_frames = int(max_seconds * meta.fps)
    else:
        trimmed_frames = meta.frame_count

    out_dir = out_dir or os.path.join(config.OUT_DIR, os.path.splitext(os.path.basename(video_path))[0])
    ensure_dir(out_dir)

    if detector is None:
        detector = build_detector(detector_kind)
    else:
        detector_kind = getattr(detector, "name", detector_kind)

    step, analysed_fps = _frame_times(meta, analysis_fps)
    stabilizer = VerdictStabilizer()
    smoother = DetectionSmoother()
    closing = ClosingRateTracker()
    segments: list[VerdictSegment] = []
    current_segment: VerdictSegment | None = None

    empty_run = 0.0
    last_state: _HeldState | None = None
    pending: list = []
    verdict_counts: dict[str, int] = {}
    latencies: list[float] = []
    analysed_frames = 0
    annotated_path = os.path.join(out_dir, "annotated.mp4")
    writer = (
        VideoWriter(annotated_path, meta.width, meta.height, meta.fps) if write_video else None
    )

    started = time.perf_counter()
    try:
        for index, frame in iter_frames(video_path):
            if index >= trimmed_frames:
                break
            if index % step != 0:
                # Not an analysis tick. Hold the frame until the next tick so its
                # overlay can be interpolated between the two surrounding states.
                # Writing it immediately would mean guessing forward; writing the
                # last state unchanged made the overlay freeze for (step-1) frames
                # at a time, which at 60 fps left 67% of output frames static.
                if writer is not None:
                    pending.append((index, frame))
                continue

            t_seconds = index / meta.fps
            frame_started = time.perf_counter()
            # Smooth boxes and coast brief dropouts before any geometry reads
            # them, so a one-sample miss neither erases a box from the display
            # nor reopens the corridor in the verdict.
            detections = smoother.update(
                detector.infer(frame, persist=True), t_seconds
            )

            gap = compute_gap(detections, meta.width, meta.height)
            centre_closed = corridor_center_covered(detections, meta.width, meta.height)

            blocking_distance = None
            if gap.blocking_ids:
                for det in detections:
                    if det.track_id in gap.blocking_ids:
                        blocking_distance = estimate_distance_m(
                            det.cls_name, det.height, meta.width
                        )
                        closing.update(det.track_id, blocking_distance)
                        break

            closing_fast = False
            if gap.blocking_ids:
                closing_fast = any(
                    closing.is_closing_fast(tid, analysed_fps) for tid in gap.blocking_ids
                )

            if gap.n_candidates == 0:
                empty_run += 1.0 / analysed_fps
            else:
                empty_run = 0.0
            empty_confirmed = empty_run >= EMPTY_CONFIRM_SECONDS

            raw = decide(
                gap,
                detection_count=len(detections),
                closing_fast=closing_fast,
                center_covered=centre_closed,
                empty_corridor_confirmed=empty_confirmed,
            )
            shown = stabilizer.update(raw.verdict, analysed_fps)

            verdict_counts[shown] = verdict_counts.get(shown, 0) + 1
            analysed_frames += 1
            latencies.append(time.perf_counter() - frame_started)

            if current_segment is None or current_segment.verdict != shown:
                if current_segment is not None:
                    current_segment.t1 = t_seconds
                    segments.append(current_segment)
                current_segment = VerdictSegment(
                    t0=t_seconds,
                    t1=t_seconds,
                    verdict=shown,
                    min_ratio=raw.gap_ratio,
                    frame_count=0,
                    track_ids=[],
                    reason=raw.reason,
                )
            current_segment.t1 = t_seconds
            current_segment.frame_count += 1
            for tid in gap.blocking_ids:
                if tid not in current_segment.track_ids:
                    current_segment.track_ids.append(tid)
            if raw.gap_ratio is not None:
                if current_segment.min_ratio is None:
                    current_segment.min_ratio = raw.gap_ratio
                else:
                    current_segment.min_ratio = min(current_segment.min_ratio, raw.gap_ratio)

            current_state = _HeldState(
                detections=detections,
                verdict=shown,
                reason=raw.reason,
                gap=gap,
                raw_verdict=raw.verdict,
            )

            if writer is not None:
                # Flush the frames held since the previous tick, sliding their
                # boxes from the previous state towards this one.
                for held_index, held_frame in pending:
                    weight = step and (held_index % step) / float(step)
                    writer.write(
                        annotate(
                            held_frame,
                            _interpolate(
                                last_state.detections if last_state else detections,
                                detections,
                                weight,
                            ),
                            last_state.verdict if last_state else shown,
                            reason=(last_state or current_state).reason,
                            gap=(last_state or current_state).gap,
                            t_seconds=held_index / meta.fps,
                            raw_verdict=(last_state or current_state).raw_verdict,
                        )
                    )
                pending.clear()
                writer.write(
                    annotate(
                        frame,
                        detections,
                        shown,
                        reason=raw.reason,
                        gap=gap,
                        t_seconds=t_seconds,
                        raw_verdict=raw.verdict,
                    )
                )
                last_state = current_state

            if progress_cb is not None:
                progress_cb(index, trimmed_frames)
    finally:
        if writer is not None:
            # Frames buffered after the final analysis tick keep the final state.
            for held_index, held_frame in pending:
                if last_state is not None:
                    held_frame = annotate(
                        held_frame,
                        last_state.detections,
                        last_state.verdict,
                        reason=last_state.reason,
                        gap=last_state.gap,
                        t_seconds=held_index / meta.fps,
                        raw_verdict=last_state.raw_verdict,
                    )
                writer.write(held_frame)
            pending.clear()
            writer.close()

    if current_segment is not None:
        current_segment.t1 = max(current_segment.t1, current_segment.t0)
        segments.append(current_segment)

    elapsed = time.perf_counter() - started
    analysed_seconds = meta.duration_s if max_seconds is None else min(meta.duration_s, max_seconds)
    summary = {
        "source": os.path.abspath(video_path),
        "detector": detector_kind,
        "video": {
            "width": meta.width,
            "height": meta.height,
            "fps": round(meta.fps, 3),
            "frame_count": meta.frame_count,
            "duration_s": round(meta.duration_s, 3),
            "codec": meta.codec,
        },
        "analysis": {
            "analysed_frames": analysed_frames,
            "frame_step": step,
            "analysed_fps": round(analysed_fps, 3),
            "elapsed_s": round(elapsed, 2),
            "wall_fps": round(analysed_frames / elapsed, 2) if elapsed > 0 else None,
            "mean_latency_ms": (
                round(1000.0 * sum(latencies) / len(latencies), 1) if latencies else None
            ),
            "p95_latency_ms": (
                round(1000.0 * sorted(latencies)[int(len(latencies) * 0.95)], 1)
                if latencies
                else None
            ),
        },
        "verdict_seconds": {
            verdict: round(count / analysed_fps, 2)
            for verdict, count in sorted(verdict_counts.items())
        },
        "verdict_share": {
            verdict: round(count / analysed_frames, 4)
            for verdict, count in sorted(verdict_counts.items())
        } if analysed_frames else {},
        "analysed_seconds": round(analysed_seconds, 2),
        "caveats": [
            "Distances are monocular estimates from assumed object heights, not sensor readings.",
            "Verdict describes free lateral space in the ego corridor, not overall traffic safety.",
            (
                f"The {detector_kind} checkpoint is third-party and was not fine-tuned on this "
                "footage; no mAP claim is made here."
            ),
        ],
    }

    events_path = os.path.join(out_dir, "events.json")
    with open(events_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": os.path.abspath(video_path),
                "detector": detector_kind,
                "thresholds": {
                    "safe_ratio": config.VERDICT_SAFE_RATIO,
                    "tight_ratio": config.VERDICT_TIGHT_RATIO,
                    "near_bucket_max_m": config.NEAR_BUCKET_MAX_M,
                    "mid_bucket_max_m": config.MID_BUCKET_MAX_M,
                    "assumed_hfov_deg": config.ASSUMED_HFOV_DEG,
                },
                "segments": [asdict(seg) for seg in segments],
            },
            fh,
            indent=2,
        )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    timeline_path = os.path.join(out_dir, "timeline.png")
    draw_timeline(segments, analysed_seconds or 1.0, timeline_path)

    return AnalysisResult(
        source=video_path,
        meta=asdict(meta),
        detector=detector_kind,
        out_dir=out_dir,
        annotated_video=annotated_path if write_video else "",
        timeline_image=timeline_path,
        events_path=events_path,
        summary_path=summary_path,
        segments=segments,
        summary=summary,
    )
