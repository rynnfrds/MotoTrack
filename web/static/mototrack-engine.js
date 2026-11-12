/**
 * MotoTrack Browser Engine — Faithful 1:1 port of src/mototrack/
 *
 * Modules ported:
 *   - config.py      (thresholds, corridor geometry, assumed heights, temporal constants)
 *   - geometry.py    (pinhole distance, corridor trapezoid, free-gap intervals, center coverage)
 *   - smoothing.py   (IoU multi-object tracker + EMA box smoother + 0.4 s dropout coasting)
 *   - verdict.py     (calibrated 1.8 / 0.4 ratios, closing rate TTC, 2.0 s median + 0.5 s hysteresis)
 *   - overlay.py     (corridor polygon tint, dashed coasted boxes, HUD panel, timeline strip)
 */

(function (global) {
  "use strict";

  const CONFIG = {
    ANALYSIS_FPS: 10.0,
    BOX_SMOOTHING_ALPHA: 0.45,
    TRACK_HOLD_SECONDS: 0.4,
    TRACK_HOLD_CONF_DECAY: 0.75,

    ASSUMED_HFOV_DEG: 78.0,
    ASSUMED_HEIGHT_M: {
      person: 1.65,
      bicycle: 1.30,
      motorcycle: 1.30,
      car: 1.60,
      bus: 2.90,
      truck: 2.90,
      train: 2.90,
      pedestrian: 1.65,
      "2-wheeler": 1.30,
      "3-wheeler": 1.90,
      "4-wheeler": 1.60,
    },
    DEFAULT_ASSUMED_HEIGHT_M: 1.60,

    NEAR_BUCKET_MAX_M: 3.0,
    MID_BUCKET_MAX_M: 8.0,

    CORRIDOR_BOTTOM_LEFT_X: 0.30,
    CORRIDOR_BOTTOM_RIGHT_X: 0.70,
    CORRIDOR_TOP_LEFT_X: 0.455,
    CORRIDOR_TOP_RIGHT_X: 0.545,
    CORRIDOR_HORIZON_Y: 0.45,

    EGO_MASK_BOTTOM_Y: 0.95,
    EGO_MASK_MIN_WIDTH_FRAC: 0.75,

    VERDICT_SAFE_RATIO: 1.8,
    VERDICT_TIGHT_RATIO: 0.4,
    TTC_DOWNGRADE_SECONDS: 1.5,
    HISTORY_LEN: 5,
    CLOSING_EPSILON_M_PER_FRAME: 0.02,

    MEDIAN_FILTER_SECONDS: 2.0,
    HYSTERESIS_SECONDS: 0.5,
    MIN_CONFIDENT_DETECTIONS: 1,
    EMPTY_CONFIRM_SECONDS: 1.0,

    VERDICT_SAFE: "SAFE",
    VERDICT_TIGHT: "TIGHT",
    VERDICT_BLOCKED: "BLOCKED",
    VERDICT_UNKNOWN: "UNKNOWN",
  };

  const SEVERITY = {
    SAFE: 0,
    TIGHT: 1,
    BLOCKED: 2,
    UNKNOWN: 3,
  };

  const BAND_ORDER = { near: 0, mid: 1, far: 2 };

  // RGB colours matching src/mototrack/overlay.py (converted from BGR)
  const VERDICT_RGB = {
    SAFE: [80, 190, 80],
    TIGHT: [230, 180, 40],
    BLOCKED: [220, 60, 60],
    UNKNOWN: [170, 170, 170],
  };

  function median(values) {
    if (!values || values.length === 0) return 0.0;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2.0;
  }

  // --- Geometry (src/mototrack/geometry.py) ----------------------------------

  function focalLengthPx(imageWidthPx, hfovDeg = CONFIG.ASSUMED_HFOV_DEG) {
    const rad = (hfovDeg * Math.PI) / 180.0;
    return (imageWidthPx / 2.0) / Math.tan(rad / 2.0);
  }

  function estimateDistanceM(clsName, bboxHeightPx, imageWidthPx, hfovDeg = CONFIG.ASSUMED_HFOV_DEG) {
    if (bboxHeightPx <= 1e-6) return Infinity;
    const realHeightM =
      CONFIG.ASSUMED_HEIGHT_M[clsName] !== undefined
        ? CONFIG.ASSUMED_HEIGHT_M[clsName]
        : CONFIG.DEFAULT_ASSUMED_HEIGHT_M;
    return (realHeightM * focalLengthPx(imageWidthPx, hfovDeg)) / bboxHeightPx;
  }

  function distanceBand(distanceM) {
    if (distanceM < CONFIG.NEAR_BUCKET_MAX_M) return "near";
    if (distanceM < CONFIG.MID_BUCKET_MAX_M) return "mid";
    return "far";
  }

  function corridorXBounds(
    yFrac,
    horizonY = CONFIG.CORRIDOR_HORIZON_Y,
    bottomLeft = CONFIG.CORRIDOR_BOTTOM_LEFT_X,
    bottomRight = CONFIG.CORRIDOR_BOTTOM_RIGHT_X,
    topLeft = CONFIG.CORRIDOR_TOP_LEFT_X,
    topRight = CONFIG.CORRIDOR_TOP_RIGHT_X
  ) {
    if (yFrac < 0.0 || yFrac > 1.0 || yFrac < horizonY) return null;
    const span = 1.0 - horizonY;
    let t = span <= 0 ? 1.0 : (yFrac - horizonY) / span;
    t = Math.min(Math.max(t, 0.0), 1.0);
    const left = topLeft + t * (bottomLeft - topLeft);
    const right = topRight + t * (bottomRight - topRight);
    return [left, right];
  }

  function corridorPolygonRelative() {
    return [
      [CONFIG.CORRIDOR_BOTTOM_LEFT_X, 1.0],
      [CONFIG.CORRIDOR_BOTTOM_RIGHT_X, 1.0],
      [CONFIG.CORRIDOR_TOP_RIGHT_X, CONFIG.CORRIDOR_HORIZON_Y],
      [CONFIG.CORRIDOR_TOP_LEFT_X, CONFIG.CORRIDOR_HORIZON_Y],
    ];
  }

  function pointInCorridor(xFrac, yFrac) {
    const bounds = corridorXBounds(yFrac);
    if (!bounds) return false;
    return xFrac >= bounds[0] && xFrac <= bounds[1];
  }

  function isEgoBodywork(det, imageHeightPx, imageWidthPx) {
    if (imageHeightPx <= 0 || imageWidthPx <= 0) return false;
    const bottomY = det.y2;
    const width = Math.max(0, det.x2 - det.x1);
    if (bottomY / imageHeightPx <= CONFIG.EGO_MASK_BOTTOM_Y) return false;
    return width >= CONFIG.EGO_MASK_MIN_WIDTH_FRAC * imageWidthPx;
  }

  function isBlockingCandidate(det, imageWidthPx, imageHeightPx) {
    if (imageWidthPx <= 0 || imageHeightPx <= 0) return false;
    const w = Math.max(0, det.x2 - det.x1);
    const h = Math.max(0, det.y2 - det.y1);
    if (w <= 0 || h <= 0) return false;
    if (isEgoBodywork(det, imageHeightPx, imageWidthPx)) return false;
    const yFrac = det.y2 / imageHeightPx;
    if (yFrac < CONFIG.CORRIDOR_HORIZON_Y || yFrac > 1.0) return false;
    // Lateral overlap, not bottom-centre containment. Portraits of a rider
    // squeezing between two trucks put each truck's bottom-centre outside the
    // narrow corridor, and containment would then report an empty corridor.
    const bounds = corridorXBounds(yFrac);
    if (!bounds) return false;
    const corridorLeftPx = bounds[0] * imageWidthPx;
    const corridorRightPx = bounds[1] * imageWidthPx;
    return det.x1 < corridorRightPx && det.x2 > corridorLeftPx;
  }

  function mergeIntervals(intervals) {
    const cleaned = [];
    for (const [a, b] of intervals) {
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      if (hi > lo) cleaned.push([lo, hi]);
    }
    if (cleaned.length === 0) return [];
    cleaned.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const merged = [cleaned[0]];
    for (let i = 1; i < cleaned.length; i++) {
      const [start, end] = cleaned[i];
      const last = merged[merged.length - 1];
      if (start <= last[1]) {
        last[1] = Math.max(last[1], end);
      } else {
        merged.push([start, end]);
      }
    }
    return merged;
  }

  function freeGaps(blocked, spanStart, spanEnd) {
    if (spanEnd <= spanStart) return [];
    const gaps = [];
    let cursor = spanStart;
    for (const [start, end] of mergeIntervals(blocked)) {
      if (end <= spanStart || start >= spanEnd) continue;
      if (start > cursor) {
        gaps.push([cursor, Math.min(start, spanEnd)]);
      }
      cursor = Math.max(cursor, end);
      if (cursor >= spanEnd) break;
    }
    if (cursor < spanEnd) {
      gaps.push([cursor, spanEnd]);
    }
    return gaps.filter(([a, b]) => b > a);
  }

  function computeGap(detections, imageWidthPx, imageHeightPx) {
    const bottomBounds = corridorXBounds(1.0) || [
      CONFIG.CORRIDOR_BOTTOM_LEFT_X,
      CONFIG.CORRIDOR_BOTTOM_RIGHT_X,
    ];
    const widestCorridorPx = (bottomBounds[1] - bottomBounds[0]) * imageWidthPx;
    const allWidths = detections
      .map((d) => Math.max(0, d.x2 - d.x1))
      .filter((w) => w > 0);

    const candidates = detections.filter((d) =>
      isBlockingCandidate(d, imageWidthPx, imageHeightPx)
    );

    if (candidates.length === 0) {
      const refW = allWidths.length ? median(allWidths) : 0.0;
      return {
        gap_px: widestCorridorPx,
        corridor_width_px: widestCorridorPx,
        reference_width_px: refW,
        scanline_y_px: imageHeightPx,
        band: null,
        near_distance_m: null,
        blocking_ids: [],
        n_candidates: 0,
        gap_m: null,
        gap_ratio: refW <= 0 ? Infinity : widestCorridorPx / refW,
      };
    }

    const distances = candidates.map((d) =>
      estimateDistanceM(d.cls_name, Math.max(0, d.y2 - d.y1), imageWidthPx)
    );
    const bands = distances.map((dist) => distanceBand(dist));
    let nearestBand = bands[0];
    for (const b of bands) {
      if (BAND_ORDER[b] < BAND_ORDER[nearestBand]) nearestBand = b;
    }

    const bandMembers = [];
    const bandDistances = [];
    for (let i = 0; i < candidates.length; i++) {
      if (bands[i] === nearestBand) {
        bandMembers.push(candidates[i]);
        bandDistances.push(distances[i]);
      }
    }

    let scanlineYPx = median(bandMembers.map((d) => d.y2));
    let yFrac = scanlineYPx / imageHeightPx;
    let bounds = corridorXBounds(yFrac);
    if (!bounds) {
      yFrac = 1.0;
      scanlineYPx = imageHeightPx;
      bounds = bottomBounds;
    }

    const spanStart = bounds[0] * imageWidthPx;
    const spanEnd = bounds[1] * imageWidthPx;
    const blocked = bandMembers.map((d) => [d.x1, d.x2]);
    const gaps = freeGaps(blocked, spanStart, spanEnd);
    let gapPx = 0.0;
    for (const [s, e] of gaps) {
      if (e - s > gapPx) gapPx = e - s;
    }

    const bandWidths = bandMembers
      .map((d) => Math.max(0, d.x2 - d.x1))
      .filter((w) => w > 0);
    const referenceWidthPx = bandWidths.length
      ? median(bandWidths)
      : allWidths.length
      ? median(allWidths)
      : 0.0;

    const nearDistance = median(bandDistances);
    const focal = focalLengthPx(imageWidthPx);
    const gapM =
      focal > 0 && Number.isFinite(nearDistance)
        ? (gapPx * nearDistance) / focal
        : null;

    const gapRatio =
      referenceWidthPx <= 0
        ? gapPx > 0
          ? Infinity
          : 0.0
        : gapPx / referenceWidthPx;

    return {
      gap_px: gapPx,
      corridor_width_px: spanEnd - spanStart,
      reference_width_px: referenceWidthPx,
      scanline_y_px: scanlineYPx,
      band: nearestBand,
      near_distance_m: Number.isFinite(nearDistance) ? nearDistance : null,
      blocking_ids: bandMembers.map((d) => d.track_id),
      n_candidates: candidates.length,
      gap_m: gapM,
      gap_ratio: gapRatio,
    };
  }

  function corridorCenterCovered(detections, imageWidthPx, imageHeightPx) {
    for (const det of detections) {
      if (!isBlockingCandidate(det, imageWidthPx, imageHeightPx)) continue;
      const dist = estimateDistanceM(
        det.cls_name,
        Math.max(0, det.y2 - det.y1),
        imageWidthPx
      );
      if (distanceBand(dist) !== "near") continue;
      const yFrac = det.y2 / imageHeightPx;
      const bounds = corridorXBounds(yFrac);
      if (!bounds) continue;
      const centerPx = ((bounds[0] + bounds[1]) / 2.0) * imageWidthPx;
      if (det.x1 <= centerPx && centerPx <= det.x2) return true;
    }
    return false;
  }

  // --- Tracker + Smoother (src/mototrack/smoothing.py) ----------------------

  function boxIoU(a, b) {
    const ix1 = Math.max(a.x1, b.x1);
    const iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2);
    const iy2 = Math.min(a.y2, b.y2);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    if (inter <= 0) return 0.0;
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    return inter / Math.max(1e-6, areaA + areaB - inter);
  }

  class SimpleByteTracker {
    constructor(iouThreshold = 0.25, maxAgeSeconds = 1.2) {
      this.iouThreshold = iouThreshold;
      this.maxAgeSeconds = maxAgeSeconds;
      this.nextId = 1;
      this.tracks = new Map(); // id -> { x1, y1, x2, y2, cls_name, t }
    }

    assign(rawDetections, tSeconds) {
      // Prune expired tracks
      for (const [id, tr] of this.tracks.entries()) {
        if (tSeconds - tr.t > this.maxAgeSeconds) {
          this.tracks.delete(id);
        }
      }

      const usedTracks = new Set();
      const assigned = [];

      for (const det of rawDetections) {
        let bestId = null;
        let bestScore = this.iouThreshold;
        for (const [id, tr] of this.tracks.entries()) {
          if (usedTracks.has(id)) continue;
          const iou = boxIoU(det, tr);
          if (iou > bestScore) {
            bestScore = iou;
            bestId = id;
          }
        }
        if (bestId === null) {
          bestId = this.nextId++;
        }
        usedTracks.add(bestId);
        this.tracks.set(bestId, {
          x1: det.x1,
          y1: det.y1,
          x2: det.x2,
          y2: det.y2,
          cls_name: det.cls_name,
          t: tSeconds,
        });
        assigned.push({ ...det, track_id: bestId, predicted: false });
      }
      return assigned;
    }
  }

  class DetectionSmoother {
    constructor(
      alpha = CONFIG.BOX_SMOOTHING_ALPHA,
      holdSeconds = CONFIG.TRACK_HOLD_SECONDS,
      confDecay = CONFIG.TRACK_HOLD_CONF_DECAY
    ) {
      this.alpha = Math.max(0.05, Math.min(1.0, alpha));
      this.holdSeconds = Math.max(0.0, holdSeconds);
      this.confDecay = Math.max(0.05, Math.min(1.0, confDecay));
      this.state = new Map();
      this.seenAt = new Map();
      this.misses = new Map();
    }

    update(detections, tSeconds) {
      const out = [];
      const currentIds = new Set();
      const a = this.alpha;
      const b = 1.0 - a;

      for (const det of detections) {
        if (det.track_id == null) {
          out.push(det);
          continue;
        }
        currentIds.add(det.track_id);
        const prev = this.state.get(det.track_id);
        let smoothed;
        if (!prev) {
          smoothed = { ...det, predicted: false };
        } else {
          smoothed = {
            ...det,
            x1: a * det.x1 + b * prev.x1,
            y1: a * det.y1 + b * prev.y1,
            x2: a * det.x2 + b * prev.x2,
            y2: a * det.y2 + b * prev.y2,
            predicted: false,
          };
        }
        this.state.set(det.track_id, smoothed);
        this.seenAt.set(det.track_id, tSeconds);
        this.misses.set(det.track_id, 0);
        out.push(smoothed);
      }

      for (const [trackId, last] of Array.from(this.state.entries())) {
        if (currentIds.has(trackId)) continue;
        const lastSeen = this.seenAt.get(trackId);
        if (lastSeen === undefined) continue;
        const age = tSeconds - lastSeen;
        if (age > this.holdSeconds || this.holdSeconds <= 0) {
          this.state.delete(trackId);
          this.seenAt.delete(trackId);
          this.misses.delete(trackId);
          continue;
        }
        const m = (this.misses.get(trackId) || 0) + 1;
        this.misses.set(trackId, m);
        const decay = Math.pow(this.confDecay, m);
        out.push({ ...last, conf: last.conf * decay, predicted: true });
      }

      return out;
    }
  }

  // --- Verdict & Stabiliser (src/mototrack/verdict.py) ----------------------

  function classifyRatio(
    gapRatio,
    safeRatio = CONFIG.VERDICT_SAFE_RATIO,
    tightRatio = CONFIG.VERDICT_TIGHT_RATIO
  ) {
    if (gapRatio >= safeRatio) return CONFIG.VERDICT_SAFE;
    if (gapRatio >= tightRatio) return CONFIG.VERDICT_TIGHT;
    return CONFIG.VERDICT_BLOCKED;
  }

  function downgrade(verdict) {
    if (verdict === CONFIG.VERDICT_SAFE) return CONFIG.VERDICT_TIGHT;
    if (verdict === CONFIG.VERDICT_TIGHT) return CONFIG.VERDICT_BLOCKED;
    return verdict;
  }

  function decide(
    gap,
    {
      detectionCount,
      closingFast,
      centerCovered,
      emptyCorridorConfirmed,
      minDetections = CONFIG.MIN_CONFIDENT_DETECTIONS,
    }
  ) {
    if (detectionCount < minDetections && gap.n_candidates === 0) {
      return {
        verdict: CONFIG.VERDICT_UNKNOWN,
        gap_ratio: null,
        gap_px: null,
        gap_m: null,
        band: null,
        blocking_ids: [],
        reason: "low_detections",
      };
    }

    if (gap.n_candidates === 0) {
      if (!emptyCorridorConfirmed) {
        return {
          verdict: CONFIG.VERDICT_UNKNOWN,
          gap_ratio: null,
          gap_px: gap.gap_px,
          gap_m: null,
          band: null,
          blocking_ids: [],
          reason: "no_obstacle_unconfirmed",
        };
      }
      return {
        verdict: CONFIG.VERDICT_SAFE,
        gap_ratio: null,
        gap_px: gap.gap_px,
        gap_m: null,
        band: null,
        blocking_ids: [],
        reason: "corridor_clear",
      };
    }

    const ratio = gap.gap_ratio;
    let verdict = classifyRatio(ratio);
    const reasons = ["gap_ratio"];

    if (centerCovered && verdict === CONFIG.VERDICT_SAFE) {
      verdict = CONFIG.VERDICT_TIGHT;
      reasons.push("center_covered");
    }

    if (
      gap.near_distance_m !== null &&
      gap.near_distance_m < CONFIG.NEAR_BUCKET_MAX_M
    ) {
      if (verdict === CONFIG.VERDICT_SAFE) {
        verdict = CONFIG.VERDICT_TIGHT;
        reasons.push("obstacle_in_near_band");
      }
    }

    if (closingFast) {
      const next = downgrade(verdict);
      if (next !== verdict) {
        verdict = next;
        reasons.push("closing_fast");
      }
    }

    return {
      verdict,
      gap_ratio: Number.isFinite(ratio) ? ratio : null,
      gap_px: gap.gap_px,
      gap_m: gap.gap_m,
      band: gap.band,
      blocking_ids: gap.blocking_ids,
      reason: reasons.join("+"),
    };
  }

  class ClosingRateTracker {
    constructor(
      historyLen = CONFIG.HISTORY_LEN,
      epsilon = CONFIG.CLOSING_EPSILON_M_PER_FRAME
    ) {
      this.historyLen = historyLen;
      this.epsilon = epsilon;
      this.history = new Map();
    }

    update(trackId, distanceM) {
      if (trackId == null || !Number.isFinite(distanceM)) return;
      let buf = this.history.get(trackId);
      if (!buf) {
        buf = [];
        this.history.set(trackId, buf);
      }
      buf.push(distanceM);
      if (buf.length > this.historyLen) buf.shift();
    }

    closingSpeed(trackId) {
      if (trackId == null) return null;
      const buf = this.history.get(trackId);
      if (!buf || buf.length < 2) return null;
      return (buf[0] - buf[buf.length - 1]) / (buf.length - 1);
    }

    isClosingFast(trackId, fps, ttcSeconds = CONFIG.TTC_DOWNGRADE_SECONDS) {
      const speed = this.closingSpeed(trackId);
      if (speed === null || speed <= this.epsilon) return false;
      const buf = this.history.get(trackId);
      if (!buf || buf.length === 0) return false;
      const distance = buf[buf.length - 1];
      if (!Number.isFinite(distance) || fps <= 0) return false;
      const ttc = distance / (speed * fps);
      return ttc < ttcSeconds;
    }
  }

  function severityToVerdict(value) {
    if (value <= SEVERITY.SAFE) return CONFIG.VERDICT_SAFE;
    if (value <= SEVERITY.TIGHT) return CONFIG.VERDICT_TIGHT;
    if (value <= SEVERITY.BLOCKED) return CONFIG.VERDICT_BLOCKED;
    return CONFIG.VERDICT_UNKNOWN;
  }

  class VerdictStabilizer {
    constructor(
      windowSeconds = CONFIG.MEDIAN_FILTER_SECONDS,
      hysteresisSeconds = CONFIG.HYSTERESIS_SECONDS
    ) {
      this.windowSeconds = windowSeconds;
      this.hysteresisSeconds = hysteresisSeconds;
      this.severityWindow = [];
      this.windowDurations = [];
      this.current = CONFIG.VERDICT_UNKNOWN;
      this.candidate = null;
      this.candidateFrames = 0;
    }

    update(rawVerdict, fps) {
      const frameDuration = fps > 0 ? 1.0 / fps : 0.05;
      const rawSeverity = SEVERITY[rawVerdict];
      this.severityWindow.push(
        rawVerdict === CONFIG.VERDICT_UNKNOWN ? null : rawSeverity
      );
      this.windowDurations.push(frameDuration);

      let total = this.windowDurations.reduce((a, b) => a + b, 0);
      while (this.windowDurations.length > 1 && total > this.windowSeconds) {
        total -= this.windowDurations.shift();
        this.severityWindow.shift();
      }

      const known = this.severityWindow.filter((s) => s !== null);
      if (known.length === 0) {
        this.current = CONFIG.VERDICT_UNKNOWN;
        this.candidate = null;
        this.candidateFrames = 0;
        return this.current;
      }

      const smoothed = severityToVerdict(median(known));
      if (smoothed === this.current) {
        this.candidate = null;
        this.candidateFrames = 0;
        return this.current;
      }

      if (smoothed === this.candidate) {
        this.candidateFrames += 1;
      } else {
        this.candidate = smoothed;
        this.candidateFrames = 1;
      }

      const holdFrames = Math.max(
        1,
        frameDuration > 0 ? Math.round(this.hysteresisSeconds / frameDuration) : 1
      );
      if (this.candidateFrames >= holdFrames) {
        this.current = this.candidate;
        this.candidate = null;
        this.candidateFrames = 0;
      }
      return this.current;
    }
  }

  // --- Interpolation & Canvas Overlay (src/mototrack/overlay.py) ------------

  function interpolateDetections(prev, nxt, weight) {
    const w = Math.min(1.0, Math.max(0.0, weight));
    if (!prev || w <= 0.0) return prev || [];
    if (!nxt || nxt.length === 0) return prev;
    const before = new Map();
    for (const d of prev) {
      if (d.track_id != null) before.set(d.track_id, d);
    }
    const after = new Map();
    for (const d of nxt) {
      if (d.track_id != null) after.set(d.track_id, d);
    }
    const blended = [];
    for (const [trackId, later] of after.entries()) {
      const earlier = before.get(trackId);
      if (!earlier) continue;
      blended.push({
        ...later,
        x1: earlier.x1 + (later.x1 - earlier.x1) * w,
        y1: earlier.y1 + (later.y1 - earlier.y1) * w,
        x2: earlier.x2 + (later.x2 - earlier.x2) * w,
        y2: earlier.y2 + (later.y2 - earlier.y2) * w,
      });
    }
    for (const det of prev) {
      if (det.track_id == null || !after.has(det.track_id)) {
        blended.push(det);
      }
    }
    return blended;
  }

  function rgbCss(rgb, alpha = 1.0) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function drawOverlayOnCanvas(ctx, width, height, state, tSeconds) {
    ctx.clearRect(0, 0, width, height);
    if (!state) return;

    const { detections, verdict, reason, gap, rawVerdict } = state;
    const color = VERDICT_RGB[verdict] || VERDICT_RGB.UNKNOWN;

    // 1. Ego corridor polygon
    const poly = corridorPolygonRelative();
    ctx.save();
    ctx.beginPath();
    poly.forEach(([rx, ry], idx) => {
      const px = rx * width;
      const py = ry * height;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = rgbCss(color, 0.20);
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = rgbCss(color, 0.95);
    ctx.stroke();
    ctx.restore();

    // 2. Detections
    const blockingSet = new Set((gap && gap.blocking_ids) || []);
    for (const det of detections || []) {
      const blocking = blockingSet.has(det.track_id);
      const boxRgb = blocking ? [220, 60, 60] : [120, 190, 120];
      ctx.save();
      ctx.lineWidth = blocking ? 3 : 2;
      ctx.strokeStyle = rgbCss(boxRgb, 0.95);
      if (det.predicted) {
        ctx.setLineDash([8, 6]);
      }
      ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
      ctx.setLineDash([]);

      const prefix = det.predicted ? "~" : "";
      const label =
        prefix +
        (det.track_id == null
          ? det.cls_name
          : `${det.cls_name} #${det.track_id}`);
      const fontPx = Math.max(12, Math.round(height * 0.022));
      ctx.font = `600 ${fontPx}px system-ui, -apple-system, sans-serif`;
      const tx = det.x1;
      const ty = Math.max(fontPx + 4, det.y1 - 6);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(label, tx, ty);
      ctx.fillStyle = rgbCss(boxRgb, 1.0);
      ctx.fillText(label, tx, ty);
      ctx.restore();
    }

    // 3. Top-left HUD panel
    const scale = Math.max(0.55, height / 720.0);
    const panelW = Math.min(Math.round(width * 0.64), Math.round(480 * scale));
    const panelH = Math.round(104 * scale);

    ctx.save();
    ctx.fillStyle = "rgba(18, 18, 18, 0.68)";
    ctx.fillRect(0, 0, panelW, panelH);
    ctx.lineWidth = 2;
    ctx.strokeStyle = rgbCss(color, 1.0);
    ctx.strokeRect(1, 1, panelW - 2, panelH - 2);

    const drawHudText = (text, x, y, sizePx, fillStyle, bold = false) => {
      ctx.font = `${bold ? "700" : "500"} ${Math.round(
        sizePx
      )}px system-ui, -apple-system, sans-serif`;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.strokeText(text, x, y);
      ctx.fillStyle = fillStyle;
      ctx.fillText(text, x, y);
    };

    drawHudText(verdict, 14, 34 * scale, 26 * scale, rgbCss(color, 1.0), true);

    let detail;
    if (!gap || gap.n_candidates === 0) {
      detail = "this frame: corridor empty";
    } else if (!Number.isFinite(gap.gap_ratio)) {
      detail = "this frame: gap unbounded";
    } else {
      detail = `this frame: gap ${gap.gap_ratio.toFixed(2)} vehicle-widths`;
    }
    drawHudText(detail, 14, 58 * scale, 14 * scale, "rgb(240,240,240)");

    const caveat =
      gap && gap.gap_m != null
        ? `~${gap.gap_m.toFixed(1)} m estimated, not a sensor`
        : "range estimate unavailable";
    drawHudText(caveat, 14, 78 * scale, 12.5 * scale, "rgb(190,190,190)");

    let stamp =
      rawVerdict && rawVerdict !== verdict
        ? `holding ${verdict} (frame reads ${rawVerdict})`
        : reason || "";
    if (tSeconds != null) {
      stamp = `t=${tSeconds.toFixed(1)}s  ${stamp}`.trim();
    }
    drawHudText(stamp, 14, 95 * scale, 11.5 * scale, "rgb(155,155,155)");
    ctx.restore();
  }

  function renderTimelineDataUrl(segments, totalSeconds, width = 1200, height = 220) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "rgb(245, 245, 245)";
    ctx.fillRect(0, 0, width, height);

    const marginL = 60;
    const marginR = 40;
    const barTop = 60;
    const barH = 60;
    const plotW = width - marginL - marginR;

    ctx.fillStyle = "rgb(30, 30, 30)";
    ctx.font = "600 18px system-ui, -apple-system, sans-serif";
    ctx.fillText("Overtaking verdict over time", marginL, 34);

    const dur = totalSeconds > 0 ? totalSeconds : 1.0;
    for (const seg of segments) {
      const x0 = marginL + Math.round(plotW * (seg.t0 / dur));
      const x1 = marginL + Math.round(plotW * (seg.t1 / dur));
      const rgb = VERDICT_RGB[seg.verdict] || VERDICT_RGB.UNKNOWN;
      ctx.fillStyle = rgbCss(rgb, 1.0);
      ctx.fillRect(x0, barTop, Math.max(2, x1 - x0), barH);
    }

    ctx.strokeStyle = "rgb(120, 120, 120)";
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, barTop, plotW, barH);

    const tickStep = Math.max(1, Math.floor(dur / 10) || 1);
    ctx.fillStyle = "rgb(80, 80, 80)";
    ctx.font = "13px system-ui, -apple-system, sans-serif";
    for (let tick = 0; tick <= Math.floor(dur); tick += tickStep) {
      const x = marginL + Math.round(plotW * (tick / dur));
      ctx.beginPath();
      ctx.moveTo(x, barTop + barH);
      ctx.lineTo(x, barTop + barH + 6);
      ctx.stroke();
      ctx.fillText(`${tick}s`, x - 10, barTop + barH + 22);
    }

    const legendY = barTop + barH + 58;
    let xCursor = marginL;
    for (const v of ["SAFE", "TIGHT", "BLOCKED", "UNKNOWN"]) {
      const rgb = VERDICT_RGB[v];
      ctx.fillStyle = rgbCss(rgb, 1.0);
      ctx.fillRect(xCursor, legendY - 12, 18, 14);
      ctx.fillStyle = "rgb(50, 50, 50)";
      ctx.font = "500 14px system-ui, -apple-system, sans-serif";
      ctx.fillText(v, xCursor + 26, legendY);
      xCursor += 150;
    }

    return canvas.toDataURL("image/png");
  }

  // --- Class taxonomy mapping for COCO-SSD ----------------------------------

  const ROAD_USER_CLASSES = new Set([
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
  ]);

  const MIXED_NAME_MAP = {
    person: "pedestrian",
    bicycle: "2-wheeler",
    motorcycle: "2-wheeler",
    car: "4-wheeler",
    bus: "4-wheeler",
    truck: "4-wheeler",
    train: "4-wheeler",
  };

  function mapPredictionsToDetections(predictions, detectorKind) {
    const out = [];
    for (const p of predictions) {
      if (!ROAD_USER_CLASSES.has(p.class)) continue;
      if (p.score < 0.25) continue;
      const [x, y, w, h] = p.bbox;
      const clsName =
        detectorKind === "mixed"
          ? MIXED_NAME_MAP[p.class] || p.class
          : p.class;
      out.push({
        track_id: null,
        cls_name: clsName,
        cls_id: 0,
        conf: p.score,
        x1: x,
        y1: y,
        x2: x + w,
        y2: y + h,
        predicted: false,
      });
    }
    return out;
  }

  // --- Synthetic Demo Clip: 30 s traffic POV simulation ----------------------

  // MediaRecorder timestamps frames by wall clock, so the recorded file is as
  // long as the draw loop actually runs. The old generator drew 12 s of frames
  // with a fixed 12 ms sleep per frame and produced a 3 s file: 30 analysed
  // frames, and a verdict timeline that never reached TIGHT or BLOCKED.
  //
  // Every actor here is placed in metres from the camera and projected through
  // the same pinhole relation geometry.py uses to estimate distance. The clip
  // the detector sees and the boxes the engine is told about therefore agree.
  const DEMO_CLIP = {
    WIDTH: 960,
    HEIGHT: 540,
    FPS: 25,
    DURATION_S: 30,
    CAMERA_HEIGHT_M: 1.35,
    ROAD_HALF_M: 6.9,
    LANE_EDGE_M: 2.3,
    DASH_WIDTH_M: 0.16,
    SPEED_MPS: 7.5,
  };

  const DEMO_LANE_DASH_M = 9.0;
  const DEMO_LANE_DASH_LENGTH_M = 3.2;

  function demoClamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function demoEase(value) {
    const v = demoClamp01(value);
    return v * v * (3 - 2 * v);
  }

  function demoLerp(a, b, t) {
    return a + (b - a) * t;
  }

  function demoProjection(width, height) {
    return {
      width,
      height,
      horizonY: height * CONFIG.CORRIDOR_HORIZON_Y,
      focal: focalLengthPx(width),
      groundK: focalLengthPx(width) * DEMO_CLIP.CAMERA_HEIGHT_M,
    };
  }

  // Screen row where the road surface at depth z metres meets the ground plane.
  function demoGroundY(proj, z) {
    return proj.horizonY + proj.groundK / Math.max(1.2, z);
  }

  // The story, in metres. Measured verdicts for this timeline (10 fps analysis,
  // same code path as app.js) are roughly:
  //   0.0-0.4 s   first frames, stabiliser has no window yet     -> UNKNOWN
  //   0.4-10.5 s  lead car 55 m out, corridor wide open          -> SAFE
  //   10.5-16.4 s ego closes, gap ratio crosses 1.8 then 0.4     -> TIGHT
  //   16.4-23.4 s lead car fills the corridor at ~5 m            -> BLOCKED
  //   23.4-30.0 s it pulled onto the right shoulder, we pass     -> SAFE
  function demoActorsAt(t) {
    // Lead car: parked ahead in our lane, we close on it. Around t=21 it pulls
    // onto the right shoulder so the corridor reopens, and the next car appears
    // far down the road as the closing shot.
    const near = demoEase((t - 0.5) / 18.0);
    const pullAside = demoEase((t - 21.3) / 2.9);
    const follow = demoEase((t - 23.5) / 6.0);
    const oncoming = demoEase(t / 26.0);
    const shoulder = demoEase((t - 10.0) / 20.0);

    const leadZ = demoLerp(demoLerp(55, 5.2, near), 9.5, pullAside);
    const leadLat = demoLerp(0.2, 7.6, pullAside);

    return [
      {
        cls_name: "car",
        conf: 0.95,
        body: "#2f6fb3",
        widthM: 1.85,
        heightM: 1.5,
        z: leadZ,
        latM: leadLat,
      },
      {
        cls_name: "car",
        conf: 0.9,
        body: "#b9bec6",
        widthM: 1.75,
        heightM: 1.45,
        z: demoLerp(52, 9.0, oncoming),
        latM: demoLerp(-5.6, -4.4, oncoming),
      },
      {
        cls_name: "motorcycle",
        conf: 0.88,
        body: "#25303c",
        widthM: 0.72,
        heightM: 1.55,
        z: demoLerp(34, 6.2, shoulder),
        latM: demoLerp(6.6, 5.8, shoulder),
      },
      {
        cls_name: "car",
        conf: 0.9,
        body: "#39424d",
        widthM: 1.8,
        heightM: 1.45,
        z: demoLerp(140, 58, follow),
        latM: 0.2,
      },
    ];
  }

  function demoBoxesAt(t, width, height) {
    const proj = demoProjection(width, height);
    const boxes = [];
    for (const actor of demoActorsAt(t)) {
      const z = Math.max(1.6, actor.z);
      const boxH = (proj.focal * actor.heightM) / z;
      const boxW = (proj.focal * actor.widthM) / z;
      const cx = width / 2 + (proj.focal * actor.latM) / z;
      const bottomY = demoGroundY(proj, z);
      const box = {
        cls_name: actor.cls_name,
        conf: actor.conf,
        x1: cx - boxW / 2,
        y1: bottomY - boxH,
        x2: cx + boxW / 2,
        y2: bottomY,
        body: actor.body,
      };
      // Skip actors that are entirely off frame: they cannot be detected.
      if (box.x2 < 0 || box.x1 > width || box.y2 < 0 || box.y1 > height) continue;
      boxes.push(box);
    }
    return boxes;
  }

  // Ground truth for the demo clip. The clip is deterministic, so the analysis
  // path can regenerate this instead of shipping a sidecar JSON.
  function demoSyntheticDetections(width, height, fps, durationS) {
    const w = width || DEMO_CLIP.WIDTH;
    const h = height || DEMO_CLIP.HEIGHT;
    const rate = fps || DEMO_CLIP.FPS;
    const total = Math.round((durationS || DEMO_CLIP.DURATION_S) * rate);
    const frames = [];
    for (let f = 0; f < total; f++) {
      const t = f / rate;
      frames.push({ t, boxes: demoBoxesAt(t, w, h) });
    }
    return frames;
  }

  function demoRand(i) {
    const value = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return value - Math.floor(value);
  }

  function demoRoundRectPath(ctx, x, y, w, h, r) {
    const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
  }

  function demoShade(hex, amount) {
    const value = parseInt(hex.slice(1), 16);
    const channels = [(value >> 16) & 255, (value >> 8) & 255, value & 255];
    const shifted = channels.map((c) => {
      const v = amount < 0 ? c * (1 + amount) : c + (255 - c) * amount;
      return Math.round(Math.max(0, Math.min(255, v)));
    });
    return `rgb(${shifted[0]}, ${shifted[1]}, ${shifted[2]})`;
  }

  function drawDemoCar(ctx, box) {
    const w = box.x2 - box.x1;
    const h = box.y2 - box.y1;
    if (w < 3 || h < 3) {
      ctx.fillStyle = box.body || "#2f6fb3";
      ctx.fillRect(box.x1, box.y1, Math.max(1, w), Math.max(1, h));
      return;
    }
    const body = box.body || "#2f6fb3";
    const r = Math.max(2, Math.min(w, h) * 0.14);

    ctx.fillStyle = "rgba(0,0,0,0.32)";
    ctx.beginPath();
    ctx.ellipse(
      box.x1 + w / 2,
      box.y2,
      w * 0.5,
      Math.max(1.5, h * 0.05),
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.fillStyle = demoShade(body, -0.35);
    demoRoundRectPath(ctx, box.x1 + w * 0.14, box.y1, w * 0.72, h * 0.52, r);
    ctx.fill();

    ctx.fillStyle = "#0f1824";
    demoRoundRectPath(
      ctx,
      box.x1 + w * 0.21,
      box.y1 + h * 0.07,
      w * 0.58,
      h * 0.28,
      r * 0.6
    );
    ctx.fill();
    ctx.fillStyle = "rgba(150, 190, 220, 0.22)";
    demoRoundRectPath(
      ctx,
      box.x1 + w * 0.23,
      box.y1 + h * 0.09,
      w * 0.54,
      h * 0.14,
      r * 0.5
    );
    ctx.fill();

    ctx.fillStyle = body;
    demoRoundRectPath(ctx, box.x1, box.y1 + h * 0.34, w, h * 0.62, r);
    ctx.fill();

    ctx.fillStyle = demoShade(body, 0.18);
    ctx.fillRect(
      box.x1 + w * 0.2,
      box.y1 + h * 0.36,
      w * 0.6,
      Math.max(1, h * 0.03)
    );

    const tailW = Math.max(2, w * 0.17);
    const tailH = Math.max(2, h * 0.09);
    const tailY = box.y1 + h * 0.52;
    if (w > 26) {
      ctx.fillStyle = "rgba(232, 69, 58, 0.22)";
      demoRoundRectPath(
        ctx,
        box.x1 + w * 0.02,
        tailY - tailH * 0.5,
        tailW * 1.6,
        tailH * 2,
        tailH * 0.5
      );
      ctx.fill();
      demoRoundRectPath(
        ctx,
        box.x2 - w * 0.02 - tailW * 1.6,
        tailY - tailH * 0.5,
        tailW * 1.6,
        tailH * 2,
        tailH * 0.5
      );
      ctx.fill();
    }
    ctx.fillStyle = "#e8453a";
    demoRoundRectPath(ctx, box.x1 + w * 0.07, tailY, tailW, tailH, tailH * 0.35);
    ctx.fill();
    demoRoundRectPath(
      ctx,
      box.x2 - w * 0.07 - tailW,
      tailY,
      tailW,
      tailH,
      tailH * 0.35
    );
    ctx.fill();

    if (w > 40 && h > 24) {
      ctx.fillStyle = "#e8eef2";
      const plateW = w * 0.26;
      const plateH = Math.max(2, h * 0.055);
      ctx.fillRect(box.x1 + w / 2 - plateW / 2, box.y1 + h * 0.66, plateW, plateH);
    }

    ctx.fillStyle = demoShade(body, -0.5);
    demoRoundRectPath(
      ctx,
      box.x1 + w * 0.03,
      box.y2 - h * 0.14,
      w * 0.94,
      h * 0.11,
      r * 0.4
    );
    ctx.fill();
  }

  function drawDemoScooter(ctx, box) {
    const w = box.x2 - box.x1;
    const h = box.y2 - box.y1;
    if (w < 2 || h < 2) {
      ctx.fillStyle = box.body || "#25303c";
      ctx.fillRect(box.x1, box.y1, Math.max(1, w), Math.max(1, h));
      return;
    }

    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath();
    ctx.ellipse(
      box.x1 + w / 2,
      box.y2,
      w * 0.45,
      Math.max(1, h * 0.02),
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.fillStyle = "#1f2731";
    demoRoundRectPath(
      ctx,
      box.x1 + w * 0.18,
      box.y1 + h * 0.18,
      w * 0.64,
      h * 0.44,
      w * 0.22
    );
    ctx.fill();

    ctx.fillStyle = "#d9e2ea";
    ctx.beginPath();
    ctx.ellipse(
      box.x1 + w * 0.5,
      box.y1 + h * 0.14,
      w * 0.26,
      h * 0.13,
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();

    ctx.fillStyle = box.body || "#25303c";
    demoRoundRectPath(
      ctx,
      box.x1 + w * 0.2,
      box.y1 + h * 0.55,
      w * 0.6,
      h * 0.32,
      w * 0.16
    );
    ctx.fill();

    ctx.fillStyle = "#e8453a";
    ctx.fillRect(
      box.x1 + w * 0.42,
      box.y1 + h * 0.6,
      Math.max(2, w * 0.16),
      Math.max(2, h * 0.05)
    );
  }

  function drawDemoRoad(ctx, proj, t) {
    const W = proj.width;
    const H = proj.height;
    const horizonY = proj.horizonY;
    const focal = proj.focal;
    const zFar = 90;
    const zBottom = proj.groundK / Math.max(1, H - horizonY);
    const xAt = (latM, z) => W / 2 + (focal * latM) / Math.max(1.2, z);

    const sky = ctx.createLinearGradient(0, 0, 0, horizonY);
    sky.addColorStop(0, "#4d7fb4");
    sky.addColorStop(0.62, "#9dc0de");
    sky.addColorStop(1, "#e7eff6");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, horizonY);

    const cover = W / 22;
    const drift = (t * DEMO_CLIP.SPEED_MPS * 0.55) % cover;
    for (let i = -1; i < 23; i++) {
      const tuftH = (0.02 + demoRand(i + 3) * 0.05) * H;
      ctx.fillStyle = i % 2 === 0 ? "#3f5140" : "#475a45";
      ctx.fillRect(i * cover - drift, horizonY - tuftH, cover * 0.95, tuftH);
    }
    ctx.fillStyle = "rgba(228, 238, 246, 0.5)";
    ctx.fillRect(0, horizonY - H * 0.01, W, H * 0.01);

    ctx.fillStyle = "#4a5a40";
    ctx.fillRect(0, horizonY, W, H - horizonY);

    const roadYFar = demoGroundY(proj, zFar);
    ctx.beginPath();
    ctx.moveTo(xAt(-DEMO_CLIP.ROAD_HALF_M, zFar), roadYFar);
    ctx.lineTo(xAt(DEMO_CLIP.ROAD_HALF_M, zFar), roadYFar);
    ctx.lineTo(xAt(DEMO_CLIP.ROAD_HALF_M, zBottom), H);
    ctx.lineTo(xAt(-DEMO_CLIP.ROAD_HALF_M, zBottom), H);
    ctx.closePath();
    ctx.fillStyle = "#3a3f45";
    ctx.fill();

    const asphalt = ctx.createLinearGradient(0, roadYFar, 0, H);
    asphalt.addColorStop(0, "rgba(255, 255, 255, 0.06)");
    asphalt.addColorStop(1, "rgba(0, 0, 0, 0.22)");
    ctx.fillStyle = asphalt;
    ctx.fill();

    ctx.lineCap = "butt";

    // Solid white line separating opposing traffic, plus the outer edge.
    ctx.strokeStyle = "rgba(236, 240, 243, 0.8)";
    for (const latM of [-DEMO_CLIP.LANE_EDGE_M, -DEMO_CLIP.ROAD_HALF_M + 0.3]) {
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(xAt(latM, zFar), demoGroundY(proj, zFar));
      ctx.lineTo(xAt(latM, zBottom), H);
      ctx.stroke();
    }

    // Dashed lane divider on the right of the ego lane, drawn as a perspective
    // quad: a fixed-pixel rectangle turned the nearest dash into a white pillar.
    const phase = (t * DEMO_CLIP.SPEED_MPS) % DEMO_LANE_DASH_M;
    const dashHalfW = (z) => Math.max(0.9, (focal * DEMO_CLIP.DASH_WIDTH_M) / 2 / z);
    ctx.fillStyle = "rgba(240, 243, 245, 0.82)";
    for (let k = 0; k < 20; k++) {
      const zNear = 6.0 + k * DEMO_LANE_DASH_M + (DEMO_LANE_DASH_M - phase);
      const zFarDash = zNear + DEMO_LANE_DASH_LENGTH_M;
      const yNear = Math.min(H + 4, demoGroundY(proj, zNear));
      const yFarDash = demoGroundY(proj, zFarDash);
      if (yNear <= yFarDash || yFarDash > H) continue;
      const xNear = W / 2 + (focal * DEMO_CLIP.LANE_EDGE_M) / zNear;
      const xFar = W / 2 + (focal * DEMO_CLIP.LANE_EDGE_M) / zFarDash;
      const wNear = dashHalfW(zNear);
      const wFar = dashHalfW(zFarDash);
      ctx.beginPath();
      ctx.moveTo(xNear - wNear, yNear);
      ctx.lineTo(xNear + wNear, yNear);
      ctx.lineTo(xFar + wFar, yFarDash);
      ctx.lineTo(xFar - wFar, yFarDash);
      ctx.closePath();
      ctx.fill();
    }

    // Light poles along the verge. They start at 11 m: a closer pole is a
    // full-height black bar running off both sides of the frame, which reads as
    // a glitch rather than a roadside object.
    const poleSpacing = 13;
    const polePhase = (t * DEMO_CLIP.SPEED_MPS) % poleSpacing;
    for (let k = 0; k < 9; k++) {
      const z = 11 + k * poleSpacing + (poleSpacing - polePhase);
      const poleY = demoGroundY(proj, z);
      if (poleY > H + 40) continue;
      const poleH = Math.min(H * 0.8, (focal * 3.4) / z);
      const poleW = Math.max(1.2, (focal * 0.12) / z);
      const armLen = Math.max(2, (focal * 0.9) / z);
      ctx.fillStyle = "#2b2f33";
      for (const side of [-1, 1]) {
        const poleX = xAt(side * (DEMO_CLIP.ROAD_HALF_M + 0.9), z);
        if (poleX < -40 || poleX > W + 40) continue;
        ctx.fillRect(poleX - poleW / 2, poleY - poleH, poleW, poleH);
        ctx.fillRect(
          side < 0 ? poleX - poleW / 2 : poleX - armLen,
          poleY - poleH,
          armLen,
          Math.max(1, poleW * 0.8)
        );
      }
    }
  }

  function drawDemoFrame(ctx, proj, t) {
    drawDemoRoad(ctx, proj, t);
    for (const box of demoBoxesAt(t, proj.width, proj.height)) {
      if (box.cls_name === "motorcycle") {
        drawDemoScooter(ctx, box);
      } else {
        drawDemoCar(ctx, box);
      }
    }
  }

  function demoPickMimeType() {
    if (typeof MediaRecorder === "undefined") return "";
    if (MediaRecorder.isTypeSupported("video/webm;codecs=vp9")) {
      return "video/webm;codecs=vp9";
    }
    return MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "";
  }

  async function generateDemoVideoBlob(options) {
    const opts = options || {};
    const onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;

    if (typeof MediaRecorder === "undefined") {
      throw new Error(
        "This browser cannot record a canvas stream (MediaRecorder unavailable)."
      );
    }

    const { WIDTH: width, HEIGHT: height, FPS: fps, DURATION_S: durationS } = DEMO_CLIP;
    const totalFrames = Math.round(durationS * fps);
    const frameMs = 1000 / fps;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const proj = demoProjection(width, height);

    const stream = canvas.captureStream(fps);
    const mimeType = demoPickMimeType();
    const chunks = [];
    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2200000 })
      : new MediaRecorder(stream);
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    };
    const stopped = new Promise((resolve) => {
      recorder.onstop = () => {
        resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
      };
    });

    drawDemoFrame(ctx, proj, 0);
    recorder.start(1000);
    const startedAt = performance.now();

    // Pace against the wall clock: one frame per 1/fps second, drifting delays
    // corrected, so the recorded file really is DURATION_S long.
    for (let f = 1; f < totalFrames; f++) {
      const wait = startedAt + f * frameMs - performance.now();
      if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
      drawDemoFrame(ctx, proj, f / fps);
      if (onProgress && f % 25 === 0) onProgress(f / totalFrames);
    }
    await new Promise((resolve) => setTimeout(resolve, frameMs * 2));
    recorder.stop();
    if (onProgress) onProgress(1);

    const blob = await stopped;
    blob._syntheticDetections = demoSyntheticDetections(width, height, fps, durationS);
    return blob;
  }

  global.MotoTrackEngine = {
    CONFIG,
    DEMO_CLIP,
    SimpleByteTracker,
    DetectionSmoother,
    ClosingRateTracker,
    VerdictStabilizer,
    computeGap,
    corridorCenterCovered,
    estimateDistanceM,
    decide,
    interpolateDetections,
    drawOverlayOnCanvas,
    renderTimelineDataUrl,
    mapPredictionsToDetections,
    generateDemoVideoBlob,
    demoBoxesAt,
    demoSyntheticDetections,
  };
})(window);

