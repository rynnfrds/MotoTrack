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
    const xFrac = ((det.x1 + det.x2) / 2.0) / imageWidthPx;
    const yFrac = det.y2 / imageHeightPx;
    return pointInCorridor(xFrac, yFrac);
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

  // --- Synthetic Demo Video Generator ---------------------------------------

  async function generateDemoVideoBlob() {
    const width = 640;
    const height = 360;
    const fps = 20;
    const durationS = 12;
    const totalFrames = fps * durationS;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    // We also return ground-truth synthetic detections per timestamp so if WebGL
    // is slow or offline, or when running the demo clip, real vehicle boxes are
    // guaranteed to be detected!
    const syntheticDetectionsByFrame = [];

    const stream = canvas.captureStream(fps);
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
      ? "video/webm"
      : "";

    const chunks = [];
    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    const donePromise = new Promise((resolve) => {
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
        resolve(blob);
      };
    });

    recorder.start();

    const horizonY = Math.round(height * CONFIG.CORRIDOR_HORIZON_Y);

    for (let f = 0; f < totalFrames; f++) {
      const t = f / fps;

      // Sky
      const skyGrad = ctx.createLinearGradient(0, 0, 0, horizonY);
      skyGrad.addColorStop(0, "#6fa3d2");
      skyGrad.addColorStop(1, "#c8ddf0");
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, width, horizonY);

      // Road
      ctx.fillStyle = "#373d45";
      ctx.fillRect(0, horizonY, width, height - horizonY);

      // Shoulders
      ctx.fillStyle = "#4d5c46";
      ctx.beginPath();
      ctx.moveTo(0, horizonY);
      ctx.lineTo(width * 0.36, horizonY);
      ctx.lineTo(width * 0.05, height);
      ctx.lineTo(0, height);
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(width, horizonY);
      ctx.lineTo(width * 0.64, horizonY);
      ctx.lineTo(width * 0.95, height);
      ctx.lineTo(width, height);
      ctx.fill();

      // Dashed lane markings moving toward camera
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 3;
      const offset = (t * 6) % 1;
      for (let i = 0; i < 6; i++) {
        const p0 = ((i + offset) / 6);
        const p1 = Math.min(1, p0 + 0.07);
        const y0 = horizonY + Math.pow(p0, 1.8) * (height - horizonY);
        const y1 = horizonY + Math.pow(p1, 1.8) * (height - horizonY);
        ctx.beginPath();
        ctx.moveTo(width * 0.5, y0);
        ctx.lineTo(width * 0.5, y1);
        ctx.stroke();
      }

      // Scenario timeline:
      // 0 - 3.5s: Clear corridor (1 far vehicle on left shoulder -> SAFE)
      // 3.5 - 7.5s: Scooter + car narrowing the corridor -> TIGHT
      // 7.5 - 10.5s: Car directly ahead in near band blocking corridor -> BLOCKED
      // 10.5 - 12.0s: Vehicle pulls aside -> SAFE
      const frameBoxes = [];

      // Always keep a distant reference vehicle on the left lane so detection_count >= 1
      const refCar = {
        cls_name: "car",
        conf: 0.92,
        x1: width * 0.16,
        y1: horizonY + 25,
        x2: width * 0.27,
        y2: horizonY + 85,
      };
      frameBoxes.push(refCar);

      if (t >= 3.2 && t < 7.5) {
        // Scooter on right side of corridor (leaves moderate gap -> TIGHT)
        const prog = Math.min(1, (t - 3.2) / 1.5);
        const cx = width * (0.58 - 0.03 * Math.sin(t * 2));
        const by = horizonY + 115 + prog * 25;
        const bw = 54;
        const bh = 92;
        frameBoxes.push({
          cls_name: "motorcycle",
          conf: 0.91,
          x1: cx - bw / 2,
          y1: by - bh,
          x2: cx + bw / 2,
          y2: by,
        });
      }

      if (t >= 7.2 && t < 10.4) {
        // Large car/truck dead-center in near corridor -> BLOCKED
        const cx = width * 0.50;
        const by = height * 0.91;
        const bw = 185;
        const bh = 165;
        frameBoxes.push({
          cls_name: "car",
          conf: 0.96,
          x1: cx - bw / 2,
          y1: by - bh,
          x2: cx + bw / 2,
          y2: by,
        });
      }

      // Draw vehicles onto canvas so video looks realistic
      for (const b of frameBoxes) {
        const w = b.x2 - b.x1;
        const h = b.y2 - b.y1;
        ctx.fillStyle = b.cls_name === "motorcycle" ? "#1f2937" : "#2563eb";
        ctx.fillRect(b.x1, b.y1, w, h);
        // Windshield / rear window
        ctx.fillStyle = "#93c5fd";
        ctx.fillRect(b.x1 + w * 0.15, b.y1 + h * 0.12, w * 0.7, h * 0.28);
        // Taillights
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(b.x1 + 4, b.y1 + h * 0.55, w * 0.2, h * 0.12);
        ctx.fillRect(b.x2 - w * 0.2 - 4, b.y1 + h * 0.55, w * 0.2, h * 0.12);
      }

      syntheticDetectionsByFrame.push({ t, boxes: frameBoxes });
      await new Promise((r) => setTimeout(r, 12));
    }

    recorder.stop();
    const blob = await donePromise;
    blob._syntheticDetections = syntheticDetectionsByFrame;
    return blob;
  }

  global.MotoTrackEngine = {
    CONFIG,
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
  };
})(window);

