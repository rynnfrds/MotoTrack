/* MotoTrack Web Application — Supports both In-Browser AI (Vercel) and FastAPI Backend */

const uploadView = document.getElementById("uploadView");
const resultView = document.getElementById("resultView");

const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const browse = document.getElementById("browse");
const filenameEl = document.getElementById("filename");
const submit = document.getElementById("submit");
const demoBtn = document.getElementById("demoBtn");
const form = document.getElementById("upload");
const errorEl = document.getElementById("error");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");
const statusPct = document.getElementById("statusPct");
const fill = document.getElementById("fill");
const livePreviewWrap = document.getElementById("livePreviewWrap");
const liveCanvas = document.getElementById("liveCanvas");

const sidebar = document.getElementById("sidebar");
const menuToggle = document.getElementById("menuToggle");
const sheetClose = document.getElementById("sheetClose");
const navAnalyse = document.getElementById("navAnalyse");
const navResult = document.getElementById("navResult");
const navResultNote = document.getElementById("navResultNote");
const screenName = document.getElementById("screenName");
const scopeLine = document.getElementById("scopeLine");

const backLink = document.getElementById("backLink");
const newClipBtn = document.getElementById("newClipBtn");
const resultTitle = document.getElementById("resultTitle");
const resultEngineBadge = document.getElementById("resultEngineBadge");
const lede = document.getElementById("lede");
const player = document.getElementById("player");
const overlayCanvas = document.getElementById("overlayCanvas");
const toggleHud = document.getElementById("toggleHud");
const frameReadout = document.getElementById("frameReadout");
const stripTrack = document.getElementById("stripTrack");
const timelineWrap = document.getElementById("timelineWrap");
const playhead = document.getElementById("playhead");
const stripStart = document.getElementById("stripStart");
const stripEnd = document.getElementById("stripEnd");
const gapChartEl = document.getElementById("gapChart");
const statsEl = document.getElementById("stats");
const segTableBody = document.querySelector("#segTable tbody");
const segCount = document.getElementById("segCount");

let selected = null;
let cocoModelPromise = null;
let activeTicks = []; // [{ t, state }] for interactive playback interpolation
let activeDuration = 1.0;
let currentResult = null;
let gapChartLabels = [];
let gapChartValues = [];

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

function setFile(file) {
  selected = file || null;
  filenameEl.textContent = file
    ? `${file.name} - ${(file.size / 1048576).toFixed(1)} MB`
    : "no file selected";
  submit.disabled = !file;
  showError("");
}

browse.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((name) => {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
});

["dragleave", "drop"].forEach((name) => {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("over");
  });
});

drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) setFile(file);
});

function setScreen(which) {
  const onResult = which === "result";
  uploadView.hidden = onResult;
  resultView.hidden = !onResult;
  navAnalyse.setAttribute("aria-current", onResult ? "false" : "page");
  navResult.setAttribute("aria-current", onResult ? "page" : "false");
  navResult.setAttribute("aria-disabled", currentResult ? "false" : "true");
  screenName.textContent = onResult ? "Result" : "Analyse";
  document.title = onResult ? "MotoTrack - result" : "MotoTrack - can I pass here?";
  closeMenu();
}

function switchToUpload() {
  player.pause();
  statusEl.hidden = true;
  livePreviewWrap.hidden = true;
  submit.disabled = !selected;
  demoBtn.disabled = false;
  setScreen("analyse");
}

function openMenu() {
  sidebar.classList.add("open");
  menuToggle.setAttribute("aria-expanded", "true");
  sheetClose.focus();
}

function closeMenu() {
  sidebar.classList.remove("open");
  menuToggle.setAttribute("aria-expanded", "false");
}

menuToggle.addEventListener("click", openMenu);
sheetClose.addEventListener("click", closeMenu);
navAnalyse.addEventListener("click", (event) => {
  event.preventDefault();
  switchToUpload();
});
navResult.addEventListener("click", (event) => {
  event.preventDefault();
  if (currentResult) setScreen("result");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMenu();
});

backLink.addEventListener("click", (e) => {
  e.preventDefault();
  switchToUpload();
});

newClipBtn.addEventListener("click", () => {
  switchToUpload();
});

async function getCocoModel() {
  if (!cocoModelPromise) {
    if (!window.cocoSsd) {
      throw new Error("TensorFlow.js COCO-SSD library did not load. Check your internet connection.");
    }
    cocoModelPromise = window.cocoSsd.load({ base: "lite_mobilenet_v2" });
  }
  return cocoModelPromise;
}

function seekVideo(video, timeSec) {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed to decode video frame"));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = timeSec;
  });
}

// MediaRecorder writes a WebM without a Duration element, so `video.duration`
// is Infinity until the browser has demuxed it. Seeking far past the end makes
// Chrome compute the real duration; without that, analysis silently truncated
// to the old 12 s fallback and the scrubber had no range.
function ensureFiniteDuration(video) {
  const known = () => Number.isFinite(video.duration) && video.duration > 0;
  if (known()) return Promise.resolve(video.duration);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      video.removeEventListener("timeupdate", onDurationChange);
      video.removeEventListener("durationchange", onDurationChange);
      clearTimeout(timer);
      try {
        video.currentTime = 0;
      } catch (error) {
        /* seeking can fail while the stream is still parsing */
      }
      resolve(value);
    };
    const onDurationChange = () => {
      if (known()) finish(video.duration);
    };
    const timer = setTimeout(() => finish(known() ? video.duration : 0), 2000);
    video.addEventListener("timeupdate", onDurationChange);
    video.addEventListener("durationchange", onDurationChange);
    try {
      video.currentTime = 1e7;
    } catch (error) {
      finish(0);
    }
  });
}

function declaredDuration(file) {
  if (Number.isFinite(file._durationSeconds) && file._durationSeconds > 0) {
    return file._durationSeconds;
  }
  const frames = file._syntheticDetections;
  if (Array.isArray(frames) && frames.length > 1) {
    const span = frames[frames.length - 1].t - frames[0].t;
    if (Number.isFinite(span) && span > 0) return span;
  }
  return 0;
}

async function runInBrowserAnalysis(file, detectorKind, maxSeconds) {
  const Engine = window.MotoTrackEngine;
  const CONFIG = Engine.CONFIG;

  statusEl.hidden = false;
  livePreviewWrap.hidden = false;
  fill.style.width = "2%";
  statusPct.textContent = "2%";

  const useSynthetic = Array.isArray(file._syntheticDetections);
  let model = null;
  if (!useSynthetic) {
    statusText.textContent = "Loading COCO-SSD WebGL detector...";
    model = await getCocoModel();
  }

  const videoUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = videoUrl;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Browser could not decode this video format. Use MP4 (H.264) or WebM."));
  });

  const width = video.videoWidth || 640;
  const height = video.videoHeight || 360;
  const probedDuration = await ensureFiniteDuration(video);
  const rawDuration = probedDuration || declaredDuration(file) || 12.0;
  const analysedSeconds = Math.min(rawDuration, maxSeconds);
  const analysisFps = CONFIG.ANALYSIS_FPS;
  const totalTicks = Math.max(1, Math.floor(analysedSeconds * analysisFps));

  liveCanvas.width = width;
  liveCanvas.height = height;
  const liveCtx = liveCanvas.getContext("2d");

  const workCanvas = document.createElement("canvas");
  workCanvas.width = width;
  workCanvas.height = height;
  const workCtx = workCanvas.getContext("2d");

  const tracker = new Engine.SimpleByteTracker();
  const smoother = new Engine.DetectionSmoother();
  const closing = new Engine.ClosingRateTracker();
  const stabilizer = new Engine.VerdictStabilizer();

  const segments = [];
  let currentSegment = null;
  let emptyRun = 0.0;
  const verdictCounts = {};
  const latencies = [];
  const ticks = [];

  const startedAt = performance.now();

  for (let i = 0; i < totalTicks; i++) {
    const tSeconds = Math.min(analysedSeconds - 0.001, i / analysisFps);
    await seekVideo(video, Math.max(0, tSeconds));
    workCtx.drawImage(video, 0, 0, width, height);

    const tickStart = performance.now();
    let rawBoxes;

    if (useSynthetic) {
      // Find closest synthetic frame
      let best = file._syntheticDetections[0];
      for (const item of file._syntheticDetections) {
        if (Math.abs(item.t - tSeconds) < Math.abs(best.t - tSeconds)) {
          best = item;
        }
      }
      rawBoxes = best.boxes.map((b) => ({
        ...b,
        cls_name:
          detectorKind === "mixed" && b.cls_name === "motorcycle"
            ? "2-wheeler"
            : detectorKind === "mixed" && b.cls_name === "car"
            ? "4-wheeler"
            : b.cls_name,
      }));
    } else {
      const preds = await model.detect(workCanvas);
      rawBoxes = Engine.mapPredictionsToDetections(preds, detectorKind);
    }

    const tracked = tracker.assign(rawBoxes, tSeconds);
    const detections = smoother.update(tracked, tSeconds);

    const gap = Engine.computeGap(detections, width, height);
    const centreClosed = Engine.corridorCenterCovered(detections, width, height);

    if (gap.blocking_ids && gap.blocking_ids.length > 0) {
      for (const det of detections) {
        if (gap.blocking_ids.includes(det.track_id)) {
          const dist = Engine.estimateDistanceM(
            det.cls_name,
            Math.max(0, det.y2 - det.y1),
            width
          );
          closing.update(det.track_id, dist);
          break;
        }
      }
    }

    let closingFast = false;
    if (gap.blocking_ids && gap.blocking_ids.length > 0) {
      closingFast = gap.blocking_ids.some((tid) =>
        closing.isClosingFast(tid, analysisFps)
      );
    }

    if (gap.n_candidates === 0) {
      emptyRun += 1.0 / analysisFps;
    } else {
      emptyRun = 0.0;
    }
    const emptyConfirmed = emptyRun >= CONFIG.EMPTY_CONFIRM_SECONDS;

    const raw = Engine.decide(gap, {
      detectionCount: detections.length,
      closingFast,
      centerCovered: centreClosed,
      emptyCorridorConfirmed: emptyConfirmed,
    });
    const shown = stabilizer.update(raw.verdict, analysisFps);

    verdictCounts[shown] = (verdictCounts[shown] || 0) + 1;
    latencies.push(performance.now() - tickStart);

    if (!currentSegment || currentSegment.verdict !== shown) {
      if (currentSegment) {
        currentSegment.t1 = Number(tSeconds.toFixed(2));
        segments.push(currentSegment);
      }
      currentSegment = {
        t0: Number(tSeconds.toFixed(2)),
        t1: Number(tSeconds.toFixed(2)),
        verdict: shown,
        min_ratio: raw.gap_ratio !== null ? Number(raw.gap_ratio.toFixed(3)) : null,
        frame_count: 0,
        track_ids: [],
        reason: raw.reason,
      };
    }
    currentSegment.t1 = Number(tSeconds.toFixed(2));
    currentSegment.frame_count += 1;
    for (const tid of gap.blocking_ids || []) {
      if (tid != null && !currentSegment.track_ids.includes(tid)) {
        currentSegment.track_ids.push(tid);
      }
    }
    if (raw.gap_ratio !== null) {
      if (currentSegment.min_ratio === null) {
        currentSegment.min_ratio = Number(raw.gap_ratio.toFixed(3));
      } else {
        currentSegment.min_ratio = Number(
          Math.min(currentSegment.min_ratio, raw.gap_ratio).toFixed(3)
        );
      }
    }

    const state = {
      detections,
      verdict: shown,
      reason: raw.reason,
      gap,
      rawVerdict: raw.verdict,
    };
    ticks.push({ t: tSeconds, state });

    // Draw live preview
    liveCtx.drawImage(workCanvas, 0, 0, width, height);
    Engine.drawOverlayOnCanvas(liveCtx, width, height, state, tSeconds);

    const pct = Math.min(99, Math.round(((i + 1) / totalTicks) * 100));
    fill.style.width = `${pct}%`;
    statusPct.textContent = `${pct}%`;
    statusText.textContent = `Analysing frame ${i + 1}/${totalTicks} (${shown})`;
  }

  if (currentSegment) {
    currentSegment.t1 = Number(Math.max(currentSegment.t1, analysedSeconds).toFixed(2));
    segments.push(currentSegment);
  }

  const elapsedS = (performance.now() - startedAt) / 1000.0;
  const sortedLat = latencies.slice().sort((a, b) => a - b);
  const meanLat =
    latencies.length > 0
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length
      : 0;
  const p95Lat =
    sortedLat.length > 0
      ? sortedLat[Math.floor(sortedLat.length * 0.95)]
      : 0;

  const verdictSeconds = {};
  const verdictShare = {};
  for (const [v, count] of Object.entries(verdictCounts)) {
    verdictSeconds[v] = Number((count / analysisFps).toFixed(2));
    verdictShare[v] = Number((count / totalTicks).toFixed(4));
  }

  const summary = {
    source: file.name,
    detector: detectorKind,
    engine: useSynthetic ? "browser-simulation" : "tfjs-coco-ssd",
    video: {
      width,
      height,
      fps: 30,
      frame_count: Math.round(rawDuration * 30),
      duration_s: Number(rawDuration.toFixed(2)),
    },
    analysis: {
      analysed_frames: totalTicks,
      analysed_fps: analysisFps,
      elapsed_s: Number(elapsedS.toFixed(2)),
      wall_fps: elapsedS > 0 ? Number((totalTicks / elapsedS).toFixed(2)) : null,
      mean_latency_ms: Number(meanLat.toFixed(1)),
      p95_latency_ms: Number(p95Lat.toFixed(1)),
    },
    verdict_seconds: verdictSeconds,
    verdict_share: verdictShare,
    analysed_seconds: Number(analysedSeconds.toFixed(2)),
    caveats: [
      "Distances are monocular estimates from assumed object heights, not sensor readings.",
      "Verdict describes free lateral space in the ego corridor, not overall traffic safety.",
    ],
  };

  const eventsObj = {
    source: file.name,
    detector: detectorKind,
    thresholds: {
      safe_ratio: CONFIG.VERDICT_SAFE_RATIO,
      tight_ratio: CONFIG.VERDICT_TIGHT_RATIO,
      near_bucket_max_m: CONFIG.NEAR_BUCKET_MAX_M,
      mid_bucket_max_m: CONFIG.MID_BUCKET_MAX_M,
      assumed_hfov_deg: CONFIG.ASSUMED_HFOV_DEG,
    },
    segments,
  };

  const timelineDataUrl = Engine.renderTimelineDataUrl(segments, analysedSeconds);

  return {
    filename: file.name,
    engineLabel: useSynthetic ? "Built-in Demo Simulation" : "Browser WebGL AI (COCO-SSD)",
    videoUrl,
    width,
    height,
    analysedSeconds,
    ticks,
    segments,
    summary,
    eventsObj,
    timelineDataUrl,
  };
}

// --- Remote FastAPI Backend Mode (Optional) -------------------------------

async function runRemoteBackendAnalysis(baseUrl, file, detectorKind) {
  const cleanBase = baseUrl.replace(/\/+$/, "");
  statusEl.hidden = false;
  livePreviewWrap.hidden = true;
  statusText.textContent = "Uploading to FastAPI server...";
  fill.style.width = "5%";

  const body = new FormData();
  body.append("file", file);
  body.append("detector", detectorKind);

  const response = await fetch(`${cleanBase}/api/jobs`, { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Upload rejected by backend");
  }

  const jobId = payload.id;
  while (true) {
    const res = await fetch(`${cleanBase}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error("Lost connection to backend job");
    const job = await res.json();
    const pct = Math.round((job.progress || 0) * 100);
    fill.style.width = `${pct}%`;
    statusPct.textContent = `${pct}%`;
    statusText.textContent = job.status === "queued" ? "Queued on server..." : `Server Analysing (${pct}%)`;

    if (job.status === "failed") {
      throw new Error(job.error || "Server analysis failed");
    }
    if (job.status === "done") {
      const evRes = await fetch(`${cleanBase}${job.result.events}`);
      const eventsObj = evRes.ok ? await evRes.json() : { segments: [] };
      return {
        filename: file.name,
        engineLabel: `FastAPI Server (${detectorKind})`,
        videoUrl: `${cleanBase}${job.result.video}`,
        width: (job.summary && job.summary.video && job.summary.video.width) || 640,
        height: (job.summary && job.summary.video && job.summary.video.height) || 360,
        analysedSeconds: (job.summary && job.summary.analysed_seconds) || 15,
        ticks: null, // Server already burned overlay into the MP4
        segments: eventsObj.segments || [],
        summary: job.summary || {},
        eventsObj,
        timelineDataUrl: `${cleanBase}${job.result.timeline}`,
      };
    }
    await new Promise((r) => setTimeout(r, 900));
  }
}

// --- Result Rendering & Interactive HUD Sync -------------------------------

function stat(label, value) {
  const wrap = document.createElement("div");
  wrap.className = "stat-row";
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrap.append(dt, dd);
  return wrap;
}

function drawStrip(segments, totalSeconds) {
  const total = totalSeconds > 0 ? totalSeconds : 1;
  stripTrack.replaceChildren();
  for (const seg of segments || []) {
    const width = Math.max(0.2, ((seg.t1 - seg.t0) / total) * 100);
    const el = document.createElement("span");
    el.className = `strip-seg ${seg.verdict}`;
    el.style.width = `${width}%`;
    el.title = `${seg.t0.toFixed(1)}-${seg.t1.toFixed(1)} s: ${seg.verdict}`;
    stripTrack.appendChild(el);
  }
}

// The gap ratio over the clip: the one series that decides the verdict, in the style's one chart hue.
function drawGapChart(ticks, totalSeconds) {
  if (!window.Charts || !gapChartEl) return;
  if (!ticks || ticks.length < 2) {
    gapChartEl.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = "<b>No per-frame series</b><span>A server run bakes the overlay into the video and returns segments only.</span>";
    gapChartEl.appendChild(empty);
    return;
  }
  const stride = Math.max(1, Math.round(ticks.length / 60));
  const labels = [];
  const values = [];
  for (let i = 0; i < ticks.length; i += stride) {
    const ratio = ticks[i].state.gap && ticks[i].state.gap.gap_ratio;
    labels.push(`${ticks[i].t.toFixed(1)}`);
    values.push(Number.isFinite(ratio) ? Math.min(4, Number(ratio.toFixed(2))) : null);
  }
  Charts.line(gapChartEl, {
    labels,
    series: [{ name: "Free gap", values }],
    format: (v) => `${v.toFixed(1)}\u00d7`,
    tickFormat: (v) => `${v.toFixed(0)}\u00d7`,
    height: 180,
    max: 4,
    reference: [
      { value: 1.8, label: "SAFE 1.8" },
      { value: 0.4, label: "BLOCKED 0.4" },
    ],
  });
}

function stateAtTime(t) {
  if (!activeTicks || activeTicks.length === 0) return null;
  if (t <= activeTicks[0].t) return activeTicks[0].state;
  const last = activeTicks[activeTicks.length - 1];
  if (t >= last.t) return last.state;

  // Binary search surrounding ticks and interpolate boxes!
  let lo = 0;
  let hi = activeTicks.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (activeTicks[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = activeTicks[lo];
  const b = activeTicks[hi];
  const span = Math.max(1e-6, b.t - a.t);
  const weight = (t - a.t) / span;
  const blendedDetections = window.MotoTrackEngine.interpolateDetections(
    a.state.detections,
    b.state.detections,
    weight
  );
  return {
    ...a.state,
    detections: blendedDetections,
  };
}

function redrawPlayerOverlay() {
  const ctx = overlayCanvas.getContext("2d");
  const t = player.currentTime || 0;
  frameReadout.textContent = `t=${t.toFixed(1)} s`;

  const duration = player.duration || activeDuration || 1.0;
  if (duration > 0 && Number.isFinite(duration)) {
    const frac = Math.min(100, (t / duration) * 100);
    playhead.style.left = `${frac}%`;
    timelineWrap.setAttribute("aria-valuenow", String(Math.round(frac)));
    timelineWrap.setAttribute("aria-valuetext", `${t.toFixed(1)} s of ${duration.toFixed(1)} s`);
  }

  if (!activeTicks || !toggleHud.checked) {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    return;
  }
  const st = stateAtTime(t);
  window.MotoTrackEngine.drawOverlayOnCanvas(
    ctx,
    overlayCanvas.width,
    overlayCanvas.height,
    st,
    t
  );
}

function showResultView(res) {
  currentResult = res;
  setScreen("result");
  navResult.setAttribute("aria-disabled", "false");
  navResultNote.textContent = "ready";

  resultTitle.textContent = res.filename;
  resultEngineBadge.textContent = res.engineLabel;
  scopeLine.textContent = res.engineLabel;

  activeTicks = res.ticks;
  activeDuration = res.analysedSeconds || 1.0;

  overlayCanvas.width = res.width;
  overlayCanvas.height = res.height;
  toggleHud.parentElement.hidden = !res.ticks;

  player.src = res.videoUrl;
  playhead.hidden = false;
  stripStart.textContent = "0 s";
  stripEnd.textContent = `${activeDuration.toFixed(1)} s`;

  // A recorded WebM has no Duration element, so the native scrubber would show
  // no total length. Resolve it once, then rewind so playback starts at 0.
  ensureFiniteDuration(player).then((duration) => {
    if (duration > 0) activeDuration = duration;
    redrawPlayerOverlay();
  });

  const summary = res.summary || {};
  const analysis = summary.analysis || {};
  const video = summary.video || {};
  const shares = summary.verdict_share || {};
  const seconds = summary.verdict_seconds || {};

  const sharePct = (key) => Math.round((shares[key] || 0) * 100);
  const safeShare = sharePct("SAFE");
  const tightShare = sharePct("TIGHT");
  const blockedShare = sharePct("BLOCKED");
  const unknownShare = sharePct("UNKNOWN");

  lede.textContent =
    `${safeShare}% of analysed time read SAFE, ${tightShare}% TIGHT, ${blockedShare}% BLOCKED` +
    `${unknownShare ? `, ${unknownShare}% UNKNOWN` : ""}.`;

  document.getElementById("kpiSafe").innerHTML = `${safeShare}<small>%</small>`;
  document.getElementById("kpiTight").innerHTML = `${tightShare}<small>%</small>`;
  document.getElementById("kpiBlocked").innerHTML = `${blockedShare}<small>%</small>`;
  document.getElementById("kpiUnknown").innerHTML = `${unknownShare}<small>%</small>`;
  const secs = (key) => `${(seconds[key] || 0).toFixed(1)} s of ${res.analysedSeconds.toFixed(1)} s`;
  document.getElementById("kpiSafeFoot").textContent = secs("SAFE");
  document.getElementById("kpiTightFoot").textContent = secs("TIGHT");
  document.getElementById("kpiBlockedFoot").textContent = secs("BLOCKED");
  document.getElementById("kpiUnknownFoot").textContent = secs("UNKNOWN");

  drawStrip(res.segments, res.analysedSeconds);
  drawGapChart(res.ticks, res.analysedSeconds);

  statsEl.replaceChildren(
    stat("Frames analysed", `${analysis.analysed_frames || 0} at ${analysis.analysed_fps || 10} fps`),
    stat("Throughput", `${analysis.wall_fps || 0} frames/s`),
    stat("Mean latency", `${analysis.mean_latency_ms || 0} ms`),
    stat("p95 latency", `${analysis.p95_latency_ms || 0} ms`),
    stat("Source", `${video.width || res.width} \u00d7 ${video.height || res.height}`),
    stat("Detector", summary.detector || "coco"),
    stat("Engine", summary.engine || "in-browser")
  );

  const segments = res.segments || [];
  segCount.textContent = segments.length === 1 ? "1 segment" : `${segments.length} segments`;
  segTableBody.replaceChildren();
  for (const seg of segments) {
    const tr = document.createElement("tr");
    tr.title = "Jump to this segment";
    const ids = seg.track_ids && seg.track_ids.length
      ? seg.track_ids.map((id) => "#" + id).join(", ")
      : "\u2014";
    tr.innerHTML = `
      <td class="cell-head seg-time" data-label="Time">${seg.t0.toFixed(1)} \u2013 ${seg.t1.toFixed(1)} s</td>
      <td class="cell-head" data-label="Verdict"><span class="pill ${seg.verdict}">${seg.verdict}</span></td>
      <td class="n" data-label="Min gap">${seg.min_ratio != null ? seg.min_ratio.toFixed(2) : "\u2014"}</td>
      <td class="n" data-label="Frames">${seg.frame_count}</td>
      <td data-label="Blocking"><span class="track-ids">${ids}</span></td>
      <td data-label="Reason"><span class="reason">${seg.reason || "\u2014"}</span></td>
    `;
    tr.addEventListener("click", () => {
      player.currentTime = seg.t0;
      redrawPlayerOverlay();
    });
    segTableBody.appendChild(tr);
  }

  // Export links
  const eventsBlob = new Blob([JSON.stringify(res.eventsObj, null, 2)], {
    type: "application/json",
  });
  const summaryBlob = new Blob([JSON.stringify(res.summary, null, 2)], {
    type: "application/json",
  });
  document.getElementById("dlEvents").href = URL.createObjectURL(eventsBlob);
  document.getElementById("dlSummary").href = URL.createObjectURL(summaryBlob);
  document.getElementById("dlTimeline").href = res.timelineDataUrl;
  document.getElementById("dlVideo").href = res.videoUrl;

  player.onloadeddata = () => redrawPlayerOverlay();
}

player.addEventListener("timeupdate", redrawPlayerOverlay);
player.addEventListener("seeked", redrawPlayerOverlay);
toggleHud.addEventListener("change", redrawPlayerOverlay);

// Smooth animation frame loop while playing
function animationLoop() {
  if (!resultView.hidden && !player.paused && !player.ended) {
    redrawPlayerOverlay();
  }
  requestAnimationFrame(animationLoop);
}
requestAnimationFrame(animationLoop);

function seekFromPointer(clientX) {
  const rect = timelineWrap.getBoundingClientRect();
  if (!rect.width) return;
  const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  const dur = player.duration || activeDuration || 1.0;
  if (Number.isFinite(dur) && dur > 0) {
    player.currentTime = frac * dur;
    redrawPlayerOverlay();
  }
}

timelineWrap.addEventListener("click", (event) => seekFromPointer(event.clientX));
timelineWrap.addEventListener("keydown", (event) => {
  const dur = player.duration || activeDuration || 1.0;
  if (!Number.isFinite(dur) || dur <= 0) return;
  const step = event.shiftKey ? 5 : 1;
  if (event.key === "ArrowRight") {
    player.currentTime = Math.min(dur, (player.currentTime || 0) + step);
  } else if (event.key === "ArrowLeft") {
    player.currentTime = Math.max(0, (player.currentTime || 0) - step);
  } else if (event.key === "Home") {
    player.currentTime = 0;
  } else if (event.key === "End") {
    player.currentTime = dur;
  } else {
    return;
  }
  event.preventDefault();
  redrawPlayerOverlay();
});

// --- Form Submission & Demo Trigger ---------------------------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selected) return;

  submit.disabled = true;
  demoBtn.disabled = true;
  showError("");

  const detectorKind = document.getElementById("detector").value;
  const maxSeconds = parseFloat(document.getElementById("maxSeconds").value) || 30;
  const backendUrl = (document.getElementById("backendUrl").value || "").trim();

  try {
    let result;
    if (backendUrl) {
      result = await runRemoteBackendAnalysis(backendUrl, selected, detectorKind);
    } else {
      result = await runInBrowserAnalysis(selected, detectorKind, maxSeconds);
    }
    showResultView(result);
  } catch (error) {
    statusEl.hidden = true;
    livePreviewWrap.hidden = true;
    submit.disabled = false;
    demoBtn.disabled = false;
    showError(error.message || String(error));
  }
});

demoBtn.addEventListener("click", async () => {
  submit.disabled = true;
  demoBtn.disabled = true;
  showError("");
  statusEl.hidden = false;
  livePreviewWrap.hidden = true;

  const demoClip = window.MotoTrackEngine.DEMO_CLIP;
  const demoSeconds = demoClip ? demoClip.DURATION_S : 30;
  statusText.textContent = `Recording ${demoSeconds}s traffic POV simulation...`;
  statusPct.textContent = "0%";
  fill.style.width = "0%";

  try {
    const blob = await window.MotoTrackEngine.generateDemoVideoBlob({
      onProgress: (fraction) => {
        const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
        fill.style.width = `${pct}%`;
        statusPct.textContent = `${pct}%`;
        statusText.textContent = `Recording ${demoSeconds}s traffic POV simulation...`;
      },
    });
    const demoFile = new File([blob], "demo_traffic_pov.webm", {
      type: blob.type || "video/webm",
    });
    demoFile._syntheticDetections = blob._syntheticDetections;
    setFile(demoFile);

    const detectorKind = document.getElementById("detector").value;
    const result = await runInBrowserAnalysis(demoFile, detectorKind, demoSeconds);
    showResultView(result);
  } catch (error) {
    statusEl.hidden = true;
    livePreviewWrap.hidden = true;
    submit.disabled = !selected;
    demoBtn.disabled = false;
    showError(error.message || String(error));
  }
});

