const jobId = window.JOB_ID;
const lede = document.getElementById("lede");
const loading = document.getElementById("loading");
const fill = document.getElementById("fill");
const progressText = document.getElementById("progressText");
const done = document.getElementById("done");
const failed = document.getElementById("failed");
const failText = document.getElementById("failText");
const player = document.getElementById("player");
const timeline = document.getElementById("timeline");
const playhead = document.getElementById("playhead");
const statsEl = document.getElementById("stats");

const VERDICT_CLASS = { SAFE: "safe", TIGHT: "tight", BLOCKED: "blocked", UNKNOWN: "unknown" };

function stat(label, value) {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrap.append(dt, dd);
  return wrap;
}

function render(summary) {
  const analysis = summary.analysis || {};
  const video = summary.video || {};
  const shares = summary.verdict_share || {};

  const blockedShare = Math.round((shares.BLOCKED || 0) * 100);
  const safeShare = Math.round((shares.SAFE || 0) * 100);
  lede.textContent = `${safeShare}% of analysed time read SAFE, ${blockedShare}% read BLOCKED.`;

  statsEl.replaceChildren(
    stat("Verdict SAFE", `${safeShare}% of time`),
    stat("Verdict BLOCKED", `${blockedShare}% of time`),
    stat("Verdict TIGHT", `${Math.round((shares.TIGHT || 0) * 100)}% of time`),
    stat("Verdict UNKNOWN", `${Math.round((shares.UNKNOWN || 0) * 100)}% of time`),
    stat("Analysed", `${analysis.analysed_frames || 0} frames`),
    stat("Throughput", `${analysis.wall_fps || 0} fps`),
    stat("Latency", `${analysis.mean_latency_ms || 0} ms/frame`),
    stat("Source", `${video.width}x${video.height}, ${video.duration_s}s`),
    stat("Detector", summary.detector || "unknown"),
  );
}

function markFailure(message) {
  loading.hidden = true;
  failed.hidden = false;
  failText.textContent = message;
}

function poll() {
  fetch(`/api/jobs/${jobId}`)
    .then((response) => {
      if (!response.ok) throw new Error("No such job - it may have been cleaned up.");
      return response.json();
    })
    .then((job) => {
      const pct = Math.round((job.progress || 0) * 100);
      fill.style.width = `${pct}%`;
      progressText.textContent = job.status === "queued" ? "Queued..." : `Analysing - ${pct}%`;

      if (job.status === "failed") {
        markFailure(job.error || "analysis failed");
        return;
      }
      if (job.status === "done") {
        show(job);
        return;
      }
      setTimeout(poll, 1000);
    })
    .catch((error) => markFailure(error.message));
}

function show(job) {
  loading.hidden = true;
  done.hidden = false;
  player.src = job.result.video;
  player.poster = "";
  timeline.src = job.result.timeline;
  document.getElementById("dlEvents").href = job.result.events;
  document.getElementById("dlSummary").href = job.result.summary;
  document.getElementById("dlVideo").href = job.result.video;
  if (job.summary) render(job.summary);

  timeline.addEventListener("load", () => { playhead.hidden = false; });
  player.addEventListener("timeupdate", () => {
    const duration = player.duration;
    if (!duration || !isFinite(duration)) return;
    playhead.style.left = `${(player.currentTime / duration) * 100}%`;
  });
  player.addEventListener("click", () => {
    const rect = timeline.getBoundingClientRect();
    if (!rect.width) return;
  });
}

poll();