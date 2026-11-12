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
  wrap.className = "stat-row";
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  wrap.append(dt, dd);
  return wrap;
}

const sidebar = document.getElementById("sidebar");
const menuToggle = document.getElementById("menuToggle");
const sheetClose = document.getElementById("sheetClose");
function closeMenu() {
  sidebar.classList.remove("open");
  if (menuToggle) menuToggle.setAttribute("aria-expanded", "false");
}
if (menuToggle) {
  menuToggle.addEventListener("click", () => {
    sidebar.classList.add("open");
    menuToggle.setAttribute("aria-expanded", "true");
    if (sheetClose) sheetClose.focus();
  });
}
if (sheetClose) sheetClose.addEventListener("click", closeMenu);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMenu();
});

function render(summary) {
  const analysis = summary.analysis || {};
  const video = summary.video || {};
  const shares = summary.verdict_share || {};

  const pct = (key) => Math.round((shares[key] || 0) * 100);
  const safeShare = pct("SAFE");
  const tightShare = pct("TIGHT");
  const blockedShare = pct("BLOCKED");
  const unknownShare = pct("UNKNOWN");
  lede.textContent =
    `${safeShare}% of analysed time read SAFE, ${tightShare}% TIGHT, ${blockedShare}% BLOCKED` +
    `${unknownShare ? `, ${unknownShare}% UNKNOWN` : ""}.`;

  document.getElementById("kpiSafe").innerHTML = `${safeShare}<small>%</small>`;
  document.getElementById("kpiTight").innerHTML = `${tightShare}<small>%</small>`;
  document.getElementById("kpiBlocked").innerHTML = `${blockedShare}<small>%</small>`;
  document.getElementById("kpiUnknown").innerHTML = `${unknownShare}<small>%</small>`;

  statsEl.replaceChildren(
    stat("Frames analysed", `${analysis.analysed_frames || 0} at ${analysis.analysed_fps || 10} fps`),
    stat("Throughput", `${analysis.wall_fps || 0} frames/s`),
    stat("Mean latency", `${analysis.mean_latency_ms || 0} ms`),
    stat("Source", `${video.width || "?"} \u00d7 ${video.height || "?"}`),
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

  if (timeline && timeline.tagName === "IMG") {
    timeline.addEventListener("load", () => { playhead.hidden = false; });
    if (timeline.complete) playhead.hidden = false;
  }
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