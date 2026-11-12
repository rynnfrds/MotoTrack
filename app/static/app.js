const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const browse = document.getElementById("browse");
const filenameEl = document.getElementById("filename");
const submit = document.getElementById("submit");
const form = document.getElementById("upload");
const errorEl = document.getElementById("error");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");
const statusPct = document.getElementById("statusPct");
const fill = document.getElementById("fill");

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

let selected = null;

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

function setFile(file) {
  selected = file || null;
  filenameEl.textContent = file ? `${file.name} - ${(file.size / 1048576).toFixed(1)} MB` : "no file selected";
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

async function poll(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) throw new Error("lost track of the job");
  const job = await response.json();

  const pct = Math.round((job.progress || 0) * 100);
  fill.style.width = `${pct}%`;
  statusPct.textContent = `${pct}%`;

  if (job.status === "done") {
    statusText.textContent = "Finished";
    fill.style.width = "100%";
    window.location.href = `/results/${jobId}`;
    return;
  }

  if (job.status === "failed") {
    statusEl.hidden = true;
    submit.disabled = false;
    showError(job.error || "analysis failed");
    return;
  }

  statusText.textContent = job.status === "queued" ? "Queued" : "Analysing";
  setTimeout(() => poll(jobId).catch((error) => {
    statusEl.hidden = true;
    submit.disabled = false;
    showError(error.message);
  }), 900);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selected) return;

  submit.disabled = true;
  showError("");
  statusEl.hidden = false;
  statusText.textContent = "Uploading";
  statusPct.textContent = "4%";
  fill.style.width = "4%";

  const body = new FormData();
  body.append("file", selected);
  body.append("detector", document.getElementById("detector").value);

  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "upload rejected");
    }
    statusText.textContent = "Queued";
    poll(payload.id).catch((error) => {
      statusEl.hidden = true;
      submit.disabled = false;
      showError(error.message);
    });
  } catch (error) {
    statusEl.hidden = true;
    submit.disabled = false;
    showError(error.message);
  }
});