"""FastAPI app: upload a clip, get an annotated video and a verdict timeline."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mototrack import config
from mototrack.io import VideoError, probe
from mototrack.pipeline import analyse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = config.JOBS_DIR
ALLOWED = set(config.ALLOWED_UPLOAD_EXTENSIONS)

app = FastAPI(title="MotoTrack", docs_url="/api/docs")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@dataclass
class Job:
    id: str
    filename: str
    detector: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def dir(self) -> str:
        return os.path.join(JOBS_DIR, self.id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "detector": self.detector,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _run_job(job: Job, video_path: str) -> None:
    try:
        job.status = "running"
        job.message = "analysing"

        def progress(index: int, total: int) -> None:
            if total > 0:
                job.progress = min(0.999, index / total)

        result = analyse(
            video_path,
            out_dir=os.path.join(job.dir, "out"),
            detector_kind=job.detector,
            progress_cb=progress,
            max_seconds=config.MAX_WEB_VIDEO_SECONDS,
        )
        job.progress = 1.0
        job.status = "done"
        job.message = "finished"
        job.result_summary = result.summary  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
    finally:
        job.finished_at = time.time()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "detectors": sorted(config.MODEL_FILES),
            "default_detector": config.DEFAULT_DETECTOR,
            "max_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_seconds": int(config.MAX_WEB_VIDEO_SECONDS),
        },
    )


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    detector: str = Form(config.DEFAULT_DETECTOR),
):
    if detector not in config.MODEL_FILES:
        raise HTTPException(400, f"unknown detector {detector!r}")

    name = os.path.basename(file.filename or "")
    extension = os.path.splitext(name)[1].lower()
    if extension not in ALLOWED:
        raise HTTPException(
            400,
            f"unsupported file type {extension or chr(40) + chr(110) + chr(111) + chr(110) + chr(101) + chr(41)}; "
            f"use one of {sorted(ALLOWED)}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, f"source{extension}")

    size = 0
    try:
        with open(video_path, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"file too large; limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"could not store upload: {exc}") from exc

    if size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "uploaded file is empty")

    try:
        meta = probe(video_path)
    except VideoError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"not a readable video: {exc}") from exc

    job = Job(id=job_id, filename=name, detector=detector)
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_job, args=(job, video_path), daemon=True, name=f"job-{job_id}"
    )
    thread.start()

    return {
        "id": job_id,
        "meta": {
            "width": meta.width,
            "height": meta.height,
            "fps": round(meta.fps, 2),
            "duration_s": round(meta.duration_s, 2),
        },
        "truncated_to_seconds": (
            config.MAX_WEB_VIDEO_SECONDS
            if meta.duration_s > config.MAX_WEB_VIDEO_SECONDS
            else None
        ),
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _get_job(job_id)
    payload = job.to_dict()
    if job.status == "done":
        payload["result"] = {
            "video": f"/media/{job_id}/out/annotated.mp4",
            "timeline": f"/media/{job_id}/out/timeline.png",
            "events": f"/api/jobs/{job_id}/events",
            "summary": f"/api/jobs/{job_id}/summary",
        }
        payload["summary"] = getattr(job, "result_summary", None)
    return payload


def _get_job(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


@app.get("/api/jobs/{job_id}/events")
async def get_events(job_id: str):
    job = _get_job(job_id)
    path = os.path.join(job.dir, "out", "events.json")
    if not os.path.exists(path):
        raise HTTPException(404, "events not ready")
    with open(path, encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/api/jobs/{job_id}/summary")
async def get_summary(job_id: str):
    job = _get_job(job_id)
    path = os.path.join(job.dir, "out", "summary.json")
    if not os.path.exists(path):
        raise HTTPException(404, "summary not ready")
    with open(path, encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    job = _get_job(job_id)
    return templates.TemplateResponse(
        request, "result.html", {"job": job, "job_dict": job.to_dict()}
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "port": config.DEFAULT_PORT, "detectors": sorted(config.MODEL_FILES)}


os.makedirs(JOBS_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=JOBS_DIR), name="media")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
