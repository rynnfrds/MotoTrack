"""API tests run in-process: no port is bound, no server is started."""

from __future__ import annotations

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mototrack import config  # noqa: E402

FAKE_VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Redirect job storage before importing the app so the real app/jobs stays clean.
    config.JOBS_DIR = str(tmp_path_factory.mktemp("jobs"))
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_detectors(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert set(payload["detectors"]) == set(config.MODEL_FILES)


def test_index_page_lists_detectors(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Bisa nyalip" in response.text
    for detector in config.MODEL_FILES:
        assert detector in response.text


def test_rejects_disallowed_extension(client):
    response = client.post(
        "/api/jobs",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"detector": "coco"},
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]


def test_rejects_unknown_detector(client):
    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", io.BytesIO(FAKE_VIDEO), "video/mp4")},
        data={"detector": "nope"},
    )
    assert response.status_code == 400
    assert "unknown detector" in response.json()["detail"]


def test_rejects_empty_file(client):
    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
        data={"detector": "coco"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_bytes_that_are_not_video(client):
    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", io.BytesIO(b"this is not a video" * 100), "video/mp4")},
        data={"detector": "coco"},
    )
    assert response.status_code == 400
    assert "not a readable video" in response.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/events").status_code == 404
    assert client.get("/results/deadbeef").status_code == 404


def test_oversize_upload_is_rejected(client, tmp_path):
    original = config.MAX_UPLOAD_BYTES
    config.MAX_UPLOAD_BYTES = 1024
    try:
        response = client.post(
            "/api/jobs",
            files={"file": ("clip.mp4", io.BytesIO(FAKE_VIDEO * 4), "video/mp4")},
            data={"detector": "coco"},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]
    finally:
        config.MAX_UPLOAD_BYTES = original

def test_upload_over_one_megabyte_is_parsed(client):
    """A realistic upload must get past multipart parsing.

    Starlette's multipart parser defaults max_part_size to 1 MiB for non-file
    parts and spools file data through a SpooledTemporaryFile, so an upload above
    1 MB exercises a different code path (disk spooling) than a tiny one. A real
    clip is always above that threshold, so pin the behaviour.
    """
    big = FAKE_VIDEO * 600  # ~1.24 MB, over the 1 MiB default part size
    assert len(big) > 1024 * 1024

    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", io.BytesIO(big), "video/mp4")},
        data={"detector": "coco"},
    )
    # It is not a valid video, so a 400 is expected -- but the reason must be the
    # video itself, never multipart parsing.
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "parsing the body" not in detail, (
        f"multipart parsing rejected a legal-size upload: {detail}"
    )
    assert "not a readable video" in detail