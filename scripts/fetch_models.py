#!/usr/bin/env python
"""Download the pretrained checkpoints MotoTrack uses.

Weights are third-party (Ultralytics AGPL-3.0, Visaitech AGPL-3.0) and are not
vendored into the repo. They land in models/, which is gitignored.

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py --detector mixed
    python scripts/fetch_models.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mototrack import config  # noqa: E402


def download(url: str, dest: str, approx_bytes: int | None = None) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if approx_bytes is None or size == approx_bytes:
            print(f"already present: {dest} ({size:,} bytes)")
            return dest
        print(f"size mismatch on {dest} ({size:,} != {approx_bytes:,}), re-downloading")

    tmp = dest + ".part"
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    size = os.path.getsize(tmp)
    if approx_bytes is not None and size != approx_bytes:
        os.remove(tmp)
        raise RuntimeError(
            f"size mismatch after download: got {size:,} expected {approx_bytes:,}"
        )
    os.replace(tmp, dest)
    print(f"saved {dest} ({size:,} bytes)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        choices=sorted(config.MODEL_FILES),
        default=config.DEFAULT_DETECTOR,
        help="which detector checkpoint to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--all", action="store_true", help="fetch every model including depth"
    )
    args = parser.parse_args(argv)

    kinds = sorted(config.MODEL_FILES) if args.all else [args.detector]
    for kind in kinds:
        spec = config.MODEL_FILES[kind]
        download(
            spec["url"],
            os.path.join(config.MODELS_DIR, spec["filename"]),
            spec["approx_bytes"],
        )
    if args.all:
        spec = config.DEPTH_MODEL
        download(
            spec["url"],
            os.path.join(config.MODELS_DIR, spec["filename"]),
            spec["approx_bytes"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
