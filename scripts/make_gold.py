#!/usr/bin/env python
"""Build a gold-labelling sheet from one or more clips.

Scoring needs a human answer key. This samples frames, renders a contact sheet
so labels can be assigned quickly, and writes a CSV to fill in.

Usage:
    python scripts/make_gold.py data/raw/clip_001.mp4 --count 60
    # then open data/gold/sheet_001.jpg and edit data/gold/gold.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mototrack import config  # noqa: E402
from mototrack.evaluate import GOLD_COLUMNS  # noqa: E402
from mototrack.io import iter_frames, probe  # noqa: E402


def sample_indices(frame_count: int, count: int) -> list[int]:
    """Evenly spaced frame indices, deterministic so the sheet can be re-made."""
    if count <= 0 or frame_count <= 0:
        return []
    if count >= frame_count:
        return list(range(frame_count))
    step = frame_count / float(count)
    return [int(i * step) for i in range(count)]


def write_contact_sheet(clip: str, indices: list[int], out_path: str, cols: int = 6) -> str:
    """Tile the sampled frames into one image with their frame numbers burned in."""
    import cv2
    import numpy as np

    wanted = set(indices)
    picked: dict[int, "np.ndarray"] = {}
    for index, frame in iter_frames(clip):
        if index in wanted:
            picked[index] = frame
        if len(picked) == len(wanted):
            break

    if not picked:
        raise RuntimeError(f"could not read any of the requested frames from {clip}")

    ordered = [picked[i] for i in indices if i in picked]
    labels = [i for i in indices if i in picked]
    thumb_w = 320
    first = ordered[0]
    thumb_h = int(first.shape[0] * thumb_w / first.shape[1])
    rows = (len(ordered) + cols - 1) // cols

    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 30, dtype=np.uint8)
    for slot, (thumb, frame_index) in enumerate(zip(ordered, labels)):
        resized = cv2.resize(thumb, (thumb_w, thumb_h))
        r, c = divmod(slot, cols)
        sheet[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = resized
        cv2.putText(
            sheet, f"f{frame_index}", (c * thumb_w + 6, r * thumb_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            sheet, f"f{frame_index}", (c * thumb_w + 6, r * thumb_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


def append_rows(gold_path: str, clip: str, indices: list[int]) -> int:
    """Add unlabelled rows for this clip, keeping any existing labels intact."""
    existing: dict[tuple, str] = {}
    if os.path.exists(gold_path):
        with open(gold_path, newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                key = (raw.get("video", ""), raw.get("frame", ""))
                existing[key] = raw.get("verdict", "")

    added = 0
    for frame_index in indices:
        key = (clip, str(frame_index))
        if key not in existing:
            existing[key] = ""
            added += 1

    os.makedirs(os.path.dirname(os.path.abspath(gold_path)) or ".", exist_ok=True)
    with open(gold_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(GOLD_COLUMNS)
        for (video, frame), verdict in sorted(existing.items()):
            writer.writerow([video, frame, verdict])
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", help="one or more video files")
    parser.add_argument("--count", type=int, default=60, help="frames per clip (default: 60)")
    parser.add_argument("--gold", default=os.path.join(config.DATA_DIR, "gold", "gold.csv"))
    parser.add_argument("--sheets", default=os.path.join(config.DATA_DIR, "gold"))
    args = parser.parse_args(argv)

    for clip_index, clip in enumerate(args.clips, start=1):
        meta = probe(clip)
        indices = sample_indices(meta.frame_count, args.count)
        if not indices:
            print(f"{clip}: no frames to sample", file=sys.stderr)
            continue
        sheet_path = os.path.join(
            args.sheets, f"sheet_{clip_index:02d}_{os.path.splitext(os.path.basename(clip))[0]}.jpg"
        )
        write_contact_sheet(clip, indices, sheet_path)
        added = append_rows(args.gold, clip, indices)
        print(f"{clip}: {len(indices)} frames sampled, {added} new rows")
        print(f"  contact sheet: {sheet_path}")

    print(f"\nNow edit {args.gold} and set verdict to SAFE / TIGHT / BLOCKED per row.")
    print("Then score it with: mototrack eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
