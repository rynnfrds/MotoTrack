"""Sweep verdict thresholds against the gold set and report the best split.

The starting thresholds (1.6 / 1.2 vehicle-widths) were guesses. This script
measures every labelled frame once, then evaluates candidate threshold pairs
against the human labels, so the shipped defaults are fitted rather than
asserted.

It reports 3-class accuracy AND a conservative alternative: the thresholds
that maximise accuracy while never calling a frame SAFE that the human called
BLOCKED. Missing a blocked corridor is the expensive error, so both are shown.

Usage:
    python scripts/calibrate.py --gold data/gold/gold.csv --detector mixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from mototrack import config  # noqa: E402
from mototrack.detectors import build_detector  # noqa: E402
from mototrack.evaluate import _frame_at, load_gold  # noqa: E402
from mototrack.geometry import compute_gap, corridor_center_covered  # noqa: E402

SAFE = config.VERDICT_SAFE
TIGHT = config.VERDICT_TIGHT
BLOCKED = config.VERDICT_BLOCKED


def measure(gold_path: str, detector_kind: str) -> list:
    """One row per labelled frame: gap ratio plus the flags decide() consults."""
    rows = load_gold(gold_path)
    detector = build_detector(detector_kind)
    measured = []
    for row in rows:
        frame = _frame_at(row.video, row.frame)
        height, width = frame.shape[:2]
        detections = detector.infer(frame, persist=False)
        gap = compute_gap(detections, width, height)
        measured.append(
            {
                "truth": row.verdict,
                "ratio": gap.gap_ratio,
                "n_candidates": gap.n_candidates,
                "band": gap.band,
                "center_covered": corridor_center_covered(detections, width, height),
            }
        )
    return measured


def classify(ratio: float, safe_ratio: float, tight_ratio: float) -> str:
    if ratio >= safe_ratio:
        return SAFE
    if ratio >= tight_ratio:
        return TIGHT
    return BLOCKED


def score(measured: list, safe_ratio: float, tight_ratio: float) -> dict:
    correct = 0
    scored = 0
    unsafe_safe = 0
    binary_correct = 0
    for row in measured:
        if row["n_candidates"] == 0:
            predicted = SAFE
        else:
            predicted = classify(row["ratio"], safe_ratio, tight_ratio)
            if row["center_covered"] and predicted == SAFE:
                predicted = TIGHT
        scored += 1
        if predicted == row["truth"]:
            correct += 1
        if (predicted == SAFE) == (row["truth"] == SAFE):
            binary_correct += 1
        if predicted == SAFE and row["truth"] == BLOCKED:
            unsafe_safe += 1
    return {
        "safe_ratio": round(safe_ratio, 3),
        "tight_ratio": round(tight_ratio, 3),
        "accuracy": correct / scored if scored else 0.0,
        "binary_accuracy": binary_correct / scored if scored else 0.0,
        "unsafe_safe": unsafe_safe,
        "scored": scored,
    }


def sweep(measured: list) -> list:
    results = []
    for safe_ratio in [round(1.0 + 0.1 * i, 2) for i in range(26)]:
        for tight_ratio in [round(0.4 + 0.1 * i, 2) for i in range(30)]:
            if tight_ratio >= safe_ratio:
                continue
            results.append(score(measured, safe_ratio, tight_ratio))
    return results


def fmt_row(name: str, row: dict) -> str:
    cells = [
        name.ljust(22),
        format(row["safe_ratio"], "7.2f"),
        format(row["tight_ratio"], "7.2f"),
        format(100.0 * row["accuracy"], "9.1f") + "%",
        format(100.0 * row["binary_accuracy"], "8.1f") + "%",
        format(row["unsafe_safe"], "13d"),
    ]
    return "".join(cells)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=os.path.join(config.DATA_DIR, "gold", "gold.csv"))
    parser.add_argument(
        "--detector", choices=sorted(config.MODEL_FILES), default=config.DEFAULT_DETECTOR
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    measured = measure(args.gold, args.detector)
    if not measured:
        print("gold set is empty", file=sys.stderr)
        return 1

    results = sweep(measured)
    best_accuracy = max(results, key=lambda r: (r["accuracy"], r["binary_accuracy"]))
    safest = [r for r in results if r["unsafe_safe"] == 0] or results
    best_conservative = max(safest, key=lambda r: (r["accuracy"], r["binary_accuracy"]))
    current = score(measured, config.VERDICT_SAFE_RATIO, config.VERDICT_TIGHT_RATIO)

    print("gold set : " + os.path.abspath(args.gold))
    print("detector : " + args.detector)
    print("frames   : " + str(len(measured)))
    print("")
    header = (
        "source".ljust(22) + "safe".rjust(7) + "tight".rjust(7)
        + "3-class".rjust(10) + "binary".rjust(9) + "unsafe SAFE".rjust(13)
    )
    print(header)
    for name, row in (
        ("current defaults", current),
        ("best accuracy", best_accuracy),
        ("no unsafe SAFE", best_conservative),
    ):
        print(fmt_row(name, row))
    print("")
    print("unsafe SAFE = frames a human called BLOCKED that the rule called SAFE.")
    print("Shipping the conservative row trades accuracy for never claiming a false clear road.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "detector": args.detector,
                    "frames": len(measured),
                    "current": current,
                    "best_accuracy": best_accuracy,
                    "best_conservative": best_conservative,
                    "top10": sorted(results, key=lambda r: r["accuracy"], reverse=True)[:10],
                },
                fh,
                indent=2,
            )
        print("")
        print("written to " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
