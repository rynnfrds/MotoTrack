"""Score a detector against hand-labelled frames.

The gold set is a CSV the user fills in by eye: one row per sampled frame with
the verdict a human would give. Everything here compares those labels with what
the pipeline reports for the same frame.

Two caveats are baked into the report rather than left implicit:

* Detector accuracy is not claimed. There is no labelled bounding-box test split
  for this footage, so no mAP is reported. Only verdict agreement with a human.
* Binary and 3-class accuracy are reported separately, because SAFE vs not-SAFE
  is the decision a rider actually makes, while TIGHT vs BLOCKED is a nuance.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from . import config
from .detectors import build_detector
from .geometry import compute_gap, corridor_center_covered
from .verdict import decide

GOLD_COLUMNS = ("video", "frame", "verdict")
VALID_LABELS = (config.VERDICT_SAFE, config.VERDICT_TIGHT, config.VERDICT_BLOCKED)


@dataclass
class GoldRow:
    video: str
    frame: int
    verdict: str


@dataclass
class ConfusionMatrix:
    labels: tuple = VALID_LABELS
    counts: dict = field(default_factory=dict)

    def add(self, truth: str, predicted: str) -> None:
        key = (truth, predicted)
        self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, truth: str, predicted: str) -> int:
        return self.counts.get((truth, predicted), 0)

    def recall(self, label: str) -> float | None:
        total = sum(self.get(label, p) for p in self.labels)
        return self.get(label, label) / total if total else None

    def precision(self, label: str) -> float | None:
        total = sum(self.get(t, label) for t in self.labels)
        return self.get(label, label) / total if total else None

    def f1(self, label: str) -> float | None:
        precision, recall = self.precision(label), self.recall(label)
        if not precision or not recall:
            return None
        return 2 * precision * recall / (precision + recall)

    def accuracy(self) -> float | None:
        total = sum(sum(self.get(t, p) for p in self.labels) for t in self.labels)
        correct = sum(self.get(x, x) for x in self.labels)
        return correct / total if total else None


def load_gold(path: str) -> list[GoldRow]:
    """Read the gold CSV, rejecting rows a human would have to fix."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"gold set not found: {path}\n"
            "create it with: python scripts/make_gold.py"
        )
    rows: list[GoldRow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in GOLD_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"gold CSV missing columns: {missing}")
        for line_no, raw in enumerate(reader, start=2):
            label = (raw.get("verdict") or "").strip().upper()
            if not label:
                continue
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_no}: verdict {label!r} is not one of {VALID_LABELS}"
                )
            try:
                frame_index = int(raw["frame"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}:{line_no}: frame {raw.get(chr(102) + chr(114) + chr(97) + chr(109) + chr(101))!r} is not an integer"
                ) from None
            rows.append(GoldRow(video=raw["video"].strip(), frame=frame_index, verdict=label))
    return rows


def _frame_at(video_path: str, frame_index: int):
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame {frame_index} of {video_path}")
        return frame
    finally:
        cap.release()


def predict_frame(detector, frame) -> str:
    """Verdict for a single still frame, without temporal context.

    A gold row is one frame, so there is no empty-run or closing-rate history to
    consult. A visibly empty corridor counts as confirmed: the human labelling
    the frame can see it directly.
    """
    height, width = frame.shape[:2]
    detections = detector.infer(frame, persist=False)
    gap = compute_gap(detections, width, height)
    result = decide(
        gap,
        detection_count=len(detections),
        closing_fast=False,
        center_covered=corridor_center_covered(detections, width, height),
        empty_corridor_confirmed=gap.n_candidates == 0,
    )
    return result.verdict


def evaluate(gold_path: str, detector_kind: str = config.DEFAULT_DETECTOR) -> dict:
    """Score one detector against the gold set."""
    rows = load_gold(gold_path)
    if not rows:
        return {"total": 0, "detector": detector_kind, "error": "gold set is empty"}

    detector = build_detector(detector_kind)
    three_class = ConfusionMatrix()
    binary_correct = 0
    binary_total = 0
    unknown = 0
    per_frame: list[dict] = []
    checked_videos: set[str] = set()

    for row in rows:
        if row.video not in checked_videos:
            if not os.path.exists(row.video):
                raise FileNotFoundError(f"gold row references missing video: {row.video}")
            checked_videos.add(row.video)
        still = _frame_at(row.video, row.frame)
        predicted = predict_frame(detector, still)
        if predicted == config.VERDICT_UNKNOWN:
            unknown += 1
            continue
        three_class.add(row.verdict, predicted)
        binary_total += 1
        if (row.verdict == config.VERDICT_SAFE) == (predicted == config.VERDICT_SAFE):
            binary_correct += 1
        per_frame.append(
            {
                "video": row.video,
                "frame": row.frame,
                "truth": row.verdict,
                "predicted": predicted,
            }
        )

    return {
        "detector": detector_kind,
        "gold": os.path.abspath(gold_path),
        "total": len(rows),
        "scored": binary_total,
        "unknown": unknown,
        "binary_accuracy": (binary_correct / binary_total) if binary_total else None,
        "three_class_accuracy": three_class.accuracy(),
        "confusion": {
            truth: {predicted: three_class.get(truth, predicted) for predicted in VALID_LABELS}
            for truth in VALID_LABELS
        },
        "per_class": {
            label: {
                "precision": three_class.precision(label),
                "recall": three_class.recall(label),
                "f1": three_class.f1(label),
                "support": sum(three_class.get(label, p) for p in VALID_LABELS),
            }
            for label in VALID_LABELS
        },
        "frames": per_frame,
        "caveats": [
            "Metrics describe verdict agreement with one human labeller, not detector mAP.",
            "No labelled bounding-box split exists for this footage, so no mAP is claimed.",
            "Single-frame scoring omits the temporal stabilisation used in the full pipeline.",
        ],
    }


def compare_detectors(gold_path: str) -> dict:
    """Run every detector on the same gold set so the winner is empirical."""
    results = {kind: evaluate(gold_path, kind) for kind in sorted(config.MODEL_FILES)}
    ranked = sorted(
        (k for k, v in results.items() if v.get("binary_accuracy") is not None),
        key=lambda k: results[k]["binary_accuracy"],
        reverse=True,
    )
    return {"gold": os.path.abspath(gold_path), "results": results, "ranking": ranked}


def fmt_pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{100.0 * value:5.1f}%"


def format_report(report: dict) -> str:
    if report.get("total", 0) == 0:
        return "no labelled frames to score: " + str(report.get("error", ""))

    detector = report["detector"]
    gold = report["gold"]
    total = report["total"]
    scored = report["scored"]
    unknown = report["unknown"]
    confusion = report["confusion"]
    per_class = report["per_class"]

    lines = [
        f"detector     : {detector}",
        f"gold set     : {gold}",
        f"frames       : {total} labelled, {scored} scored, {unknown} UNKNOWN",
        f"binary acc   : {fmt_pct(report['binary_accuracy'])}  (SAFE vs not-SAFE)",
        f"3-class acc  : {fmt_pct(report['three_class_accuracy'])}",
        "",
        "confusion (rows truth, cols predicted)",
    ]
    header = " " * 14 + "".join(f"{label:>10}" for label in VALID_LABELS)
    lines.append(header)
    for truth in VALID_LABELS:
        cells = "".join(f"{confusion[truth][pred]:>10}" for pred in VALID_LABELS)
        lines.append(f"{truth:<14}{cells}")

    lines.append("")
    lines.append(f"{'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for label in VALID_LABELS:
        metrics = per_class[label]
        precision = fmt_pct(metrics["precision"])
        recall = fmt_pct(metrics["recall"])
        f1 = fmt_pct(metrics["f1"])
        support = metrics["support"]
        lines.append(f"{label:<10}{precision:>8}{recall:>8}{f1:>8}{support:>9}")

    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"note: {caveat}")
    return "\n".join(lines)


def format_comparison(report: dict) -> str:
    lines = [f"gold set: {report['gold']}", ""]
    lines.append(f"{'detector':<10}{'binary acc':>12}{'3-class acc':>14}{'scored':>9}{'unknown':>9}")
    for kind, result in report["results"].items():
        if result.get("total", 0) == 0:
            lines.append(f"{kind:<10}{'n/a':>12}")
            continue
        binary = fmt_pct(result["binary_accuracy"])
        three = fmt_pct(result["three_class_accuracy"])
        lines.append(
            f"{kind:<10}{binary:>12}{three:>14}{result['scored']:>9}{result['unknown']:>9}"
        )
    lines.append("")
    if report["ranking"]:
        lines.append(f"best on this gold set: {report['ranking'][0]}")
    else:
        lines.append("no detector produced a scored verdict")
    lines.append("")
    lines.append(
        "This ranking is evidence from your own footage, "
        "not a claim about either model in general."
    )
    return "\n".join(lines)
