"""Pluggable detector backends.

Two pretrained checkpoints are supported and both are optional at runtime:

* "coco"  - Ultralytics YOLO26n on COCO. Broad, well tested, weak on the
  scooter-dense mixed traffic common in Indonesia.
* "mixed" - a YOLO26n fine-tune for South Asian mixed traffic
  (pedestrian / 2-wheeler / 3-wheeler / 4-wheeler). Narrower but closer to the
  target domain.

Neither is declared the winner here. The evaluation harness runs both against
the same gold set and docs/METRICS.md records which one won.

Both backends yield the same Detection record so downstream geometry does not
care which model produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    """One tracked object in one frame, in absolute pixel coordinates."""

    track_id: int | None
    cls_name: str
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    # True when the box was coasted through a detector dropout rather than
    # observed. Drawn dashed and still counted as an obstacle, because missing a
    # real vehicle is worse than briefly over-reporting one.
    predicted: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def bottom_y(self) -> float:
        return self.y2

    @property
    def bottom_center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2.0


class Detector(Protocol):
    """Contract every backend must satisfy."""

    name: str

    def infer(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Return tracked detections for one BGR frame."""
        ...

    def reset(self) -> None:
        """Drop tracker state between videos."""
        ...


def model_path(kind: str) -> str:
    spec = config.MODEL_FILES[kind]
    return os.path.join(config.MODELS_DIR, spec["filename"])


class _UltralyticsDetector:
    """Shared Ultralytics plumbing; subclasses only declare their class policy."""

    name = "base"
    kind = "coco"

    def __init__(
        self,
        weights: str | None = None,
        imgsz: int = config.IMGSZ,
        conf: float = config.CONF_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        tracker: str = config.DEFAULT_TRACKER,
        allowed_class_ids: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # imported lazily: torch import is slow

        self.weights_path = weights or model_path(self.kind)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"weights not found: {self.weights_path}\n"
                f"run: python scripts/fetch_models.py --detector {self.kind}"
            )
        self.model = YOLO(self.weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.allowed_class_ids = (
            tuple(allowed_class_ids) if allowed_class_ids is not None else None
        )
        self.device = device

    def _keep(self, cls_id: int) -> bool:
        if self.allowed_class_ids is None:
            return True
        return cls_id in self.allowed_class_ids

    def infer(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=persist,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = results[0].names
        ids = boxes.id
        id_values: Iterable[int | None]
        if ids is None:
            id_values = [None] * len(boxes)
        else:
            id_values = [int(v) for v in ids.tolist()]

        detections: list[Detection] = []
        for track_id, cls_id, conf, xyxy in zip(
            id_values,
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
        ):
            cls_id = int(cls_id)
            if not self._keep(cls_id):
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            detections.append(
                Detection(
                    track_id=track_id,
                    cls_name=str(names.get(cls_id, str(cls_id))),
                    cls_id=cls_id,
                    conf=float(conf),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def reset(self) -> None:
        # Ultralytics keeps predictor state on the model. Dropping it forces a
        # fresh tracker for the next video instead of carrying stale track ids.
        try:
            self.model.predictor = None
        except AttributeError:  # pragma: no cover - ultralytics internals
            pass


class CocoDetector(_UltralyticsDetector):
    """YOLO26n on COCO, filtered to vehicle-ish obstacle classes."""

    name = "coco"
    kind = config.DETECTOR_COCO

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("allowed_class_ids", config.COCO_OBSTACLE_CLASS_IDS)
        super().__init__(**kwargs)


class MixedTrafficDetector(_UltralyticsDetector):
    """YOLO26n fine-tune for South Asian mixed traffic (all classes count)."""

    name = "mixed"
    kind = config.DETECTOR_MIXED

    def __init__(self, **kwargs) -> None:
        # Every class in this taxonomy is a road user; keep them all.
        kwargs.setdefault("allowed_class_ids", None)
        super().__init__(**kwargs)


DETECTORS = {
    config.DETECTOR_COCO: CocoDetector,
    config.DETECTOR_MIXED: MixedTrafficDetector,
}


def build_detector(kind: str = config.DEFAULT_DETECTOR, **kwargs) -> Detector:
    try:
        cls = DETECTORS[kind]
    except KeyError:
        raise ValueError(
            f"unknown detector {kind!r}; choose from {sorted(DETECTORS)}"
        ) from None
    return cls(**kwargs)
