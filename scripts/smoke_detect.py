"""Smoke test: both detectors load and produce Detections on a still image."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import cv2, numpy as np
from mototrack.detectors import build_detector

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if img_path:
    frame = cv2.imread(img_path)
    assert frame is not None, f"cannot read {img_path}"
else:
    # synthetic nonsense frame: just proves the plumbing runs end to end
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[200:400, 250:420] = (40, 90, 160)

for kind in ("coco", "mixed"):
    t0 = time.perf_counter()
    det = build_detector(kind)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = det.infer(frame, persist=False)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"{kind}: load {load_s:.2f}s  infer {infer_ms:.0f}ms  detections {len(out)}")
    for d in out[:6]:
        print(f"   id={d.track_id} {d.cls_name} conf={d.conf:.2f} box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})")
