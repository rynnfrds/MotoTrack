import sys
sys.path.insert(0, "src")
import cv2
from mototrack.detectors import build_detector
from mototrack.geometry import compute_gap, is_blocking_candidate, corridor_x_bounds, estimate_distance_m, distance_band

cap = cv2.VideoCapture("data/raw/clip_001.mp4")
det = build_detector("mixed")
for want in (107, 195, 263):
    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    ds = det.infer(frame, persist=False)
    gap = compute_gap(ds, w, h)
    print("--- frame", want, w, "x", h, " dets=", len(ds))
    for d in ds:
        dist = estimate_distance_m(d.cls_name, d.height, w)
        print("  ", d.cls_name, "h=", round(d.height), "d=", round(dist,1), "band=", distance_band(dist),
              "bx=", round(d.bottom_center_x), "(", round(d.bottom_center_x/w,2), ")",
              "by=", round(d.bottom_y), "(", round(d.bottom_y/h,2), ")",
              "blocking=", is_blocking_candidate(d, w, h))
    print("  => candidates=", gap.n_candidates, "band=", gap.band, "gap_px=", round(gap.gap_px),
          "ref_w=", round(gap.reference_width_px), "ratio=", round(gap.gap_ratio,2))
    b = corridor_x_bounds(1.0)
    print("  corridor bottom:", round(b[0]*w), "..", round(b[1]*w))
