# Model card

## What this system is

A demonstration pipeline that estimates whether an ego motorcycle has lateral
space to pass, from a single forward-facing camera. It pairs a pretrained object
detector with explicit monocular geometry and a rule layer.

It is a learning project. It is not a rider-assistance system, not a safety
component, and must not be used to make or inform real riding decisions.

## Components

### Detector: `coco` (default)

- Checkpoint: `yolo26n.pt` from the official Ultralytics assets release.
- Architecture: YOLO26n, COCO-pretrained, 80 classes.
- Filtered downstream to road-user classes (person, bicycle, car, motorcycle,
  bus, train, truck).
- Size on disk: 5.5 MB.
- Source: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`

### Detector: `mixed`

- Checkpoint: `visaitech/vehicle-mixed-traffic-yolo26n` (`best.pt`).
- Architecture: YOLO26n fine-tuned on the Visaitech Mixed-Traffic Vehicle
  Detection dataset (v0.1).
- Classes: `pedestrian`, `2-wheeler`, `3-wheeler`, `4-wheeler`.
- Size on disk: 5.38 MB.
- Source: `https://huggingface.co/visaitech/vehicle-mixed-traffic-yolo26n`

Its published model card reports, on a 54-image validation split held out by
source video: **mAP50 0.719, recall 0.734** on original frames, and **mAP50
0.629** on the anonymised public frames. It is an early v0.1 baseline trained on
239 images: weak on pedestrians, untested at night. Those are the authors'
numbers, not ours, and they are not a claim about this footage.

### Depth model (present, not used by the default path)

`yolo26n-depth.pt` (13.0 MB) is downloaded by `fetch_models.py --all` for future
experiments. The shipped pipeline uses the pinhole estimate below instead.

## How the decision is made

1. Detect and track each frame (ByteTrack by default).
2. Estimate range per object with the pinhole similar-triangles relation:
   `distance_m = (real_height_m * focal_px) / box_height_px`, with
   `focal_px = (width_px / 2) / tan(hfov / 2)` and an assumed horizontal field
   of view of 78 degrees. Assumed heights per class live in `config.py`.
3. Keep objects whose **bottom-centre** sits inside the ego corridor, which is a
   trapezoid fixed in relative frame coordinates. Bottom-centre is used because
   the centre of a distant box drifts upward while its wheels stay on the road.
4. At the nearest occupied depth band, subtract the blocked x-intervals from the
   corridor to get the free gaps, and take the widest.
5. Normalise: `gap_ratio = widest_free_gap_px / median_vehicle_width_px` **at the
   same band**. Normalising by vehicle width makes the number scale-invariant,
   so it survives zoom and perspective without calibration.
6. Classify against calibrated thresholds, then apply two overrides: a vehicle
   sitting on the corridor centre line caps the verdict at `TIGHT`, and a fast
   closing rate (TTC under ~1.5 s) drops the verdict one severity level.
7. Stabilise with a 2 s median filter and 0.5 s hysteresis.

### Why the thresholds are fitted, not chosen

The first values were guesses (1.6 / 1.2). `scripts/calibrate.py` swept 780
threshold pairs against the hand-labelled gold set and selected **1.8 / 0.4** as
the pair maximising agreement while never calling a frame `SAFE` that the human
called `BLOCKED`. That constraint is deliberate: a false `SAFE` invites a rider
to commit to a manoeuvre they cannot abort, while a false `BLOCKED` only costs
time. `tests/test_verdict.py` asserts the shipped numbers so they cannot silently
revert to guesses.

## Failure modes, explicitly

| Situation | Behaviour |
|---|---|
| Night, rain, tunnel, heavy glare | Detection count collapses; verdict becomes `UNKNOWN` with reason `low_detections` rather than guessing. |
| Camera roll while leaning | The corridor is frame-fixed, so a lean rotates the road out of it. Documented, not corrected by default. |
| Oncoming traffic | Not modelled. The verdict only describes space ahead in your corridor. |
| Side-lane entry, vehicles about to change lane | Not modelled. There is no motion prediction of other road users. |
| A bus where a car is expected | `4-wheeler` unifies sedans through buses, so assumed height is wrong and the metre estimate degrades. The band and the ratio are more robust than the metre figure. |
| Ego bike bodywork in frame | Discounted only when the box is anchored at the bottom edge **and** spans 75% or more of frame width, so a genuinely close vehicle is never masked away. |
| Portrait 9:16 footage | Supported; the corridor is defined in relative coordinates. |

## Training data and provenance

No checkpoint shipped with this project was trained by it. Both are third-party.

`scripts/colab_train.ipynb` provides an optional fine-tuning path: it starts from
`yolo26n.pt` and trains on the Visaitech mixed-traffic dataset (239 train / 54 val
labelled frames; the 203 raw clips, CC-BY-4.0, are not needed). Training it is not
required for any feature described here.

The notebook includes a fix for a real defect in that dataset: its published
`data.yaml` sets `path: .`, which Ultralytics resolves against the current working
directory, so training fails with `images not found` unless the path is rewritten
to an absolute one.

## Licence

The Ultralytics package and both checkpoints are **AGPL-3.0**. This project
inherits that obligation: if it is ever offered as a network service, the source
must be made available. Weights are not vendored into the repository; they are
fetched by `scripts/fetch_models.py` into a gitignored `models/` directory.

## Intended use

Learning, portfolio demonstration, and a worked example of measuring a
monocular-geometry heuristic honestly: state the assumptions, fit the thresholds
to labelled data, and report accuracy against a human rather than claiming a
detector benchmark.