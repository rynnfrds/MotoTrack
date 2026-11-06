# Metrics

Everything below was measured on this machine, on one clip, against labels from
one person. It is reported at that resolution on purpose: there is no claim of
general detector accuracy here.

## Test conditions

| Item | Value |
|---|---|
| Clip | `clip_001.mp4` from the Visaitech mixed-traffic archive (anonymised dashcam, South Asia) |
| Resolution and rate | 640x480, 20 fps, 14.65 s, 293 frames |
| Analysed | 147 frames at 10 fps analysis rate |
| Gold set | 29 frames sampled evenly, labelled by eye against SAFE / TIGHT / BLOCKED |
| Machine | Windows, CPU only (no CUDA device present) |
| Python / torch / ultralytics | 3.12 / 2.15.0+cpu / 8.4.202 |

## Detector comparison

Same gold set, same geometry, same rules. Only the checkpoint differs:

| Detector | Binary accuracy (SAFE vs not-SAFE) | 3-class accuracy | Scored | UNKNOWN |
|---|---|---|---|---|
| `coco` | 41.7% | 33.3% | 12 | 17 |
| `mixed` | **69.0%** | **41.4%** | 29 | 0 |

The `coco` row is misleading in a useful way: it abstains on 17 of 29 frames
(`UNKNOWN`) rather than guessing, which is why its percentage is computed over
only 12 frames. The mixed-traffic checkpoint found something to say on every
frame and was right more often. On this footage, `mixed` is the better default.

That sentence is scoped to this footage. It is not a claim that the mixed model
is better in general.

## Effect of calibration

| Thresholds | 3-class accuracy | Binary accuracy | Frames called SAFE that a human called BLOCKED |
|---|---|---|---|
| Guessed (1.6 / 1.2) | 20.7% | 62.1% | 0 |
| Fitted (1.8 / 0.4) | **41.4%** | **69.0%** | 0 |

Sweeping 780 threshold pairs against 29 labels roughly doubled 3-class accuracy.
The dangerous error - calling a blocked corridor safe - stayed at zero both
before and after, and the sweep selected the fitted pair only from candidates
that kept it at zero.

With 29 frames this fit is weak evidence. It is a starting point that is
honestly derived, not a validated calibration. Re-run `scripts/calibrate.py`
whenever the gold set grows; the numbers will move.

## Per-class breakdown (`mixed`, fitted thresholds)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SAFE | 7.1% | 33.3% | 11.8% | 3 |
| TIGHT | 100.0% | 8.7% | 16.0% | 23 |
| BLOCKED | 0.0% | 0.0% | n/a | 3 |

Read this honestly: the model is not distinguishing three classes well. It is
mostly calling things `BLOCKED`, which happens to be safe but is not useful, and
`TIGHT` recall is poor. The 3-class accuracy of 41.4% is far from usable.

The binary number (69%) is the one that matters for the decision the system is
actually trying to support, and it is still not good enough for anything but a
demo.

## Runtime

| Metric | Value |
|---|---|
| Mean latency | 94.2 ms per analysed frame |
| p95 latency | 86.4 ms |
| Wall throughput | 8.5 fps at `imgsz=640`, CPU |
| Analysis rate | 10 fps regardless of source rate |
| A 15 s clip end to end | ~17 s |

A 60 s clip takes roughly 70 s on this machine. The web UI caps uploads at 90
seconds for that reason.

## Display stability (flicker)

Frame-by-frame detection jitters, and tracks briefly vanish and reappear. On video
that reads as the whole overlay strobing, which looks like a weak model but is a
display and tracking problem. Two defects were found and fixed, and both are now
covered by regression tests.

### Defect 1: skipped frames were written un-annotated

Analysis runs at 10 fps regardless of source rate. Frames between analysis ticks
were written to the output video **raw**, so boxes and the HUD appeared on one
frame and vanished on the next. At 60 fps that was 5 of every 6 output frames.

Measured on a 60 fps clip, after the fix: **0 of 360 output frames** lacked the
overlay, down from 5-in-6. Regression tests in `tests/test_pipeline.py` fail if a
raw frame is ever written again.

### Defect 2: boxes jittered and blinked out

| Metric | Before | After | Change |
|---|---|---|---|
| Mean box jitter (px/frame) | 6.52 | 2.24 | **66% less** |
| Disappear-then-reappear events | 56 | 10 | **82% fewer** |

Measured over 180 analysed frames of a 60 fps clip with the `mixed` detector.
Fixes: an exponential moving average on each track's box corners
(`BOX_SMOOTHING_ALPHA = 0.45`), and coasting a missed track for
`TRACK_HOLD_SECONDS = 0.4`, covering the observed 0.1-0.3 s dropouts without
hiding an object that genuinely left.

Coasted boxes are drawn **dashed with a `~` prefix** so an inferred box is never
mistaken for an observed one, and their confidence decays each missed sample so
they expire quickly.

**This does not change measured accuracy.** Scoring is per-frame and stateless, so
the gold-set numbers above are identical before and after. Smoothing changes what
the video looks like, not how often the verdict is right. That distinction matters:
it is easy to make a demo look calmer and assume it got smarter.

Coasting fails toward caution: a coasted box still counts as an obstacle, because
forgetting a real vehicle is worse than briefly over-reporting one.

### Defect 3: the overlay was frozen between analysis ticks

Holding the last analysed state fixed the strobing but introduced the opposite
problem: at 10 analysed fps on a 60 fps source, 5 of every 6 output frames showed
a **byte-identical** overlay. Boxes advanced in visible steps, which reads as the
tracking lagging behind the video.

Fix: frames between ticks are buffered and written only once the *next* analysis
tick is known, with each box's corners interpolated between the two surrounding
states. Interpolation matches tracks by id; a track that appears or disappears
between ticks is held at its known position rather than slid in from an unrelated
object's coordinates.

| Metric | Held last state | Interpolated |
|---|---|---|
| Output frame pairs with a static overlay | 67% | **10%** |
| Median changed pixels per frame | 1 | **3,636** |

Measured on 360 output frames of a 60 fps clip. The verification runs against the
video served by the web API, not just the CLI, so the deployed path is covered.
Regression tests in `tests/test_overlay_smoothness.py` fail if frames stop being
interpolated (confirmed: the probe drops the moving-frame ratio to 15%).

**Note on cause and effect:** none of the three display defects change verdict
accuracy. The gold-set numbers are identical throughout. They change what the
video looks like, which is what makes a demo feel broken even when the decision
logic is unchanged.

## What is deliberately not measured

- **Detector mAP.** No labelled bounding-box test split exists for this footage,
  so no mAP figure is claimed for either checkpoint on this domain.
- **Real-world overtaking safety.** Nobody has validated that a `SAFE` verdict
  corresponds to a genuinely safe pass. That would need instrumented riding and
  is out of scope for a learning project.
- **Cross-clip generalisation.** One clip, one location, one lighting condition.
- **Night and rain.** Untested.

## How to reproduce

```bash
.venv/Scripts/python.exe scripts/make_gold.py data/raw/clip_001.mp4 --count 30
# label data/gold/gold.csv by eye
.venv/Scripts/python.exe -m mototrack.cli compare-detectors
.venv/Scripts/python.exe scripts/calibrate.py --detector mixed
```