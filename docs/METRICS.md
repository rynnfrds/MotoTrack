# Metrics

Everything below was measured on this machine, on one clip, against labels from
one person. It is reported at that resolution on purpose: there is no claim of
general detector accuracy here.

## Test conditions

| Item | Value |
|---|---|
| Clip | `clip_001.mp4`, portrait ego-POV motovlog footage from a public YouTube Short (`youtube.com/shorts/5zzcCtHc3xg`) |
| Resolution and rate | 720x1280, 29.92 fps, 37.22 s, 1113 frames |
| Analysed | 371 frames at 10 fps analysis rate (frame step 3) |
| Gold set | 30 frames sampled evenly, labelled by eye against SAFE / TIGHT / BLOCKED |
| Machine | Windows, CPU only (no CUDA device present) |
| Python / torch / ultralytics | 3.12 / 2.15.0+cpu / 8.4.202 |

The clip is third-party footage. It is not redistributed in this repository and
`data/raw/` is gitignored; see "Provenance and licence" below.

## Detector comparison

Same gold set, same geometry, same rules. Only the checkpoint differs:

| Detector | Binary accuracy (SAFE vs not-SAFE) | 3-class accuracy | Scored | UNKNOWN |
|---|---|---|---|---|
| `coco` | **100.0%** | **41.7%** | 24 | 6 |
| `mixed` | 74.1% | 37.0% | 27 | 3 |

The `coco` row abstains on 6 of 30 frames (`UNKNOWN`) rather than guessing, which
is why its percentage is computed over 24 frames. On this footage `coco` is the
better default, which reverses the ranking the previous clip produced. That
reversal is the point: the winner is a property of the footage, not of the model.

Read the 100% honestly. It means the rule never contradicted the labeller on the
SAFE-vs-not-SAFE question across 24 frames, and nothing more. The 3-class number
on the same frames is 41.7%, and the per-class table below shows why.

## Effect of calibration

| Thresholds | 3-class accuracy | Binary accuracy | Frames called SAFE that a human called BLOCKED |
|---|---|---|---|
| Guessed (1.6 / 1.2) | 56.7% | 100.0% | 0 |
| Fitted (1.8 / 0.4) | 53.3% | 100.0% | 0 |

Two things stand out, and both are reported rather than smoothed over.

First, the sweep no longer improves on the shipped thresholds. It reports a
**plateau**: every `safe_ratio` from 1.0 to 1.8 with `tight_ratio` 0.4 scores
identically. The shipped pair sits at the conservative end of that plateau
because the data cannot distinguish the candidates within it. Moving to 1.0
would lower the bar for calling a corridor clear on no evidence, so it was not
adopted.

Second, the numbers in this table are computed over all 30 rows with an empty
corridor scored as SAFE, so they use a different denominator from the detector
table above (24 scored rows). Both are stated instead of picking the flattering
one.

With 30 frames this fit is weak evidence. It is a starting point that is
honestly derived, not a validated calibration. Re-run `scripts/calibrate.py`
whenever the gold set grows; the numbers will move.

## Per-class breakdown (`coco`, shipped thresholds)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SAFE | 100.0% | 100.0% | 100.0% | 7 |
| TIGHT | 100.0% | 6.7% | 12.5% | 15 |
| BLOCKED | 12.5% | 100.0% | 22.2% | 2 |

The model is not distinguishing three classes. It separates SAFE from not-SAFE
perfectly on this set, then collapses almost every non-SAFE frame into BLOCKED:
TIGHT recall is 6.7%, so 14 of 15 tight frames are reported as blocked. That
direction is safe and useless at the same time. A rider told BLOCKED when the
slot is merely narrow loses the opportunity; the system never does the opposite.

No frame a human called BLOCKED was reported SAFE. That is the one line this
project treats as non-negotiable, and it held before and after calibration.

## Confusion matrix (`coco`)

Rows are the human label, columns the prediction.

|  | SAFE | TIGHT | BLOCKED |
|---|---|---|---|
| **SAFE** | 7 | 0 | 0 |
| **TIGHT** | 0 | 1 | 14 |
| **BLOCKED** | 0 | 0 | 2 |

## Runtime

| Metric | `coco` | `mixed` |
|---|---|---|
| Mean latency | 50.2 ms per analysed frame | 53.4 ms per analysed frame |
| Wall throughput | 8.36 fps at `imgsz=640`, CPU | 7.73 fps |
| Analysis rate | 10 fps regardless of source rate | same |
| This 37 s clip end to end | 44.35 s | 48.01 s |

These were taken from consecutive runs on the same machine, and that matters: the
first run after a fresh model load recorded 106.7 ms per frame, then 81.0, then
50.2 for the same work. Reporting only the best number would be a lie of
selection, and reporting only the worst would misrepresent steady state, so both
checkpoints here come from the warm runs and the spread is written down.

The clip is portrait 720x1280, so more pixels are pushed through the detector per
frame than the previous 640x480 clip. A 90 s clip takes roughly 100 s. The web UI
caps uploads at 90 seconds for that reason.

## Verdict distribution on the default clip

| Verdict | Duration | Share of analysed time |
|---|---|---|
| BLOCKED | 22.56 s | 60.7% |
| SAFE | 9.52 s | 25.6% |
| UNKNOWN | 3.91 s | 10.5% |
| TIGHT | 1.20 s | 3.2% |

Recovered as 8 stable segments by `events.json`. The clip is a rider threading
between lorries on a two-lane road, so the corridor is genuinely closed for most
of its length. Unlike the previous clip, this one does produce SAFE stretches:
the last 7 seconds run beside a parked queue with the lane clear.

## Display stability (flicker)

Frame-by-frame detection jitters, and tracks briefly vanish and reappear. On video
that reads as the whole overlay strobing, which looks like a weak model but is a
display and tracking problem. Three defects were found and fixed, and all three
are covered by regression tests.

### Defect 1: skipped frames were written un-annotated

Analysis runs at 10 fps regardless of source rate. Frames between analysis ticks
were written to the output video **raw**, so boxes and the HUD appeared on one
frame and vanished on the next. At 60 fps that was 5 of every 6 output frames.

Measured on a 60 fps test clip, after the fix: **0 of 360 output frames** lacked
the overlay, down from 5-in-6. Regression tests in `tests/test_pipeline.py` fail
if a raw frame is ever written again.

### Defect 2: boxes jittered and blinked out

Fixes: an exponential moving average on each track's box corners
(`BOX_SMOOTHING_ALPHA = 0.45`), and coasting a missed track for
`TRACK_HOLD_SECONDS = 0.4`, covering the observed 0.1-0.3 s dropouts without
hiding an object that genuinely left.

On the current default clip, measured by `scripts/measure_display.py`:

| Metric | No smoothing | `alpha = 0.45` | Change |
|---|---|---|---|
| Mean box corner movement | 15.33 px per analysed frame | 12.11 px per analysed frame | **21% less** |

The reduction is smaller than the 66% recorded on the previous 60 fps clip, and
the reason is honest: this footage is a handheld portrait shot of the rider
moving fast past stationary lorries, so apparent box motion is dominated by ego
motion rather than by detector noise. A 0.45 EMA cannot remove motion that is
really there.

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

On the current default clip (1113 output frames, step 3):

| Metric | Result |
|---|---|
| Output frame pairs with a byte-identical HUD region | 2 of 1112 (**0.2%**) |

The previous clip recorded 67% held-state frames and 10% after interpolation. The
current figure is lower because the source runs at 29.9 fps with a step of 3, so
two of every three output frames are interpolated, and because the HUD reads the
per-frame gap measurement, which changes constantly on this footage.

Regression tests in `tests/test_overlay_smoothness.py` fail if frames stop being
interpolated (confirmed: the probe drops the moving-frame ratio to 15%).

**Note on cause and effect:** none of the three display defects change verdict
accuracy. The gold-set numbers are identical throughout. They change what the
video looks like, which is what makes a demo feel broken even when the decision
logic is unchanged.

### Defect 4: side traffic was invisible to the corridor test

This one did change accuracy, and it was found by replacing the clip.

The blocking-candidate test asked whether a box's **bottom-centre point** sits
inside the corridor. That holds for a dashcam looking down a lane, where side
traffic is outside the corridor by definition. It fails for ego-POV footage of a
motorcycle squeezing between two lorries: each lorry's bottom-centre falls
outside the narrow corridor, so both were discarded, the corridor was reported
empty, and the verdict came out SAFE while the rider was threading a slot about
1.5 m wide.

On the 30-frame gold set, before and after the fix:

| Detector | Binary accuracy before | Binary accuracy after | 3-class before | 3-class after |
|---|---|---|---|---|
| `coco` | 45.8% | **100.0%** | 37.5% | 41.7% |
| `mixed` | 37.0% | **74.1%** | 33.3% | 37.0% |

The test now asks whether the box's **horizontal extent overlaps** the corridor
at the object's ground-contact row, plus a horizon check so gantries and bridge
decks above the road are excluded. Three regression tests cover it: a lorry
beside the corridor must still narrow it, an object entirely outside must not
count, and an object above the horizon must not count.

Porting the same change to the browser engine
(`web/static/mototrack-engine.js`) keeps the deployed path consistent with the
CLI. The browser engine's accuracy is still unmeasured.

## Provenance and licence

The default clip is third-party footage obtained from a public YouTube Short
(`youtube.com/shorts/5zzcCtHc3xg`, "macet nyalip motovlog viral"). It is **not**
openly licensed, it is not redistributed in this repository, and `data/raw/` plus
`data/gold/` are gitignored. Anyone reproducing these numbers needs to obtain the
clip themselves and re-label the gold set, because frame indices refer to this
exact file.

The detector checkpoints are a separate matter: two pretrained weights under
AGPL-3.0, fetched by `scripts/fetch_models.py`, never vendored.

## What is deliberately not measured

- **Detector mAP.** No labelled bounding-box test split exists for this footage,
  so no mAP figure is claimed for either checkpoint on this domain.
- **Real-world overtaking safety.** Nobody has validated that a `SAFE` verdict
  corresponds to a genuinely safe pass. That would need instrumented riding and
  is out of scope for a learning project.
- **Cross-clip generalisation.** One clip, one route, one lighting condition.
- **Night and rain.** Untested.
- **The browser engine's accuracy.** The TensorFlow.js path shares the geometry
  but not the detector, and nothing here has been measured for it.

## How to reproduce

```bash
.venv/Scripts/python.exe scripts/make_gold.py data/raw/clip_001.mp4 --count 30
# label data/gold/gold.csv by eye
.venv/Scripts/python.exe -m mototrack.cli compare-detectors
.venv/Scripts/python.exe scripts/calibrate.py --detector coco
.venv/Scripts/python.exe scripts/measure_display.py
```
