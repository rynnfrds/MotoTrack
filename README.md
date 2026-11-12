# MotoTrack

Ego-POV motorcycle footage in, a per-moment answer out: **can I pass here, or not?**

Upload a motovlog clip and every moment gets a verdict:

| Verdict | Meaning |
|---|---|
| `SAFE` | The corridor ahead is clear enough to commit to a pass. |
| `TIGHT` | Space exists, but it is narrow. |
| `BLOCKED` | The corridor is closed or a vehicle occupies the line. |
| `UNKNOWN` | The detector could not see well enough to have an opinion. |

This is a learning, portfolio-scale project. It is **not** a rider-assistance
system and must not be used as one. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## What it does

```
video -> YOLO26n detect + track -> corridor geometry -> gap ratio -> verdict
      -> annotated mp4 + events.json + summary.json + timeline.png
```

- **Detect and track** road users with a pretrained YOLO26n checkpoint.
- **Project an ego corridor** as a trapezoid fixed in relative frame coordinates,
  so 16:9 clips and 9:16 reels share one code path.
- **Measure the free lateral gap** inside that corridor at the nearest depth band,
  normalised by vehicle width. Normalising by vehicle width makes the number
  scale-invariant, so no camera calibration is needed.
- **Stabilise** the verdict with a median filter plus hysteresis, so the label
  does not flicker on single-frame detection glitches.
- **Stabilise the display** too: each track's box is smoothed, briefly coasted
  through detector dropouts (drawn dashed while inferred), and interpolated between
  analysis ticks, so boxes slide at video rate instead of strobing or stepping.
- **Score itself** against a hand-labelled gold set, and fit its own thresholds
  from that data rather than shipping guesses.

## Quickstart

Requires Python 3.12 and `ffmpeg` on PATH.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# weights are third-party and gitignored; this fetches them
.venv/Scripts/python.exe scripts/fetch_models.py --all
```

Analyse a clip:

```bash
.venv/Scripts/python.exe -m mototrack.cli analyze path/to/clip.mp4 --detector mixed
```

Outputs land in `out/<clip>/`:

- `annotated.mp4` - H.264, browser-playable, corridor tinted by verdict
- `events.json` - verdict segments with timestamps and blocking track ids
- `summary.json` - verdict time share, throughput, latency
- `timeline.png` - the verdict strip over time

## Running it

Two ways to run the same interface. The analysis is identical; only the shell
differs.

### Windows app (no browser, no terminal)

Double-click **`Start MotoTrack.bat`**. A native window opens with the UI
already loaded. This needs the optional extra once:

```bash
uv pip install --python .venv/Scripts/python.exe pywebview
```

Under the hood it starts the same FastAPI app on loopback and shows it in a
WebView2 window, so there is no server to start first and no port to remember.

### Browser (Local FastAPI)

```bash
.venv/Scripts/python.exe -m mototrack.cli web
# -> http://127.0.0.1:8770
```

Either way: upload, pick a detector, watch progress, then land on a page with the
annotated video, a verdict timeline, the numbers, and JSON export.

**Port 8770 is the default.** Port 8000 is deliberately avoided. Override with
`--port` or the `MOTOTRACK_PORT` environment variable; the CLI checks the port is
free before binding and tells you what is available if it is not. The desktop
window picks the next free port automatically rather than refusing to start.

### Deploying to Vercel

A standalone web version lives in [`web/`](web/) with a matching [`vercel.json`](vercel.json).
Because Vercel serverless functions cap bundle sizes at 250 MB and lack persistent
background workers or `ffmpeg` for long PyTorch/Ultralytics video transcoding, the
Vercel web app runs **100% client-side in the browser**:

- **In-Browser AI Engine (default on Vercel):** Uses TensorFlow.js (`COCO-SSD` WebGL)
  paired with a faithful JavaScript port ([`web/static/mototrack-engine.js`](web/static/mototrack-engine.js))
  of MotoTrack's pinhole distance estimation, relative ego-corridor trapezoid,
  scale-invariant gap ratio, EMA box smoothing + 0.4 s dropout coasting, and
  calibrated `1.8 / 0.4` verdict hysteresis. Also includes a 1-click **Demo Traffic Clip**
  generator: a deterministic 30 s POV clip (960x540, 25 fps, drawn live with
  `MediaRecorder`) whose actors are placed in metres and projected through the
  same pinhole model the engine uses, so SAFE / TIGHT / BLOCKED all occur and the
  boxes land where the vehicles are. Recording a real 30 s file costs 30 s of wall
  clock; the progress bar shows the recording pass, then the 10 fps analysis pass.
- **Hybrid FastAPI Backend (optional):** Expand *Execution Mode & Remote Backend*
  in the UI and paste your local or tunnelled `mototrack web` URL to run native
  YOLO26n checkpoints (`coco` or `mixed`) on your own server.

To deploy to Vercel:

1. Push this repository to GitHub / GitLab / Bitbucket, or run `npx vercel` from the project directory.
2. Vercel automatically detects [`vercel.json`](vercel.json) and serves `web/` as the output directory.

## Detectors

Two pretrained checkpoints, both optional at runtime, swapped with `--detector`:

| Name | Checkpoint | Why |
|---|---|---|
| `coco` (default) | `yolo26n.pt`, COCO | Broad and well tested, filtered to road-user classes. |
| `mixed` | `visaitech/vehicle-mixed-traffic-yolo26n` | Fine-tuned for South Asian mixed traffic; closer to Indonesian road conditions. |

Neither is declared the winner in advance. `compare-detectors` runs both against
the same gold set and reports which won on your footage:

```bash
.venv/Scripts/python.exe -m mototrack.cli compare-detectors
```

## Measuring it honestly

```bash
# 1. sample frames and render a contact sheet
.venv/Scripts/python.exe scripts/make_gold.py data/raw/clip_001.mp4 --count 60
# 2. label each row in data/gold/gold.csv as SAFE / TIGHT / BLOCKED
# 3. score
.venv/Scripts/python.exe -m mototrack.cli eval --detector mixed
# 4. fit thresholds to that data instead of guessing them
.venv/Scripts/python.exe scripts/calibrate.py --detector mixed
```

Current numbers are in [docs/METRICS.md](docs/METRICS.md). They are measured on
one clip with one labeller and are reported as such.

## Fine-tuning (optional)

Nothing above needs training. If you later want to test whether fine-tuning beats
the pretrained baseline on your own footage, `scripts/colab_train.ipynb` runs it
on Colab free tier: it downloads only the labelled frames (skips the 12 GB of raw
clips), fixes the dataset's `data.yaml` path bug, fine-tunes from `yolo26n.pt`,
and exports `best.pt` into `models/`. Then re-run `compare-detectors` and see
whether it actually won.

A negative result is a valid outcome. Write it down either way.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

115 tests. Geometry, verdict logic and temporal smoothing are pure functions
tested against hand-computed layouts; the pipeline and overlay tests decode the
real output video; the API tests run in-process via `TestClient` and never bind
a port.

## Layout

```
src/mototrack/
  config.py      every tunable, plus the port policy
  detectors.py   pluggable backends, one Detection type
  geometry.py    pinhole range, corridor, free-gap measurement
  verdict.py     thresholds, hysteresis, closing rate
  pipeline.py    per-frame loop and output writing
  io.py          probing, frame streaming, H.264 transcode
  overlay.py     corridor, boxes, HUD, timeline image
  evaluate.py    gold-set scoring and confusion matrices
  cli.py         analyze / preview-corridor / eval / compare-detectors / web
app/             FastAPI upload UI
scripts/         fetch_models, make_gold, calibrate, smoke_detect
```

## Limits, stated up front

- Ranges are monocular estimates from assumed object heights with an assumed 78
  degree field of view. They are reported as bands and caveated in the HUD.
- "Safe" means free space in *your* corridor. It says nothing about oncoming
  traffic, a vehicle about to change lanes, or road surface.
- Pretrained weights were not fine-tuned on this footage. No mAP is claimed.
- CPU-only throughout: expect roughly 85-95 ms per analysed frame.

Licensing: the Ultralytics package and both checkpoints are AGPL-3.0. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md).