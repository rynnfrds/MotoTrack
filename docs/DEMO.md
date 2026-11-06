# Demo runbook

## Start the app

```bash
.venv/Scripts/python.exe -m mototrack.cli web
```

Opens `http://127.0.0.1:8770`. Port 8000 is reserved by another local
application and is never used by this project.

If 8770 is taken:

```bash
.venv/Scripts/python.exe -m mototrack.cli web --port 8771
# or
$env:MOTOTRACK_PORT = "8771"; .venv/Scripts/python.exe -m mototrack.cli web
```

The CLI checks the port first and prints free alternatives instead of crashing
after a half-start.

## If the page does not change after a code fix

The server keeps the code it started with. Uvicorn without `--reload` does not
re-read files, so a fix can be correct on disk and still invisible in the browser.
Stop the server (Ctrl+C in its terminal) and start it again:

```bash
.venv/Scripts/python.exe -m mototrack.cli web
```

Then hard-refresh the page (Ctrl+Shift+R) so the browser drops the cached
`app.js` and `style.css`. Both steps are needed; either one alone leaves you
looking at old behaviour.

## Windows app instead of a browser

Double-click `Start MotoTrack.bat`. A native window opens with the UI loaded; no
terminal, no URL to remember. First time only:

```bash
uv pip install --python .venv/Scripts/python.exe pywebview
```

It is the same FastAPI app on loopback, so everything below applies unchanged.
The window picks the next free port automatically instead of refusing to start,
which matters when port 8770 is already busy.

## A demo that shows the thing working

1. Open the UI. Drop a motovlog clip, pick `mixed`, press Analyse.
2. Watch the progress bar. It reports real analysis progress, roughly 8.5 fps.
3. On the result page: the annotated video plays with the corridor tinted by
   verdict, obstacle boxes in red when they block the corridor, and a HUD that
   shows the verdict plus the evidence that produced it.
4. Scrub the video. The playhead tracks across the verdict timeline underneath.
5. Export `events.json` and point at a segment: it lists `t0`, `t1`, verdict,
   the minimum gap ratio in that run, and which track ids were in the way.

## What to say when someone asks how accurate it is

Point at `docs/METRICS.md` and read it out: 69% binary accuracy on 29
hand-labelled frames from one clip, thresholds fitted on that same data, 3-class
accuracy of 41% which is not usable. The honest framing is the interesting part.

## Known-awkward moments in a live demo

- **First run is slow.** The first detection after startup pays model load.
  Warm it up before an audience by analysing a short clip once.
- **Verdict holds briefly at the start.** The stabiliser reports `UNKNOWN` until
  it has a window of evidence. That is intended, not a bug.
- **A held verdict can disagree with the frame underneath** near a transition.
  The HUD says so explicitly: `holding SAFE (frame reads TIGHT)`.

## Regenerating the artefacts

```bash
.venv/Scripts/python.exe -m mototrack.cli analyze data/raw/clip_001.mp4 \
  --detector mixed -o out/demo
```

Produces `annotated.mp4`, `events.json`, `summary.json`, `timeline.png` in
`out/demo/`.