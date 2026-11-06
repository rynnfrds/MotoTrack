"""Command line entry point for MotoTrack."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from . import config


def port_in_use(host: str, port: int) -> bool:
    """True when binding host:port fails, i.e. something already listens there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def suggest_ports(host: str, start: int, count: int = 5) -> list[int]:
    free: list[int] = []
    candidate = start
    while len(free) < count and candidate < start + 200:
        if not port_in_use(host, candidate):
            free.append(candidate)
        candidate += 1
    return free


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import analyse

    last_pct = [-10]

    def progress(index: int, total: int) -> None:
        if total <= 0:
            return
        pct = 100.0 * index / total
        if pct - last_pct[0] < 5.0 and pct < 99.5:
            return
        last_pct[0] = pct
        sys.stderr.write(chr(13) + f"  processing {pct:5.1f}%")
        sys.stderr.flush()

    result = analyse(
        args.video,
        out_dir=args.out,
        detector_kind=args.detector,
        analysis_fps=args.analysis_fps,
        write_video=not args.no_video,
        progress_cb=progress,
        max_seconds=args.max_seconds,
    )
    sys.stderr.write(chr(13) + " " * 34 + chr(13))
    analysis = result.summary["analysis"]
    shares = result.summary["verdict_share"]

    print(f"source     : {result.source}")
    print(f"detector   : {result.detector}")
    print(f"output dir : {result.out_dir}")
    if result.annotated_video:
        print(f"annotated  : {result.annotated_video}")
    print(f"timeline   : {result.timeline_image}")
    print(f"events     : {result.events_path}")
    print(f"summary    : {result.summary_path}")
    print(f"frames     : {analysis['analysed_frames']} analysed")
    print(f"speed      : {analysis['wall_fps']} fps wall, {analysis['mean_latency_ms']} ms/frame")
    for verdict in (config.VERDICT_SAFE, config.VERDICT_TIGHT, config.VERDICT_BLOCKED, config.VERDICT_UNKNOWN):
        share = shares.get(verdict, 0.0)
        print(f"  {verdict:<8}: {100.0 * share:5.1f}% of analysed time")
    return 0


def cmd_preview_corridor(args: argparse.Namespace) -> int:
    """Render the ego corridor over one frame so thresholds can be eyeballed."""
    import cv2

    from .geometry import compute_gap, corridor_polygon_relative
    from .io import probe
    from .overlay import annotate

    meta = probe(args.video)
    cap = cv2.VideoCapture(args.video)
    try:
        if args.time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print("could not read a frame at that timestamp", file=sys.stderr)
            return 2
    finally:
        cap.release()

    from .detectors import build_detector

    detector = build_detector(args.detector)
    detections = detector.infer(frame, persist=False)
    gap = compute_gap(detections, frame.shape[1], frame.shape[0])
    out = annotate(frame, detections, config.VERDICT_UNKNOWN, gap=gap)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}  ({meta.width}x{meta.height}, corridor bottom edge "
          f"{config.CORRIDOR_BOTTOM_LEFT_X:.2f}-{config.CORRIDOR_BOTTOM_RIGHT_X:.2f} of width)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, format_report

    report = evaluate(args.gold, args.detector)
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport written to {args.json}")
    return 0 if report.get("total", 0) else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from .evaluate import compare_detectors, format_comparison

    report = compare_detectors(args.gold)
    print(format_comparison(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport written to {args.json}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host
    port = config.resolve_port(args.port)
    if port_in_use(host, port):
        alternatives = suggest_ports(host, port + 1)
        print(f"port {port} on {host} is already in use.", file=sys.stderr)
        if alternatives:
            print(f"try one of: {alternatives}  (or set {config.PORT_ENV_VAR})", file=sys.stderr)
        return 2
    print(f"MotoTrack web UI: http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mototrack",
        description="Ego-POV overtaking feasibility detector.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse a clip and write outputs")
    analyze.add_argument("video", help="path to the video file")
    analyze.add_argument("-o", "--out", default=None, help="output directory")
    analyze.add_argument(
        "--detector",
        choices=sorted(config.MODEL_FILES),
        default=config.DEFAULT_DETECTOR,
        help="which checkpoint to run (default: %(default)s)",
    )
    analyze.add_argument(
        "--analysis-fps",
        type=float,
        default=config.ANALYSIS_FPS,
        help="frames analysed per second of video (default: %(default)s)",
    )
    analyze.add_argument("--no-video", action="store_true", help="skip the annotated video")
    analyze.add_argument(
        "--max-seconds", type=float, default=None, help="stop after N seconds of source"
    )
    analyze.set_defaults(func=cmd_analyze)

    preview = sub.add_parser("preview-corridor", help="draw the corridor over one frame")
    preview.add_argument("video")
    preview.add_argument("-o", "--out", default="out/corridor_preview.jpg")
    preview.add_argument("--time", type=float, default=1.0, help="timestamp in seconds")
    preview.add_argument(
        "--detector", choices=sorted(config.MODEL_FILES), default=config.DEFAULT_DETECTOR
    )
    preview.set_defaults(func=cmd_preview_corridor)

    ev = sub.add_parser("eval", help="score a detector against the gold set")
    ev.add_argument("--gold", default=os.path.join(config.DATA_DIR, "gold", "gold.csv"))
    ev.add_argument(
        "--detector", choices=sorted(config.MODEL_FILES), default=config.DEFAULT_DETECTOR
    )
    ev.add_argument("--json", default=None, help="also write the report as JSON")
    ev.set_defaults(func=cmd_eval)

    cmp_ = sub.add_parser("compare-detectors", help="score both detectors on one gold set")
    cmp_.add_argument("--gold", default=os.path.join(config.DATA_DIR, "gold", "gold.csv"))
    cmp_.add_argument("--json", default=None)
    cmp_.set_defaults(func=cmd_compare)

    web = sub.add_parser("web", help="serve the upload UI")
    web.add_argument("--host", default=config.DEFAULT_HOST)
    web.add_argument(
        "--port", type=int, default=None, help=f"default {config.DEFAULT_PORT} or ${config.PORT_ENV_VAR}"
    )
    web.set_defaults(func=cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
