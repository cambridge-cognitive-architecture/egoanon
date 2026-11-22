"""python -m egoanon scan|render"""
import argparse
from pathlib import Path

from .detect import ModelDetector
from .plan import load_plan, save_plan
from .render import LEVELS
from .video import render_video, scan_video


def main() -> None:
    ap = argparse.ArgumentParser(prog="egoanon")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="detect and track; writes a plan")
    s.add_argument("video", type=Path)
    s.add_argument("--weights", type=Path, required=True)
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--stride", type=int, default=1)
    r = sub.add_parser("render", help="render one level from a plan")
    r.add_argument("video", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--level", choices=LEVELS, default="blur")
    a = ap.parse_args()
    if a.command == "scan":
        if a.plan.exists():
            ap.error(f"{a.plan} already exists")
        plan = scan_video(a.video, ModelDetector(a.weights), a.stride)
        save_plan(plan, a.plan)
        print(f"{plan.n_frames} frames, {len(plan.detections)} with a region")
    else:
        m = render_video(a.video, load_plan(a.plan), a.output, a.level)
        print(f"{m['n_frames']} frames, {m['n_frames_covered']} covered. Review before sharing.")


if __name__ == "__main__":
    main()
