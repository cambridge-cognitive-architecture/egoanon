"""python -m egoanon scan|render"""
import argparse
from pathlib import Path

from .detect import ModelDetector
from .offline import offline
from .plan import load_plan, save_plan
from .render import LEVELS
from .scene import MODEL_LEVELS
from .video import render_video, scan_video


def main() -> None:
    ap = argparse.ArgumentParser(prog="egoanon")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="detect and track; writes a plan")
    s.add_argument("video", type=Path)
    s.add_argument("--weights", type=Path, required=True)
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--bodies", action="store_true",
                   help="also find whole people; needed by inpaint and depth")
    r = sub.add_parser("render", help="render one level from a plan")
    r.add_argument("video", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--level", choices=(*LEVELS, *MODEL_LEVELS, "record"), default="blur")
    r.add_argument("--lama", type=Path, help="inpaint, depth: local big-lama.pt")
    r.add_argument("--pose", type=Path, help="inpaint, depth: pose weights; draws skeletons")
    r.add_argument("--depth-model", type=Path, help="depth: local Depth Anything V2 folder")
    r.add_argument("--clipseg", type=Path, help="depth: local CLIPSeg folder")
    r.add_argument("--labels", action="store_true", help="depth: draw object labels")
    r.add_argument("--vlm", type=Path, help="record: local mlx-vlm model folder")
    r.add_argument("--window", type=float, default=10.0, help="record: seconds per line")
    a = ap.parse_args()
    if a.command == "scan":
        if a.plan.exists():
            ap.error(f"{a.plan} already exists")
        plan = scan_video(a.video, ModelDetector(a.weights, bodies=a.bodies), a.stride)
        save_plan(plan, a.plan)
        print(f"{plan.n_frames} frames, {len(plan.detections)} with a region")
    else:
        plan = load_plan(a.plan)
        if a.level == "record":
            if a.vlm is None:
                ap.error("--level record needs --vlm")
            from .record import MlxVlm, render_record
            with offline():
                m = render_record(a.video, plan, a.output, MlxVlm(a.vlm), a.window)
            print(f"{m['n_windows']} windows, {m['n_withheld']} withheld. The model can name objects that are not there: check every line.")
            return
        if a.level in MODEL_LEVELS:
            if plan.params.get("bodies") is not True:
                ap.error(f"--level {a.level} needs a plan from: scan --bodies")
            from .models import load_models
            with offline():
                models = load_models(a.level, a.lama, a.depth_model, a.clipseg, a.pose, a.labels)
                m = render_video(a.video, plan, a.output, a.level, models)
        else:
            m = render_video(a.video, plan, a.output, a.level)
        print(f"{m['n_frames']} frames, {m['n_frames_covered']} covered. Review before sharing.")


if __name__ == "__main__":
    main()
