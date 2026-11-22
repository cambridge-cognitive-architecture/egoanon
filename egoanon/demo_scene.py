"""Synthetic demos of inpaint, depth and record, using stand-in models.

Everything is drawn here: a room, a stick figure and a screen. No model is loaded.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .plan import Region
from .record import render_record
from .render import apply_level
from .scene import Pose, SceneModels, SceneObject
from .video import scan_video

W, H, N, FPS = 640, 420, 60, 10
FLOOR = 250
FIGURE, SCREEN, SIGN = (255, 0, 255), (255, 255, 0), (0, 230, 255)
DOOR, DESK = (60, 100, 140), (95, 130, 110)
POINTS = np.array([
    (0, -86), (0, 0), (0, 0), (0, 0), (0, 0), (-20, -58), (20, -58), (-31, -24), (31, -24),
    (-36, 8), (36, 8), (-13, 12), (13, 12), (-16, 56), (16, 56), (-19, 100), (19, 100)], float)


def figure_points(i: int) -> np.ndarray:
    seen = POINTS.any(axis=1)[:, None]
    return np.where(seen, POINTS + (170 + 3 * i, 215), 0)


def synthetic_frame(i: int) -> np.ndarray:
    im = np.full((H, W, 3), (205, 215, 220), np.uint8)
    cv2.rectangle(im, (0, FLOOR), (W, H), (150, 170, 185), -1)
    for x in range(-300, W + 300, 70):
        cv2.line(im, (x, H), (W // 2 + (x - W // 2) // 3, FLOOR), (130, 150, 165), 1)
    cv2.rectangle(im, (40, 90), (115, FLOOR + 10), DOOR, -1)
    cv2.rectangle(im, (420, 225), (600, 305), DESK, -1)
    cv2.rectangle(im, (450, 130), (565, 208), SCREEN, -1)
    for y in range(142, 202, 12):
        cv2.line(im, (460, y), (555, y), (60, 60, 60), 2)
    cv2.rectangle(im, (245, 50), (335, 90), SIGN, -1)
    cv2.line(im, (255, 70), (325, 70), (60, 60, 60), 3)
    p = figure_points(i).astype(int).tolist()
    for a, b in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)):
        cv2.line(im, tuple(p[a]), tuple(p[b]), FIGURE, 12)
    cv2.fillPoly(im, [np.array([p[5], p[6], p[12], p[11]], np.int32)], FIGURE)
    cv2.circle(im, tuple(p[0]), 22, FIGURE, -1)
    return im


def write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        writer.write(synthetic_frame(i))
    writer.release()


def near(image: np.ndarray, colour, tolerance: int = 70) -> np.ndarray:
    return np.abs(image.astype(int) - colour).max(axis=2) < tolerance


class ColourDetector:
    """Finds the marker colours. A stand-in, not a privacy detector."""
    params = {"bodies": True}

    def __call__(self, image: np.ndarray) -> list[Region]:
        out = []
        for colour, source, pad in ((FIGURE, "body", 6), (SCREEN, "screen", 0)):
            found = near(image, colour)
            if found.any():
                x, y, w, h = cv2.boundingRect(found.astype(np.uint8))
                out.append(Region(x - pad, y - pad, w + 2 * pad, h + 2 * pad, source))
        return out


def standin_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(image, mask.astype(np.uint8), 7, cv2.INPAINT_TELEA)


def standin_depth(image: np.ndarray) -> np.ndarray:
    y = np.arange(H, dtype=float)[:, None]
    depth = np.repeat(0.2 + np.maximum(y - FLOOR, 0) / (H - FLOOR), W, axis=1)
    depth[near(image, DOOR, 25)] = 0.12
    depth[near(image, DESK, 25)] = 0.75
    depth[near(image, SIGN, 25)] = 0.28
    return depth


def standin_models(frame: int, skeletons: bool = False, labels: bool = False) -> SceneModels:
    models = SceneModels(inpaint=standin_inpaint, depth=standin_depth,
                         signs=lambda image: near(image, SIGN))
    if skeletons:
        points = figure_points(frame)
        seen = points[points.any(axis=1)]
        box = (*(seen.min(0) - 25), *(seen.max(0) + 25))
        models.pose = lambda image: [Pose(box, points)]
    if labels:
        models.objects = lambda image: [SceneObject("door", near(image, DOOR, 25)),
                                        SceneObject("desk", near(image, DESK, 25), "box")]
    return models


class StandInVlm:
    """Canned replies: one passes, one always fails, one passes on the second try."""

    def __init__(self):
        self.window = self.tries = 0

    def __call__(self, image, prompt: str, temperature: float) -> str:
        if temperature == 0.0:
            self.window, self.tries = self.window + 1, 0
        self.tries += 1
        if self.window == 1:
            return "door, desk, screen"
        if self.window == 2 or self.tries == 1:
            return "someone near a desk"
        return "desk, door"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("level", choices=("inpaint", "depth", "record"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path(f"docs/demo_{a.level}.png")
    i = 20
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "synthetic.mp4"
        write_clip(clip)
        plan = scan_video(clip, ColourDetector())
        fig, axes = plt.subplots(1, 2 if a.level == "record" else 3, figsize=(9, 2.5),
                                 layout="constrained")
        frame = synthetic_frame(i)
        panels = [("synthetic input", frame)]
        if a.level == "inpaint":
            for title, skeletons in (("inpaint", False), ("inpaint + skeletons", True)):
                panels.append((title, apply_level(frame, plan.regions_at(i), "inpaint",
                                                  standin_models(i, skeletons))))
        elif a.level == "depth":
            for title, extra in (("depth", False), ("depth + skeletons, labels", True)):
                panels.append((title, apply_level(frame, plan.regions_at(i), "depth",
                                                  standin_models(i, extra, extra))))
        else:
            path = Path(d) / "record.json"
            render_record(clip, plan, path, StandInVlm(), window_s=2.0)
            import json
            for k, row in enumerate(json.loads(path.read_text())["windows"]):
                y = 0.78 - 0.26 * k
                axes[1].text(0.02, y, f"{row['start_s']:.0f}-{row['end_s']:.0f} s", family="monospace")
                axes[1].text(0.22, y, row["scene"], family="monospace", fontsize=9)
            axes[1].set_title("record: 2 s windows", loc="left", fontsize=10)
            axes[1].set_axis_off()
        for ax, (title, image) in zip(axes, panels):
            ax.imshow(image[..., ::-1])
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_axis_off()
        fig.text(0.995, 0.005, "synthetic, stand-in models", ha="right", va="bottom",
                 fontsize=7, color="#777777")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, facecolor="white", metadata={"Software": None})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
