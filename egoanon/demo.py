"""Self-generated clip: coloured shapes standing in for a face and a screen."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .plan import Region
from .render import LEVELS, apply_level
from .video import render_video, scan_video

W, H, N = 480, 320, 40


def synthetic_frame(i: int) -> np.ndarray:
    im = np.full((H, W, 3), (205, 215, 220), np.uint8)
    cv2.rectangle(im, (0, 220), (W, H), (150, 170, 185), -1)
    for x in range(-200, W + 200, 60):
        cv2.line(im, (x, H), (W // 2 + (x - W // 2) // 3, 220), (130, 150, 165), 1)
    # cyan patterned panel = "screen"
    cv2.rectangle(im, (300, 70), (430, 160), (255, 255, 0), -1)
    for y in range(82, 155, 14):
        cv2.line(im, (312, y), (418, y), (60, 60, 60), 3)
    # magenta patterned disc = "face"
    cx = 70 + 4 * i
    cv2.circle(im, (cx, 135), 34, (255, 0, 255), -1)
    for dx in (-15, 0, 15):
        cv2.line(im, (cx + dx, 115), (cx + dx, 155), (20, 20, 20), 4)
    return im


def colour_detector(image: np.ndarray) -> list[Region]:
    """Finds the two marker colours. A stand-in, not a privacy detector."""
    out = []
    for colour, source in (((255, 0, 255), "face"), ((255, 255, 0), "screen")):
        near = np.abs(image.astype(int) - colour).max(axis=2) < 70
        if near.any():
            ys, xs = np.nonzero(near)
            out.append(Region(float(xs.min()), float(ys.min()), float(np.ptp(xs) + 1),
                              float(np.ptp(ys) + 1), source))
    return out


def write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    for i in range(N):
        writer.write(synthetic_frame(i))
    writer.release()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/demo.png"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        clip = Path(d) / "synthetic.mp4"
        write_clip(clip)
        plan = scan_video(clip, colour_detector)
        render_video(clip, plan, Path(d) / "blurred.mp4")
        cap = cv2.VideoCapture(str(clip))
        for _ in range(21):
            _, frame = cap.read()
        cap.release()
        fig, axes = plt.subplots(2, 3, figsize=(9, 4.6), layout="constrained")
        panels = [("synthetic input", frame)]
        panels += [(level, apply_level(frame, plan.regions_at(20), level)) for level in LEVELS]
        for ax, (title, image) in zip(axes.flat, panels):
            ax.imshow(image[..., ::-1])
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_axis_off()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.out, dpi=120, facecolor="white", metadata={"Software": None})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
