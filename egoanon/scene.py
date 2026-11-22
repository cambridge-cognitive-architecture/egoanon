"""Levels that need models: people removed, and depth of the people-free frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .plan import Region
from .render import region_mask

MODEL_LEVELS = ("inpaint", "depth")
MASK_DILATE = SIGN_DILATE = 9
DEPTH_BLUR_SIGMA = 25
CLIP_PERCENTILES = (2.0, 98.0)
LABEL_MIN_PX = 800  # at 1600 x 1200; scaled with the frame
LABEL_FONT_PX = 56  # at 1600 wide

# COCO-17 limbs and trunk. Every keypoint seen also gets a dot.
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))
POSE_COLOUR = (173, 68, 142)  # BGR
PERSON_BAR, PIN_BAR, BOX_BAR = "#8e44ad", "#9aa7b1", "#ffd54f"


@dataclass
class Pose:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    keypoints: np.ndarray                   # (17, 2); zeros where not seen
    score: float = 1.0


@dataclass
class SceneObject:
    name: str  # from a fixed vocabulary, never read from the image
    mask: np.ndarray
    kind: str = "pin"  # pin, for large structures | box, for small objects


@dataclass
class SceneModels:
    """The local models a level needs. Any callables with these signatures will do."""
    inpaint: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    depth: Callable[[np.ndarray], np.ndarray] | None = None       # larger = nearer
    signs: Callable[[np.ndarray], np.ndarray] | None = None       # boolean mask
    pose: Callable[[np.ndarray], list[Pose]] | None = None        # optional skeletons
    objects: Callable[[np.ndarray], list[SceneObject]] | None = None  # optional labels
    params: dict = field(default_factory=dict)

    def require(self, level: str) -> None:
        needed = ("inpaint",) if level == "inpaint" else ("inpaint", "depth", "signs")
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            raise ValueError(f"level {level!r} needs a local model for: {', '.join(missing)}")


def dilate(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def removal_mask(regions: list[Region], width: int, height: int) -> np.ndarray:
    """Everything that is replaced: people, faces, screens and manual regions."""
    return dilate(region_mask(regions, width, height)[0] > 0, MASK_DILATE)


def remove(image: np.ndarray, mask: np.ndarray, inpaint: Callable) -> np.ndarray:
    """Replace the masked pixels. The model is never given what was under the mask."""
    if not mask.any():
        return image.copy()
    hidden = image.copy()
    hidden[mask] = 0
    filled = np.asarray(inpaint(hidden, mask.copy()))
    height, width = image.shape[:2]
    if filled.ndim != 3 or filled.shape[0] < height or filled.shape[1] < width:
        raise RuntimeError(f"the inpainter returned shape {filled.shape}")
    out = image.copy()
    # A model may pad to a multiple of 8. Crop, as resizing would shift the fill.
    out[mask] = filled[:height, :width][mask]
    return out


def select_poses(poses: list[Pose], regions: list[Region], width: int, height: int) -> list[Pose]:
    """Poses to draw: duplicates go, and so does anyone in a manual region."""
    manual = region_mask([r for r in regions if r.source == "manual"], width, height)[0] > 0
    kept: list[Pose] = []
    for pose in sorted(poses, key=lambda p: -p.score):
        x0, y0, x1, y1 = np.rint(pose.box).astype(int)
        box = manual[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
        seen = pose.keypoints[pose.keypoints.sum(axis=1) > 0].astype(int)
        inside = manual[np.clip(seen[:, 1], 0, height - 1), np.clip(seen[:, 0], 0, width - 1)]
        if inside.any() or (box.size and box.mean() > 0.05):
            continue
        if any(_iou(pose.box, other.box) > 0.5 for other in kept):
            continue
        kept.append(pose)
    return kept


def _iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    both = w * h
    return both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both)


def draw_skeletons(image: np.ndarray, poses: list[Pose], colour, thickness: int, radius: int) -> None:
    for pose in poses:
        points, seen = pose.keypoints.astype(int), pose.keypoints.sum(axis=1) > 0
        for a, b in SKELETON:
            if seen[a] and seen[b]:
                cv2.line(image, tuple(map(int, points[a])), tuple(map(int, points[b])),
                         colour, thickness, cv2.LINE_AA)
        for point in points[seen]:
            cv2.circle(image, tuple(map(int, point)), radius, colour, -1, cv2.LINE_AA)


def colour_depth(depth: np.ndarray, blur_mask: np.ndarray) -> np.ndarray:
    """Near is dark purple, far is light orange; detail is blurred inside the mask."""
    from matplotlib import colormaps

    d = np.asarray(depth, np.float64)
    if d.shape != blur_mask.shape or not np.isfinite(d).all():
        raise RuntimeError("the depth map has the wrong shape or is not finite")
    span = d.max() - d.min()
    far = 1.0 - (d - d.min()) / span if span > 0 else np.zeros_like(d)
    low, high = np.percentile(far, CLIP_PERCENTILES)
    stretched = np.clip((far - low) / max(high - low, 1e-9), 0, 1)
    value = cv2.equalizeHist((stretched * 255).astype(np.uint8)).astype(np.float64) / 255.0
    smooth = cv2.GaussianBlur(value.astype(np.float32), (0, 0), DEPTH_BLUR_SIGMA)
    value[blur_mask] = smooth[blur_mask]
    rgb = (colormaps["magma"](np.clip(value, 0, 1))[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _largest(mask: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n < 2:
        return np.zeros(mask.shape, bool), 0
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best, int(stats[best, cv2.CC_STAT_AREA])


def _anchor(mask: np.ndarray) -> tuple[int, int]:
    """The centroid, or the nearest point of the mask if that falls outside it."""
    ys, xs = np.nonzero(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    if not mask[cy, cx]:
        nearest = int(((xs - cx) ** 2 + (ys - cy) ** 2).argmin())
        cx, cy = int(xs[nearest]), int(ys[nearest])
    return cx, cy


def draw_labels(canvas: np.ndarray, objects: list[SceneObject], people: np.ndarray) -> np.ndarray:
    """Pins and boxes. A label with no free place is left out."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    height, width = canvas.shape[:2]
    s = width / 1600
    minimum = LABEL_MIN_PX * width * height / (1600 * 1200)
    items = []  # name, rank, anchor, box, bar colour
    for obj in objects:
        mask, area = _largest(obj.mask)
        if area < max(1, minimum):
            continue
        if obj.kind == "box":
            size = max(3, round(21 * s)) | 1
            core = cv2.erode(mask.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
            if core.sum() >= minimum * 300 / LABEL_MIN_PX:
                mask = _largest(core)[0]
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            items.append((obj.name, (0, -w), None, (x, y, x + w, y + h), BOX_BAR))
        else:
            items.append((obj.name, (1, -area), _anchor(mask), None, PIN_BAR))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(people.astype(np.uint8), 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= minimum:
            items.append(("person", (1, -area), _anchor(labels == i), None, PERSON_BAR))
    if not items:
        return canvas

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"),
                              max(12, round(LABEL_FONT_PX * s)))
    pad, gap, dot = max(3, round(17 * s)), max(3, round(6 * s)), max(3, round(11 * s))
    used: list[tuple[int, int, int, int]] = []
    placed = []
    for name, _, anchor, box, bar in sorted(items, key=lambda item: item[1]):
        left, top, right, bottom = font.getbbox(name)
        w, h = int(right - left + 2 * pad), int(bottom - top + pad)
        if box:
            x0, y0, x1, y1 = box
            options = [(x0, y0 - h - gap), (x0, y1 + gap), (x1 - w, y0 - h - gap),
                       (x1 - w, y1 + gap), (x1 + gap, y0), (x0 - w - gap, y0),
                       (x0, y0 - 2 * h - gap)]
        else:
            ax, ay = anchor
            options = []
            for d in (dot + gap, *(round(v * s) for v in (40, 70, 110, 160, 220))):
                options += [(ax + d, ay - h // 2), (ax - w - d, ay - h // 2),
                            (ax - w // 2, ay + d), (ax - w // 2, ay - h - d)]
        for x, y in options:
            if x < 1 or y < 1 or x + w >= width or y + h >= height:
                continue
            if all(x + w <= u[0] or u[2] <= x or y + h <= u[1] or u[3] <= y for u in used):
                used.append((x, y, x + w, y + h))
                placed.append((name, anchor, box, bar, (x, y, x + w, y + h), (left, top)))
                break

    picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    for _, anchor, box, bar, _, _ in placed:  # marks first, so that no chip is covered
        if box:
            draw.rectangle(box, outline="white", width=max(3, round(7 * s)))
            draw.rectangle(box, outline=bar, width=max(1, round(3 * s)))
        else:
            ax, ay = anchor
            draw.ellipse((ax - dot, ay - dot, ax + dot, ay + dot), fill=bar, outline="white", width=2)
    for name, _, _, bar, (x0, y0, x1, y1), (left, top) in placed:
        draw.rectangle((x0, y0, x1, y1), fill="white", outline=(176, 172, 165), width=1)
        draw.rectangle((x0, y0, x0 + max(2, round(10 * s)), y1), fill=bar)
        draw.text((x0 + pad - left, y0 + pad // 2 - top), name, font=font, fill=(27, 27, 27))
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def render_scene(image: np.ndarray, regions: list[Region], level: str,
                 models: SceneModels | None) -> np.ndarray:
    if models is None:
        raise ValueError(f"level {level!r} needs models; see SceneModels")
    models.require(level)
    height, width = image.shape[:2]
    poses = models.pose(image) if models.pose else []
    # A person only the pose model found is removed too: it can only add cover.
    regions = regions + [Region(x0, y0, x1 - x0, y1 - y0, "body", p.score)
                         for p in poses for x0, y0, x1, y1 in [p.box]]
    clean = remove(image, removal_mask(regions, width, height), models.inpaint)
    drawn = select_poses(poses, regions, width, height)
    if level == "inpaint":
        draw_skeletons(clean, drawn, POSE_COLOUR, 5, 7)
        return clean

    # Both models see only the people-free frame, before anything is drawn on it.
    signs = np.asarray(models.signs(clean), bool)
    if signs.shape != (height, width):
        raise RuntimeError(f"the sign model returned shape {signs.shape}")
    hard = [r for r in regions if r.source in ("screen", "manual")]
    blur = dilate(signs | (region_mask(hard, width, height)[0] > 0), SIGN_DILATE)
    out = colour_depth(models.depth(clean), blur)
    draw_skeletons(out, drawn, (255, 255, 255), 6, 8)
    if models.objects:
        people = region_mask([r for r in regions if r.source == "body"], width, height)[0] > 0
        out = draw_labels(out, models.objects(clean), people)
    return out
