"""Render levels that need no model, and the entry point for those that do."""
from __future__ import annotations

import cv2
import numpy as np

from .plan import MARGIN, Region

LEVELS = ("blur", "pixelate", "block", "silhouette", "boxes")
COLOURS = {  # BGR
    "face": (168, 79, 123), "person": (181, 109, 47), "tag": (90, 160, 224),
    "screen": (140, 122, 31), "manual": (90, 90, 90),
}
FEATHER = 31


def _shape(region: Region, width: int, height: int, filled=True):
    g = region.grow(MARGIN.get(region.source, MARGIN["manual"]), width, height)
    x0, y0 = int(np.floor(g.x)), int(np.floor(g.y))
    x1, y1 = int(np.ceil(g.x + g.w)), int(np.ceil(g.y + g.h))
    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
    axes = (max(4, (x1 - x0 + 1) // 2), max(4, (y1 - y0 + 1) // 2))

    def draw(canvas, value):
        thickness = -1 if filled else 2
        if region.source in ("screen", "manual", "body"):  # corners matter: use rectangles
            cv2.rectangle(canvas, (x0, y0), (x1, y1), value, thickness)
        else:
            cv2.ellipse(canvas, centre, axes, 0, 0, 360, value, thickness)
    return draw, g


def region_mask(regions, width: int, height: int) -> tuple[np.ndarray, float]:
    """The union mask, and the short side of the largest region."""
    mask = np.zeros((height, width), np.float32)
    largest = 0.0
    for r in regions:
        draw, g = _shape(r, width, height)
        draw(mask, 1.0)
        largest = max(largest, min(g.w, g.h))
    return mask, largest


def feather_outward(mask: np.ndarray) -> np.ndarray:
    """Soften the edge without weakening the inside.

    Blurring the mask alone would drop a small region's peak well below one,
    leaving a distant face partly visible.
    """
    return np.maximum(mask, cv2.GaussianBlur(mask, (FEATHER, FEATHER), 0))[..., None]


def apply_level(image: np.ndarray, regions: list[Region], level: str = "blur",
                models=None) -> np.ndarray:
    from .scene import MODEL_LEVELS, render_scene

    if level in MODEL_LEVELS:
        return render_scene(image, regions, level, models)
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS + MODEL_LEVELS}")
    # Whole-body regions are only for the levels that remove people.
    regions = [r for r in regions if r.source != "body"]
    height, width = image.shape[:2]
    if level in ("silhouette", "boxes"):
        # No source pixel survives these two levels.
        out = np.full_like(image, 235)
        for r in regions:
            _shape(r, width, height, filled=level == "silhouette")[0](
                out, COLOURS.get(r.source, COLOURS["manual"]))
        return out
    if not regions:
        return image.copy()
    mask, largest = region_mask(regions, width, height)
    if level == "block":
        out = image.copy()
        out[mask > 0] = 0
        return out
    if level == "pixelate":
        n = max(8, int(largest / 4))
        small = cv2.resize(image, (max(1, width // n), max(1, height // n)),
                           interpolation=cv2.INTER_AREA)
        covered = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        # A mosaic alone can be attacked, so it is blurred as well.
        k = (n // 2) | 1
        covered = cv2.GaussianBlur(covered, (k, k), 0)
    else:
        k = max(21, int(largest * 0.55)) | 1
        covered = cv2.GaussianBlur(image, (k, k), 0)
    alpha = feather_outward(mask)
    return (image * (1 - alpha) + covered * alpha).astype(np.uint8)
