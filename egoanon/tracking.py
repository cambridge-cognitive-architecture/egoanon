"""IoU tracking-by-detection and filling of short detection dropouts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .plan import Region


def _iou_matrix(a: list[Region], b: list[Region]) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b)); zero where the union is empty."""
    p = np.array([(r.x, r.y, r.w, r.h) for r in a], dtype=float).reshape(-1, 4)
    q = np.array([(r.x, r.y, r.w, r.h) for r in b], dtype=float).reshape(-1, 4)
    start = np.maximum(p[:, None, :2], q[None, :, :2])
    end = np.minimum((p[:, :2] + p[:, 2:])[:, None], (q[:, :2] + q[:, 2:])[None])
    inter = np.prod(np.clip(end - start, 0, None), axis=2)
    area_p = np.prod(np.clip(p[:, 2:], 0, None), axis=1)
    area_q = np.prod(np.clip(q[:, 2:], 0, None), axis=1)
    union = area_p[:, None] + area_q[None] - inter
    return np.divide(inter, union, out=np.zeros_like(union), where=union > 0)


@dataclass
class _Track:
    id: int
    last: Region
    missed: int = 0


class IouTracker:
    """Link detections across frames by optimal assignment on box IoU."""

    def __init__(self, min_iou: float = 0.3, max_missed: int = 20):
        self.min_iou = min_iou
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 0

    def step(self, regions: list[Region]) -> list[Region]:
        """Advance one frame and return copies of `regions` with track_id set.

        Call this for every frame, including those without detections.
        """
        last = [t.last for t in self._tracks]
        iou = _iou_matrix(last, regions)
        same_source = np.array(
            [[t.source == r.source for r in regions] for t in last], dtype=bool
        ).reshape(iou.shape)
        allowed = same_source & (iou >= self.min_iou)

        # Forbidden pairs weigh zero, so they cannot displace an acceptable match.
        rows, cols = linear_sum_assignment(-np.where(allowed, iou, 0.0))
        matched = {c: self._tracks[r] for r, c in zip(rows, cols) if allowed[r, c]}

        for track in self._tracks:
            track.missed += 1

        tracked = []
        for i, region in enumerate(regions):
            track = matched.get(i)
            if track is None:
                track = _Track(self._next_id, region)
                self._next_id += 1
                self._tracks.append(track)
            out = replace(region, track_id=track.id)
            track.last, track.missed = replace(out), 0
            tracked.append(out)

        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return tracked


def interpolate_dropouts(
    detections: dict[int, list[Region]], max_gap: int = 15
) -> dict[int, list[Region]]:
    """Return a copy of `detections` with short gaps within tracks filled linearly.

    A gap between frames f0 and f1 is filled when 1 < f1 - f0 <= max_gap.
    Regions of different sources are never joined, even if they share an id.
    """
    filled = {frame: [replace(r) for r in regs] for frame, regs in detections.items()}

    sightings: defaultdict[tuple[str, int], dict[int, Region]] = defaultdict(dict)
    for frame in sorted(detections):
        for region in detections[frame]:
            if region.track_id >= 0:
                sightings[region.source, region.track_id].setdefault(frame, region)

    for seen in sightings.values():
        frames = list(seen)
        for f0, f1 in zip(frames, frames[1:]):
            if not 1 < f1 - f0 <= max_gap:
                continue
            a, b = seen[f0], seen[f1]
            for frame in range(f0 + 1, f1):
                t = (frame - f0) / (f1 - f0)
                between = replace(
                    a,
                    x=a.x + t * (b.x - a.x),
                    y=a.y + t * (b.y - a.y),
                    w=a.w + t * (b.w - a.w),
                    h=a.h + t * (b.h - a.h),
                    score=min(a.score, b.score),
                )
                filled.setdefault(frame, []).append(between)

    return dict(sorted(filled.items()))
