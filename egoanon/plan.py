"""Regions, and the plan that records what was examined and found."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PAD_FRAMES = 3
# How far each kind of box is grown before it is covered, as a fraction of it.
MARGIN = {"face": 0.28, "person": 0.28, "tag": 0.28, "manual": 0.28, "screen": 0.42}


@dataclass
class Region:
    """One region on one frame, in pixels."""
    x: float
    y: float
    w: float
    h: float
    source: str = "face"  # face | person | tag | screen | manual
    score: float = 1.0
    track_id: int = -1

    def grow(self, margin: float, width: int, height: int) -> "Region":
        """Enlarge by a fraction of the box, clipping each edge to the frame."""
        dx, dy = self.w * margin / 2, self.h * margin / 2
        x0, x1 = max(0.0, self.x - dx), min(float(width), self.x + self.w + dx)
        y0, y1 = max(0.0, self.y - dy), min(float(height), self.y + self.h + dy)
        return Region(x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0),
                      self.source, self.score, self.track_id)


@dataclass
class ManualRegion:
    """A hand-drawn box over an inclusive frame range."""
    x: float
    y: float
    w: float
    h: float
    frame_from: int
    frame_to: int
    note: str = ""


@dataclass
class BlurPlan:
    n_frames: int
    fps: float
    width: int
    height: int
    frame_from: int
    frame_to: int
    source_sha256: str
    detections: dict[int, list[Region]] = field(default_factory=dict)
    manual: list[ManualRegion] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def pad(self) -> int:
        """Frames each detection is carried either way; never less than the stride."""
        return max(DEFAULT_PAD_FRAMES, int(self.params.get("stride", 1)))

    @property
    def is_whole_video(self) -> bool:
        return self.frame_from == 0 and self.frame_to == self.n_frames - 1

    def covers(self, frame: int) -> bool:
        return self.frame_from <= frame <= self.frame_to

    def regions_at(self, frame: int) -> list[Region]:
        if not self.covers(frame):
            raise ValueError(f"frame {frame} was never scanned")
        out = []
        for f in range(frame - self.pad, frame + self.pad + 1):
            out.extend(self.detections.get(f, []))
        for m in self.manual:
            if m.frame_from <= frame <= m.frame_to:
                out.append(Region(m.x, m.y, m.w, m.h, "manual"))
        return out


def save_plan(plan: BlurPlan, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(plan), indent=1), encoding="utf-8")


def load_plan(path: str | Path) -> BlurPlan:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["detections"] = {int(k): [Region(**r) for r in v] for k, v in d["detections"].items()}
    d["manual"] = [ManualRegion(**m) for m in d["manual"]]
    return BlurPlan(**d)
