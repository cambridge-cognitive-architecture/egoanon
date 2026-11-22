"""Scan once, render from the plan, refuse anything that was not examined."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import cv2

from .plan import BlurPlan, Region
from .render import LEVELS, apply_level
from .tracking import IouTracker, interpolate_dropouts

SCHEMA = "egoanon/1"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_video(video: str | Path, detector: Callable[..., list[Region]],
               stride: int = 1, max_gap: int = 15) -> BlurPlan:
    """Walk the video from the start, without seeking, which can be off by frames."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = IouTracker(max_missed=max_gap)
    motion = 0.04 * (stride - 1)
    found: dict[int, list[Region]] = {}
    i = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if i % stride == 0:
                regions = tracker.step(detector(image))
                if regions:
                    found[i] = [r.grow(motion, width, height) if motion else r for r in regions]
            i += 1
    finally:
        cap.release()
    if i == 0:
        raise ValueError("no frame could be read")
    return BlurPlan(max(n, i), fps, width, height, 0, i - 1, sha256(video),
                    interpolate_dropouts(found, max_gap), [],
                    dict(stride=stride, max_gap=max_gap, **getattr(detector, "params", {})))


def render_video(video: str | Path, plan: BlurPlan, output: str | Path,
                 level: str = "blur") -> dict:
    video, output = Path(video), Path(output)
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    if not plan.is_whole_video:
        raise ValueError("the plan does not cover the whole video")
    if sha256(video) != plan.source_sha256:
        raise ValueError("the plan was made for a different video")
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"),
                             plan.fps, (plan.width, plan.height))
    i = covered = 0
    try:
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError("cannot open the video reader or writer")
        while True:
            ok, image = cap.read()
            if not ok:
                break
            regions = plan.regions_at(i)
            writer.write(apply_level(image, regions, level))
            covered += bool(regions)
            i += 1
    finally:
        cap.release()
        writer.release()
    if i != plan.n_frames:
        raise RuntimeError(f"read {i} of {plan.n_frames} frames; output left as {partial.name}")
    partial.rename(output)
    # File names and paths often carry identifiers, so only hashes are recorded.
    manifest = dict(schema=SCHEMA, level=level, source_sha256=plan.source_sha256,
                    output_sha256=sha256(output), n_frames=i, n_frames_covered=covered,
                    params=plan.params,
                    caveat="Automatic detection misses things. A person must review the output.")
    manifest_path(output).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def manifest_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_manifest.json")


def is_anonymised(video: str | Path) -> bool:
    """True only if a manifest vouches for exactly these bytes."""
    video = Path(video)
    if not manifest_path(video).exists():
        return False
    m = json.loads(manifest_path(video).read_text(encoding="utf-8"))
    return m.get("schema") == SCHEMA and m.get("output_sha256") == sha256(video)
