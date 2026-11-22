"""The "record" level: screened text, one line per window, and no image.

OFFLINE ONLY. The language model is shown original frames, people and screens
included, so it must run on the machine that holds the video. No frame and no
rejected text is written or printed. The kept text can still be wrong: these
models name objects that are not in the frame, so a person must check every line.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import cv2

from .offline import go_offline
from .plan import BlurPlan
from .screen import MAX_WORDS, MIN_WORDS, OBJECT_WORDS, screen
from .video import SCHEMA, check_source, manifest_path, sha256

WINDOW_S = 10.0
FRAME_WIDTH = 1024
MAX_TOKENS = 40
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2)
WITHHELD = "[withheld: failed the screen]"
PROMPT = "\n".join([
    "This is one frame from a workplace. List AT MOST 6 WORDS naming the "
    "objects in the ROOM, separated by commas (for example: desk, door, "
    "computer, equipment).",
    "Name only physical objects you can actually see. Do not name the room "
    "itself and do not use the words room, fittings or layout.",
    "Do NOT mention any person, people, staff, or any body part, clothing, "
    "mask, glove or uniform, and do NOT mention anyone doing anything.",
    "Do NOT mention any text, number, name, label, letter or anything "
    "written on a screen, sign, badge or document.",
    "Do NOT mention any medical condition, symptom, diagnosis, treatment, "
    "procedure or medication.",
    "Reply with the list only, no full stop, no preamble."])


def describe(image, vlm: Callable) -> dict:
    """Ask until a line passes the screen. Rejected text is dropped, never returned."""
    reasons: set[str] = set()
    for attempt, temperature in enumerate(TEMPERATURES, 1):
        text = (vlm(image, PROMPT, temperature) or "").strip().strip('"').strip()
        line = text.split("\n")[0].rstrip(".").strip()
        failed = screen(line)
        if not failed:
            return dict(scene=line, passed=True, attempts=attempt, reject_reasons=sorted(reasons))
        reasons.update(failed)
    return dict(scene=WITHHELD, passed=False, attempts=len(TEMPERATURES),
                reject_reasons=sorted(reasons))


def window_frames(n_frames: int, fps: float, window_s: float) -> list[tuple[int, int, int]]:
    """(first, last, sampled) for each window; the middle frame is sampled."""
    step = window_s * fps
    if not step >= 1:
        raise ValueError("a window must be at least one frame long")
    out = []
    for k in range(math.ceil(n_frames / step)):
        first, last = round(k * step), min(n_frames, round((k + 1) * step)) - 1
        if first <= last:
            out.append((first, last, (first + last) // 2))
    return out


def render_record(video: str | Path, plan: BlurPlan, output: str | Path, vlm: Callable,
                  window_s: float = WINDOW_S) -> dict:
    video, output = Path(video), Path(output)
    if output.suffix != ".json":
        raise ValueError("a record is written to a .json file")
    check_source(video, plan, output)
    windows = {w[2]: w for w in window_frames(plan.n_frames, plan.fps, window_s)}
    cap = cv2.VideoCapture(str(video))
    rows, i = [], 0
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot open the video")
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if i in windows:
                height, width = image.shape[:2]
                size = (FRAME_WIDTH, 2 * round(height * FRAME_WIDTH / width / 2))
                small = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                first, last, _ = windows[i]
                rows.append(dict(start_s=round(first / plan.fps, 3),
                                 end_s=round((last + 1) / plan.fps, 3), **describe(small, vlm)))
            i += 1
    finally:
        cap.release()
    if i != plan.n_frames:
        raise RuntimeError(f"read {i} of {plan.n_frames} frames; no record was written")

    record = dict(schema=SCHEMA, level="record", window_s=window_s, windows=rows,
                  screen=dict(rule="every word on the object list", object_words=len(OBJECT_WORDS),
                              min_words=MIN_WORDS, max_words=MAX_WORDS),
                  caveat="Model output: it can name objects that are not present "
                         "(hallucination). A person must check every line against the video.")
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
    partial.rename(output)
    manifest = dict(schema=SCHEMA, level="record", source_sha256=plan.source_sha256,
                    output_sha256=sha256(output), n_frames=i, n_windows=len(rows),
                    n_withheld=sum(not r["passed"] for r in rows),
                    models=getattr(vlm, "params", {}), caveat=record["caveat"])
    manifest_path(output).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


class MlxVlm:
    """A local vision-language model through mlx-vlm (Apple silicon)."""

    def __init__(self, folder):
        from .models import fingerprint, local_dir

        path = local_dir(folder, "the language model")
        go_offline()
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(str(path))
        self.config = getattr(self.model, "config", None) or load_config(str(path))
        self._generate, self._template = generate, apply_chat_template
        self.params = dict(weights_sha256=dict(vlm=fingerprint(path)))

    def __call__(self, image, prompt: str, temperature: float) -> str:
        from PIL import Image

        text = self._template(self.processor, self.config, prompt, num_images=1)
        return self._generate(self.model, self.processor, text,
                              image=[Image.fromarray(image[..., ::-1].copy())],
                              max_tokens=MAX_TOKENS, temperature=temperature, verbose=False).text
