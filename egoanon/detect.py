"""Detection is a union of cautious sources.

faces    MediaPipe BlazeFace (full range)
persons  the head band of every person, so people facing away are covered
tags     the torso band, where a badge or lanyard sits
screens  tv, laptop and phone classes

A detector that cannot load, or that errors, stops the run.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .plan import Region
from .yolo import YoloObjects

HEAD_FRACTION, HEAD_WIDTH = 0.32, 0.62
TORSO_FROM, TORSO_TO = 0.28, 0.62
# On a head-mounted camera a wide "person" along the bottom edge is usually the
# wearer's own arm. Other wide boxes (seated, lying, bending) are covered whole.
PERSON_MAX_ASPECT = 1.3
ARM_BOTTOM_FRAC, ARM_TOP_FRAC = 0.98, 0.50


def go_offline() -> None:
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["ULTRALYTICS_ANALYTICS"] = "False"


def person_regions(corners, width: int, height: int, score: float = 1.0,
                   tags: bool = True) -> list[Region]:
    left, top, right, bottom = corners
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return []
    if w / h > PERSON_MAX_ASPECT:
        own_arm = bottom >= ARM_BOTTOM_FRAC * height and top >= ARM_TOP_FRAC * height
        return [] if own_arm else [Region(left, top, w, h, "person", score)]
    head_w = w * HEAD_WIDTH
    out = [Region((left + right - head_w) / 2, top, head_w, h * HEAD_FRACTION, "person", score)]
    if tags:
        out.append(Region(left, top + h * TORSO_FROM, w, h * (TORSO_TO - TORSO_FROM), "tag", score))
    return out


class ModelDetector:
    def __init__(self, weights: str | Path, face_conf: float = 0.20,
                 person_conf: float = 0.40, screen_conf: float = 0.25):
        go_offline()  # must run before ultralytics is imported
        self.objects = YoloObjects(weights, person_conf, screen_conf)
        import mediapipe as mp

        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=face_conf)
        self.params = dict(weights=Path(weights).name, face_conf=face_conf,
                           person_conf=person_conf, screen_conf=screen_conf)

    def __call__(self, image: np.ndarray) -> list[Region]:
        import cv2

        height, width = image.shape[:2]
        out = []
        found = self.face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for d in found.detections or []:
            b = d.location_data.relative_bounding_box
            out.append(Region(b.xmin * width, b.ymin * height, b.width * width, b.height * height,
                              "face", float(d.score[0]) if d.score else 1.0))
        persons, screens = self.objects(image)
        for *corners, score in persons.tolist():
            out.extend(person_regions(corners, width, height, score))
        for left, top, right, bottom, score in screens.tolist():
            out.append(Region(left, top, right - left, bottom - top, "screen", score))
        return out
