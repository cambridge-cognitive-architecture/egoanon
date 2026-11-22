"""Person and screen detection through the Ultralytics Python API."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SCREEN_CLASSES = ("tv", "laptop", "cell phone", "monitor", "screen")


class YoloObjects:
    """Detect persons and screens with a local YOLO weights file."""

    def __init__(
        self,
        weights: str | os.PathLike,
        person_conf: float = 0.40,
        screen_conf: float = 0.25,
    ):
        """Load existing weights; nothing is ever downloaded."""
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {path}")

        os.environ["YOLO_OFFLINE"] = "1"  # read by ultralytics when it is imported
        from ultralytics import YOLO

        # An absolute path cannot be mistaken for the name of a downloadable model.
        self.model = YOLO(str(path.absolute()))
        names = self.model.names
        self._person_ids = [i for i, name in names.items() if name == "person"]
        self._screen_ids = [i for i, name in names.items() if name in SCREEN_CLASSES]
        if not self._person_ids:
            raise ValueError('The model has no "person" class')
        if not self._screen_ids:
            raise ValueError(f"The model has none of the screen classes {SCREEN_CLASSES}")

        self.person_conf = person_conf
        self.screen_conf = screen_conf

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (persons, screens), each (N, 5): x0, y0, x1, y1, confidence."""
        result = self.model.predict(
            image,
            classes=self._person_ids + self._screen_ids,
            conf=min(self.person_conf, self.screen_conf),
            verbose=False,
        )[0]
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()

        def select(ids: list[int], threshold: float) -> np.ndarray:
            keep = np.isin(cls, ids) & (conf >= threshold)
            return np.column_stack((xyxy[keep], conf[keep])).astype(float)

        return (
            select(self._person_ids, self.person_conf),
            select(self._screen_ids, self.screen_conf),
        )
