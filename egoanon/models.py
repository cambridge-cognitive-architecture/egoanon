"""Adapters for the optional models. Each takes a local path and never downloads."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .offline import go_offline
from .scene import Pose, SceneModels, SceneObject

SIGN_PROMPTS = ("a computer screen", "a monitor", "a wall sign", "a printed notice",
                "a whiteboard with writing")
THRESHOLD = 0.4
CLIPSEG_WIDTH = 512
# name: (prompts, how it is drawn). Only these names can appear as labels.
OBJECT_LABELS = {
    "door": (("double doors", "a door"), "pin"),
    "desk": (("a desk", "a counter"), "pin"),
    "curtain": (("a curtain",), "pin"),
    "cabinet": (("a cabinet", "a cupboard"), "pin"),
    "monitor": (("a computer monitor",), "box"),
    "chair": (("a chair",), "box"),
}


def local_file(path, what: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path.absolute()


def local_dir(path, what: str) -> Path:
    path = Path(path).expanduser()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{what} must be a local model folder with a config.json: {path}")
    return path.absolute()


def fingerprint(path: Path) -> str:
    """Identify weights by content, as paths can carry identifiers."""
    files = [path] if path.is_file() else sorted(
        f for f in path.iterdir() if f.suffix in (".safetensors", ".bin") or f.name == "config.json")
    h = hashlib.sha256()
    for file in files:
        with open(file, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class LamaInpainter:
    """LaMa through simple-lama-inpainting. There is no fallback inpainter."""

    def __init__(self, weights):
        path = local_file(weights, "LaMa weights")
        go_offline()
        from simple_lama_inpainting import SimpleLama

        # The package downloads unless LAMA_MODEL names a file.
        before = os.environ.get("LAMA_MODEL")
        os.environ["LAMA_MODEL"] = str(path)
        try:
            self.model = SimpleLama()
        finally:
            if before is None:
                del os.environ["LAMA_MODEL"]
            else:
                os.environ["LAMA_MODEL"] = before

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = self.model(Image.fromarray(image[..., ::-1].copy()),
                         Image.fromarray(mask.astype(np.uint8) * 255))
        return np.asarray(out)[..., ::-1]


class YoloPose:
    def __init__(self, weights, conf: float = 0.30, imgsz: int = 1600):
        path = local_file(weights, "YOLO pose weights")
        go_offline()
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        if self.model.task != "pose":
            raise ValueError("these weights are not a pose model")
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, image: np.ndarray) -> list[Pose]:
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if not len(result.boxes):
            return []
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        points = result.keypoints.xy.cpu().numpy()
        return [Pose(tuple(b), k, float(c))
                for b, k, c in zip(boxes, points, result.boxes.conf.cpu().numpy())]


class DepthAnything:
    """Depth Anything V2 from a local folder. Larger values are nearer."""

    def __init__(self, folder):
        path = local_dir(folder, "Depth Anything V2")
        go_offline()
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True).eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(image[..., ::-1].copy()), return_tensors="pt")
        with self.torch.no_grad():
            depth = self.model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                depth[None], size=image.shape[:2], mode="bicubic", align_corners=False)
        return depth[0, 0].numpy()


class ClipSeg:
    """Text-prompted segmentation, for screens and signs and for object labels."""

    def __init__(self, folder):
        path = local_dir(folder, "CLIPSeg")
        go_offline()
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(path, local_files_only=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            path, local_files_only=True).eval().to(self.device)

    def probability(self, image: np.ndarray, prompts: tuple[str, ...]) -> np.ndarray:
        """The highest probability over the prompts, at the image's size."""
        import cv2

        height, width = image.shape[:2]
        size = (CLIPSEG_WIDTH, max(1, int(height * CLIPSEG_WIDTH / width)))
        small = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=list(prompts), images=[small] * len(prompts),
                                padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            logits = logits.reshape(len(prompts), 1, *logits.shape[-2:])
            logits = self.torch.nn.functional.interpolate(
                logits, size=small.shape[:2], mode="bilinear", align_corners=False)
            best = self.torch.sigmoid(logits[:, 0]).max(0).values.cpu().numpy()
        return cv2.resize(best.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    def signs(self, image: np.ndarray) -> np.ndarray:
        return self.probability(image, SIGN_PROMPTS) > THRESHOLD

    def objects(self, image: np.ndarray, labels: dict | None = None) -> list[SceneObject]:
        return [SceneObject(name, self.probability(image, prompts) > THRESHOLD, kind)
                for name, (prompts, kind) in (labels or OBJECT_LABELS).items()]


def load_models(level: str, lama=None, depth=None, clipseg=None, pose=None,
                labels: bool = False) -> SceneModels:
    needed = {"lama": lama, **({"depth": depth, "clipseg": clipseg} if level == "depth" else {})}
    missing = [name for name, value in needed.items() if not value]
    if missing:
        raise ValueError(f"level {level!r} needs: {', '.join(missing)}")
    if labels and level != "depth":
        raise ValueError("object labels are only drawn on the depth level")

    models = SceneModels(inpaint=LamaInpainter(lama))
    used = dict(lama=lama)
    if level == "depth":
        segmenter = ClipSeg(clipseg)
        models.depth, models.signs = DepthAnything(depth), segmenter.signs
        used.update(depth=depth, clipseg=clipseg)
        if labels:
            models.objects = segmenter.objects
    if pose:
        models.pose = YoloPose(pose)
        used["pose"] = pose
    models.params = dict(weights_sha256={k: fingerprint(Path(v).expanduser()) for k, v in used.items()},
                         skeletons=bool(pose), labels=labels)
    return models
