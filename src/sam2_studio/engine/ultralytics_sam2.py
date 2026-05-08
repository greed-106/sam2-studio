from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from sam2_studio.engine.backend import MaskResult
from sam2_studio.interaction.prompts import PromptBatch


class UltralyticsSam2Backend:
    """Small adapter around Ultralytics SAM2 image predictor."""

    def __init__(self, imgsz: int = 1024, conf: float = 0.25) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self._predictor = None
        self._model_path: str | None = None
        self._image_key: str | None = None
        self._lock = threading.Lock()

    def load_model(self, model_path: str, device: str = "", half: bool = False) -> None:
        with self._lock:
            resolved = str(Path(model_path).expanduser().resolve())
            if self._predictor is not None and self._model_path == resolved:
                return

            from ultralytics.models.sam import SAM2Predictor

            overrides = {
                "model": resolved,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "save": False,
                "verbose": False,
                "half": half,
            }
            if device:
                overrides["device"] = device
            self._predictor = SAM2Predictor(overrides=overrides)
            self._model_path = resolved
            self._image_key = None

    def segment_image(
        self,
        image_rgb: np.ndarray,
        image_key: str,
        object_id: int,
        prompts: PromptBatch,
    ) -> MaskResult:
        if prompts.is_empty():
            raise ValueError("At least one point or box prompt is required.")

        with self._lock:
            if self._predictor is None:
                raise RuntimeError("SAM2 model is not loaded.")

            image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
            if self._image_key != image_key:
                self._predictor.reset_image()
                self._predictor.set_image(image_bgr)
                self._image_key = image_key

            kwargs = prompts.to_ultralytics()
            with torch.inference_mode():
                results = self._predictor(
                    points=kwargs["points"],
                    labels=kwargs["labels"],
                    bboxes=kwargs["bboxes"],
                )

            if not results or results[0].masks is None:
                raise RuntimeError("SAM2 did not return any mask for the prompt.")

            result = results[0]
            masks = result.masks.data.detach().cpu().numpy().astype(bool)
            scores = None
            boxes = None
            if result.boxes is not None and len(result.boxes) > 0:
                scores = result.boxes.conf.detach().cpu().numpy()
                boxes = result.boxes.xyxy.detach().cpu().numpy()

            if masks.shape[0] > 1:
                best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
                masks = masks[best_idx : best_idx + 1]
                if scores is not None:
                    scores = scores[best_idx : best_idx + 1]
                if boxes is not None:
                    boxes = boxes[best_idx : best_idx + 1]

            return MaskResult(object_id=object_id, masks=masks, scores=scores, boxes_xyxy=boxes)
