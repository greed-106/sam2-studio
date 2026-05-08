from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sam2_studio.interaction.prompts import PromptBatch


@dataclass
class MaskResult:
    object_id: int
    masks: np.ndarray
    scores: np.ndarray | None = None
    boxes_xyxy: np.ndarray | None = None


@dataclass
class FrameMaskResult:
    frame_idx: int
    object_id: int
    mask: np.ndarray
    score: float | None = None
    box_xyxy: np.ndarray | None = None


class Sam2Backend(Protocol):
    def load_model(self, model_path: str, device: str = "", half: bool = False) -> None: ...

    def segment_image(
        self,
        image_rgb: np.ndarray,
        image_key: str,
        object_id: int,
        prompts: PromptBatch,
    ) -> MaskResult: ...
