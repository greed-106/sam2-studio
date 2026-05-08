from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sam2_studio.interaction.prompts import PromptBatch


Color = tuple[int, int, int]


DEFAULT_COLORS: list[Color] = [
    (0, 204, 102),
    (255, 99, 71),
    (65, 105, 225),
    (255, 193, 7),
    (156, 39, 176),
    (0, 188, 212),
]


@dataclass
class ObjectState:
    object_id: int
    name: str
    color: Color
    visible: bool = True


class AnnotationSession:
    def __init__(self) -> None:
        self.media_path: str | None = None
        self.media_kind: str | None = None
        self.frame_names: list[str] = []
        self.current_frame_idx = 0
        self.current_image_rgb: np.ndarray | None = None
        self.objects: dict[int, ObjectState] = {}
        self.current_object_id = 0
        self.prompts: dict[int, dict[int, PromptBatch]] = {}
        self.masks: dict[int, dict[int, np.ndarray]] = {}
        self.mask_undo_stack: list[tuple[int, int, np.ndarray | None]] = []
        self.add_object(0)

    def reset_media(self, media_path: str, media_kind: str, frame_names: list[str] | None = None) -> None:
        self.media_path = media_path
        self.media_kind = media_kind
        self.frame_names = list(frame_names or [])
        self.current_frame_idx = 0
        self.current_image_rgb = None
        self.prompts.clear()
        self.masks.clear()
        self.mask_undo_stack.clear()

    def add_object(self, object_id: int | None = None) -> ObjectState:
        if object_id is None:
            object_id = max(self.objects, default=-1) + 1
        color = DEFAULT_COLORS[object_id % len(DEFAULT_COLORS)]
        state = ObjectState(object_id=object_id, name=f"object {object_id}", color=color)
        self.objects[object_id] = state
        self.current_object_id = object_id
        return state

    def set_frame(self, frame_idx: int, image_rgb: np.ndarray) -> None:
        self.current_frame_idx = int(frame_idx)
        self.current_image_rgb = image_rgb

    def prompt_batch(self, frame_idx: int | None = None, object_id: int | None = None) -> PromptBatch:
        frame = self.current_frame_idx if frame_idx is None else int(frame_idx)
        obj = self.current_object_id if object_id is None else int(object_id)
        return self.prompts.setdefault(frame, {}).setdefault(obj, PromptBatch())

    def set_mask(self, frame_idx: int, object_id: int, mask: np.ndarray) -> None:
        self.masks.setdefault(int(frame_idx), {})[int(object_id)] = mask.astype(bool)

    def clear_mask(self, frame_idx: int, object_id: int) -> None:
        frame_masks = self.masks.get(int(frame_idx))
        if frame_masks is None:
            return
        frame_masks.pop(int(object_id), None)
        if not frame_masks:
            self.masks.pop(int(frame_idx), None)

    def push_mask_undo(self, frame_idx: int, object_id: int, previous_mask: np.ndarray | None) -> None:
        previous = None if previous_mask is None else previous_mask.astype(bool).copy()
        self.mask_undo_stack.append((int(frame_idx), int(object_id), previous))

    def undo_mask_edit(self) -> bool:
        if not self.mask_undo_stack:
            return False
        frame_idx, object_id, previous_mask = self.mask_undo_stack.pop()
        if previous_mask is None:
            self.clear_mask(frame_idx, object_id)
        else:
            self.set_mask(frame_idx, object_id, previous_mask)
        return True

    def undo_last_prompt(self, frame_idx: int | None = None, object_id: int | None = None) -> bool:
        frame = self.current_frame_idx if frame_idx is None else int(frame_idx)
        obj = self.current_object_id if object_id is None else int(object_id)
        prompts = self.prompts.get(frame, {}).get(obj)
        if prompts is None:
            return False
        if prompts.points:
            prompts.points.pop()
        elif prompts.box is not None:
            prompts.box = None
        else:
            return False
        if prompts.is_empty():
            self.prompts.get(frame, {}).pop(obj, None)
        self.clear_mask(frame, obj)
        return True

    def current_masks(self) -> list[tuple[int, np.ndarray, Color]]:
        frame_masks = self.masks.get(self.current_frame_idx, {})
        return self.mask_items_for_frame(self.current_frame_idx, visible_only=True)

    def mask_items_for_frame(self, frame_idx: int, visible_only: bool = False) -> list[tuple[int, np.ndarray, Color]]:
        frame_masks = self.masks.get(int(frame_idx), {})
        items: list[tuple[int, np.ndarray, Color]] = []
        for object_id, mask in frame_masks.items():
            obj = self.objects.get(object_id)
            if obj is not None and (obj.visible or not visible_only):
                items.append((object_id, mask, obj.color))
        return items

    def current_prompts(self) -> list[tuple[int, PromptBatch, Color]]:
        frame_prompts = self.prompts.get(self.current_frame_idx, {})
        items: list[tuple[int, PromptBatch, Color]] = []
        for object_id, prompts in frame_prompts.items():
            obj = self.objects.get(object_id)
            if obj is not None and obj.visible:
                items.append((object_id, prompts, obj.color))
        return items

    def clear_annotations(self) -> None:
        self.prompts.clear()
        self.masks.clear()
        self.mask_undo_stack.clear()
