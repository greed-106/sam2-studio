from __future__ import annotations

import gc
import traceback
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from sam2_studio.engine.backend import FrameMaskResult, MaskResult, Sam2Backend
from sam2_studio.interaction.prompts import PromptBatch
from sam2_studio.media.source import list_image_sequence


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class PropagationSignals(QObject):
    frame_ready = Signal(object)
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)


class LoadModelTask(QRunnable):
    def __init__(self, backend: Sam2Backend, model_path: str, device: str = "", half: bool = False) -> None:
        super().__init__()
        self.backend = backend
        self.model_path = model_path
        self.device = device
        self.half = half
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.backend.load_model(self.model_path, self.device, self.half)
            self.signals.finished.emit(self.model_path)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class SegmentTask(QRunnable):
    def __init__(
        self,
        backend: Sam2Backend,
        image_rgb: np.ndarray,
        image_key: str,
        generation: int,
        request_id: int,
        frame_idx: int,
        object_id: int,
        prompts: PromptBatch,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.image_rgb = np.ascontiguousarray(image_rgb.copy())
        self.image_key = image_key
        self.generation = generation
        self.request_id = request_id
        self.frame_idx = frame_idx
        self.object_id = object_id
        self.prompts = prompts.clone()
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result: MaskResult = self.backend.segment_image(
                self.image_rgb,
                self.image_key,
                self.object_id,
                self.prompts,
            )
            self.signals.finished.emit((self.generation, self.request_id, self.image_key, self.frame_idx, result))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class ExportTask(QRunnable):
    def __init__(self, label: str, export_func) -> None:
        super().__init__()
        self.label = label
        self.export_func = export_func
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit((self.label, self.export_func()))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class VideoPropagationTask(QRunnable):
    """Run SAM2 video propagation from the current frame in a worker thread.

    The task creates a temporary MJPG clip that starts at the prompted frame. This keeps
    Ultralytics' high-level `SAM2VideoPredictor` on its supported first-frame prompt path
    while still allowing the UI to propagate from an arbitrary frame.
    """

    def __init__(
        self,
        model_path: str,
        video_path: str,
        start_frame_idx: int,
        frame_count: int,
        fps: float,
        object_id: int,
        prompts: PromptBatch,
        direction: str,
        generation: int,
        request_id: int,
        imgsz: int = 1024,
        conf: float = 0.25,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.start_frame_idx = int(start_frame_idx)
        self.frame_count = int(frame_count)
        self.fps = float(fps or 25.0)
        self.object_id = int(object_id)
        self.prompts = prompts.clone()
        self.direction = direction
        self.generation = generation
        self.request_id = request_id
        self.imgsz = imgsz
        self.conf = conf
        self._cancelled = False
        self.signals = PropagationSignals()

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            if self.prompts.is_empty():
                raise ValueError("At least one prompt is required before propagation.")
            frame_indices = self._frame_indices()
            if not frame_indices:
                raise ValueError("No frames are available for propagation.")
            if self._cancelled:
                raise RuntimeError("Propagation cancelled.")
            if self._can_use_original_video(frame_indices):
                self._run_predictor(frame_indices, Path(self.video_path))
                self.signals.finished.emit((self.generation, self.request_id, self.direction))
                return
            with tempfile.TemporaryDirectory(prefix="sam2-studio-prop-") as tmpdir:
                clip_path = Path(tmpdir) / "clip.avi"
                self._write_clip(frame_indices, clip_path)
                self._run_predictor(frame_indices, clip_path)
            self.signals.finished.emit((self.generation, self.request_id, self.direction))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())

    def _frame_indices(self) -> list[int]:
        if self.direction == "forward":
            return list(range(self.start_frame_idx, self.frame_count))
        if self.direction == "reverse":
            return list(range(self.start_frame_idx, -1, -1))
        raise ValueError(f"Unsupported propagation direction: {self.direction}")

    def _can_use_original_video(self, frame_indices: list[int]) -> bool:
        return (
            self.direction == "forward"
            and self.start_frame_idx == 0
            and not Path(self.video_path).is_dir()
            and len(frame_indices) == self.frame_count
        )

    def _write_clip(self, frame_indices: list[int], clip_path: Path) -> None:
        if Path(self.video_path).is_dir():
            self._write_image_sequence_clip(frame_indices, clip_path)
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                raise ValueError("Could not create temporary propagation clip.")
            try:
                for frame_idx in frame_indices:
                    if self._cancelled:
                        raise RuntimeError("Propagation cancelled.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise ValueError(f"Could not read frame {frame_idx} from {self.video_path}")
                    writer.write(frame)
            finally:
                writer.release()
        finally:
            cap.release()

    def _write_image_sequence_clip(self, frame_indices: list[int], clip_path: Path) -> None:
        frame_paths = list_image_sequence(self.video_path)
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Could not read image sequence frame: {frame_paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("Could not create temporary propagation clip.")
        try:
            for frame_idx in frame_indices:
                if self._cancelled:
                    raise RuntimeError("Propagation cancelled.")
                frame_path = frame_paths[frame_idx]
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Could not read image sequence frame: {frame_path}")
                if frame.shape[:2] != (height, width):
                    raise ValueError(f"Image sequence frame shape mismatch: {frame_path}")
                writer.write(frame)
        finally:
            writer.release()

    def _run_predictor(self, frame_indices: list[int], clip_path: Path) -> None:
        if self._cancelled:
            raise RuntimeError("Propagation cancelled.")
        from ultralytics.models.sam import SAM2VideoPredictor

        kwargs = self.prompts.to_ultralytics()
        if self._cancelled:
            raise RuntimeError("Propagation cancelled.")
        predictor = None
        try:
            predictor = SAM2VideoPredictor(
                overrides={
                    "model": self.model_path,
                    "imgsz": self.imgsz,
                    "conf": self.conf,
                    "save": False,
                    "verbose": False,
                }
            )
            total = len(frame_indices)
            for clip_idx, result in enumerate(
                predictor(
                    source=str(clip_path),
                    stream=True,
                    points=kwargs["points"],
                    labels=kwargs["labels"],
                    bboxes=kwargs["bboxes"],
                )
            ):
                if self._cancelled:
                    raise RuntimeError("Propagation cancelled.")
                if clip_idx >= total:
                    break
                if result.masks is None or len(result.masks) == 0:
                    self.signals.progress.emit(min(clip_idx + 1, total), total)
                    continue
                masks = result.masks.data.detach().cpu().numpy().astype(bool)
                score = None
                box = None
                if result.boxes is not None and len(result.boxes) > 0:
                    scores = result.boxes.conf.detach().cpu().numpy()
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    score = float(scores[0]) if len(scores) else None
                    box = boxes[0] if len(boxes) else None
                frame_result = FrameMaskResult(
                    frame_idx=frame_indices[clip_idx],
                    object_id=self.object_id,
                    mask=masks[0],
                    score=score,
                    box_xyxy=box,
                )
                self.signals.frame_ready.emit((self.generation, self.request_id, frame_result))
                self.signals.progress.emit(min(clip_idx + 1, total), total)
        finally:
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
