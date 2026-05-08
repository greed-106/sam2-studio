from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class FramePacket:
    index: int
    image_rgb: np.ndarray


class ImageSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> np.ndarray:
        image_bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {self.path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_image_sequence(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Image sequence path is not a directory: {root}")
    frames = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    frames.sort(key=natural_sort_key)
    if not frames:
        raise ValueError(f"No supported images were found in directory: {root}")
    return frames


class VideoSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")
        self._lock = threading.Lock()
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, frame_idx: int) -> FramePacket:
        with self._lock:
            bounded = max(0, min(int(frame_idx), max(0, self.frame_count - 1)))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, bounded)
            ok, image_bgr = self._cap.read()
        if not ok or image_bgr is None:
            raise ValueError(f"Could not read frame {frame_idx} from {self.path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return FramePacket(index=bounded, image_rgb=image_rgb)

    def close(self) -> None:
        with self._lock:
            self._cap.release()


class ImageSequenceSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frame_paths = list_image_sequence(self.path)
        self.frame_names = [frame.name for frame in self.frame_paths]
        self.frame_count = len(self.frame_paths)
        self.fps = 10.0
        first = ImageSource(self.frame_paths[0]).read()
        self.height, self.width = first.shape[:2]

    def read_frame(self, frame_idx: int) -> FramePacket:
        bounded = max(0, min(int(frame_idx), self.frame_count - 1))
        image_rgb = ImageSource(self.frame_paths[bounded]).read()
        if image_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(f"Image sequence frame shape mismatch: {self.frame_paths[bounded]}")
        return FramePacket(index=bounded, image_rgb=np.ascontiguousarray(image_rgb))

    def close(self) -> None:
        return None


def classify_media(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_dir():
        return "image_sequence"
    suffix = resolved.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported media extension: {suffix}")


def open_video_source(path: str | Path) -> VideoSource | ImageSequenceSource:
    if Path(path).is_dir():
        return ImageSequenceSource(path)
    return VideoSource(path)
