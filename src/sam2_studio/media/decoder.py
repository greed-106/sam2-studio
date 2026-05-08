from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from sam2_studio.media.source import FramePacket, ImageSequenceSource


class VideoPlaybackThread(QThread):
    frame_ready = Signal(object)
    failed = Signal(str)
    playback_finished = Signal()

    def __init__(self, video_path: str | Path, fps: float, parent=None) -> None:
        super().__init__(parent)
        self.video_path = str(video_path)
        self.fps = fps or 25.0
        self._lock = threading.Lock()
        self._playing = False
        self._stop = False
        self._seek_to: int | None = 0
        self._frame_inflight = False

    def play(self) -> None:
        with self._lock:
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def request_seek(self, frame_idx: int, play_after_seek: bool | None = None) -> None:
        with self._lock:
            self._seek_to = max(0, int(frame_idx))
            if play_after_seek is not None:
                self._playing = play_after_seek

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._playing = False

    def ack_frame(self) -> None:
        with self._lock:
            self._frame_inflight = False

    def _emit_frame(self, packet: FramePacket, force: bool = False) -> None:
        with self._lock:
            if self._frame_inflight and not force:
                return
            self._frame_inflight = True
        self.frame_ready.emit(packet)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.failed.emit(f"Could not open video: {self.video_path}")
            return

        frame_idx = 0
        frame_interval = 1.0 / max(1.0, self.fps)
        try:
            while True:
                with self._lock:
                    should_stop = self._stop
                    playing = self._playing
                    seek_to = self._seek_to
                    self._seek_to = None

                if should_stop:
                    break

                if seek_to is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    frame_idx = seek_to
                    ok, image_bgr = cap.read()
                    if ok and image_bgr is not None:
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                        self._emit_frame(FramePacket(index=frame_idx, image_rgb=image_rgb), force=True)
                        frame_idx += 1
                    else:
                        self.failed.emit(f"Could not read frame {seek_to} from {self.video_path}")

                if not playing:
                    self.msleep(10)
                    continue

                with self._lock:
                    frame_inflight = self._frame_inflight
                if frame_inflight:
                    self.msleep(1)
                    continue

                started = time.perf_counter()
                ok, image_bgr = cap.read()
                if not ok or image_bgr is None:
                    with self._lock:
                        self._playing = False
                    self.playback_finished.emit()
                    self.msleep(10)
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                self._emit_frame(FramePacket(index=frame_idx, image_rgb=image_rgb))
                frame_idx += 1

                elapsed = time.perf_counter() - started
                sleep_for = max(0.0, frame_interval - elapsed)
                if sleep_for:
                    self.msleep(int(sleep_for * 1000))
        finally:
            cap.release()


class ImageSequencePlaybackThread(QThread):
    frame_ready = Signal(object)
    failed = Signal(str)
    playback_finished = Signal()

    def __init__(self, sequence_path: str | Path, fps: float, parent=None) -> None:
        super().__init__(parent)
        self.sequence_path = str(sequence_path)
        self.fps = fps or 10.0
        self._lock = threading.Lock()
        self._playing = False
        self._stop = False
        self._seek_to: int | None = 0
        self._frame_inflight = False

    def play(self) -> None:
        with self._lock:
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def request_seek(self, frame_idx: int, play_after_seek: bool | None = None) -> None:
        with self._lock:
            self._seek_to = max(0, int(frame_idx))
            if play_after_seek is not None:
                self._playing = play_after_seek

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._playing = False

    def ack_frame(self) -> None:
        with self._lock:
            self._frame_inflight = False

    def _emit_frame(self, packet: FramePacket, force: bool = False) -> None:
        with self._lock:
            if self._frame_inflight and not force:
                return
            self._frame_inflight = True
        self.frame_ready.emit(packet)

    def run(self) -> None:
        try:
            source = ImageSequenceSource(self.sequence_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        frame_idx = 0
        frame_interval = 1.0 / max(1.0, self.fps)
        try:
            while True:
                with self._lock:
                    should_stop = self._stop
                    playing = self._playing
                    seek_to = self._seek_to
                    self._seek_to = None

                if should_stop:
                    break

                if seek_to is not None:
                    frame_idx = min(max(0, seek_to), source.frame_count - 1)
                    try:
                        packet = source.read_frame(frame_idx)
                    except Exception as exc:
                        self.failed.emit(str(exc))
                    else:
                        self._emit_frame(packet, force=True)
                        frame_idx = packet.index + 1

                if not playing:
                    self.msleep(10)
                    continue

                if frame_idx >= source.frame_count:
                    with self._lock:
                        self._playing = False
                    self.playback_finished.emit()
                    self.msleep(10)
                    continue

                with self._lock:
                    frame_inflight = self._frame_inflight
                if frame_inflight:
                    self.msleep(1)
                    continue

                started = time.perf_counter()
                try:
                    packet = source.read_frame(frame_idx)
                except Exception as exc:
                    self.failed.emit(str(exc))
                    with self._lock:
                        self._playing = False
                    self.msleep(10)
                    continue

                self._emit_frame(packet)
                frame_idx = packet.index + 1

                elapsed = time.perf_counter() - started
                sleep_for = max(0.0, frame_interval - elapsed)
                if sleep_for:
                    self.msleep(int(sleep_for * 1000))
        finally:
            source.close()
