from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


class PlaybackBar(QWidget):
    play_toggled = Signal(bool)
    seek_requested = Signal(int)
    step_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._playing = False
        self.previous_button = QPushButton("上一帧")
        self.play_button = QPushButton("播放")
        self.next_button = QPushButton("下一帧")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("0 / 0")
        self.marker_label = QLabel("P:0 M:0")
        self.slider.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.next_button.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.previous_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.marker_label)

        self.previous_button.clicked.connect(lambda: self.step_requested.emit(-1))
        self.play_button.clicked.connect(self._on_play_clicked)
        self.next_button.clicked.connect(lambda: self.step_requested.emit(1))
        self.slider.sliderReleased.connect(self._on_slider_released)

    def configure(self, frame_count: int) -> None:
        enabled = frame_count > 1
        self.slider.setEnabled(enabled)
        self.previous_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.play_button.setText("播放")
        self._playing = False
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, max(0, frame_count - 1))
            self.slider.setValue(0)
        self.set_position(0, frame_count)
        self.set_markers(set(), set())

    def set_position(self, frame_idx: int, frame_count: int | None = None) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(frame_idx)
        total = frame_count if frame_count is not None else self.slider.maximum() + 1
        self.frame_label.setText(f"{frame_idx + 1} / {max(1, total)}")

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.play_button.setText("暂停" if playing else "播放")

    def set_markers(self, prompted_frames: set[int], mask_frames: set[int]) -> None:
        self.marker_label.setText(f"P:{len(prompted_frames)} M:{len(mask_frames)}")
        self.marker_label.setToolTip(
            "Prompted frames: "
            + ", ".join(str(frame + 1) for frame in sorted(prompted_frames)[:20])
            + "\nMask frames: "
            + ", ".join(str(frame + 1) for frame in sorted(mask_frames)[:20])
        )

    def is_playing(self) -> bool:
        return self._playing

    def _on_play_clicked(self) -> None:
        self.set_playing(not self._playing)
        self.play_toggled.emit(self._playing)

    def _on_slider_released(self) -> None:
        self.seek_requested.emit(self.slider.value())
