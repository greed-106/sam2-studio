from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from sam2_studio.interaction.prompts import BoxPrompt, PromptBatch


def rgb_to_qimage(image_rgb: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(image_rgb)
    height, width, channels = contiguous.shape
    if channels != 3:
        raise ValueError("Expected an RGB image with three channels.")
    qimage = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()


class SamCanvas(QWidget):
    point_added = Signal(float, float, int)
    box_added = Signal(float, float, float, float)
    polygon_finished = Signal(object)
    brush_stroke = Signal(object, float, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image: QImage | None = None
        self._image_shape: tuple[int, int] | None = None
        self._overlay: QImage | None = None
        self._prompt_items: list[tuple[int, PromptBatch, tuple[int, int, int]]] = []
        self._tool = "positive"
        self._box_start: QPointF | None = None
        self._box_current: QPointF | None = None
        self._polygon_points: list[tuple[float, float]] = []
        self._brush_points: list[tuple[float, float]] = []
        self._brush_radius = 10.0

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._box_start = None
        self._box_current = None
        self._polygon_points.clear()
        self._brush_points.clear()
        self.update()

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._image = rgb_to_qimage(image_rgb)
        self._image_shape = image_rgb.shape[:2]
        self.update()

    def set_masks(self, mask_items: Iterable[tuple[int, np.ndarray, tuple[int, int, int]]]) -> None:
        items = list(mask_items)
        self._overlay = self._build_overlay(items) if items else None
        self.update()

    def set_prompts(self, prompt_items: Iterable[tuple[int, PromptBatch, tuple[int, int, int]]]) -> None:
        self._prompt_items = list(prompt_items)
        self.update()

    def _build_overlay(self, mask_items: list[tuple[int, np.ndarray, tuple[int, int, int]]]) -> QImage:
        if self._image_shape is None:
            raise ValueError("No image has been set.")
        height, width = self._image_shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        for _object_id, mask, color in mask_items:
            if mask.shape != (height, width):
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {(height, width)}")
            overlay[mask.astype(bool)] = (color[0], color[1], color[2], 110)
        qimage = QImage(overlay.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        return qimage.copy()

    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        image_w = self._image.width()
        image_h = self._image.height()
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / image_w, widget_h / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale
        return QRectF((widget_w - draw_w) / 2.0, (widget_h - draw_h) / 2.0, draw_w, draw_h)

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        if self._image is None:
            return None
        rect = self._image_rect()
        if not rect.contains(point):
            return None
        x = (point.x() - rect.x()) / rect.width() * self._image.width()
        y = (point.y() - rect.y()) / rect.height() * self._image.height()
        return float(x), float(y)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        rect = self._image_rect()
        if self._image is None:
            return QPointF()
        return QPointF(rect.x() + x / self._image.width() * rect.width(), rect.y() + y / self._image.height() * rect.height())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 22, 28))
        if self._image is None:
            painter.setPen(QColor(150, 160, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image or video to start")
            return

        target = self._image_rect()
        painter.drawImage(target, self._image)
        if self._overlay is not None:
            painter.drawImage(target, self._overlay)

        for _object_id, prompts, color in self._prompt_items:
            qcolor = QColor(*color)
            for point in prompts.points:
                pos = self._image_to_widget(point.x, point.y)
                pen = QPen(QColor(80, 255, 120) if point.label else QColor(255, 80, 80), 8)
                painter.setPen(pen)
                painter.drawPoint(pos)
            if prompts.box is not None:
                self._draw_box(painter, prompts.box, qcolor)

        if self._polygon_points:
            painter.setPen(QPen(QColor(255, 220, 90), 2, Qt.PenStyle.DashLine))
            widget_points = [self._image_to_widget(x, y) for x, y in self._polygon_points]
            for start, end in zip(widget_points, widget_points[1:]):
                painter.drawLine(start, end)
            for point in widget_points:
                painter.drawEllipse(point, 3, 3)

        if self._box_start is not None and self._box_current is not None:
            painter.setPen(QPen(QColor(255, 220, 90), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self._box_start, self._box_current).normalized())

    def _draw_box(self, painter: QPainter, box: BoxPrompt, color: QColor) -> None:
        start = self._image_to_widget(box.x1, box.y1)
        end = self._image_to_widget(box.x2, box.y2)
        painter.setPen(QPen(color, 2))
        painter.drawRect(QRectF(start, end).normalized())

    def mousePressEvent(self, event) -> None:
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        if self._tool == "box" and event.button() == Qt.MouseButton.LeftButton:
            self._box_start = event.position()
            self._box_current = event.position()
            self.update()
            return
        if self._tool == "polygon":
            if event.button() == Qt.MouseButton.LeftButton:
                self._polygon_points.append(image_point)
                self.update()
            elif event.button() == Qt.MouseButton.RightButton and len(self._polygon_points) >= 3:
                points = list(self._polygon_points)
                self._polygon_points.clear()
                self.polygon_finished.emit(points)
                self.update()
            return
        if self._tool in {"brush_add", "brush_erase"} and event.button() == Qt.MouseButton.LeftButton:
            self._brush_points = [image_point]
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._box_start is not None:
            self._box_current = event.position()
            self.update()
        if self._brush_points and self._tool in {"brush_add", "brush_erase"}:
            image_point = self._widget_to_image(event.position())
            if image_point is not None:
                self._brush_points.append(image_point)
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            if self._tool in {"brush_add", "brush_erase"} and self._brush_points:
                points = list(self._brush_points)
                self._brush_points.clear()
                self.brush_stroke.emit(points, self._brush_radius, 1 if self._tool == "brush_add" else 0)
                self.update()
                return
            self._box_start = None
            self._box_current = None
            self.update()
            return

        if self._tool == "box" and self._box_start is not None:
            start = self._widget_to_image(self._box_start)
            end = image_point
            self._box_start = None
            self._box_current = None
            if start is not None:
                self.box_added.emit(start[0], start[1], end[0], end[1])
            self.update()
            return

        if self._tool in {"brush_add", "brush_erase"} and self._brush_points:
            if image_point is not None:
                self._brush_points.append(image_point)
            points = list(self._brush_points)
            self._brush_points.clear()
            if points:
                self.brush_stroke.emit(points, self._brush_radius, 1 if self._tool == "brush_add" else 0)
            self.update()
            return

        if event.button() == Qt.MouseButton.RightButton:
            label = 0
        elif self._tool == "negative":
            label = 0
        else:
            label = 1
        self.point_added.emit(image_point[0], image_point[1], label)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._tool == "polygon" and len(self._polygon_points) >= 3:
            points = list(self._polygon_points)
            self._polygon_points.clear()
            self.polygon_finished.emit(points)
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._box_start = None
            self._box_current = None
            self._polygon_points.clear()
            self._brush_points.clear()
            self.update()
            return
        super().keyPressEvent(event)
