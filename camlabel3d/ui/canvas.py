"""Interactive frame canvas for prompt drawing and result preview."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from camlabel3d.core.models import PointPrompt, PromptMode

from .image_utils import pil_to_qimage


class FrameCanvas(QWidget):
    """Aspect-preserving image canvas with prompt drawing support."""

    boxCompleted = Signal(object)
    pointAdded = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        self._image_size = (0, 0)
        self._prompt_mode = PromptMode.TEXT
        self._point_label = 1
        self._prompt_box: tuple[float, float, float, float] | None = None
        self._prompt_points: list[PointPrompt] = []
        self._detection_boxes: list[tuple[float, float, float, float]] = []
        self._highlight_box: tuple[float, float, float, float] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None

    def clear(self) -> None:
        self._pixmap = None
        self._image_size = (0, 0)
        self._detection_boxes = []
        self._highlight_box = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_image(self, image: Image.Image | np.ndarray) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        self.set_qimage(pil_to_qimage(image))

    def set_qimage(self, qimage: QImage) -> None:
        """Install an already converted image with minimal GUI-thread work."""

        self._pixmap = QPixmap.fromImage(qimage)
        self._image_size = (qimage.width(), qimage.height())
        self.update()

    def set_prompt_mode(self, mode: PromptMode) -> None:
        self._prompt_mode = mode
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_point_label(self, label: int) -> None:
        self._point_label = 1 if int(label) else 0

    def set_prompt_box(self, box: tuple[float, float, float, float] | None) -> None:
        self._prompt_box = box
        self.update()

    def set_prompt_points(self, points: Iterable[PointPrompt]) -> None:
        self._prompt_points = list(points)
        self.update()

    def set_detection_boxes(self, boxes: Iterable[tuple[float, float, float, float]]) -> None:
        self._detection_boxes = list(boxes)
        self.update()

    def set_highlight_box(self, box: tuple[float, float, float, float] | None) -> None:
        self._highlight_box = box
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        if self._pixmap is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video or image folder to begin.")
            return

        target = self._target_rect()
        painter.drawPixmap(target.toRect(), self._pixmap)

        detection_pen = QPen(QColor(88, 190, 255), 2)
        highlight_pen = QPen(QColor(32, 255, 32), 3)
        prompt_pen = QPen(QColor(255, 72, 72), 3)
        temp_pen = QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine)
        point_pen = QPen(QColor(255, 255, 255), 1)

        for box in self._detection_boxes:
            painter.setPen(detection_pen)
            painter.drawRect(self._image_box_to_widget_rect(box))

        if self._highlight_box is not None:
            painter.setPen(highlight_pen)
            painter.drawRect(self._image_box_to_widget_rect(self._highlight_box))

        if self._prompt_box is not None:
            painter.setPen(prompt_pen)
            painter.drawRect(self._image_box_to_widget_rect(self._prompt_box))

        if self._drag_start is not None and self._drag_current is not None:
            painter.setPen(temp_pen)
            painter.drawRect(self._image_box_to_widget_rect((*self._drag_start, *self._drag_current)))

        for point in self._prompt_points:
            widget_point = self._image_to_widget(point.x, point.y)
            color = QColor(255, 64, 64) if point.label == 1 else QColor(170, 170, 170)
            painter.setPen(point_pen)
            painter.setBrush(color)
            painter.drawEllipse(widget_point, 5, 5)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        pos = self._widget_to_image(event.position())
        if pos is None:
            return
        if self._prompt_mode in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE):
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_start = pos
                self._drag_current = pos
                self.update()
        elif self._prompt_mode == PromptMode.POINT:
            if event.button() == Qt.MouseButton.LeftButton:
                point = PointPrompt(x=pos[0], y=pos[1], label=self._point_label)
                self.pointAdded.emit(point)
            elif event.button() == Qt.MouseButton.RightButton and self._prompt_points:
                self.pointAdded.emit(None)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_start is None:
            return
        pos = self._widget_to_image(event.position())
        if pos is None:
            return
        self._drag_current = pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_start is None or self._prompt_mode not in (PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE):
            return
        pos = self._widget_to_image(event.position())
        if pos is None:
            self._drag_start = None
            self._drag_current = None
            self.update()
            return
        x1, y1 = self._drag_start
        x2, y2 = pos
        self._drag_start = None
        self._drag_current = None
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.update()
            return
        self.boxCompleted.emit((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        self.update()

    def _target_rect(self) -> QRectF:
        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0:
            return QRectF()
        scale = min(self.width() / img_w, self.height() / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        offset_x = (self.width() - draw_w) / 2.0
        offset_y = (self.height() - draw_h) / 2.0
        return QRectF(offset_x, offset_y, draw_w, draw_h)

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        target = self._target_rect()
        if target.isNull() or not target.contains(point):
            return None
        rel_x = (point.x() - target.x()) / target.width()
        rel_y = (point.y() - target.y()) / target.height()
        img_w, img_h = self._image_size
        return (
            max(0.0, min(rel_x * img_w, img_w - 1)),
            max(0.0, min(rel_y * img_h, img_h - 1)),
        )

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        target = self._target_rect()
        img_w, img_h = self._image_size
        if img_w <= 0 or img_h <= 0:
            return QPointF(0.0, 0.0)
        return QPointF(
            target.x() + (x / img_w) * target.width(),
            target.y() + (y / img_h) * target.height(),
        )

    def _image_box_to_widget_rect(self, box: tuple[float, float, float, float]) -> QRectF:
        x1, y1, x2, y2 = box
        top_left = self._image_to_widget(x1, y1)
        bottom_right = self._image_to_widget(x2, y2)
        return QRectF(top_left, bottom_right).normalized()
