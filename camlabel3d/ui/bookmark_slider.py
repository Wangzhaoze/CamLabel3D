"""Slider widget with lightweight frame bookmark rendering."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


class BookmarkSlider(QSlider):
    """Horizontal slider with red bookmark markers for notable frames."""

    def __init__(self, orientation: Qt.Orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self._bookmarks: list[int] = []
        self._bookmark_color = QColor(255, 82, 82)
        self._bookmark_colors: dict[int, tuple[QColor, ...]] = {}

    def set_bookmarks(self, values: list[int] | tuple[int, ...] | set[int]) -> None:
        normalized = sorted({max(self.minimum(), min(int(value), self.maximum())) for value in values})
        colors = {value: (QColor(self._bookmark_color),) for value in normalized}
        if normalized == self._bookmarks and colors == self._bookmark_colors:
            return
        self._bookmarks = normalized
        self._bookmark_colors = colors
        self.update()

    def set_colored_bookmarks(
        self,
        values: dict[int, QColor | Sequence[QColor]],
    ) -> None:
        normalized_colors: dict[int, tuple[QColor, ...]] = {}
        for raw_value, raw_colors in values.items():
            value = max(self.minimum(), min(int(raw_value), self.maximum()))
            color_list: list[QColor] = []
            if isinstance(raw_colors, QColor):
                raw_items = [raw_colors]
            else:
                raw_items = list(raw_colors)
            for color in raw_items:
                qcolor = QColor(color)
                if not qcolor.isValid():
                    continue
                if any(existing == qcolor for existing in color_list):
                    continue
                color_list.append(qcolor)
            if color_list:
                normalized_colors[value] = tuple(color_list)
        normalized = sorted(normalized_colors.keys())
        if normalized == self._bookmarks and normalized_colors == self._bookmark_colors:
            return
        self._bookmarks = normalized
        self._bookmark_colors = normalized_colors
        self.update()

    def clear_bookmarks(self) -> None:
        if not self._bookmarks:
            return
        self._bookmarks = []
        self._bookmark_colors = {}
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if self.orientation() != Qt.Orientation.Horizontal:
            return
        if not self._bookmarks or self.maximum() <= self.minimum():
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        if groove.width() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        minimum = int(self.minimum())
        maximum = int(self.maximum())
        y_bottom = groove.bottom() + 2
        marker_height = 8
        for bookmark in self._bookmarks:
            position = QStyle.sliderPositionFromValue(minimum, maximum, int(bookmark), groove.width())
            x = groove.left() + position
            colors = self._bookmark_colors.get(bookmark, (self._bookmark_color,))
            if len(colors) == 1:
                offsets = [0]
            else:
                span = min(8, max(2, 2 * (len(colors) - 1)))
                start_offset = -span // 2
                offsets = [start_offset + 2 * index for index in range(len(colors))]
            for offset, color in zip(offsets, colors):
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(color)
                marker_x = x + offset
                painter.drawLine(marker_x, y_bottom - marker_height, marker_x, y_bottom)
                painter.drawEllipse(marker_x - 2, y_bottom - marker_height - 2, 4, 4)
