"""Interactive bird's-eye-view canvas for trajectory calibration."""

from __future__ import annotations

from math import ceil, floor, hypot

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from camlabel3d.core.bev import BEVScene, BEVViewport


class BEVCanvas(QWidget):
    """Render a stable BEV scene with wheel-based FOV zoom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self._scene: BEVScene | None = None
        self._viewport: BEVViewport | None = None

    def has_scene(self) -> bool:
        return self._scene is not None and self._viewport is not None

    def clear(self) -> None:
        self._scene = None
        self._viewport = None
        self.update()

    def set_scene(self, scene: BEVScene, *, reset_view: bool = False) -> None:
        self._scene = scene
        if reset_view or self._viewport is None:
            self._viewport = scene.viewport_seed
        self.update()

    def reset_view(self) -> None:
        if self._scene is None:
            return
        self._viewport = self._scene.viewport_seed
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._scene is None or self._viewport is None:
            event.ignore()
            return
        delta = int(event.angleDelta().y())
        if delta == 0:
            event.ignore()
            return

        steps = max(1.0, abs(delta) / 120.0)
        factor = (0.86**steps) if delta > 0 else ((1.0 / 0.86) ** steps)
        seed = self._scene.viewport_seed
        half_x = max(1.0, self._viewport.width_m * 0.5 * factor)
        z_max = max(2.0, self._viewport.z_max * factor)
        min_half_x = max(seed.grid_spacing_m * 1.5, seed.width_m * 0.12)
        max_half_x = max(seed.width_m * 4.0, min_half_x + seed.grid_spacing_m)
        min_z_max = max(seed.grid_spacing_m * 3.0, seed.z_max * 0.2)
        max_z_max = max(seed.z_max * 4.0, min_z_max + seed.grid_spacing_m)
        clamped_half_x = min(max(half_x, min_half_x), max_half_x)
        clamped_z_max = min(max(z_max, min_z_max), max_z_max)
        grid_spacing = _select_grid_spacing(max(clamped_half_x * 2.0, clamped_z_max))
        x_range = ceil(clamped_half_x / grid_spacing) * grid_spacing
        forward_max = ceil(clamped_z_max / grid_spacing) * grid_spacing
        self._viewport = BEVViewport(
            x_min=float(-x_range),
            x_max=float(x_range),
            z_min=0.0,
            z_max=float(forward_max),
            grid_spacing_m=float(grid_spacing),
        )
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#F5F5F0"))

        viewport = self._viewport
        scene = self._scene
        if viewport is None or scene is None:
            painter.setPen(QColor(86, 98, 110))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Switch to BEV after loading a source to inspect trajectories.",
            )
            return

        plot = self._plot_rect()
        painter.fillRect(plot, QColor("#FBFBF8"))
        self._draw_grid(painter, plot, viewport)
        self._draw_axes(painter, plot, viewport)
        self._draw_boxes(painter, plot, viewport, scene)
        self._draw_trajectory(painter, plot, viewport, scene)
        self._draw_current_pose(painter, plot, viewport, scene)

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(18.0, 14.0, -18.0, -22.0)

    def _draw_grid(self, painter: QPainter, plot: QRectF, viewport: BEVViewport) -> None:
        grid_pen = QPen(QColor("#B9C0C6"))
        grid_pen.setWidthF(1.0)
        grid_pen.setStyle(Qt.PenStyle.DashLine)
        grid_pen.setDashPattern([4.0, 4.0])
        painter.setPen(grid_pen)

        for value in _tick_values(viewport.x_min, viewport.x_max, viewport.grid_spacing_m):
            x_pos = self._world_to_widget(plot, viewport, value, viewport.z_min).x()
            painter.drawLine(QPointF(x_pos, plot.top()), QPointF(x_pos, plot.bottom()))
        for value in _tick_values(viewport.z_min, viewport.z_max, viewport.grid_spacing_m):
            y_pos = self._world_to_widget(plot, viewport, viewport.x_min, value).y()
            painter.drawLine(QPointF(plot.left(), y_pos), QPointF(plot.right(), y_pos))

    def _draw_axes(self, painter: QPainter, plot: QRectF, viewport: BEVViewport) -> None:
        axis_pen = QPen(QColor("#4B5966"))
        axis_pen.setWidthF(1.4)
        tick_pen = QPen(QColor("#4B5966"))
        tick_pen.setWidthF(1.0)
        text_pen = QColor("#37424D")
        painter.setPen(axis_pen)

        origin = self._world_to_widget(plot, viewport, 0.0, 0.0)
        painter.drawLine(QPointF(plot.left(), origin.y()), QPointF(plot.right(), origin.y()))
        painter.drawLine(QPointF(origin.x(), plot.top()), QPointF(origin.x(), plot.bottom()))

        painter.setPen(tick_pen)
        tick_half = 5.0
        for value in _tick_values(viewport.x_min, viewport.x_max, viewport.grid_spacing_m):
            point = self._world_to_widget(plot, viewport, value, 0.0)
            painter.drawLine(
                QPointF(point.x(), origin.y() - tick_half),
                QPointF(point.x(), origin.y() + tick_half),
            )
            if abs(value) < 1e-6:
                continue
            self._draw_text(
                painter,
                _format_tick_label(value),
                QPointF(point.x(), origin.y() + 8.0),
                text_pen,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            )

        for value in _tick_values(viewport.z_min, viewport.z_max, viewport.grid_spacing_m):
            point = self._world_to_widget(plot, viewport, 0.0, value)
            painter.drawLine(
                QPointF(origin.x() - tick_half, point.y()),
                QPointF(origin.x() + tick_half, point.y()),
            )
            self._draw_text(
                painter,
                _format_tick_label(value),
                QPointF(origin.x() - 8.0, point.y()),
                text_pen,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

    def _draw_boxes(
        self,
        painter: QPainter,
        plot: QRectF,
        viewport: BEVViewport,
        scene: BEVScene,
    ) -> None:
        current_det_id = scene.current_pose.det_id if scene.current_pose is not None else ""
        for box in scene.frame_boxes:
            is_current_focus = bool(current_det_id and box.det_id == current_det_id)
            edge_color = QColor("#4D8AC6") if box.is_focus else QColor("#8795A1")
            face_color = QColor("#D8E7F6") if box.is_focus else QColor("#E7EBEF")
            face_color.setAlphaF(0.9 if is_current_focus else (0.55 if box.is_focus else 0.35))
            box_pen = QPen(edge_color)
            box_pen.setWidthF(2.4 if is_current_focus else (1.8 if box.is_focus else 1.2))
            painter.setPen(box_pen)
            painter.setBrush(face_color)
            polygon = QPolygonF(
                [self._world_to_widget(plot, viewport, x_value, z_value) for x_value, z_value in box.corners_xz]
            )
            painter.drawPolygon(polygon)

    def _draw_trajectory(
        self,
        painter: QPainter,
        plot: QRectF,
        viewport: BEVViewport,
        scene: BEVScene,
    ) -> None:
        if not scene.trajectory_samples:
            return
        history = [
            self._world_to_widget(plot, viewport, sample.x, sample.z)
            for sample in scene.trajectory_samples
        ]
        if len(history) >= 2:
            history_pen = QPen(QColor("#147AD6"))
            history_pen.setWidthF(2.4)
            history_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(history_pen)
            painter.drawPolyline(QPolygonF(history))

        point_color = QColor("#5AA7E8")
        point_color.setAlphaF(0.7)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(point_color)
        for sample in scene.trajectory_samples:
            painter.drawEllipse(self._world_to_widget(plot, viewport, sample.x, sample.z), 3.2, 3.2)

    def _draw_current_pose(
        self,
        painter: QPainter,
        plot: QRectF,
        viewport: BEVViewport,
        scene: BEVScene,
    ) -> None:
        if scene.current_pose is None:
            return
        current = scene.current_pose
        center = self._world_to_widget(plot, viewport, current.center_x, current.center_z)
        marker_pen = QPen(QColor("#FFFFFF"))
        marker_pen.setWidthF(1.0)
        painter.setPen(marker_pen)
        painter.setBrush(QColor("#D64541"))
        painter.drawEllipse(center, 4.6, 4.6)

        start = self._world_to_widget(plot, viewport, current.yaw_arrow.start_x, current.yaw_arrow.start_z)
        end = self._world_to_widget(plot, viewport, current.yaw_arrow.end_x, current.yaw_arrow.end_z)
        arrow_pen = QPen(QColor("#D64541"))
        arrow_pen.setWidthF(2.2)
        arrow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arrow_pen)
        painter.setBrush(QColor("#D64541"))
        painter.drawLine(start, end)

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = hypot(dx, dy)
        if length <= 1e-6:
            return
        unit_x = dx / length
        unit_y = dy / length
        perp_x = -unit_y
        perp_y = unit_x
        head_length = min(16.0, max(10.0, length * 0.22))
        head_width = head_length * 0.62
        base = QPointF(end.x() - unit_x * head_length, end.y() - unit_y * head_length)
        left = QPointF(base.x() + perp_x * head_width * 0.5, base.y() + perp_y * head_width * 0.5)
        right = QPointF(base.x() - perp_x * head_width * 0.5, base.y() - perp_y * head_width * 0.5)
        painter.drawPolygon(QPolygonF([end, left, right]))

    @staticmethod
    def _draw_text(
        painter: QPainter,
        text: str,
        anchor: QPointF,
        color: QColor,
        *,
        align: Qt.AlignmentFlag,
    ) -> None:
        painter.setPen(color)
        bounds = painter.fontMetrics().boundingRect(text)
        x_pos = anchor.x()
        y_pos = anchor.y()
        if align & Qt.AlignmentFlag.AlignHCenter:
            x_pos -= bounds.width() * 0.5
        elif align & Qt.AlignmentFlag.AlignRight:
            x_pos -= bounds.width()
        if align & Qt.AlignmentFlag.AlignVCenter:
            y_pos += bounds.height() * 0.3
        elif align & Qt.AlignmentFlag.AlignTop:
            y_pos += bounds.height()
        painter.drawText(QPointF(x_pos, y_pos), text)

    @staticmethod
    def _world_to_widget(plot: QRectF, viewport: BEVViewport, x_value: float, z_value: float) -> QPointF:
        x_ratio = (float(x_value) - viewport.x_min) / max(1e-6, viewport.width_m)
        z_ratio = (float(z_value) - viewport.z_min) / max(1e-6, viewport.height_m)
        return QPointF(
            plot.left() + x_ratio * plot.width(),
            plot.bottom() - z_ratio * plot.height(),
        )


def _tick_values(minimum: float, maximum: float, spacing: float) -> list[float]:
    if spacing <= 0.0:
        return [float(minimum), float(maximum)]
    start = floor(minimum / spacing)
    end = ceil(maximum / spacing)
    values: list[float] = []
    for index in range(start, end + 1):
        value = index * spacing
        if minimum - 1e-6 <= value <= maximum + 1e-6:
            values.append(float(value))
    return values


def _format_tick_label(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:.1f}"


def _select_grid_spacing(span_m: float) -> float:
    if span_m <= 30.0:
        return 2.0
    if span_m <= 80.0:
        return 5.0
    return 10.0
