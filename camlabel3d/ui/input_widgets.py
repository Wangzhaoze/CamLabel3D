"""UI input widgets that avoid accidental mouse-wheel value changes."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelSpinBox(QSpinBox):
    """Ignore mouse-wheel input so scrolling does not change numeric values."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Ignore mouse-wheel input so scrolling does not change numeric values."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.ignore()


class MenuWheelComboBox(QComboBox):
    """Allow mouse-wheel scrolling only while the combo popup list is open."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.view().isVisible():
            super().wheelEvent(event)
            return
        event.ignore()
