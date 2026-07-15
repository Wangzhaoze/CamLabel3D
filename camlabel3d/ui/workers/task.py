"""Generic one-shot worker for bounded blocking application use cases."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    resultReady = Signal(object)
    taskFailed = Signal(str)

    def __init__(self, function: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self.function = function

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            self.resultReady.emit(self.function())
        except Exception as exc:
            self.taskFailed.emit(str(exc))
