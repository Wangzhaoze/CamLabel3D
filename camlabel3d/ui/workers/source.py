"""Background source discovery/opening worker."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Signal


class SourceLoadWorker(QThread):
    """Run blocking media/catalog I/O before handing ownership to the UI."""

    sourceLoaded = Signal(object)
    sourceFailed = Signal(str)

    def __init__(self, loader: Callable[[Callable[[], bool]], object], parent=None) -> None:
        super().__init__(parent)
        self.loader = loader

    def cancel(self) -> None:
        """Request cooperative cancellation of source discovery."""

        self.requestInterruption()

    def run(self) -> None:  # noqa: D401, N802 - Qt naming
        try:
            payload = self.loader(self.isInterruptionRequested)
            if self.isInterruptionRequested():
                provider = getattr(payload, "provider", None)
                if provider is not None:
                    try:
                        provider.close()
                    except Exception:
                        pass
                return
            self.sourceLoaded.emit(payload)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.sourceFailed.emit(str(exc))
