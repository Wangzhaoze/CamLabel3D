"""Desktop application entrypoint."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from camlabel3d.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CamLabel3D")
    window = MainWindow()
    window.show()
    return app.exec()
