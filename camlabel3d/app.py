"""Desktop application entrypoint."""

from __future__ import annotations

import sys

from camlabel3d.runtime_config import RuntimeConfig, configure_process_environment


def main() -> int:
    runtime_config = RuntimeConfig.from_env()
    configure_process_environment(runtime_config)

    from PySide6.QtWidgets import QApplication

    from camlabel3d.application import ApplicationContext
    from camlabel3d.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CamLabel3D")
    application_context = ApplicationContext.create(runtime_config)
    window = MainWindow(application_context=application_context)
    window.show()
    return app.exec()
