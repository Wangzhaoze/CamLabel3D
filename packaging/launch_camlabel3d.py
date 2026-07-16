from __future__ import annotations

import sys

from launcher_common import run_python


def main() -> int:
    args = sys.argv[1:]
    if "--self-test" in args:
        return run_python(
            [
                "-c",
                "import camlabel3d.app; from PySide6.QtWidgets import QApplication; print('SELF_TEST_OK')",
            ],
            windowed=False,
        )
    return run_python(["-m", "camlabel3d", *args], windowed=True)


if __name__ == "__main__":
    raise SystemExit(main())
