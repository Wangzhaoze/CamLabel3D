from __future__ import annotations

import sys

from launcher_common import run_python


def main() -> int:
    args = sys.argv[1:]
    if "--self-test" in args:
        return run_python(
            ["-m", "camlabel3d.postprocess_cli", "--help"],
            windowed=False,
        )
    return run_python(["-m", "camlabel3d.postprocess_cli", *args], windowed=False)


if __name__ == "__main__":
    raise SystemExit(main())
