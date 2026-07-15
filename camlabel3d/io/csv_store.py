"""CSV persistence for the unified CamLabel3D annotation file."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from camlabel3d.core.models import CSV_FIELD_ORDER, DetectionRecord


class CSVStore:
    """Load and save unified CamLabel3D CSV files."""

    def __init__(self, path: str | Path, backup_enabled: bool = True) -> None:
        self.path = Path(path).resolve()
        self.backup_enabled = bool(backup_enabled)

    def load_records(self) -> list[DetectionRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [DetectionRecord.from_row(row) for row in reader]

    def save_records(self, records: Sequence[DetectionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        bak_path = self.path.with_suffix(self.path.suffix + ".bak")
        sorted_records = sorted(records, key=DetectionRecord.sort_key)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_ORDER)
                writer.writeheader()
                for record in sorted_records:
                    writer.writerow(record.to_row())
                handle.flush()
                os.fsync(handle.fileno())

            if self.backup_enabled and self.path.exists():
                shutil.copy2(self.path, bak_path)
            os.replace(tmp_path, self.path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
