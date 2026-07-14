"""CSV persistence for the unified CamLabel3D annotation file."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

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

    def save_records(self, records: list[DetectionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        bak_path = self.path.with_suffix(self.path.suffix + ".bak")
        sorted_records = sorted(records, key=DetectionRecord.sort_key)

        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_ORDER)
            writer.writeheader()
            for record in sorted_records:
                writer.writerow(record.to_row())

        if self.backup_enabled and self.path.exists():
            shutil.copy2(self.path, bak_path)
        os.replace(tmp_path, self.path)
