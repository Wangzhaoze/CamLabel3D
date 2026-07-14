from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from camlabel3d.core.models import SourceMode
from camlabel3d.core.source_config import DatasetConfigStore


class DatasetConfigStoreTests(unittest.TestCase):
    def test_ensure_exists_writes_default_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "dataset_sources.json"
            store = DatasetConfigStore(config_path)
            store.ensure_exists()

            self.assertTrue(config_path.exists())
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIn("result_root", payload)
            self.assertEqual(payload["datasets"][0]["id"], "RAMPCNN")

    def test_discover_recordings_only_keeps_matching_media_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "rec_02" / "images_0").mkdir(parents=True)
            (root / "rec_10" / "images_0").mkdir(parents=True)
            (root / "ANNO_RA").mkdir(parents=True)

            config_path = root / "dataset_sources.json"
            config_path.write_text(
                json.dumps(
                    {
                        "result_root": str(root / "results"),
                        "datasets": [
                            {
                                "id": "RAMPCNN",
                                "display_name": "RAMPCNN",
                                "root_path": str(root),
                                "media_subdir": "images_0",
                                "default_intrinsics": {
                                    "fx": 1.0,
                                    "fy": 2.0,
                                    "cx": 3.0,
                                    "cy": 4.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = DatasetConfigStore(config_path)
            config = store.load()
            recordings = store.discover_recordings(config, "RAMPCNN")

            self.assertEqual(recordings, ["rec_02", "rec_10"])

    def test_output_paths_follow_daily_naming_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "dataset_sources.json"
            config_path.write_text(
                json.dumps(
                    {
                        "result_root": str(root / "annotation_results"),
                        "datasets": [
                            {
                                "id": "RAMPCNN",
                                "display_name": "RAMPCNN",
                                "root_path": "D:/Datasets/RAMPCNN",
                                "media_subdir": "images_0",
                                "default_intrinsics": {
                                    "fx": 1.0,
                                    "fy": 2.0,
                                    "cx": 3.0,
                                    "cy": 4.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = DatasetConfigStore(config_path)
            config = store.load()
            day = date(2026, 7, 13)

            dataset_csv = store.resolve_output_csv_path(
                config=config,
                source_mode=SourceMode.DATASET,
                dataset_id="RAMPCNN",
                recording_id="2019_04_09_bms1000",
                on_date=day,
            )
            video_csv = store.resolve_output_csv_path(
                config=config,
                source_mode=SourceMode.VIDEO,
                manual_source_path=root / "demo_video.mp4",
                on_date=day,
            )
            folder_csv = store.resolve_output_csv_path(
                config=config,
                source_mode=SourceMode.IMAGE_FOLDER,
                manual_source_path=root / "images_0",
                on_date=day,
            )

            expected_dir = root / "annotation_results" / "2026-07-13"
            self.assertEqual(dataset_csv, expected_dir / "RAMPCNN__2019_04_09_bms1000.camlabel3d.csv")
            self.assertEqual(video_csv, expected_dir / "video__demo_video.camlabel3d.csv")
            self.assertEqual(folder_csv, expected_dir / "folder__images_0.camlabel3d.csv")
            self.assertTrue(expected_dir.exists())

    def test_seed_output_session_files_copies_most_recent_matching_session_into_today(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "dataset_sources.json"
            config_path.write_text(
                json.dumps(
                    {
                        "result_root": str(root / "annotation_results"),
                        "datasets": [
                            {
                                "id": "RAMPCNN",
                                "display_name": "RAMPCNN",
                                "root_path": str(root / "dataset_root"),
                                "media_subdir": "images_0",
                                "default_intrinsics": {
                                    "fx": 1.0,
                                    "fy": 2.0,
                                    "cx": 3.0,
                                    "cy": 4.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = DatasetConfigStore(config_path)
            config = store.load()
            prior_day = date(2026, 7, 13)
            today = date(2026, 7, 14)

            prior_raw = store.resolve_output_csv_path(
                config=config,
                source_mode=SourceMode.DATASET,
                dataset_id="RAMPCNN",
                recording_id="2019_04_09_bms1000",
                on_date=prior_day,
            )
            prior_latest = prior_raw.with_name("RAMPCNN__2019_04_09_bms1000.latest.camlabel3d.csv")
            prior_raw.write_text("raw-records\n", encoding="utf-8")
            prior_latest.write_text("latest-records\n", encoding="utf-8")

            today_raw = store.seed_output_session_files(
                config=config,
                source_mode=SourceMode.DATASET,
                dataset_id="RAMPCNN",
                recording_id="2019_04_09_bms1000",
                on_date=today,
            )
            today_latest = today_raw.with_name("RAMPCNN__2019_04_09_bms1000.latest.camlabel3d.csv")

            self.assertEqual(today_raw.read_text(encoding="utf-8"), "raw-records\n")
            self.assertEqual(today_latest.read_text(encoding="utf-8"), "latest-records\n")
            self.assertIn("2026-07-14", str(today_raw))


if __name__ == "__main__":
    unittest.main()
