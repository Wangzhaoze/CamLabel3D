from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from camlabel3d.core.models import DetectionRecord
from camlabel3d.io.csv_store import CSVStore


def make_record() -> DetectionRecord:
    return DetectionRecord(
        frame_index=3,
        category="car",
        score=0.91,
        score_2d=0.93,
        score_3d=0.74,
        box2d_x1=10.0,
        box2d_y1=20.0,
        box2d_x2=110.0,
        box2d_y2=180.0,
        center_x=1.0,
        center_y=2.0,
        center_z=3.0,
        yaw_deg=12.0,
        pitch_deg=-3.0,
        roll_deg=1.5,
        size_w=1.5,
        size_l=4.2,
        size_h=1.8,
        det_id="det-1",
        source_id="source-1",
        source_type="image_folder",
        source_path="D:/data/images",
        dataset_id="",
        recording_id="",
        timestamp_ms=None,
        image_path="D:/data/images/frame_0003.png",
        prompt_mode="Text",
        prompt_label="",
        prompt_payload_json='{"mode":"Text","texts":["car"]}',
        prompt_group_id="group-1",
        input_fx=1000.0,
        input_fy=1000.0,
        input_cx=640.0,
        input_cy=360.0,
        pred_fx=980.0,
        pred_fy=970.0,
        pred_cx=638.0,
        pred_cy=358.0,
        use_actual_intrinsics=True,
        is_enabled=True,
        track_id="",
        track_status="",
    )


class CSVStoreTests(unittest.TestCase):
    def test_roundtrip_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "results.csv"
            store = CSVStore(csv_path)
            store.save_records([make_record()])

            loaded = store.load_records()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].category, "car")
            self.assertAlmostEqual(loaded[0].score, 0.91, places=6)
            self.assertAlmostEqual(loaded[0].yaw_deg, 12.0, places=6)
            self.assertTrue(loaded[0].is_enabled)
            self.assertEqual(loaded[0].source_id, "")
            self.assertEqual(loaded[0].prompt_payload_json, "")
            self.assertTrue(bool(loaded[0].det_id))
            self.assertTrue(loaded[0].is_visible)

    def test_runtime_visibility_defaults_to_enabled_without_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "results.csv"
            store = CSVStore(csv_path)
            record = make_record()
            record.is_visible = False
            store.save_records([record])

            loaded = store.load_records()

            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0].is_enabled)
            self.assertTrue(loaded[0].is_visible)

    def test_second_save_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "results.csv"
            store = CSVStore(csv_path)
            store.save_records([make_record()])
            updated = make_record()
            updated.center_x = 5.0
            store.save_records([updated])

            backup_path = csv_path.with_suffix(".csv.bak")
            self.assertTrue(backup_path.exists())

    def test_legacy_quaternion_rows_load_and_rewrite_to_compact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "legacy.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame_index,category,score,score_2d,score_3d,box2d_x1,box2d_y1,box2d_x2,box2d_y2,center_x,center_y,center_z,size_w,size_l,size_h,quat_w,quat_x,quat_y,quat_z,is_enabled,track_id,track_status",
                        "3,car,0.91,0.93,0.74,10,20,110,180,1,2,3,1.5,4.2,1.8,1,0,0,0,1,,",
                    ]
                ),
                encoding="utf-8",
            )
            store = CSVStore(csv_path)
            loaded = store.load_records()

            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(loaded[0].yaw_deg, 0.0, places=6)
            self.assertAlmostEqual(loaded[0].pitch_deg, 0.0, places=6)
            self.assertAlmostEqual(loaded[0].roll_deg, 0.0, places=6)

            store.save_records(loaded)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("yaw_deg", header)
            self.assertNotIn("quat_w", header)


if __name__ == "__main__":
    unittest.main()
