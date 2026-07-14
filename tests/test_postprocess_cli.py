from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from camlabel3d.core.models import DetectionRecord
from camlabel3d.io.csv_store import CSVStore
from camlabel3d.postprocess_cli import main


def make_record(
    frame_index: int,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 12.0,
    yaw_deg: float = 0.0,
    size_w: float = 2.0,
    size_l: float = 4.5,
    size_h: float = 1.6,
    score: float = 0.9,
    score_3d: float = 0.9,
    track_id: str = "1",
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category="car",
        score=score,
        score_2d=score,
        score_3d=score_3d,
        box2d_x1=600.0 + frame_index,
        box2d_y1=300.0,
        box2d_x2=680.0 + frame_index,
        box2d_y2=420.0,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        yaw_deg=yaw_deg,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=size_w,
        size_l=size_l,
        size_h=size_h,
        is_enabled=True,
        track_id=track_id,
        track_status="auto" if track_id else "",
    )


class PostprocessCliTests(unittest.TestCase):
    def test_analyze_writes_outlier_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            report_path = Path(tmp_dir) / "demo.outliers.json"
            CSVStore(csv_path, backup_enabled=False).save_records(
                [
                    make_record(0, center_x=0.0, center_z=10.0, yaw_deg=10.0),
                    make_record(1, center_x=4.0, center_z=15.0, yaw_deg=170.0),
                    make_record(2, center_x=0.3, center_z=10.3, yaw_deg=12.0),
                ]
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "analyze",
                        "--csv",
                        str(csv_path),
                        "--rule",
                        "yaw_spike",
                        "--output",
                        str(report_path),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("Outlier analysis complete", stdout.getvalue())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["hit_count"], 1)
            self.assertEqual(payload["hits"][0]["frame_index"], 1)
            self.assertEqual(payload["hits"][0]["rule_id"], "yaw_spike")

    def test_apply_operation_writes_new_csv_without_overwriting_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            output_path = Path(tmp_dir) / "demo.post.camlabel3d.csv"
            CSVStore(csv_path, backup_enabled=False).save_records(
                [
                    make_record(0, size_w=2.0, size_l=4.0, size_h=1.5, score_3d=0.9),
                    make_record(1, size_w=5.0, size_l=8.0, size_h=2.5, score_3d=0.2),
                    make_record(2, size_w=2.1, size_l=4.1, size_h=1.55, score_3d=0.8),
                ]
            )
            original = CSVStore(csv_path).load_records()

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "apply",
                        "--csv",
                        str(csv_path),
                        "--scope",
                        "current_frame",
                        "--frame-index",
                        "1",
                        "--operation",
                        "fix_track_size",
                        "--output",
                        str(output_path),
                        "--fx",
                        "800",
                        "--fy",
                        "800",
                        "--cx",
                        "640",
                        "--cy",
                        "360",
                        "--image-width",
                        "1280",
                        "--image-height",
                        "720",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertIn("Postprocessing complete", stdout.getvalue())
            updated = CSVStore(output_path, backup_enabled=False).load_records()
            self.assertAlmostEqual(updated[1].size_w, 2.1, places=6)
            self.assertAlmostEqual(updated[1].size_l, 4.1, places=6)
            self.assertAlmostEqual(updated[1].size_h, 1.55, places=6)
            reloaded_original = CSVStore(csv_path, backup_enabled=False).load_records()
            self.assertAlmostEqual(reloaded_original[1].size_w, original[1].size_w, places=6)
            self.assertFalse(output_path.with_suffix(output_path.suffix + ".bak").exists())


if __name__ == "__main__":
    unittest.main()
