from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.postprocess import BBoxFilterRule, FilterConfig, PostprocessSession, WorkflowStage
from camlabel3d.io.csv_store import CSVStore


def make_record(
    frame_index: int = 0,
    category: str = "car",
    score: float = 0.9,
    score_3d: float = 0.8,
    center_z: float = 12.0,
    track_id: str = "",
    track_status: str = "",
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category=category,
        score=score,
        score_2d=score,
        score_3d=score_3d,
        box2d_x1=10.0 + frame_index,
        box2d_y1=20.0,
        box2d_x2=110.0 + frame_index,
        box2d_y2=180.0,
        center_x=0.2 * frame_index,
        center_y=0.0,
        center_z=center_z,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=1.8,
        size_l=4.2,
        size_h=1.6,
        track_id=track_id,
        track_status=track_status,
        is_enabled=True,
    )


class PostprocessSessionTests(unittest.TestCase):
    def test_activate_and_start_postprocessing_switches_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            CSVStore(raw_path, backup_enabled=False).save_records([make_record()])

            session = PostprocessSession()
            stage, active_path, records = session.activate(raw_path)
            self.assertEqual(stage, WorkflowStage.DETECTION)
            self.assertEqual(active_path, raw_path.resolve())
            self.assertEqual(len(records), 1)

            latest_path, latest_records = session.start_postprocessing(records)
            self.assertEqual(session.stage, WorkflowStage.POSTPROCESSING)
            self.assertTrue(latest_path.exists())
            self.assertTrue(str(latest_path).endswith(".latest.camlabel3d.csv"))
            self.assertEqual(len(latest_records), 1)

    def test_activate_selected_latest_csv_enters_postprocessing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            latest_path = Path(tmp_dir) / "demo.latest.camlabel3d.csv"
            CSVStore(raw_path, backup_enabled=False).save_records([make_record(track_id="1")])
            CSVStore(latest_path, backup_enabled=False).save_records([make_record(track_id="7")])

            session = PostprocessSession()
            stage, active_path, records = session.activate(raw_path, selected_csv_path=latest_path)

            self.assertEqual(stage, WorkflowStage.POSTPROCESSING)
            self.assertEqual(active_path, latest_path.resolve())
            self.assertEqual(session.raw_path, raw_path.resolve())
            self.assertEqual(session.latest_path, latest_path.resolve())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].track_id, "7")

    def test_blank_raw_csv_does_not_enable_postprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "blank.camlabel3d.csv"
            CSVStore(raw_path, backup_enabled=False).save_records([])

            session = PostprocessSession()
            stage, active_path, records = session.activate(raw_path)

            self.assertEqual(stage, WorkflowStage.DETECTION)
            self.assertEqual(active_path, raw_path.resolve())
            self.assertEqual(records, [])
            self.assertFalse(session.can_start_postprocessing(records))

    def test_configure_source_does_not_auto_load_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            latest_path = Path(tmp_dir) / "demo.latest.camlabel3d.csv"
            CSVStore(raw_path, backup_enabled=False).save_records([make_record(track_id="1")])
            CSVStore(latest_path, backup_enabled=False).save_records([make_record(track_id="7")])

            session = PostprocessSession()
            stage, active_path = session.configure_source(raw_path)

            self.assertEqual(stage, WorkflowStage.DETECTION)
            self.assertEqual(active_path, raw_path.resolve())
            self.assertEqual(session.raw_path, raw_path.resolve())
            self.assertEqual(session.latest_path, latest_path.resolve())
            self.assertEqual(session.active_path, raw_path.resolve())

    def test_reset_to_raw_restores_original_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "demo.camlabel3d.csv"
            CSVStore(raw_path, backup_enabled=False).save_records([make_record(track_id="")])

            session = PostprocessSession()
            _, _, records = session.activate(raw_path)
            _, records = session.start_postprocessing(records)
            records[0].track_id = "5"
            session.save_records(records)

            _, restored = session.reset_to_raw()
            self.assertEqual(restored[0].track_id, "")

    def test_filter_merge_lock_and_unlock_update_records(self) -> None:
        records = [
            make_record(frame_index=0, score=0.9, track_id="1", track_status="auto"),
            make_record(frame_index=2, score=0.4, track_id="2", track_status="auto"),
            make_record(frame_index=4, score=0.7, track_id="2", track_status="auto"),
        ]
        session = PostprocessSession()

        disabled = session.apply_filter(records, FilterConfig(min_score=0.5))
        self.assertEqual(disabled, 1)
        self.assertFalse(records[1].is_enabled)
        self.assertFalse(records[1].is_visible)

        locked = session.lock_track(records, "1")
        self.assertEqual(locked, 1)
        self.assertEqual(records[0].track_status, "locked")

        unlocked = session.unlock_track(records, "1")
        self.assertEqual(unlocked, 1)
        self.assertEqual(records[0].track_status, "auto")

        merged = session.merge_tracks(records, "2", "1")
        self.assertEqual(merged, 1)
        self.assertEqual(records[2].track_id, "1")
        self.assertEqual(records[2].track_status, "manual")

        summaries = session.build_track_summaries(records)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].track_id, "1")
        self.assertEqual(summaries[0].enabled_count, 2)

    def test_delete_track_disables_and_hides_rows(self) -> None:
        records = [
            make_record(frame_index=0, track_id="9", track_status="auto"),
            make_record(frame_index=1, track_id="9", track_status="auto"),
        ]
        session = PostprocessSession()

        disabled = session.delete_track(records, "9")

        self.assertEqual(disabled, 2)
        self.assertTrue(all(not record.is_enabled for record in records))
        self.assertTrue(all(not record.is_visible for record in records))

    def test_generic_filter_rules_use_and_semantics_and_only_disable_rows(self) -> None:
        records = [
            make_record(frame_index=0, score=0.9, center_z=12.0),
            make_record(frame_index=1, score=0.4, center_z=10.0),
            make_record(frame_index=2, score=0.8, center_z=30.0),
        ]
        session = PostprocessSession()

        disabled = session.apply_filter(
            records,
            FilterConfig(
                rules=(
                    BBoxFilterRule("score", min_enabled=True, min_value=0.5),
                    BBoxFilterRule("center_z", max_enabled=True, max_value=20.0),
                )
            ),
        )

        self.assertEqual(disabled, 2)
        self.assertTrue(records[0].is_enabled)
        self.assertFalse(records[1].is_enabled)
        self.assertFalse(records[2].is_enabled)
        self.assertFalse(records[1].is_visible)
        self.assertFalse(records[2].is_visible)
        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
