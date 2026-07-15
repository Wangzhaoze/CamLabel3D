from __future__ import annotations

import unittest

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.processing import (
    OperationScope,
    OutlierScope,
    ProcessingContext,
    ProcessingEngine,
    TrackBatchEditRequest,
    TrackBatchOperationKind,
    apply_track_batch_edit,
    build_default_bulk_operation_registry,
    build_default_outlier_registry,
)


def make_record(
    frame_index: int,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 12.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    size_w: float = 2.0,
    size_l: float = 4.5,
    size_h: float = 1.6,
    score: float = 0.9,
    score_3d: float = 0.9,
    track_id: str = "1",
    category: str = "car",
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category=category,
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
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        size_w=size_w,
        size_l=size_l,
        size_h=size_h,
        is_enabled=True,
        track_id=track_id,
        track_status="auto" if track_id else "",
    )


class ProcessingEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ProcessingEngine()

    def test_default_registries_include_builtin_rules_and_operations(self) -> None:
        outlier_ids = [rule.rule_id for rule in build_default_outlier_registry().all()]
        operation_ids = [operation.operation_id for operation in build_default_bulk_operation_registry().all()]
        self.assertEqual(
            outlier_ids,
            ["yaw_spike", "pitch_spike", "roll_spike", "size_spike", "center_spike"],
        )
        self.assertEqual(operation_ids, ["smooth_angles", "fix_track_size", "smooth_track_centers"])

    def test_single_record_can_hit_multiple_outlier_rules(self) -> None:
        records = [
            make_record(0, center_x=0.0, center_z=10.0, yaw_deg=10.0, size_w=2.0, size_l=4.5),
            make_record(1, center_x=4.0, center_z=15.0, yaw_deg=170.0, size_w=5.0, size_l=8.0, score_3d=0.2),
            make_record(2, center_x=0.4, center_z=10.4, yaw_deg=12.0, size_w=2.1, size_l=4.6),
        ]
        context = ProcessingContext(records=records)

        hits = self.engine.analyze_outliers(
            records=records,
            scope=OutlierScope.GLOBAL,
            enabled_rule_ids=["yaw_spike", "size_spike", "center_spike"],
            params_by_rule={},
            context=context,
        )

        rule_ids = {hit.rule_id for hit in hits if hit.frame_index == 1}
        self.assertEqual(rule_ids, {"yaw_spike", "size_spike", "center_spike"})

    def test_pitch_and_roll_spikes_are_reported_independently(self) -> None:
        pitch_records = [
            make_record(0, pitch_deg=2.0),
            make_record(1, pitch_deg=92.0),
            make_record(2, pitch_deg=4.0),
        ]
        roll_records = [
            make_record(0, roll_deg=-3.0, track_id="2"),
            make_record(1, roll_deg=105.0, track_id="2"),
            make_record(2, roll_deg=-5.0, track_id="2"),
        ]
        records = pitch_records + roll_records
        context = ProcessingContext(records=records)

        hits = self.engine.analyze_outliers(
            records=records,
            scope=OutlierScope.GLOBAL,
            enabled_rule_ids=["pitch_spike", "roll_spike"],
            params_by_rule={},
            context=context,
        )

        hit_ids = {(hit.rule_id, hit.track_id, hit.frame_index) for hit in hits}
        self.assertIn(("pitch_spike", "1", 1), hit_ids)
        self.assertIn(("roll_spike", "2", 1), hit_ids)

    def test_scope_filtering_limits_visible_hits(self) -> None:
        records = [
            make_record(0, center_x=0.0, center_z=10.0, yaw_deg=10.0, track_id="1"),
            make_record(1, center_x=4.0, center_z=15.0, yaw_deg=170.0, track_id="1"),
            make_record(2, center_x=0.4, center_z=10.4, yaw_deg=12.0, track_id="1"),
            make_record(1, center_x=2.0, center_z=20.0, yaw_deg=0.0, track_id="2"),
        ]
        context = ProcessingContext(records=records, current_frame_index=1, selected_track_id="1")
        hits = self.engine.analyze_outliers(
            records=records,
            scope=OutlierScope.GLOBAL,
            enabled_rule_ids=["yaw_spike"],
            params_by_rule={},
            context=context,
        )

        frame_hits = self.engine.filter_hits_for_scope(hits, OutlierScope.CURRENT_FRAME, context)
        track_hits = self.engine.filter_hits_for_scope(hits, OutlierScope.SELECTED_TRACK, context)

        self.assertTrue(all(hit.frame_index == 1 for hit in frame_hits))
        self.assertTrue(all(hit.track_id == "1" for hit in track_hits))

    def test_fix_track_size_can_target_current_frame_scope(self) -> None:
        records = [
            make_record(0, size_w=2.0, size_l=4.0, size_h=1.5, score_3d=0.9),
            make_record(1, size_w=5.0, size_l=8.0, size_h=2.5, score_3d=0.2),
            make_record(2, size_w=2.1, size_l=4.1, size_h=1.55, score_3d=0.8),
        ]
        context = ProcessingContext(records=records, current_frame_index=1)

        result = self.engine.apply_operation(
            operation_id="fix_track_size",
            records=records,
            scope=OperationScope.CURRENT_FRAME,
            params={},
            context=context,
        )

        self.assertEqual(result.updated_count, 1)
        self.assertAlmostEqual(records[1].size_w, 2.1, places=6)
        self.assertAlmostEqual(records[1].size_l, 4.1, places=6)
        self.assertAlmostEqual(records[1].size_h, 1.55, places=6)
        self.assertAlmostEqual(records[0].size_w, 2.0, places=6)

    def test_smooth_angles_changes_angles_only(self) -> None:
        records = [
            make_record(0, yaw_deg=0.0, center_x=0.0),
            make_record(1, yaw_deg=90.0, center_x=1.0),
            make_record(2, yaw_deg=0.0, center_x=2.0),
        ]
        original_centers = [(record.center_x, record.center_y, record.center_z) for record in records]
        context = ProcessingContext(records=records, selected_track_id="1")

        self.engine.apply_operation(
            operation_id="smooth_angles",
            records=records,
            scope=OperationScope.SELECTED_TRACK,
            params={},
            context=context,
        )

        self.assertNotAlmostEqual(records[1].yaw_deg, 90.0, places=4)
        self.assertEqual(original_centers, [(record.center_x, record.center_y, record.center_z) for record in records])

    def test_smooth_track_centers_changes_centers_only(self) -> None:
        records = [
            make_record(0, center_x=0.0, center_z=10.0),
            make_record(1, center_x=10.0, center_z=25.0),
            make_record(2, center_x=0.5, center_z=10.5),
        ]
        original_sizes = [(record.size_w, record.size_l, record.size_h) for record in records]
        context = ProcessingContext(records=records, selected_track_id="1")

        self.engine.apply_operation(
            operation_id="smooth_track_centers",
            records=records,
            scope=OperationScope.SELECTED_TRACK,
            params={},
            context=context,
        )

        self.assertNotAlmostEqual(records[1].center_x, 10.0, places=4)
        self.assertEqual(original_sizes, [(record.size_w, record.size_l, record.size_h) for record in records])

    def test_track_batch_add_updates_only_selected_track_range(self) -> None:
        records = [
            make_record(0, center_x=1.0, track_id="1"),
            make_record(1, center_x=2.0, track_id="1"),
            make_record(2, center_x=3.0, track_id="1"),
            make_record(1, center_x=9.0, track_id="2"),
        ]
        context = ProcessingContext(records=records)

        result = apply_track_batch_edit(
            records,
            TrackBatchEditRequest(
                track_id="1",
                field_name="center_x",
                operation=TrackBatchOperationKind.ADD,
                frame_start=1,
                frame_end=2,
                operand=5.0,
            ),
            context,
        )

        self.assertEqual(result.updated_count, 2)
        self.assertAlmostEqual(records[0].center_x, 1.0, places=6)
        self.assertAlmostEqual(records[1].center_x, 7.0, places=6)
        self.assertAlmostEqual(records[2].center_x, 8.0, places=6)
        self.assertAlmostEqual(records[3].center_x, 9.0, places=6)

    def test_track_batch_smooth_yaw_uses_angle_unwrap(self) -> None:
        records = [
            make_record(0, yaw_deg=170.0, track_id="1"),
            make_record(1, yaw_deg=-170.0, track_id="1"),
            make_record(2, yaw_deg=170.0, track_id="1"),
        ]
        context = ProcessingContext(records=records)

        result = apply_track_batch_edit(
            records,
            TrackBatchEditRequest(
                track_id="1",
                field_name="yaw_deg",
                operation=TrackBatchOperationKind.SMOOTH,
                frame_start=0,
                frame_end=2,
                smooth_window=3,
            ),
            context,
        )

        self.assertEqual(result.updated_count, 3)
        self.assertLess(abs(abs(records[1].yaw_deg) - 180.0), 15.0)

    def test_track_batch_divide_by_zero_is_rejected(self) -> None:
        records = [make_record(0, center_z=10.0, track_id="1")]
        context = ProcessingContext(records=records)

        with self.assertRaisesRegex(ValueError, "Division by zero"):
            apply_track_batch_edit(
                records,
                TrackBatchEditRequest(
                    track_id="1",
                    field_name="center_z",
                    operation=TrackBatchOperationKind.DIVIDE,
                    frame_start=0,
                    frame_end=0,
                    operand=0.0,
                ),
                context,
            )


if __name__ == "__main__":
    unittest.main()
