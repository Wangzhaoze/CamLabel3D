from __future__ import annotations

import unittest

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.tracking import TrackingConfig, TrackingEngine


def make_record(
    frame_index: int,
    center_x: float,
    center_z: float,
    box2d_x1: float,
    box2d_y1: float,
    box2d_x2: float,
    box2d_y2: float,
    category: str = "car",
    track_id: str = "",
    track_status: str = "",
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category=category,
        score=0.9,
        score_2d=0.9,
        score_3d=0.9,
        box2d_x1=box2d_x1,
        box2d_y1=box2d_y1,
        box2d_x2=box2d_x2,
        box2d_y2=box2d_y2,
        center_x=center_x,
        center_y=0.0,
        center_z=center_z,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.5,
        size_h=1.6,
        is_enabled=True,
        track_id=track_id,
        track_status=track_status,
    )


class TrackingEngineTests(unittest.TestCase):
    def test_tracking_keeps_consistent_id_for_same_target(self) -> None:
        records = [
            make_record(0, 0.0, 15.0, 600.0, 320.0, 680.0, 420.0, track_id="1"),
            make_record(1, 0.2, 15.3, 604.0, 320.0, 684.0, 420.0, track_id="2"),
            make_record(2, 0.4, 15.6, 608.0, 320.0, 688.0, 420.0, track_id="3"),
        ]

        result = TrackingEngine().run(records, TrackingConfig(max_gap_frames=2, max_center_distance_m=3.0))

        track_ids = [record.track_id for record in result]
        self.assertEqual(track_ids, ["1", "1", "1"])
        self.assertTrue(all(record.track_status == "auto" for record in result))

    def test_tracking_respects_locked_track_ids(self) -> None:
        records = [
            make_record(0, 0.0, 12.0, 610.0, 320.0, 690.0, 420.0, track_id="7", track_status="locked"),
            make_record(2, 0.3, 12.4, 614.0, 320.0, 694.0, 420.0),
        ]

        result = TrackingEngine().run(records, TrackingConfig(max_gap_frames=3, max_center_distance_m=3.0))

        self.assertEqual(result[0].track_id, "7")
        self.assertEqual(result[0].track_status, "locked")
        self.assertEqual(result[1].track_id, "7")
        self.assertEqual(result[1].track_status, "auto")

    def test_tracking_does_not_associate_across_categories(self) -> None:
        records = [
            make_record(0, 0.0, 10.0, 600.0, 320.0, 680.0, 420.0, category="car", track_id="1"),
            make_record(1, 0.1, 10.1, 601.0, 320.0, 681.0, 420.0, category="pedestrian", track_id="2"),
        ]

        result = TrackingEngine().run(records, TrackingConfig(max_gap_frames=2, max_center_distance_m=3.0))

        self.assertEqual(result[0].track_id, "1")
        self.assertEqual(result[1].track_id, "2")
        self.assertEqual(result[0].track_status, "auto")
        self.assertEqual(result[1].track_status, "auto")


if __name__ == "__main__":
    unittest.main()
