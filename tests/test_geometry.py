from __future__ import annotations

import unittest

import numpy as np

from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.core.geometry import (
    box10_quaternion_to_corners,
    boxes10_quaternion_to_corners,
    project_box3d_heading_line,
    project_points_opencv,
)
from camlabel3d.core.models import DetectionRecord


class GeometryTests(unittest.TestCase):
    def test_vectorized_corners_match_vis4d_opencv_convention(self) -> None:
        try:
            import torch
            from vis4d.data.const import AxisMode
            from vis4d.op.box.box3d import boxes3d_to_corners
        except ImportError:
            self.skipTest("vis4d reference implementation is unavailable")

        rng = np.random.default_rng(20260715)
        boxes = np.empty((32, 10), dtype=np.float64)
        boxes[:, :3] = rng.uniform([-5.0, -2.0, 3.0], [5.0, 2.0, 30.0], size=(32, 3))
        boxes[:, 3:6] = rng.uniform(0.2, 6.0, size=(32, 3))
        quaternions = rng.normal(size=(32, 4))
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        boxes[:, 6:10] = quaternions

        expected = boxes3d_to_corners(
            torch.tensor(boxes, dtype=torch.float64),
            AxisMode.OPENCV,
        ).cpu().numpy()
        actual = boxes10_quaternion_to_corners(boxes)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_euler_roundtrip_keeps_box_orientation(self) -> None:
        record = DetectionRecord(
            frame_index=0,
            category="car",
            score=0.9,
            score_2d=0.8,
            score_3d=0.7,
            box2d_x1=0.0,
            box2d_y1=0.0,
            box2d_x2=1.0,
            box2d_y2=1.0,
            center_x=1.0,
            center_y=0.5,
            center_z=12.0,
            yaw_deg=20.0,
            pitch_deg=-5.0,
            roll_deg=3.0,
            size_w=1.8,
            size_l=4.5,
            size_h=1.6,
        )

        box3d = DetectorAdapter.record_to_box3d_quaternion(record)
        converted = DetectionRecord.from_row(
            {
                "frame_index": "0",
                "category": "car",
                "score": "0.9",
                "score_2d": "0.8",
                "score_3d": "0.7",
                "box2d_x1": "0",
                "box2d_y1": "0",
                "box2d_x2": "1",
                "box2d_y2": "1",
                "center_x": "1.0",
                "center_y": "0.5",
                "center_z": "12.0",
                "size_w": "1.8",
                "size_l": "4.5",
                "size_h": "1.6",
                "quat_w": str(float(box3d[6])),
                "quat_x": str(float(box3d[7])),
                "quat_y": str(float(box3d[8])),
                "quat_z": str(float(box3d[9])),
            }
        )

        self.assertAlmostEqual(converted.yaw_deg, record.yaw_deg, places=4)
        self.assertAlmostEqual(converted.pitch_deg, record.pitch_deg, places=4)
        self.assertAlmostEqual(converted.roll_deg, record.roll_deg, places=4)

    def test_project_record_to_box2d_returns_valid_bounds(self) -> None:
        record = DetectionRecord(
            frame_index=0,
            category="car",
            score=0.9,
            score_2d=0.8,
            score_3d=0.7,
            box2d_x1=0.0,
            box2d_y1=0.0,
            box2d_x2=1.0,
            box2d_y2=1.0,
            center_x=0.0,
            center_y=0.0,
            center_z=15.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            size_w=2.0,
            size_l=4.0,
            size_h=1.5,
        )
        intrinsics = np.array(
            [
                [800.0, 0.0, 640.0],
                [0.0, 800.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        x1, y1, x2, y2 = DetectorAdapter.project_record_to_box2d(
            record=record,
            intrinsics=intrinsics,
            image_shape=(720, 1280),
        )

        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, 1279.0)
        self.assertLessEqual(y2, 719.0)

    def test_project_box3d_heading_line_returns_screen_segment(self) -> None:
        record = DetectionRecord(
            frame_index=0,
            category="car",
            score=0.9,
            score_2d=0.8,
            score_3d=0.7,
            box2d_x1=0.0,
            box2d_y1=0.0,
            box2d_x2=1.0,
            box2d_y2=1.0,
            center_x=0.0,
            center_y=0.0,
            center_z=15.0,
            yaw_deg=25.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            size_w=2.0,
            size_l=4.0,
            size_h=1.5,
        )
        intrinsics = np.array(
            [
                [800.0, 0.0, 640.0],
                [0.0, 800.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        start_point, end_point = project_box3d_heading_line(
            DetectorAdapter.record_to_box3d_quaternion(record),
            intrinsics,
        )

        self.assertNotEqual(start_point, end_point)
        self.assertTrue(all(np.isfinite(value) for value in (*start_point, *end_point)))

        corners_2d = project_points_opencv(
            box10_quaternion_to_corners(DetectorAdapter.record_to_box3d_quaternion(record)),
            intrinsics,
        )
        expected_start = (corners_2d[0] + corners_2d[1] + corners_2d[2] + corners_2d[3]) * 0.25
        expected_end = (corners_2d[0] + corners_2d[1]) * 0.5
        self.assertAlmostEqual(start_point[0], float(expected_start[0]), places=5)
        self.assertAlmostEqual(start_point[1], float(expected_start[1]), places=5)
        self.assertAlmostEqual(end_point[0], float(expected_end[0]), places=5)
        self.assertAlmostEqual(end_point[1], float(expected_end[1]), places=5)


if __name__ == "__main__":
    unittest.main()
