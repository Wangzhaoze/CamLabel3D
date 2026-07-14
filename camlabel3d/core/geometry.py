"""Geometry helpers for CamLabel3D 3D boxes."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

NEAR_CLIP_Z = 0.15

_BOX_EDGES = [
    (0, 1), (1, 5), (5, 4), (4, 0),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (2, 3), (3, 7), (7, 6), (6, 2),
]
def normalize_quaternion_wxyz(quat: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(quat), dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return array / norm


def quaternion_wxyz_to_euler_yxz_deg(quat: Iterable[float]) -> tuple[float, float, float]:
    rotation = quaternion_wxyz_to_matrix(quat)
    pitch_rad = np.arcsin(np.clip(-rotation[1, 2], -1.0, 1.0))
    cos_pitch = float(np.cos(pitch_rad))
    if abs(cos_pitch) > 1e-6:
        yaw_rad = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        roll_rad = float(np.arctan2(rotation[1, 0], rotation[1, 1]))
    else:
        yaw_rad = float(np.arctan2(-rotation[2, 0], rotation[0, 0]))
        roll_rad = 0.0
    return (
        float(np.degrees(yaw_rad)),
        float(np.degrees(pitch_rad)),
        float(np.degrees(roll_rad)),
    )


def euler_yxz_deg_to_quaternion_wxyz(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    rotation = euler_yxz_deg_to_matrix(yaw_deg, pitch_deg, roll_deg)
    return matrix_to_quaternion_wxyz(rotation)


def quaternion_wxyz_to_matrix(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_yxz_deg_to_matrix(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    yaw_rad = float(np.radians(yaw_deg))
    pitch_rad = float(np.radians(pitch_deg))
    roll_rad = float(np.radians(roll_deg))

    cy, sy = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    cp, sp = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    cr, sr = float(np.cos(roll_rad)), float(np.sin(roll_rad))

    return np.array(
        [
            [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
            [cp * sr, cp * cr, -sp],
            [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion_wxyz([w, x, y, z])


def box9d_to_box10_quaternion(
    center_x: float,
    center_y: float,
    center_z: float,
    size_w: float,
    size_l: float,
    size_h: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    quat = euler_yxz_deg_to_quaternion_wxyz(yaw_deg, pitch_deg, roll_deg)
    return np.array(
        [
            float(center_x),
            float(center_y),
            float(center_z),
            float(size_w),
            float(size_l),
            float(size_h),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ],
        dtype=np.float32,
    )


def box10_quaternion_to_box9d(box3d: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(box3d), dtype=np.float64)
    yaw_deg, pitch_deg, roll_deg = quaternion_wxyz_to_euler_yxz_deg(values[6:10])
    return {
        "center_x": float(values[0]),
        "center_y": float(values[1]),
        "center_z": float(values[2]),
        "size_w": float(values[3]),
        "size_l": float(values[4]),
        "size_h": float(values[5]),
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
    }


def boxes10_quaternion_to_corners(boxes3d: Iterable[Iterable[float]]) -> np.ndarray:
    boxes_array = np.asarray(list(boxes3d), dtype=np.float64)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 10:
        raise ValueError(f"Expected Nx10 boxes, got {boxes_array.shape}")

    try:  # pragma: no cover - exercised in runtime when torch+vis4d are available
        import torch
        from vis4d.data.const import AxisMode
        from vis4d.op.box.box3d import boxes3d_to_corners

        box_tensor = torch.tensor(boxes_array, dtype=torch.float32)
        corners = boxes3d_to_corners(box_tensor, AxisMode.OPENCV)
        return corners.detach().cpu().numpy()
    except Exception:
        pass

    corners_all: list[np.ndarray] = []
    for box in boxes_array:
        center = np.asarray(box[:3], dtype=np.float64)
        width = float(box[3])
        length = float(box[4])
        height = float(box[5])
        rotation = quaternion_wxyz_to_matrix(box[6:10])

        half_w = width * 0.5
        half_l = length * 0.5
        half_h = height * 0.5
        # Match the vis4d corner ordering used by WildDet3D's tracking demo:
        # front face = 0-1-5-4, back face = 2-3-7-6.
        local_corners = np.array(
            [
                [-half_w, half_h, half_l],
                [half_w, half_h, half_l],
                [-half_w, half_h, -half_l],
                [half_w, half_h, -half_l],
                [-half_w, -half_h, half_l],
                [half_w, -half_h, half_l],
                [-half_w, -half_h, -half_l],
                [half_w, -half_h, -half_l],
            ],
            dtype=np.float64,
        )
        rotated = local_corners @ rotation.T
        corners_all.append(rotated + center)
    return np.asarray(corners_all, dtype=np.float64)


def box10_quaternion_to_corners(box3d: Iterable[float]) -> np.ndarray:
    return boxes10_quaternion_to_corners([list(box3d)])[0]


def project_points_opencv(
    points_xyz: np.ndarray,
    intrinsics: np.ndarray,
    near_clip: float = NEAR_CLIP_Z,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    if np.any(~np.isfinite(points)):
        raise ValueError("3D points contain non-finite values.")
    if np.any(points[:, 2] <= near_clip):
        raise ValueError("3D box cannot be projected because part of it is behind the camera.")

    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    projected = np.stack([u, v], axis=1)
    if np.any(~np.isfinite(projected)):
        raise ValueError("Projected 2D points contain non-finite values.")
    return projected


def project_box3d_quaternion_to_2d_bounds(
    box3d: Iterable[float],
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    near_clip: float = NEAR_CLIP_Z,
) -> tuple[float, float, float, float]:
    corners_3d = box10_quaternion_to_corners(box3d)
    corners_2d = project_points_opencv(corners_3d, intrinsics, near_clip=near_clip)

    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = float(np.min(corners_2d[:, 0]))
    y1 = float(np.min(corners_2d[:, 1]))
    x2 = float(np.max(corners_2d[:, 0]))
    y2 = float(np.max(corners_2d[:, 1]))

    if x2 < 0 or y2 < 0 or x1 > width - 1 or y1 > height - 1:
        raise ValueError("Projected 2D box is completely outside the image.")

    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Projected 2D box is degenerate.")
    return (x1, y1, x2, y2)


def project_box3d_heading_line(
    box3d: Iterable[float],
    intrinsics: np.ndarray,
    near_clip: float = NEAR_CLIP_Z,
) -> tuple[tuple[float, float], tuple[float, float]]:
    corners_3d = box10_quaternion_to_corners(box3d)
    corners_2d = project_points_opencv(corners_3d, intrinsics, near_clip=near_clip)

    center_bottom_front = (corners_2d[0] + corners_2d[1]) * 0.5
    center_bottom = (corners_2d[0] + corners_2d[1] + corners_2d[2] + corners_2d[3]) * 0.25
    start = (float(center_bottom[0]), float(center_bottom[1]))
    end = (float(center_bottom_front[0]), float(center_bottom_front[1]))
    return start, end


def draw_box3d_overlay(
    image: Image.Image,
    box3d: Iterable[float],
    intrinsics: np.ndarray,
    color_rgb: tuple[int, int, int] = (32, 255, 32),
    line_width: int = 3,
    near_clip: float = NEAR_CLIP_Z,
) -> Image.Image:
    import cv2

    corners_3d = box10_quaternion_to_corners(box3d)
    corners_2d = project_points_opencv(corners_3d, intrinsics, near_clip=near_clip)

    canvas_rgb = np.asarray(image.convert("RGB")).copy()
    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

    for start_idx, end_idx in _BOX_EDGES:
        p1 = corners_2d[start_idx]
        p2 = corners_2d[end_idx]
        cv2.line(
            canvas_bgr,
            (int(round(float(p1[0]))), int(round(float(p1[1])))),
            (int(round(float(p2[0]))), int(round(float(p2[1])))),
            color_bgr,
            thickness=int(line_width),
            lineType=cv2.LINE_AA,
        )
    return Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))


def draw_box3d_heading_overlay(
    image: Image.Image,
    box3d: Iterable[float],
    intrinsics: np.ndarray,
    color_rgb: tuple[int, int, int] = (88, 190, 255),
    line_width: int = 3,
    near_clip: float = NEAR_CLIP_Z,
) -> Image.Image:
    import cv2

    start, end = project_box3d_heading_line(
        box3d=box3d,
        intrinsics=intrinsics,
        near_clip=near_clip,
    )

    canvas_rgb = np.asarray(image.convert("RGB")).copy()
    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    start_xy = (int(round(start[0])), int(round(start[1])))
    end_xy = (int(round(end[0])), int(round(end[1])))
    cv2.line(
        canvas_bgr,
        start_xy,
        end_xy,
        color_bgr,
        thickness=int(line_width),
        lineType=cv2.LINE_AA,
    )
    return Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
