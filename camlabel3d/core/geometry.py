"""Geometry helpers for CamLabel3D 3D boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

NEAR_CLIP_Z = 0.15

_BOX_EDGES = [
    (0, 1), (1, 5), (5, 4), (4, 0),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (2, 3), (3, 7), (7, 6), (6, 2),
]


@dataclass(frozen=True)
class Box3DOverlay:
    box3d: Iterable[float]
    color_rgb: tuple[int, int, int]
    line_width: int = 2
    draw_edges: bool = True
    draw_heading: bool = True
    label: str = ""
    heading_color_rgb: tuple[int, int, int] | None = None


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
    if boxes_array.size == 0:
        return np.empty((0, 8, 3), dtype=np.float64)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 10:
        raise ValueError(f"Expected Nx10 boxes, got {boxes_array.shape}")

    # Vectorized NumPy implementation of the vis4d OPENCV convention. Preview
    # rendering calls this at pointer-event frequency, where constructing Torch
    # tensors and dispatching many tiny CPU kernels is disproportionately slow.
    quaternions = boxes_array[:, 6:10]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    fallback = np.zeros_like(quaternions)
    fallback[:, 0] = 1.0
    quaternions = np.where(norms > 1e-8, quaternions / np.maximum(norms, 1e-8), fallback)
    w, x, y, z = (quaternions[:, offset] for offset in range(4))
    rotations = np.empty((len(boxes_array), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    half_width = boxes_array[:, 3] * 0.5
    half_length = boxes_array[:, 4] * 0.5
    half_height = boxes_array[:, 5] * 0.5
    signs = np.array(
        [
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # vis4d's OPENCV axis convention maps box length to camera X and box
    # width to camera Z; preserving that mapping is essential for correct BBX.
    dimensions = np.stack([half_length, half_height, half_width], axis=1)
    local_corners = signs[None, :, :] * dimensions[:, None, :]
    rotated = np.einsum("nij,nkj->nki", rotations, local_corners, optimize=True)
    return rotated + boxes_array[:, None, :3]


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
    return draw_box3d_overlays(
        image,
        [Box3DOverlay(box3d, color_rgb, line_width, draw_edges=True, draw_heading=False)],
        intrinsics,
        near_clip=near_clip,
    )


def draw_box3d_heading_overlay(
    image: Image.Image,
    box3d: Iterable[float],
    intrinsics: np.ndarray,
    color_rgb: tuple[int, int, int] = (88, 190, 255),
    line_width: int = 3,
    near_clip: float = NEAR_CLIP_Z,
) -> Image.Image:
    return draw_box3d_overlays(
        image,
        [Box3DOverlay(box3d, color_rgb, line_width, draw_edges=False, draw_heading=True)],
        intrinsics,
        near_clip=near_clip,
    )


def render_box3d_scene_rgb(
    image_rgb: np.ndarray,
    overlays: Iterable[Box3DOverlay],
    intrinsics: np.ndarray,
    *,
    prompt_box: tuple[float, float, float, float] | None = None,
    prompt_points: Iterable[tuple[float, float, int]] = (),
    near_clip: float = NEAR_CLIP_Z,
) -> np.ndarray:
    """Render a full-resolution RGB frame and all 3D annotations in one pass.

    OpenCV drawing functions operate on channel values without requiring a BGR
    image, so keeping this canvas RGB avoids two full-frame color conversions.
    Geometry is projected as one NumPy batch and labels are drawn in the same
    pass. The result remains an owning, contiguous source-resolution array.
    """

    import cv2

    source = np.asarray(image_rgb)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {source.shape}")
    if source.dtype != np.uint8:
        source = (
            source * 255.0
            if np.issubdtype(source.dtype, np.floating) and float(np.nanmax(source)) <= 1.0
            else source
        )
        source = np.clip(source, 0, 255).astype(np.uint8)
    canvas = np.array(source, dtype=np.uint8, order="C", copy=True)
    height, width = canvas.shape[:2]
    overlay_items = tuple(overlays)
    if overlay_items:
        boxes = np.asarray([list(item.box3d) for item in overlay_items], dtype=np.float64)
        corners_all = boxes10_quaternion_to_corners(boxes)
        k = np.asarray(intrinsics, dtype=np.float64)
        front_mask = np.all(corners_all[..., 2] >= near_clip, axis=1)
        projected_all = np.zeros((*corners_all.shape[:2], 2), dtype=np.float64)
        front_corners = corners_all[front_mask]
        if len(front_corners) > 0:
            front_depth = front_corners[..., 2]
            projected_all[front_mask, :, 0] = (
                float(k[0, 0]) * front_corners[..., 0] / front_depth + float(k[0, 2])
            )
            projected_all[front_mask, :, 1] = (
                float(k[1, 1]) * front_corners[..., 1] / front_depth + float(k[1, 2])
            )
        pixel_all = np.clip(
            np.rint(projected_all),
            -2_000_000_000,
            2_000_000_000,
        ).astype(np.int32)

        def project(point: np.ndarray) -> tuple[float, float] | None:
            z_value = float(point[2])
            if z_value < near_clip:
                return None
            u_value = float(k[0, 0]) * float(point[0]) / z_value + float(k[0, 2])
            v_value = float(k[1, 1]) * float(point[1]) / z_value + float(k[1, 2])
            if not np.isfinite(u_value) or not np.isfinite(v_value):
                return None
            return u_value, v_value

        def draw_segment(
            point_a: np.ndarray,
            point_b: np.ndarray,
            color: tuple[int, int, int],
            thickness: int,
        ) -> None:
            a = np.asarray(point_a, dtype=np.float64)
            b = np.asarray(point_b, dtype=np.float64)
            if float(a[2]) < near_clip and float(b[2]) < near_clip:
                return
            if float(a[2]) < near_clip:
                ratio = (near_clip - float(a[2])) / (float(b[2]) - float(a[2]))
                a = a + ratio * (b - a)
                a[2] = near_clip
            elif float(b[2]) < near_clip:
                ratio = (near_clip - float(b[2])) / (float(a[2]) - float(b[2]))
                b = b + ratio * (a - b)
                b[2] = near_clip
            projected_a = project(a)
            projected_b = project(b)
            if projected_a is None or projected_b is None:
                return
            limit = 2_000_000_000
            pixel_a = (
                int(np.clip(round(projected_a[0]), -limit, limit)),
                int(np.clip(round(projected_a[1]), -limit, limit)),
            )
            pixel_b = (
                int(np.clip(round(projected_b[0]), -limit, limit)),
                int(np.clip(round(projected_b[1]), -limit, limit)),
            )
            visible, clipped_a, clipped_b = cv2.clipLine((0, 0, width, height), pixel_a, pixel_b)
            if visible:
                cv2.line(
                    canvas,
                    clipped_a,
                    clipped_b,
                    tuple(int(channel) for channel in color),
                    thickness=max(1, int(thickness)),
                    lineType=cv2.LINE_AA,
                )

        for box_index, (overlay, box, corners) in enumerate(zip(overlay_items, boxes, corners_all)):
            edge_color = tuple(int(channel) for channel in overlay.color_rgb)
            thickness = max(1, int(overlay.line_width))
            heading_color = overlay.heading_color_rgb or edge_color
            if front_mask[box_index]:
                # Normal camera-facing boxes take a vectorized projection path
                # and a few batched OpenCV calls. Only near-plane crossings use
                # the exact per-segment clipping path below.
                pixels = pixel_all[box_index]
                projected = projected_all[box_index]
                if overlay.draw_edges:
                    cv2.polylines(
                        canvas,
                        [pixels[[0, 1, 5, 4]], pixels[[2, 3, 7, 6]]],
                        True,
                        edge_color,
                        thickness,
                        cv2.LINE_AA,
                    )
                    connectors = [
                        pixels[[start_index, end_index]]
                        for start_index, end_index in ((0, 2), (1, 3), (4, 6), (5, 7))
                    ]
                    cv2.polylines(
                        canvas,
                        connectors,
                        False,
                        edge_color,
                        thickness,
                        cv2.LINE_AA,
                    )
                if overlay.draw_heading:
                    bottom_center = tuple(
                        np.rint(projected[[0, 1, 2, 3]].mean(axis=0)).astype(np.int32)
                    )
                    bottom_front_center = tuple(
                        np.rint(projected[[0, 1]].mean(axis=0)).astype(np.int32)
                    )
                    cv2.line(
                        canvas,
                        bottom_center,
                        bottom_front_center,
                        heading_color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA,
                    )
            else:
                if overlay.draw_edges:
                    for start_index, end_index in _BOX_EDGES:
                        draw_segment(corners[start_index], corners[end_index], edge_color, thickness)
                if overlay.draw_heading:
                    bottom_center_3d = np.mean(corners[[0, 1, 2, 3]], axis=0)
                    bottom_front_center_3d = np.mean(corners[[0, 1]], axis=0)
                    draw_segment(
                        bottom_center_3d,
                        bottom_front_center_3d,
                        heading_color,
                        thickness,
                    )

            if overlay.label:
                anchor = project(box[:3])
                if (
                    anchor is not None
                    and -50.0 <= anchor[0] <= width + 50.0
                    and -50.0 <= anchor[1] <= height + 50.0
                ):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.4, min(0.65, min(width, height) / 900.0))
                    font_thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        overlay.label,
                        font,
                        font_scale,
                        font_thickness,
                    )
                    label_x = int(round(anchor[0] - text_width * 0.5))
                    label_y = int(round(anchor[1] - text_height * 0.5))
                    label_x = max(2, min(label_x, max(2, width - text_width - 8)))
                    label_y = max(text_height + 6, min(label_y, max(text_height + 6, height - baseline - 4)))
                    cv2.rectangle(
                        canvas,
                        (label_x - 4, label_y - text_height - 4),
                        (label_x + text_width + 4, label_y + baseline + 3),
                        edge_color,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        overlay.label,
                        (label_x, label_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA,
                    )

    if prompt_box is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in prompt_box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 72, 72), thickness=3, lineType=cv2.LINE_AA)
    for point_x, point_y, point_label in prompt_points:
        color = (255, 64, 64) if int(point_label) == 1 else (170, 170, 170)
        cv2.circle(
            canvas,
            (int(round(point_x)), int(round(point_y))),
            radius=6,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def draw_box3d_overlays(
    image: Image.Image,
    overlays: Iterable[Box3DOverlay],
    intrinsics: np.ndarray,
    near_clip: float = NEAR_CLIP_Z,
) -> Image.Image:
    """Draw many edge/heading overlays with one full-image color conversion."""

    import cv2

    canvas_rgb = np.asarray(image.convert("RGB")).copy()
    canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
    for overlay in overlays:
        corners_3d = box10_quaternion_to_corners(overlay.box3d)
        corners_2d = project_points_opencv(corners_3d, intrinsics, near_clip=near_clip)
        color_rgb = overlay.color_rgb
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        thickness = int(overlay.line_width)
        if overlay.draw_edges:
            for start_idx, end_idx in _BOX_EDGES:
                p1 = corners_2d[start_idx]
                p2 = corners_2d[end_idx]
                cv2.line(
                    canvas_bgr,
                    (int(round(float(p1[0]))), int(round(float(p1[1])))),
                    (int(round(float(p2[0]))), int(round(float(p2[1])))),
                    color_bgr,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
        if overlay.draw_heading:
            center_bottom_front = (corners_2d[0] + corners_2d[1]) * 0.5
            center_bottom = (corners_2d[0] + corners_2d[1] + corners_2d[2] + corners_2d[3]) * 0.25
            cv2.line(
                canvas_bgr,
                (int(round(float(center_bottom[0]))), int(round(float(center_bottom[1])))),
                (int(round(float(center_bottom_front[0]))), int(round(float(center_bottom_front[1])))),
                color_bgr,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    return Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
