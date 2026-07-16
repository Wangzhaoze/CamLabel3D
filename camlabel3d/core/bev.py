"""Bird's-eye-view scene building and static overview rendering."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from camlabel3d.core.detector import DetectorAdapter
from camlabel3d.core.geometry import box10_quaternion_to_corners
from camlabel3d.core.models import DetectionRecord, natural_sort_key

DEFAULT_TRAJECTORY_WINDOW_FRAMES = 30
DEFAULT_OVERVIEW_SIZE_PX = (1280, 720)


@dataclass(frozen=True, slots=True)
class BEVBox:
    det_id: str
    track_id: str
    frame_index: int
    category: str
    center_x: float
    center_z: float
    yaw_deg: float
    size_w: float
    size_l: float
    corners_xz: tuple[tuple[float, float], ...]
    is_focus: bool = False


@dataclass(frozen=True, slots=True)
class BEVTrajectorySample:
    frame_index: int
    det_id: str
    track_id: str
    x: float
    z: float
    yaw_deg: float
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class BEVArrow:
    start_x: float
    start_z: float
    end_x: float
    end_z: float


@dataclass(frozen=True, slots=True)
class BEVCurrentPose:
    frame_index: int
    det_id: str
    track_id: str
    center_x: float
    center_z: float
    box: BEVBox
    yaw_arrow: BEVArrow


@dataclass(frozen=True, slots=True)
class BEVViewport:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    grid_spacing_m: float

    @property
    def width_m(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height_m(self) -> float:
        return float(self.z_max - self.z_min)


@dataclass(frozen=True, slots=True)
class BEVScene:
    frame_index: int
    focus_track_id: str
    focus_det_id: str
    trajectory_window_frames: int
    frame_boxes: tuple[BEVBox, ...]
    trajectory_samples: tuple[BEVTrajectorySample, ...]
    current_pose: BEVCurrentPose | None
    viewport_seed: BEVViewport


def select_default_bev_track_id(records: Sequence[DetectionRecord]) -> str:
    counts = Counter(
        str(record.track_id).strip()
        for record in records
        if record.is_enabled and str(record.track_id).strip()
    )
    if not counts:
        return ""
    return min(counts.items(), key=lambda item: (-item[1], natural_sort_key(item[0])))[0]


def select_default_bev_frame_index(records: Sequence[DetectionRecord], focus_track_id: str = "") -> int:
    track_id = str(focus_track_id).strip()
    candidates = [
        record
        for record in records
        if record.is_enabled and (not track_id or str(record.track_id).strip() == track_id)
    ]
    if not candidates:
        if not records:
            return 0
        candidates = list(records)
    candidates.sort(key=lambda item: (int(item.frame_index), item.det_id))
    return int(candidates[len(candidates) // 2].frame_index)


def build_bev_scene(
    records: Sequence[DetectionRecord],
    *,
    frame_index: int,
    focus_track_id: str = "",
    focus_det_id: str = "",
    trajectory_window_frames: int = DEFAULT_TRAJECTORY_WINDOW_FRAMES,
) -> BEVScene:
    trajectory_window = max(0, int(trajectory_window_frames))
    records_by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)
    records_by_track: dict[str, list[DetectionRecord]] = defaultdict(list)
    records_by_id: dict[str, DetectionRecord] = {}
    for record in records:
        records_by_id[record.det_id] = record
        records_by_frame[int(record.frame_index)].append(record)
        track_id = str(record.track_id).strip()
        if track_id:
            records_by_track[track_id].append(record)

    resolved_focus_det_id = str(focus_det_id).strip()
    resolved_focus_track_id = str(focus_track_id).strip()
    if not resolved_focus_track_id and resolved_focus_det_id:
        focused_record = records_by_id.get(resolved_focus_det_id)
        if focused_record is not None:
            resolved_focus_track_id = str(focused_record.track_id).strip()

    current_frame_records = [
        record
        for record in sorted(
            records_by_frame.get(int(frame_index), []),
            key=lambda item: (-float(item.score), item.det_id),
        )
        if record.is_enabled and bool(record.is_visible)
    ]
    frame_boxes = tuple(
        _record_to_bev_box(record, is_focus=bool(resolved_focus_track_id and str(record.track_id).strip() == resolved_focus_track_id))
        for record in current_frame_records
    )

    trajectory_samples: tuple[BEVTrajectorySample, ...] = ()
    current_pose: BEVCurrentPose | None = None
    if resolved_focus_track_id:
        track_records = [
            record
            for record in sorted(
                records_by_track.get(resolved_focus_track_id, []),
                key=lambda item: (int(item.frame_index), item.det_id),
            )
            if record.is_enabled
        ]
        history_records = [record for record in track_records if int(record.frame_index) <= int(frame_index)]
        if trajectory_window > 0:
            start_frame = int(frame_index) - trajectory_window
            history_records = [
                record
                for record in history_records
                if start_frame <= int(record.frame_index)
            ]
        trajectory_samples = tuple(
            BEVTrajectorySample(
                frame_index=int(record.frame_index),
                det_id=record.det_id,
                track_id=resolved_focus_track_id,
                x=float(record.center_x),
                z=float(record.center_z),
                yaw_deg=float(record.yaw_deg),
                is_current=int(record.frame_index) == int(frame_index),
            )
            for record in history_records
        )
        current_record = _select_current_pose_record(
            track_records=track_records,
            frame_index=int(frame_index),
            focus_det_id=resolved_focus_det_id,
        )
        if current_record is not None:
            current_box = _record_to_bev_box(current_record, is_focus=True)
            current_pose = BEVCurrentPose(
                frame_index=int(current_record.frame_index),
                det_id=current_record.det_id,
                track_id=resolved_focus_track_id,
                center_x=float(current_box.center_x),
                center_z=float(current_box.center_z),
                box=current_box,
                yaw_arrow=_record_to_bev_arrow(current_record),
            )

    viewport_seed = _build_viewport_seed(
        frame_boxes=frame_boxes,
        trajectory_samples=trajectory_samples,
        current_pose=current_pose,
    )
    return BEVScene(
        frame_index=int(frame_index),
        focus_track_id=resolved_focus_track_id,
        focus_det_id=resolved_focus_det_id,
        trajectory_window_frames=trajectory_window,
        frame_boxes=frame_boxes,
        trajectory_samples=trajectory_samples,
        current_pose=current_pose,
        viewport_seed=viewport_seed,
    )


def render_bev_overview(
    scene: BEVScene,
    output_path: str | Path,
    *,
    title: str | None = None,
    size_px: tuple[int, int] = DEFAULT_OVERVIEW_SIZE_PX,
) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, Polygon
    from matplotlib.ticker import MultipleLocator

    width_px, height_px = (max(1, int(size_px[0])), max(1, int(size_px[1])))
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(
        figsize=(width_px / 100.0, height_px / 100.0),
        dpi=100,
        facecolor="#F5F5F0",
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_facecolor("#FBFBF8")
    axis.set_aspect("equal", adjustable="box")

    viewport = scene.viewport_seed
    axis.set_xlim(viewport.x_min, viewport.x_max)
    axis.set_ylim(viewport.z_min, viewport.z_max)

    x_ticks = np.arange(
        viewport.x_min,
        viewport.x_max + viewport.grid_spacing_m * 0.5,
        viewport.grid_spacing_m,
        dtype=np.float64,
    )
    z_ticks = np.arange(
        viewport.z_min,
        viewport.z_max + viewport.grid_spacing_m * 0.5,
        viewport.grid_spacing_m,
        dtype=np.float64,
    )
    axis.xaxis.set_major_locator(MultipleLocator(viewport.grid_spacing_m))
    axis.yaxis.set_major_locator(MultipleLocator(viewport.grid_spacing_m))
    axis.set_xticks(x_ticks)
    axis.set_yticks(z_ticks)
    axis.grid(
        True,
        which="major",
        linestyle=(0, (4, 4)),
        linewidth=0.8,
        color="#B9C0C6",
        alpha=0.85,
        zorder=0,
    )

    axis.spines["left"].set_position(("data", 0.0))
    axis.spines["bottom"].set_position(("data", 0.0))
    axis.spines["left"].set_color("#4B5966")
    axis.spines["bottom"].set_color("#4B5966")
    axis.spines["left"].set_linewidth(1.4)
    axis.spines["bottom"].set_linewidth(1.4)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    axis.tick_params(
        axis="x",
        labelsize=10,
        colors="#37424D",
        direction="inout",
        length=6,
        width=1.0,
        pad=6,
    )
    axis.tick_params(
        axis="y",
        labelsize=10,
        colors="#37424D",
        direction="inout",
        length=6,
        width=1.0,
        pad=6,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")

    for box in scene.frame_boxes:
        is_current_focus = bool(scene.current_pose is not None and box.det_id == scene.current_pose.det_id)
        edge_color = "#4D8AC6" if box.is_focus else "#8795A1"
        face_color = "#D8E7F6" if box.is_focus else "#E7EBEF"
        alpha = 0.9 if is_current_focus else (0.55 if box.is_focus else 0.35)
        line_width = 2.4 if is_current_focus else (1.8 if box.is_focus else 1.2)
        axis.add_patch(
            Polygon(
                box.corners_xz,
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=alpha,
                joinstyle="round",
            )
        )

    if scene.trajectory_samples:
        history = [(sample.x, sample.z) for sample in scene.trajectory_samples]
        if len(history) >= 2:
            axis.plot(
                [point[0] for point in history],
                [point[1] for point in history],
                color="#147AD6",
                linewidth=2.4,
                solid_capstyle="round",
                zorder=4,
            )
        axis.scatter(
            [sample.x for sample in scene.trajectory_samples],
            [sample.z for sample in scene.trajectory_samples],
            s=18,
            color="#5AA7E8",
            alpha=0.55,
            zorder=5,
        )

    if scene.current_pose is not None:
        current = scene.current_pose
        axis.scatter(
            [current.center_x],
            [current.center_z],
            s=60,
            color="#D64541",
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=7,
        )
        axis.add_patch(
            FancyArrowPatch(
                (current.yaw_arrow.start_x, current.yaw_arrow.start_z),
                (current.yaw_arrow.end_x, current.yaw_arrow.end_z),
                arrowstyle="-|>",
                mutation_scale=16.0,
                linewidth=2.2,
                color="#D64541",
                zorder=8,
                shrinkA=0.0,
                shrinkB=0.0,
            )
        )

    if title:
        axis.set_title(title, fontsize=16, color="#1F2A35", pad=12.0)
    history_line = (
        "History: all previous frames"
        if scene.trajectory_window_frames <= 0
        else f"History window: {scene.trajectory_window_frames} frames"
    )
    metadata_lines = [
        f"Frame {scene.frame_index}",
        f"Track {scene.focus_track_id or '--'}",
        history_line,
        f"Viewport {viewport.width_m:.1f}m x {viewport.height_m:.1f}m",
    ]
    axis.text(
        0.015,
        0.985,
        "\n".join(metadata_lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#30404F",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFFFFF",
            "edgecolor": "#C8D0D8",
            "alpha": 0.92,
        },
    )

    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    return output


def derive_bev_overview_path(csv_path: str | Path) -> Path:
    path = Path(csv_path).resolve()
    suffix = ".camlabel3d.csv"
    if path.name.endswith(suffix):
        stem = path.name[: -len(suffix)]
        return path.with_name(f"{stem}.bev_overview.png")
    return path.with_suffix(path.suffix + ".bev_overview.png")


def _select_current_pose_record(
    *,
    track_records: Sequence[DetectionRecord],
    frame_index: int,
    focus_det_id: str,
) -> DetectionRecord | None:
    if focus_det_id:
        for record in track_records:
            if record.det_id == focus_det_id and int(record.frame_index) == int(frame_index):
                return record
    frame_matches = [record for record in track_records if int(record.frame_index) == int(frame_index)]
    if not frame_matches:
        return None
    return max(frame_matches, key=lambda item: (float(item.score), item.det_id))


def _record_to_bev_box(record: DetectionRecord, *, is_focus: bool) -> BEVBox:
    box3d = DetectorAdapter.record_to_box3d_quaternion(record)
    corners = box10_quaternion_to_corners(box3d)
    corners_xz = tuple(
        (float(corners[index, 0]), float(corners[index, 2]))
        for index in (0, 1, 3, 2)
    )
    return BEVBox(
        det_id=record.det_id,
        track_id=str(record.track_id).strip(),
        frame_index=int(record.frame_index),
        category=record.category,
        center_x=float(record.center_x),
        center_z=float(record.center_z),
        yaw_deg=float(record.yaw_deg),
        size_w=float(record.size_w),
        size_l=float(record.size_l),
        corners_xz=corners_xz,
        is_focus=bool(is_focus),
    )


def _record_to_bev_arrow(record: DetectionRecord) -> BEVArrow:
    box3d = DetectorAdapter.record_to_box3d_quaternion(record)
    corners = box10_quaternion_to_corners(box3d)
    start = np.mean(corners[[0, 1, 2, 3]][:, [0, 2]], axis=0)
    end = np.mean(corners[[0, 1]][:, [0, 2]], axis=0)
    return BEVArrow(
        start_x=float(start[0]),
        start_z=float(start[1]),
        end_x=float(end[0]),
        end_z=float(end[1]),
    )


def _build_viewport_seed(
    *,
    frame_boxes: Iterable[BEVBox],
    trajectory_samples: Iterable[BEVTrajectorySample],
    current_pose: BEVCurrentPose | None,
) -> BEVViewport:
    xs: list[float] = []
    zs: list[float] = []
    for box in frame_boxes:
        for x_value, z_value in box.corners_xz:
            xs.append(float(x_value))
            zs.append(float(z_value))
    for sample in trajectory_samples:
        xs.append(float(sample.x))
        zs.append(float(sample.z))
    if current_pose is not None:
        xs.extend([current_pose.center_x, current_pose.yaw_arrow.end_x])
        zs.extend([current_pose.center_z, current_pose.yaw_arrow.end_z])
        for x_value, z_value in current_pose.box.corners_xz:
            xs.append(float(x_value))
            zs.append(float(z_value))

    if not xs or not zs:
        xs = [-4.0, 4.0]
        zs = [6.0, 10.0]

    lateral_extent = max(4.0, max(abs(float(value)) for value in xs))
    forward_extent = max(6.0, max(0.0, max(float(value) for value in zs)))
    reference_span = max(2.0 * lateral_extent, forward_extent, 12.0)
    lateral_padding = max(2.0, 0.14 * reference_span)
    forward_padding = max(2.0, 0.14 * reference_span)

    x_range = lateral_extent + lateral_padding
    z_max = forward_extent + forward_padding
    grid_spacing = _select_grid_spacing(max(2.0 * x_range, z_max))
    x_range = math.ceil(x_range / grid_spacing) * grid_spacing
    z_max = math.ceil(z_max / grid_spacing) * grid_spacing

    min_x_range = max(6.0, grid_spacing * 3.0)
    min_z_max = max(12.0, grid_spacing * 4.0)
    x_range = max(x_range, min_x_range)
    z_max = max(z_max, min_z_max)

    return BEVViewport(
        x_min=float(-x_range),
        x_max=float(x_range),
        z_min=0.0,
        z_max=float(z_max),
        grid_spacing_m=float(grid_spacing),
    )


def _select_grid_spacing(span_m: float) -> float:
    if span_m <= 30.0:
        return 2.0
    if span_m <= 80.0:
        return 5.0
    return 10.0
