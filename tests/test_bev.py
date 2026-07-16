from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from camlabel3d.core.bev import (
    DEFAULT_TRAJECTORY_WINDOW_FRAMES,
    build_bev_scene,
    derive_bev_overview_path,
    render_bev_overview,
    select_default_bev_frame_index,
    select_default_bev_track_id,
)
from camlabel3d.core.models import DetectionRecord


def _record(
    frame_index: int,
    *,
    det_id: str,
    track_id: str,
    center_x: float,
    center_z: float,
    yaw_deg: float = 0.0,
    size_w: float = 2.0,
    size_l: float = 4.0,
    is_enabled: bool = True,
    is_visible: bool | None = None,
    score: float = 0.9,
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category="car",
        score=score,
        score_2d=score,
        score_3d=score,
        box2d_x1=0.0,
        box2d_y1=0.0,
        box2d_x2=10.0,
        box2d_y2=10.0,
        center_x=center_x,
        center_y=0.0,
        center_z=center_z,
        yaw_deg=yaw_deg,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=size_w,
        size_l=size_l,
        size_h=1.5,
        is_enabled=is_enabled,
        is_visible=is_visible,
        track_id=track_id,
        track_status="auto" if track_id else "",
        det_id=det_id,
    )


def test_build_bev_scene_filters_frame_boxes_and_resolves_focus_track_from_det_id() -> None:
    focus_prev = _record(9, det_id="t1-9", track_id="7", center_x=0.0, center_z=10.0)
    focus_curr = _record(10, det_id="t1-10", track_id="7", center_x=1.0, center_z=11.0)
    focus_future = _record(11, det_id="t1-11", track_id="7", center_x=2.0, center_z=12.0)
    hidden_curr = _record(
        10,
        det_id="t2-10",
        track_id="2",
        center_x=5.0,
        center_z=15.0,
        is_visible=False,
    )
    disabled_curr = _record(
        10,
        det_id="t3-10",
        track_id="3",
        center_x=-4.0,
        center_z=16.0,
        is_enabled=False,
    )

    scene = build_bev_scene(
        [focus_prev, focus_curr, focus_future, hidden_curr, disabled_curr],
        frame_index=10,
        focus_det_id="t1-10",
    )

    assert scene.focus_track_id == "7"
    assert [box.det_id for box in scene.frame_boxes] == ["t1-10"]
    assert [sample.det_id for sample in scene.trajectory_samples] == ["t1-9", "t1-10"]
    assert scene.current_pose is not None
    assert scene.current_pose.det_id == "t1-10"
    assert scene.viewport_seed.grid_spacing_m == 2.0
    assert scene.viewport_seed.z_min == 0.0
    assert scene.viewport_seed.x_min == -scene.viewport_seed.x_max


def test_build_bev_scene_can_include_full_history_when_window_is_zero() -> None:
    records = [
        _record(1, det_id="h1", track_id="7", center_x=0.0, center_z=8.0),
        _record(2, det_id="h2", track_id="7", center_x=0.5, center_z=8.5),
        _record(3, det_id="h3", track_id="7", center_x=1.0, center_z=9.0),
        _record(4, det_id="f4", track_id="7", center_x=1.5, center_z=9.5),
    ]

    scene = build_bev_scene(
        records,
        frame_index=3,
        focus_track_id="7",
        trajectory_window_frames=0,
    )

    assert [sample.det_id for sample in scene.trajectory_samples] == ["h1", "h2", "h3"]


def test_bev_box_corners_and_arrow_follow_existing_yaw_convention() -> None:
    current = _record(
        10,
        det_id="focus",
        track_id="focus-track",
        center_x=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        size_w=2.0,
        size_l=4.0,
    )
    scene = build_bev_scene([current], frame_index=10, focus_track_id="focus-track")

    assert scene.current_pose is not None
    assert scene.current_pose.box.corners_xz == ((2.0, 9.0), (2.0, 11.0), (-2.0, 11.0), (-2.0, 9.0))
    assert scene.current_pose.yaw_arrow.start_x == 0.0
    assert scene.current_pose.yaw_arrow.start_z == 10.0
    assert scene.current_pose.yaw_arrow.end_x == 2.0
    assert scene.current_pose.yaw_arrow.end_z == 10.0


def test_default_track_and_frame_selection_choose_largest_enabled_track_and_its_median_frame() -> None:
    records = [
        _record(1, det_id="a1", track_id="A", center_x=0.0, center_z=10.0),
        _record(2, det_id="a2", track_id="A", center_x=1.0, center_z=10.5),
        _record(3, det_id="a3", track_id="A", center_x=2.0, center_z=11.0),
        _record(4, det_id="b1", track_id="B", center_x=0.0, center_z=12.0),
        _record(5, det_id="c1", track_id="", center_x=0.0, center_z=13.0),
    ]

    assert select_default_bev_track_id(records) == "A"
    assert select_default_bev_frame_index(records, focus_track_id="A") == 2
    assert select_default_bev_frame_index([], focus_track_id="A") == 0


def test_render_bev_overview_writes_png_with_expected_size() -> None:
    records = [
        _record(9, det_id="t1-9", track_id="7", center_x=0.0, center_z=10.0),
        _record(10, det_id="t1-10", track_id="7", center_x=1.0, center_z=11.0),
        _record(11, det_id="t1-11", track_id="7", center_x=2.0, center_z=12.0),
    ]
    scene = build_bev_scene(
        records,
        frame_index=10,
        focus_track_id="7",
        trajectory_window_frames=DEFAULT_TRAJECTORY_WINDOW_FRAMES,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "overview.png"
        rendered_path = render_bev_overview(scene, output_path)

        assert rendered_path == output_path.resolve()
        assert rendered_path.exists()
        with Image.open(rendered_path) as image:
            assert image.size == (1280, 720)


def test_derive_bev_overview_path_reuses_camlabel3d_suffix() -> None:
    path = derive_bev_overview_path("D:/demo/sample.camlabel3d.csv")
    assert path.name == "sample.bev_overview.png"
