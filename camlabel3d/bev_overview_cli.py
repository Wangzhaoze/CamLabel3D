"""Command-line entrypoint for static BEV trajectory overview rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

from camlabel3d.core.bev import (
    DEFAULT_TRAJECTORY_WINDOW_FRAMES,
    build_bev_scene,
    derive_bev_overview_path,
    render_bev_overview,
    select_default_bev_frame_index,
    select_default_bev_track_id,
)
from camlabel3d.io.csv_store import CSVStore


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a static BEV trajectory overview PNG from a CamLabel3D CSV file."
    )
    parser.add_argument("--csv-path", required=True, help="Input CamLabel3D CSV path.")
    parser.add_argument("--track-id", default="", help="Focus track ID. Defaults to the largest enabled track.")
    parser.add_argument("--frame-index", type=int, default=None, help="Focus frame index. Defaults to the track median.")
    parser.add_argument(
        "--trajectory-window",
        type=int,
        default=DEFAULT_TRAJECTORY_WINDOW_FRAMES,
        help="Maximum history window size in frames before the focus frame. Use 0 for full history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path).resolve()
    records = CSVStore(csv_path).load_records()
    if not records:
        raise SystemExit(f"No detections found in {csv_path}")
    if int(args.trajectory_window) < 0:
        raise SystemExit("--trajectory-window must be >= 0.")

    requested_track_id = str(args.track_id or "").strip()
    focus_track_id = requested_track_id or select_default_bev_track_id(records)
    if requested_track_id and not any(str(record.track_id).strip() == requested_track_id for record in records):
        raise SystemExit(f"Track '{requested_track_id}' was not found in {csv_path}.")

    frame_index = (
        int(args.frame_index)
        if args.frame_index is not None
        else select_default_bev_frame_index(records, focus_track_id=focus_track_id)
    )
    scene = build_bev_scene(
        records,
        frame_index=frame_index,
        focus_track_id=focus_track_id,
        trajectory_window_frames=int(args.trajectory_window),
    )
    output_path = derive_bev_overview_path(csv_path)
    render_bev_overview(scene, output_path)
    print(
        "BEV overview written: "
        f"{output_path} "
        f"(track={scene.focus_track_id or '--'}, frame={scene.frame_index}, "
        f"history_window={scene.trajectory_window_frames if scene.trajectory_window_frames > 0 else 'all'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
