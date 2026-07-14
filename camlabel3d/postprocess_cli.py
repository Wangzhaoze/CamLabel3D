"""Command-line entrypoint for CamLabel3D CSV postprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from camlabel3d.core import (
    DetectorAdapter,
    OutlierScope,
    ProcessingContext,
    ProcessingEngine,
    ProcessingScope,
    hits_to_report_json,
)
from camlabel3d.io.csv_store import CSVStore


def _derive_report_path(csv_path: Path) -> Path:
    suffix = ".camlabel3d.csv"
    if csv_path.name.endswith(suffix):
        stem = csv_path.name[: -len(suffix)]
        return csv_path.with_name(f"{stem}.outliers.json")
    return csv_path.with_suffix(csv_path.suffix + ".outliers.json")


def _derive_post_path(csv_path: Path) -> Path:
    suffix = ".camlabel3d.csv"
    if csv_path.name.endswith(suffix):
        stem = csv_path.name[: -len(suffix)]
        return csv_path.with_name(f"{stem}.post{suffix}")
    return csv_path.with_suffix(csv_path.suffix + ".post.csv")


def _parse_scope(raw: str) -> ProcessingScope:
    lookup = {
        "current_frame": ProcessingScope.CURRENT_FRAME,
        "selected_track": ProcessingScope.SELECTED_TRACK,
        "global": ProcessingScope.GLOBAL,
    }
    return lookup[str(raw)]


def _parse_param_overrides(values: list[str]) -> dict[str, dict[str, float | int]]:
    overrides: dict[str, dict[str, float | int]] = {}
    for value in values:
        left, raw_number = value.split("=", 1)
        prefix, key = left.split(".", 1)
        overrides.setdefault(prefix.strip(), {})[key.strip()] = float(raw_number)
    return overrides


def _build_context(
    records,
    *,
    scope: ProcessingScope,
    frame_index: int | None,
    track_id: str,
    reproject_record=None,
) -> ProcessingContext:
    del scope
    return ProcessingContext(
        records=records,
        current_frame_index=frame_index,
        selected_track_id=track_id.strip(),
        reproject_record=reproject_record,
    )


def _build_cli_reprojector(args) -> callable:
    intrinsics = np.array(
        [
            [float(args.fx), 0.0, float(args.cx)],
            [0.0, float(args.fy), float(args.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    image_shape = (int(args.image_height), int(args.image_width))

    def reproject_record(record) -> None:
        x1, y1, x2, y2 = DetectorAdapter.project_record_to_box2d(
            record=record,
            intrinsics=intrinsics,
            image_shape=image_shape,
        )
        record.box2d_x1 = x1
        record.box2d_y1 = y1
        record.box2d_x2 = x2
        record.box2d_y2 = y2

    return reproject_record


def _require_scope_inputs(args, scope: ProcessingScope) -> None:
    if scope == ProcessingScope.CURRENT_FRAME and args.frame_index is None:
        raise SystemExit("--frame-index is required when --scope current_frame is used.")
    if scope == ProcessingScope.SELECTED_TRACK and not str(args.track_id or "").strip():
        raise SystemExit("--track-id is required when --scope selected_track is used.")


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze and postprocess CamLabel3D CSV files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_scope_arguments(target) -> None:
        target.add_argument("--csv", required=True, help="Input CamLabel3D CSV path.")
        target.add_argument(
            "--scope",
            choices=["current_frame", "selected_track", "global"],
            default="global",
            help="Processing scope.",
        )
        target.add_argument("--frame-index", type=int, default=None, help="Frame index for current_frame scope.")
        target.add_argument("--track-id", default="", help="Track ID for selected_track scope.")
        target.add_argument(
            "--rule",
            action="append",
            default=[],
            help="Enable one outlier rule by ID. Repeat to enable multiple rules. Defaults to all built-ins.",
        )
        target.add_argument(
            "--param",
            action="append",
            default=[],
            help="Override parameters with PREFIX.KEY=VALUE, for example yaw_spike.yaw_jump_deg=70.",
        )

    analyze = subparsers.add_parser("analyze", help="Analyze a CSV and output an outlier report JSON.")
    add_common_scope_arguments(analyze)
    analyze.add_argument("--output", default="", help="Optional report JSON path.")

    apply = subparsers.add_parser("apply", help="Apply outlier fixes and/or bulk operations to a CSV.")
    add_common_scope_arguments(apply)
    apply.add_argument(
        "--operation",
        action="append",
        default=[],
        help="Bulk operation ID to apply. Repeat to run multiple operations in order.",
    )
    apply.add_argument(
        "--fix-outliers",
        action="store_true",
        help="Apply all fixable outlier rules within the selected scope before running operations.",
    )
    apply.add_argument("--output", default="", help="Optional output CSV path.")
    apply.add_argument("--fx", type=float, default=None, help="Actual camera fx used for 2D reprojection.")
    apply.add_argument("--fy", type=float, default=None, help="Actual camera fy used for 2D reprojection.")
    apply.add_argument("--cx", type=float, default=None, help="Actual camera cx used for 2D reprojection.")
    apply.add_argument("--cy", type=float, default=None, help="Actual camera cy used for 2D reprojection.")
    apply.add_argument("--image-width", type=int, default=None, help="Image width used for 2D reprojection.")
    apply.add_argument("--image-height", type=int, default=None, help="Image height used for 2D reprojection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)

    engine = ProcessingEngine()
    csv_path = Path(args.csv).resolve()
    records = CSVStore(csv_path).load_records()
    scope = _parse_scope(args.scope)
    _require_scope_inputs(args, scope)

    enabled_rules = args.rule or [rule.rule_id for rule in engine.outlier_registry.all()]
    params_by_prefix = _parse_param_overrides(args.param)

    if args.command == "analyze":
        context = _build_context(
            records,
            scope=scope,
            frame_index=args.frame_index,
            track_id=args.track_id,
        )
        hits = engine.analyze_outliers(
            records=records,
            scope=OutlierScope(scope),
            enabled_rule_ids=enabled_rules,
            params_by_rule=params_by_prefix,
            context=context,
        )
        output_path = Path(args.output).resolve() if args.output else _derive_report_path(csv_path)
        output_path.write_text(
            hits_to_report_json(
                hits,
                csv_path=str(csv_path),
                scope=scope.value,
                enabled_rule_ids=enabled_rules,
            ),
            encoding="utf-8",
        )
        print(f"Outlier analysis complete: {len(hits)} hits -> {output_path}")
        return 0

    if not args.fix_outliers and not args.operation:
        raise SystemExit("Nothing to apply. Use --fix-outliers and/or --operation.")
    reprojection_args = (args.fx, args.fy, args.cx, args.cy, args.image_width, args.image_height)
    if any(value is None for value in reprojection_args):
        raise SystemExit("Applying geometry changes requires --fx --fy --cx --cy --image-width --image-height.")

    context = _build_context(
        records,
        scope=scope,
        frame_index=args.frame_index,
        track_id=args.track_id,
        reproject_record=_build_cli_reprojector(args),
    )

    messages: list[str] = []
    if args.fix_outliers:
        hits = engine.analyze_outliers(
            records=records,
            scope=OutlierScope(scope),
            enabled_rule_ids=enabled_rules,
            params_by_rule=params_by_prefix,
            context=context,
        )
        result = engine.fix_hits(records, hits, params_by_prefix, context)
        messages.append(result.message)

    for operation_id in args.operation:
        result = engine.apply_operation(
            operation_id=operation_id,
            records=records,
            scope=scope,
            params=params_by_prefix.get(operation_id),
            context=context,
        )
        messages.append(result.message)

    output_path = Path(args.output).resolve() if args.output else _derive_post_path(csv_path)
    CSVStore(output_path, backup_enabled=False).save_records(records)
    print(f"Postprocessing complete -> {output_path}")
    for message in messages:
        print(f"- {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
