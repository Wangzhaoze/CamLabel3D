from __future__ import annotations

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.processing import OutlierScope, ProcessingContext, ProcessingEngine


def _record(
    frame_index: int,
    *,
    center_x: float,
    center_z: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    size_w: float,
    size_l: float,
    size_h: float,
    score_3d: float,
) -> DetectionRecord:
    return DetectionRecord(
        frame_index=frame_index,
        category="car",
        score=0.9,
        score_2d=0.9,
        score_3d=score_3d,
        box2d_x1=0.0,
        box2d_y1=0.0,
        box2d_x2=10.0,
        box2d_y2=10.0,
        center_x=center_x,
        center_y=0.0,
        center_z=center_z,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        size_w=size_w,
        size_l=size_l,
        size_h=size_h,
        track_id="1",
        track_status="auto",
        det_id=f"det-{frame_index}",
    )


def test_parallel_rule_analysis_matches_single_worker_results() -> None:
    records = [
        _record(
            0,
            center_x=0.0,
            center_z=10.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            size_w=2.0,
            size_l=4.0,
            size_h=1.5,
            score_3d=0.9,
        ),
        _record(
            1,
            center_x=8.0,
            center_z=20.0,
            yaw_deg=120.0,
            pitch_deg=100.0,
            roll_deg=-110.0,
            size_w=5.0,
            size_l=8.0,
            size_h=3.0,
            score_3d=0.1,
        ),
        _record(
            2,
            center_x=0.0,
            center_z=10.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            size_w=2.0,
            size_l=4.0,
            size_h=1.5,
            score_3d=0.9,
        ),
    ]
    rule_ids = ["center_spike", "roll_spike", "size_spike", "yaw_spike", "pitch_spike"]

    serial_hits = ProcessingEngine(max_workers=1).analyze_outliers(
        records,
        OutlierScope.GLOBAL,
        rule_ids,
        {},
        ProcessingContext(records=records),
    )
    parallel_hits = ProcessingEngine(max_workers=4).analyze_outliers(
        records,
        OutlierScope.GLOBAL,
        rule_ids,
        {},
        ProcessingContext(records=records),
    )

    assert {hit.rule_id for hit in serial_hits} == set(rule_ids)
    assert [hit.to_report_row() for hit in parallel_hits] == [hit.to_report_row() for hit in serial_hits]
