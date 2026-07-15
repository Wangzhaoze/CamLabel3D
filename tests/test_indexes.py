from __future__ import annotations

from camlabel3d.application.indexes import OutlierIndex, RecordIndex
from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.processing import OutlierHit


def _record(
    det_id: str,
    frame_index: int,
    *,
    score: float,
    track_id: str,
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
        center_x=0.0,
        center_y=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
        track_id=track_id,
        det_id=det_id,
    )


def _hit(rule_id: str, det_id: str, frame_index: int, track_id: str) -> OutlierHit:
    return OutlierHit(
        rule_id=rule_id,
        frame_index=frame_index,
        det_id=det_id,
        track_id=track_id,
        category="car",
        severity=2.0,
        message="test hit",
        fixable=True,
    )


def test_record_index_builds_stable_lookup_orders_and_isolates_returned_lists() -> None:
    low_score = _record("b", 2, score=0.5, track_id="2")
    high_score = _record("a", 2, score=0.9, track_id=" 1 ")
    earlier = _record("c", 0, score=0.7, track_id="1")
    untracked = _record("d", 1, score=0.8, track_id="   ")

    index = RecordIndex([low_score, high_score, earlier, untracked])

    assert index.revision == 1
    assert index.by_id("a") is high_score
    assert [record.det_id for record in index.for_frame(2)] == ["a", "b"]
    assert [record.det_id for record in index.for_track(" 1 ")] == ["c", "a"]
    assert index.for_track("") == []

    returned = index.for_frame(2)
    returned.clear()
    assert [record.det_id for record in index.for_frame(2)] == ["a", "b"]

    index.rebuild([untracked])
    assert index.revision == 2
    assert index.by_id("a") is None
    assert [record.det_id for record in index.for_frame(1)] == ["d"]


def test_outlier_index_groups_one_hit_across_all_relevant_dimensions() -> None:
    yaw = _hit("yaw_spike", "a", 4, "1")
    size = _hit("size_spike", "a", 4, "1")
    untracked = _hit("center_spike", "b", 9, "  ")

    index = OutlierIndex.build(hit for hit in (yaw, size, untracked))

    assert index.hits == [yaw, size, untracked]
    assert index.by_det_id == {"a": [yaw, size], "b": [untracked]}
    assert index.by_frame == {4: [yaw, size], 9: [untracked]}
    assert index.by_track_id == {"1": [yaw, size]}
    assert index.by_rule_id == {
        "yaw_spike": [yaw],
        "size_spike": [size],
        "center_spike": [untracked],
    }
    assert index.frames == {4, 9}
