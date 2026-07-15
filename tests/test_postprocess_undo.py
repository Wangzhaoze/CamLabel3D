from __future__ import annotations

import pytest

from camlabel3d.core.models import DetectionRecord
from camlabel3d.core.postprocess import PostprocessSession


def _record(value: float) -> DetectionRecord:
    return DetectionRecord(
        frame_index=int(value),
        category="car",
        score=0.9,
        score_2d=0.9,
        score_3d=0.9,
        box2d_x1=0.0,
        box2d_y1=0.0,
        box2d_x2=10.0,
        box2d_y2=10.0,
        center_x=value,
        center_y=0.0,
        center_z=10.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
        det_id=f"det-{value}",
    )


def test_undo_limit_keeps_only_the_newest_snapshots_in_lifo_order() -> None:
    session = PostprocessSession(undo_limit=2)
    for value in (1.0, 2.0, 3.0):
        session.push_undo_snapshot([_record(value)])

    assert session.undo()[0].center_x == 3.0
    assert session.undo()[0].center_x == 2.0
    assert session.has_undo() is False
    with pytest.raises(ValueError, match="Nothing to undo"):
        session.undo()


def test_undo_limit_has_a_minimum_of_one() -> None:
    session = PostprocessSession(undo_limit=0)
    session.push_undo_snapshot([_record(1.0)])
    session.push_undo_snapshot([_record(2.0)])

    assert session.undo_limit == 1
    assert session.undo()[0].center_x == 2.0
    assert session.has_undo() is False


def test_default_snapshot_clones_records_and_detaches_the_list() -> None:
    original_record = _record(1.0)
    source = [original_record]
    session = PostprocessSession()

    session.push_undo_snapshot(source)
    original_record.center_x = 99.0
    source.append(_record(2.0))
    restored = session.undo()

    assert len(restored) == 1
    assert restored[0].center_x == 1.0
    assert restored[0] is not original_record


def test_snapshot_owned_reuses_owned_records_but_detaches_the_list_container() -> None:
    owned_record = _record(1.0)
    owned_snapshot = [owned_record]
    session = PostprocessSession()

    session.push_undo_snapshot(owned_snapshot, snapshot_owned=True)
    owned_snapshot.append(_record(2.0))
    restored = session.undo()

    assert restored == [owned_record]
    assert restored is not owned_snapshot
    assert restored[0] is owned_record
