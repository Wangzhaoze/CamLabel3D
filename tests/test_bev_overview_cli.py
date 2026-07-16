from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from camlabel3d.bev_overview_cli import main
from camlabel3d.core.models import DetectionRecord
from camlabel3d.io.csv_store import CSVStore


def _record(
    frame_index: int,
    *,
    det_id: str,
    track_id: str,
    center_x: float,
    center_z: float,
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
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        size_w=2.0,
        size_l=4.0,
        size_h=1.5,
        is_enabled=True,
        track_id=track_id,
        track_status="auto" if track_id else "",
        det_id=det_id,
    )


def test_bev_overview_cli_uses_defaults_and_writes_png() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "demo.camlabel3d.csv"
        CSVStore(csv_path, backup_enabled=False).save_records(
            [
                _record(1, det_id="b-1", track_id="B", center_x=0.0, center_z=15.0),
                _record(2, det_id="b-2", track_id="B", center_x=1.0, center_z=16.0),
                _record(3, det_id="b-3", track_id="B", center_x=2.0, center_z=17.0),
                _record(4, det_id="a-4", track_id="A", center_x=-2.0, center_z=12.0),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["--csv-path", str(csv_path)])

        output_path = csv_path.with_name("demo.bev_overview.png")
        assert code == 0
        assert output_path.exists()
        assert "track=B" in stdout.getvalue()
        assert "frame=2" in stdout.getvalue()
        with Image.open(output_path) as image:
            assert image.size == (1280, 720)


def test_bev_overview_cli_rejects_unknown_track_id() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "demo.camlabel3d.csv"
        CSVStore(csv_path, backup_enabled=False).save_records(
            [_record(1, det_id="b-1", track_id="B", center_x=0.0, center_z=15.0)]
        )

        try:
            main(["--csv-path", str(csv_path), "--track-id", "missing"])
        except SystemExit as exc:
            assert "Track 'missing' was not found" in str(exc)
        else:
            raise AssertionError("Expected SystemExit for an unknown track ID")
