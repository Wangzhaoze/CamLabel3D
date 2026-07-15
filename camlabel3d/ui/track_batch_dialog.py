"""Track-scoped batch CSV editor dialog."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camlabel3d.core import (
    DetectionRecord,
    TRACK_BATCH_NUMERIC_FIELDS,
    TrackBatchEditRequest,
    TrackBatchOperationKind,
    TrackSummary,
    track_batch_records,
)

from .input_widgets import MenuWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox

TRACK_BATCH_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("Frame", "frame_index", "int"),
    ("Category", "category", "str"),
    ("Score", "score", "float"),
    ("2D Score", "score_2d", "float"),
    ("3D Score", "score_3d", "float"),
    ("center_x", "center_x", "float"),
    ("center_y", "center_y", "float"),
    ("center_z", "center_z", "float"),
    ("yaw_deg", "yaw_deg", "float"),
    ("pitch_deg", "pitch_deg", "float"),
    ("roll_deg", "roll_deg", "float"),
    ("size_w", "size_w", "float"),
    ("size_l", "size_l", "float"),
    ("size_h", "size_h", "float"),
    ("Enabled", "is_enabled", "bool"),
    ("Track Status", "track_status", "str"),
)
TRACK_BATCH_EDITABLE_FIELDS = {
    "category",
    "score",
    "score_2d",
    "score_3d",
    "box2d_x1",
    "box2d_y1",
    "box2d_x2",
    "box2d_y2",
    "center_x",
    "center_y",
    "center_z",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "size_w",
    "size_l",
    "size_h",
    "is_enabled",
}
ITEM_DET_ID_ROLE = int(Qt.ItemDataRole.UserRole)
ITEM_ORIGINAL_VALUE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
TrackBatchTableEdit = tuple[str, str, str, object]
TRACK_BATCH_COLUMN_WIDTHS: tuple[int, ...] = (
    70,   # Frame
    140,  # Category
    90,   # Score
    90,   # 2D Score
    90,   # 3D Score
    95,   # center_x
    95,   # center_y
    95,   # center_z
    95,   # yaw
    95,   # pitch
    95,   # roll
    95,   # size_w
    95,   # size_l
    95,   # size_h
    70,   # enabled
    120,  # status
)


class TrackBatchEditorDialog(QDialog):
    """Modeless dialog for Excel-style edits on one track across a frame range."""

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        on_apply: Callable[[TrackBatchEditRequest], str | None] | None = None,
        on_undo: Callable[[str | None], str | None] | None = None,
        on_commit_edits: Callable[[list[TrackBatchTableEdit], str | None], str | None] | None = None,
        on_jump_frame: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Track Batch CSV Editor")
        self.resize(1280, 720)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self._on_apply = on_apply
        self._on_undo = on_undo
        self._on_commit_edits = on_commit_edits
        self._on_jump_frame = on_jump_frame
        self._records: list[DetectionRecord] = []
        self._track_summaries: list[TrackSummary] = []
        self._internal_change = False
        self._has_pending_table_edits = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.hint_label = QLabel(
            "Select one track, choose a frame range, then apply a numeric batch edit. "
            "Smoothing uses a centered moving average. Arithmetic operations support add/subtract/multiply/divide.",
            self,
        )
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)

        source_group = QGroupBox("Track Scope", self)
        source_layout = QFormLayout(source_group)
        source_layout.setContentsMargins(8, 10, 8, 8)
        source_layout.setSpacing(6)

        self.track_combo = MenuWheelComboBox(source_group)
        source_layout.addRow("Track ID", self.track_combo)

        range_row = QWidget(source_group)
        range_layout = QHBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(6)
        self.start_frame_spin = NoWheelSpinBox()
        self.start_frame_spin.setRange(0, 10_000_000)
        self.end_frame_spin = NoWheelSpinBox()
        self.end_frame_spin.setRange(0, 10_000_000)
        self.reset_range_btn = QPushButton("Use Full Track Range", range_row)
        range_layout.addWidget(self.start_frame_spin)
        range_layout.addWidget(QLabel("to", range_row))
        range_layout.addWidget(self.end_frame_spin)
        range_layout.addWidget(self.reset_range_btn)
        source_layout.addRow("Frames", range_row)

        self.range_info_label = QLabel("No track selected.", source_group)
        self.range_info_label.setWordWrap(True)
        source_layout.addRow("Info", self.range_info_label)
        root.addWidget(source_group)

        operation_group = QGroupBox("Batch Operation", self)
        operation_layout = QFormLayout(operation_group)
        operation_layout.setContentsMargins(8, 10, 8, 8)
        operation_layout.setSpacing(6)

        self.field_combo = MenuWheelComboBox(operation_group)
        for label, field_name in TRACK_BATCH_NUMERIC_FIELDS:
            self.field_combo.addItem(label, field_name)
        operation_layout.addRow("Column", self.field_combo)

        self.operation_combo = MenuWheelComboBox(operation_group)
        self.operation_combo.addItem("Smooth", TrackBatchOperationKind.SMOOTH)
        self.operation_combo.addItem("Add", TrackBatchOperationKind.ADD)
        self.operation_combo.addItem("Subtract", TrackBatchOperationKind.SUBTRACT)
        self.operation_combo.addItem("Multiply", TrackBatchOperationKind.MULTIPLY)
        self.operation_combo.addItem("Divide", TrackBatchOperationKind.DIVIDE)
        operation_layout.addRow("Operation", self.operation_combo)

        self.operand_spin = NoWheelDoubleSpinBox()
        self.operand_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.operand_spin.setDecimals(6)
        self.operand_spin.setSingleStep(0.1)
        self.operand_spin.setValue(0.0)
        operation_layout.addRow("Operand", self.operand_spin)

        self.smooth_window_spin = NoWheelSpinBox()
        self.smooth_window_spin.setRange(1, 999)
        self.smooth_window_spin.setValue(5)
        operation_layout.addRow("Smooth Window", self.smooth_window_spin)

        self.apply_btn = QPushButton("Apply Batch Operation", operation_group)
        operation_layout.addRow("", self.apply_btn)
        root.addWidget(operation_group)

        table_group = QGroupBox("Track Rows", self)
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(8, 10, 8, 8)
        table_layout.setSpacing(6)

        self.table = QTableWidget(table_group)
        self.table.setColumnCount(len(TRACK_BATCH_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([header for header, _, _ in TRACK_BATCH_TABLE_COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for index in range(len(TRACK_BATCH_TABLE_COLUMNS)):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(index, TRACK_BATCH_COLUMN_WIDTHS[index])
        table_layout.addWidget(self.table)

        self.table_status_label = QLabel("No rows loaded.", table_group)
        self.table_status_label.setWordWrap(True)
        table_layout.addWidget(self.table_status_label)
        root.addWidget(table_group, 1)

        footer = QWidget(self)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self.status_label = QLabel("", footer)
        self.status_label.setWordWrap(True)
        footer_layout.addWidget(self.status_label, 1)
        self.update_btn = QPushButton("Update", footer)
        footer_layout.addWidget(self.update_btn)
        self.undo_btn = QPushButton("Undo", footer)
        footer_layout.addWidget(self.undo_btn)
        self.close_btn = QPushButton("Close", footer)
        footer_layout.addWidget(self.close_btn)
        root.addWidget(footer)

        self.track_combo.currentIndexChanged.connect(self._on_track_changed)
        self.start_frame_spin.valueChanged.connect(self._on_range_changed)
        self.end_frame_spin.valueChanged.connect(self._on_range_changed)
        self.reset_range_btn.clicked.connect(self._reset_full_track_range)
        self.operation_combo.currentIndexChanged.connect(self._update_operation_ui)
        self.apply_btn.clicked.connect(self._apply_current_request)
        self.update_btn.clicked.connect(self._request_update)
        self.undo_btn.clicked.connect(self._request_undo)
        self.close_btn.clicked.connect(self.close)
        self.table.currentCellChanged.connect(self._on_table_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)

        self._update_operation_ui()
        self._update_action_state()

    def set_records(
        self,
        *,
        records: Sequence[DetectionRecord],
        track_summaries: Sequence[TrackSummary],
        selected_track_id: str | None = None,
    ) -> None:
        previous_track_id = self.current_track_id()
        previous_start = int(self.start_frame_spin.value())
        previous_end = int(self.end_frame_spin.value())
        preserve_range = selected_track_id is None and bool(previous_track_id)
        target_track_id = str(selected_track_id or previous_track_id or "").strip()

        self._records = list(records)
        self._track_summaries = list(track_summaries)
        summary_by_id = {summary.track_id: summary for summary in self._track_summaries}
        track_ids = [summary.track_id for summary in self._track_summaries]
        for record in self._records:
            track_id = str(record.track_id).strip()
            if track_id and track_id not in track_ids:
                track_ids.append(track_id)
        if not target_track_id and track_ids:
            target_track_id = track_ids[0]
        if target_track_id and target_track_id not in track_ids:
            target_track_id = track_ids[0] if track_ids else ""

        self._internal_change = True
        with QSignalBlocker(self.track_combo):
            self.track_combo.clear()
            for track_id in track_ids:
                summary = summary_by_id.get(track_id)
                if summary is not None:
                    label = f"{summary.track_id} | {summary.category} | {summary.first_frame}-{summary.last_frame}"
                else:
                    label = self._fallback_track_label(track_id)
                self.track_combo.addItem(label, track_id)
            if target_track_id:
                index = self.track_combo.findData(target_track_id)
                if index >= 0:
                    self.track_combo.setCurrentIndex(index)
        self._internal_change = False

        self._sync_range_to_track(
            preserve_values=preserve_range and target_track_id == previous_track_id,
            previous_start=previous_start,
            previous_end=previous_end,
        )
        self._refresh_table()
        self._update_action_state()

    def current_track_id(self) -> str:
        value = self.track_combo.currentData() or self.track_combo.currentText()
        return str(value or "").strip()

    def _track_records(self, *, selected_range_only: bool, enabled_only: bool = False) -> list[DetectionRecord]:
        track_id = self.current_track_id()
        if not track_id:
            return []
        if selected_range_only:
            return track_batch_records(
                self._records,
                track_id=track_id,
                frame_start=int(self.start_frame_spin.value()),
                frame_end=int(self.end_frame_spin.value()),
                enabled_only=enabled_only,
            )
        return track_batch_records(self._records, track_id=track_id, enabled_only=enabled_only)

    def _track_bounds(self, track_id: str) -> tuple[int, int] | None:
        track_records = track_batch_records(self._records, track_id=track_id, enabled_only=False)
        if not track_records:
            return None
        return (int(track_records[0].frame_index), int(track_records[-1].frame_index))

    def _fallback_track_label(self, track_id: str) -> str:
        track_records = track_batch_records(self._records, track_id=track_id, enabled_only=False)
        if not track_records:
            return track_id
        category = track_records[0].category
        start_frame = int(track_records[0].frame_index)
        end_frame = int(track_records[-1].frame_index)
        enabled_count = sum(1 for record in track_records if record.is_enabled)
        return f"{track_id} | {category} | {start_frame}-{end_frame} | enabled {enabled_count}/{len(track_records)}"

    def _sync_range_to_track(
        self,
        *,
        preserve_values: bool,
        previous_start: int,
        previous_end: int,
    ) -> None:
        track_id = self.current_track_id()
        bounds = self._track_bounds(track_id)
        self._internal_change = True
        with QSignalBlocker(self.start_frame_spin), QSignalBlocker(self.end_frame_spin):
            if bounds is None:
                self.start_frame_spin.setRange(0, 0)
                self.end_frame_spin.setRange(0, 0)
                self.start_frame_spin.setValue(0)
                self.end_frame_spin.setValue(0)
                self.range_info_label.setText("No enabled detections are available for this track.")
            else:
                minimum, maximum = bounds
                self.start_frame_spin.setRange(minimum, maximum)
                self.end_frame_spin.setRange(minimum, maximum)
                if preserve_values:
                    start_value = min(max(previous_start, minimum), maximum)
                    end_value = min(max(previous_end, minimum), maximum)
                else:
                    start_value = minimum
                    end_value = maximum
                self.start_frame_spin.setValue(start_value)
                self.end_frame_spin.setValue(end_value)
                self.range_info_label.setText(
                    f"Full track span: frames {minimum} to {maximum}. "
                    f"Batch operations use the selected frame range, while the table below always shows the full track."
                )
        self._internal_change = False

    def _update_operation_ui(self) -> None:
        operation = self.operation_combo.currentData()
        is_smooth = operation == TrackBatchOperationKind.SMOOTH
        self.operand_spin.setEnabled(not is_smooth)
        self.smooth_window_spin.setEnabled(is_smooth)
        if is_smooth:
            self.status_label.setText("Smooth applies a centered moving average over the selected frame range.")
        else:
            self.status_label.setText("")
        self._update_action_state()

    def _on_track_changed(self) -> None:
        if self._internal_change:
            return
        self._sync_range_to_track(preserve_values=False, previous_start=0, previous_end=0)
        self._refresh_table()
        self._update_action_state()

    def _reset_full_track_range(self) -> None:
        bounds = self._track_bounds(self.current_track_id())
        if bounds is None:
            return
        self._internal_change = True
        with QSignalBlocker(self.start_frame_spin), QSignalBlocker(self.end_frame_spin):
            self.start_frame_spin.setValue(bounds[0])
            self.end_frame_spin.setValue(bounds[1])
        self._internal_change = False
        self._update_action_state()

    def _on_range_changed(self) -> None:
        if self._internal_change:
            return
        self._refresh_table_status()
        self._update_action_state()

    def _refresh_table(self) -> None:
        track_rows = self._track_records(selected_range_only=False, enabled_only=False)
        self._internal_change = True
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(track_rows))
            for row_index, record in enumerate(track_rows):
                for col_index, (_, field_name, value_kind) in enumerate(TRACK_BATCH_TABLE_COLUMNS):
                    item = QTableWidgetItem()
                    item.setData(ITEM_DET_ID_ROLE, record.det_id)
                    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    editable = field_name in TRACK_BATCH_EDITABLE_FIELDS
                    if value_kind == "bool":
                        if editable:
                            flags |= Qt.ItemFlag.ItemIsUserCheckable
                        item.setFlags(flags)
                        checked = bool(getattr(record, field_name))
                        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                        item.setData(ITEM_ORIGINAL_VALUE_ROLE, checked)
                    else:
                        if editable:
                            flags |= Qt.ItemFlag.ItemIsEditable
                        item.setFlags(flags)
                        text = self._format_field(record, field_name, value_kind)
                        item.setText(text)
                        item.setData(ITEM_ORIGINAL_VALUE_ROLE, text)
                    self.table.setItem(row_index, col_index, item)
            self.table.clearSelection()
        self._internal_change = False
        self._has_pending_table_edits = False
        self._refresh_table_status()

    def sync_from_records(
        self,
        *,
        records: Sequence[DetectionRecord],
        track_summaries: Sequence[TrackSummary],
        selected_track_id: str | None = None,
        changed_fields: Sequence[str] | None = None,
    ) -> None:
        current_track_id = self.current_track_id()
        target_track_id = str(selected_track_id or current_track_id or "").strip()
        previous_rows = self.table.rowCount()
        self._records = list(records)
        self._track_summaries = list(track_summaries)
        track_rows = self._track_records(selected_range_only=False, enabled_only=False)
        changed_field_set = {str(field_name) for field_name in (changed_fields or []) if str(field_name).strip()}
        requires_full_reset = any(field_name in {"category", "is_enabled", "track_status"} for field_name in changed_field_set)
        if (
            not target_track_id
            or target_track_id != current_track_id
            or previous_rows != len(track_rows)
            or self._has_pending_table_edits
            or requires_full_reset
        ):
            self.set_records(
                records=records,
                track_summaries=track_summaries,
                selected_track_id=selected_track_id,
            )
            return

        self._internal_change = True
        with QSignalBlocker(self.table):
            for row_index, record in enumerate(track_rows):
                for col_index, (_, field_name, value_kind) in enumerate(TRACK_BATCH_TABLE_COLUMNS):
                    if changed_field_set and field_name not in changed_field_set:
                        continue
                    item = self.table.item(row_index, col_index)
                    if item is None:
                        self._internal_change = False
                        self.set_records(
                            records=records,
                            track_summaries=track_summaries,
                            selected_track_id=selected_track_id,
                        )
                        return
                    item.setData(ITEM_DET_ID_ROLE, record.det_id)
                    if value_kind == "bool":
                        checked = bool(getattr(record, field_name))
                        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                        item.setData(ITEM_ORIGINAL_VALUE_ROLE, checked)
                    else:
                        text = self._format_field(record, field_name, value_kind)
                        item.setText(text)
                        item.setData(ITEM_ORIGINAL_VALUE_ROLE, text)
        self._internal_change = False
        self._has_pending_table_edits = False
        self._refresh_table_status()

    def _refresh_table_status(self) -> None:
        track_rows = self._track_records(selected_range_only=False, enabled_only=False)
        range_rows = self._track_records(selected_range_only=True, enabled_only=False)

        if not self.current_track_id():
            self.table_status_label.setText("Select one track first.")
        elif not track_rows:
            self.table_status_label.setText("No rows exist for the selected track.")
        else:
            self.table_status_label.setText(
                f"Showing all {len(track_rows)} row(s) for track {self.current_track_id()}. "
                f"Batch operations will apply only to frames {self.start_frame_spin.value()} to {self.end_frame_spin.value()} "
                f"({len(range_rows)} row(s) in range)."
            )

    @staticmethod
    def _format_field(record: DetectionRecord, field_name: str, value_kind: str) -> str:
        value = getattr(record, field_name)
        if value_kind == "float":
            return f"{float(value):.6f}"
        if value_kind == "int":
            return str(int(value))
        if value_kind == "bool":
            return "1" if bool(value) else "0"
        return str(value)

    def _on_table_selection_changed(self) -> None:
        if self._on_jump_frame is None:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        frame_text = item.text().strip()
        if not frame_text:
            return
        self._on_jump_frame(int(frame_text))

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._internal_change:
            return
        column = item.column()
        if not (0 <= column < len(TRACK_BATCH_TABLE_COLUMNS)):
            return
        _, field_name, value_kind = TRACK_BATCH_TABLE_COLUMNS[column]
        if field_name not in TRACK_BATCH_EDITABLE_FIELDS:
            return
        self._has_pending_table_edits = True
        self.status_label.setText("Pending table edits. Click Update to apply them.")
        self._update_action_state()

    def _apply_current_request(self) -> None:
        if self._on_apply is None:
            return
        track_id = self.current_track_id()
        if not track_id:
            self.status_label.setText("Select a track before applying a batch edit.")
            return
        field_name = str(self.field_combo.currentData() or "")
        operation = self.operation_combo.currentData()
        if not field_name or operation is None:
            self.status_label.setText("Select both a column and an operation first.")
            return

        request = TrackBatchEditRequest(
            track_id=track_id,
            field_name=field_name,
            operation=TrackBatchOperationKind(operation),
            frame_start=int(self.start_frame_spin.value()),
            frame_end=int(self.end_frame_spin.value()),
            operand=float(self.operand_spin.value()),
            smooth_window=int(self.smooth_window_spin.value()),
        )
        message = self._on_apply(request)
        if message:
            self.status_label.setText(message)
        self._update_action_state()

    def _collect_pending_table_edits(self) -> list[TrackBatchTableEdit] | None:
        track_rows = self._track_records(selected_range_only=False, enabled_only=False)
        edits: list[TrackBatchTableEdit] = []
        for row_index, record in enumerate(track_rows):
            for col_index, (_, field_name, value_kind) in enumerate(TRACK_BATCH_TABLE_COLUMNS):
                if field_name not in TRACK_BATCH_EDITABLE_FIELDS:
                    continue
                item = self.table.item(row_index, col_index)
                if item is None:
                    continue
                det_id = str(item.data(ITEM_DET_ID_ROLE) or "").strip()
                if not det_id:
                    continue
                original_value = item.data(ITEM_ORIGINAL_VALUE_ROLE)
                if value_kind == "bool":
                    proposed_value = item.checkState() == Qt.CheckState.Checked
                    if bool(proposed_value) == bool(original_value):
                        continue
                else:
                    proposed_value = item.text().strip()
                    if str(proposed_value) == str(original_value):
                        continue
                    if value_kind == "float":
                        try:
                            numeric = float(str(proposed_value))
                        except Exception:
                            self.status_label.setText(
                                f"Invalid numeric value in row {row_index + 1}, column {field_name}."
                            )
                            return None
                        if not np.isfinite(numeric):
                            self.status_label.setText(
                                f"Non-finite numeric value in row {row_index + 1}, column {field_name}."
                            )
                            return None
                edits.append((det_id, field_name, value_kind, proposed_value))
        return edits

    def _request_update(self) -> None:
        edits = self._collect_pending_table_edits()
        if edits is None:
            return
        if not edits:
            self._refresh_table()
            self._update_action_state()
            self.status_label.setText("No local table edits to apply.")
            return
        if self._on_commit_edits is None:
            self.status_label.setText("No commit handler is available for table edits.")
            return
        message = self._on_commit_edits(edits, self.current_track_id() or None)
        self._refresh_table()
        if message:
            self.status_label.setText(message)
        self._update_action_state()

    def _request_undo(self) -> None:
        if self._on_undo is None:
            self.status_label.setText("Undo is not available here.")
            return
        message = self._on_undo(self.current_track_id() or None)
        if message:
            self.status_label.setText(message)
        self._update_action_state()

    def _update_action_state(self) -> None:
        has_track = bool(self.current_track_id())
        has_rows = bool(self._track_records(selected_range_only=False, enabled_only=False))
        has_range_rows = bool(self._track_records(selected_range_only=True, enabled_only=False))
        valid_range = int(self.start_frame_spin.value()) <= int(self.end_frame_spin.value())
        self.apply_btn.setEnabled(has_track and has_rows and has_range_rows and valid_range)
        self.update_btn.setEnabled(has_track and has_rows)
        self.undo_btn.setEnabled(True)
