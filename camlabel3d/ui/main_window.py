"""Main PySide6 window for single-frame 3D detection review."""

from __future__ import annotations

import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PySide6.QtCore import QSignalBlocker, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QStyle,
)

from camlabel3d.core import (
    BulkOperationRegistry,
    DatasetConfigStore,
    DatasetSourcesConfig,
    DetectionConfig,
    DetectionRecord,
    DetectorAdapter,
    FilterConfig,
    ImageFolderFrameProvider,
    OperationScope,
    OutlierHit,
    OutlierRuleRegistry,
    OutlierScope,
    ParameterSpec,
    PointPrompt,
    PostprocessSession,
    ProcessingContext,
    ProcessingEngine,
    ProcessingScope,
    PromptMode,
    PromptSpec,
    SourceContext,
    SourceMode,
    TrackSummary,
    TrackingConfig,
    VideoFrameProvider,
    WorkflowStage,
    build_default_bulk_operation_registry,
    build_default_outlier_registry,
    clone_records,
)

from .bookmark_slider import BookmarkSlider
from .canvas import FrameCanvas
from .input_widgets import MenuWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from .worker import DetectionWorker, TrackingWorker

RESULTS_TABLE_COLUMNS: list[tuple[str, str, str, bool]] = [
    ("Show", "is_enabled", "bool", True),
    ("Category", "category", "str", True),
    ("Score", "score", "score", False),
    ("2D Score", "score_2d", "score", False),
    ("3D Score", "score_3d", "score", False),
    ("center_x", "center_x", "float", True),
    ("center_y", "center_y", "float", True),
    ("center_z", "center_z", "float", True),
    ("yaw_deg", "yaw_deg", "float", True),
    ("pitch_deg", "pitch_deg", "float", True),
    ("roll_deg", "roll_deg", "float", True),
    ("size_w", "size_w", "float", True),
    ("size_l", "size_l", "float", True),
    ("size_h", "size_h", "float", True),
    ("track_id", "track_id", "str", True),
    ("track_status", "track_status", "str", False),
    ("Flags", "_outlier_flags", "computed", False),
]

TRACK_TABLE_COLUMNS = ("track_id", "category", "enabled_count", "first_frame", "last_frame", "status")
OUTLIER_TABLE_COLUMNS = ("frame_index", "track_id", "category", "rule_id", "severity", "fixable", "message")

GEOMETRY_RESULT_FIELDS = {
    "center_x",
    "center_y",
    "center_z",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "size_w",
    "size_l",
    "size_h",
}

OUTLIER_RULE_COLORS: dict[str, QColor] = {
    "yaw_spike": QColor(232, 88, 88),
    "pitch_spike": QColor(245, 166, 35),
    "roll_spike": QColor(170, 108, 255),
    "size_spike": QColor(64, 156, 255),
    "center_spike": QColor(74, 201, 126),
}


class ActivitySection(str, Enum):
    FILE = "File"
    DETECT = "Detect"
    TRACK = "Track"


class MainWindow(QMainWindow):
    """CamLabel3D desktop UI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CamLabel3D - WildDet3D Detection Review")
        self.resize(1840, 1100)

        self.detector = DetectorAdapter()
        self.dataset_config_store = DatasetConfigStore()
        self.postprocess_session = PostprocessSession()
        self.outlier_registry: OutlierRuleRegistry = build_default_outlier_registry()
        self.bulk_operation_registry: BulkOperationRegistry = build_default_bulk_operation_registry()
        self.processing_engine = ProcessingEngine(
            outlier_registry=self.outlier_registry,
            bulk_operation_registry=self.bulk_operation_registry,
        )
        self.dataset_config: DatasetSourcesConfig | None = None

        self.current_provider: ImageFolderFrameProvider | VideoFrameProvider | None = None
        self.current_source_context: SourceContext | None = None
        self.current_raw_csv_path: Path | None = None
        self.current_latest_csv_path: Path | None = None
        self.current_output_csv_path: Path | None = None
        self.current_source_output_csv_path: Path | None = None
        self.current_result_dir: Path | None = None
        self.current_active_source_path: Path | None = None
        self.current_annotation_csv_override_path: Path | None = None
        self.manual_source_path: Path | None = None
        self.selected_dataset_id = ""
        self.selected_recording_id = ""

        self.records: list[DetectionRecord] = []
        self.current_frame_index = 0
        self.current_prompt_box: tuple[float, float, float, float] | None = None
        self.current_prompt_points: list[PointPrompt] = []
        self.copied_range_prompt: PromptSpec | None = None
        self.current_frame_det_ids: list[str] = []

        self.detection_worker: DetectionWorker | None = None
        self.tracking_worker: TrackingWorker | None = None
        self._controls_locked = False
        self._results_table_internal_change = False
        self._outlier_rule_table_internal_change = False
        self._source_ui_guard = False
        self.current_track_ids: list[str] = []
        self.outlier_hits_global: list[OutlierHit] = []
        self.outlier_hits_visible: list[OutlierHit] = []
        self.outlier_hits_by_det_id: dict[str, list[OutlierHit]] = {}
        self.outlier_hits_by_frame: dict[int, list[OutlierHit]] = {}
        self.outlier_hits_by_track_id: dict[str, list[OutlierHit]] = {}
        self.outlier_hits_by_rule_id: dict[str, list[OutlierHit]] = {}
        self.outlier_frames: set[int] = set()
        self.outlier_rule_enabled: dict[str, bool] = {
            rule.rule_id: bool(rule.default_enabled)
            for rule in self.outlier_registry.all()
        }
        self.outlier_rule_params: dict[str, dict[str, float | int]] = {
            rule.rule_id: rule.default_params()
            for rule in self.outlier_registry.all()
        }
        self.bulk_operation_params: dict[str, dict[str, float | int]] = {
            operation.operation_id: operation.default_params()
            for operation in self.bulk_operation_registry.all()
        }
        self._outlier_param_widgets: dict[str, QWidget] = {}
        self._bulk_param_widgets: dict[str, QWidget] = {}
        self.activity_section = ActivitySection.FILE
        self._activity_buttons: dict[ActivitySection, QToolButton] = {}
        self._activity_page_indices: dict[ActivitySection, int] = {}
        self._side_panel_visible = True
        self._side_panel_last_width = 420
        self._results_panel_last_height = 280

        self.model_status_text = "Model idle (loads on detect)"

        self._build_ui()
        self._connect_signals()
        self._initialize_processing_controls()
        self._sync_prompt_ui()
        self._on_intrinsics_mode_changed()
        self._set_idle_progress()

        self._load_dataset_config(initial=True)
        self._apply_source_mode_ui()
        self._refresh_info_panel()
        self._update_action_states()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        self.activity_bar = self._build_activity_bar(central)
        root_layout.addWidget(self.activity_bar)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self.workspace_splitter.setChildrenCollapsible(True)
        root_layout.addWidget(self.workspace_splitter, stretch=1)

        self.side_panel_container = self._build_side_panel_container(central)
        self.side_panel_container.setMinimumWidth(320)

        self.main_view_splitter = QSplitter(Qt.Orientation.Vertical, central)
        self.main_view_splitter.setChildrenCollapsible(True)

        preview_container = QWidget(central)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_layout.setSpacing(8)

        self.canvas = FrameCanvas(preview_container)
        preview_layout.addWidget(self.canvas, stretch=1)

        self.frame_info_label = QLabel("Frame: -- / --", preview_container)
        self.time_info_label = QLabel("t = --", preview_container)
        info_row = QHBoxLayout()
        info_row.addWidget(self.frame_info_label)
        info_row.addStretch(1)
        info_row.addWidget(self.time_info_label)
        preview_layout.addLayout(info_row)

        self.navigation_widget = QWidget(preview_container)
        nav_layout = QHBoxLayout(self.navigation_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)
        self.prev_frame_btn = QPushButton("Prev")
        self.next_frame_btn = QPushButton("Next")
        self.frame_slider = BookmarkSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_index_spin = NoWheelSpinBox()
        self.frame_index_spin.setMinimum(0)
        self.frame_index_spin.setMaximum(0)
        nav_layout.addWidget(self.prev_frame_btn)
        nav_layout.addWidget(self.frame_slider, stretch=1)
        nav_layout.addWidget(self.next_frame_btn)
        nav_layout.addWidget(self.frame_index_spin)
        preview_layout.addWidget(self.navigation_widget)

        self.source_group = self._build_source_group(self.side_panel_container)
        self.prompt_group = self._build_prompt_group(self.side_panel_container)
        self.detection_params_group = self._build_detection_group(self.side_panel_container)
        self.run_group = self._build_run_group(self.side_panel_container)
        self.postprocess_group = self._build_postprocess_group(self.side_panel_container)
        self.outlier_group = self._build_outlier_group(self.side_panel_container)
        self.bulk_ops_group = self._build_bulk_ops_group(self.side_panel_container)
        self.track_group = self._build_track_group(self.side_panel_container)
        self.info_group = self._build_info_group(self.side_panel_container)
        self.results_group = self._build_results_group(central)

        self._populate_side_panel_pages()

        self.workspace_splitter.addWidget(self.side_panel_container)
        self.workspace_splitter.addWidget(self.main_view_splitter)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([self._side_panel_last_width, 1360])

        self.main_view_splitter.addWidget(preview_container)
        self.main_view_splitter.addWidget(self.results_group)
        self.main_view_splitter.setStretchFactor(0, 1)
        self.main_view_splitter.setStretchFactor(1, 0)
        self.main_view_splitter.setSizes([820, self._results_panel_last_height])

        self.workspace_splitter.splitterMoved.connect(self._on_workspace_splitter_moved)
        self.main_view_splitter.splitterMoved.connect(self._on_main_view_splitter_moved)
        self._set_activity_section(ActivitySection.FILE, allow_toggle=False)

        status = QStatusBar(self)
        self.setStatusBar(status)

    def _build_activity_bar(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        widget.setFixedWidth(56)
        widget.setStyleSheet(
            """
            QToolButton {
                border: none;
                padding: 10px;
                border-radius: 6px;
            }
            QToolButton:checked {
                background-color: rgba(90, 140, 220, 0.22);
            }
            """
        )
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)

        icons = {
            ActivitySection.FILE: self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            ActivitySection.DETECT: self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            ActivitySection.TRACK: self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
        }
        for section in ActivitySection:
            button = QToolButton(widget)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIcon(icons[section])
            button.setIconSize(QSize(22, 22))
            button.setToolTip(section.value)
            button.setAutoRaise(True)
            button.clicked.connect(lambda _checked=False, selected=section: self._set_activity_section(selected, allow_toggle=True))
            layout.addWidget(button)
            self._activity_buttons[section] = button
        layout.addStretch(1)
        return widget

    def _build_side_panel_container(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        self.side_panel_title_label = QLabel(widget)
        self.side_panel_title_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.side_panel_title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.side_panel_title_label)

        self.side_panel_stack = QStackedWidget(widget)
        layout.addWidget(self.side_panel_stack, stretch=1)
        return widget

    def _populate_side_panel_pages(self) -> None:
        pages = {
            ActivitySection.FILE: [self.source_group, self.info_group],
            ActivitySection.DETECT: [self.prompt_group, self.detection_params_group, self.run_group],
            ActivitySection.TRACK: [self.postprocess_group, self.outlier_group, self.bulk_ops_group, self.track_group],
        }
        for section, groups in pages.items():
            page = self._build_side_panel_page(groups, title=section.value)
            self._activity_page_indices[section] = self.side_panel_stack.addWidget(page)

    def _build_side_panel_page(self, groups: Sequence[QWidget], title: str) -> QScrollArea:
        scroll = QScrollArea(self.side_panel_stack)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName(f"{title.lower()}PanelPage")

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for group in groups:
            layout.addWidget(group)
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _set_activity_section(self, section: ActivitySection, allow_toggle: bool) -> None:
        if allow_toggle and section == self.activity_section:
            self._set_side_panel_visible(not self._side_panel_visible)
            return

        self.activity_section = section
        page_index = self._activity_page_indices.get(section, 0)
        self.side_panel_stack.setCurrentIndex(page_index)
        self.side_panel_title_label.setText(section.value.upper())
        self._set_side_panel_visible(True)
        self._sync_activity_buttons()

    def _set_side_panel_visible(self, visible: bool) -> None:
        if visible == self._side_panel_visible:
            self._sync_activity_buttons()
            return

        if not visible:
            sizes = self.workspace_splitter.sizes()
            if len(sizes) >= 2 and sizes[0] > 0:
                self._side_panel_last_width = sizes[0]
            total_width = max(sum(sizes), 1)
            self.side_panel_container.setVisible(False)
            self.workspace_splitter.setSizes([0, total_width])
            self._side_panel_visible = False
            self._sync_activity_buttons()
            return

        self.side_panel_container.setVisible(True)
        total_width = max(sum(self.workspace_splitter.sizes()), self.workspace_splitter.width(), 640)
        side_width = min(max(self._side_panel_last_width, self.side_panel_container.minimumWidth()), max(total_width - 240, 240))
        self.workspace_splitter.setSizes([side_width, max(total_width - side_width, 240)])
        self._side_panel_visible = True
        self._sync_activity_buttons()

    def _sync_activity_buttons(self) -> None:
        for section, button in self._activity_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(section == self.activity_section)

    def _on_workspace_splitter_moved(self, _pos: int, _index: int) -> None:
        if not self._side_panel_visible:
            return
        sizes = self.workspace_splitter.sizes()
        if len(sizes) >= 2 and sizes[0] > 0:
            self._side_panel_last_width = sizes[0]

    def _on_main_view_splitter_moved(self, _pos: int, _index: int) -> None:
        sizes = self.main_view_splitter.sizes()
        if len(sizes) >= 2 and sizes[1] > 0:
            self._results_panel_last_height = sizes[1]

    def _expand_results_panel_if_needed(self, force: bool = False) -> None:
        if not hasattr(self, "main_view_splitter") or not self.records:
            return
        sizes = self.main_view_splitter.sizes()
        if len(sizes) < 2:
            return
        bottom_height = sizes[1]
        if force or bottom_height <= 0:
            total_height = max(sum(sizes), self.main_view_splitter.height(), 400)
            desired_bottom = min(max(self._results_panel_last_height, 220), max(total_height - 140, 140))
            self.main_view_splitter.setSizes([max(total_height - desired_bottom, 140), desired_bottom])

    def _build_source_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("File", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        data_group = QGroupBox("Data Source", group)
        data_layout = QVBoxLayout(data_group)
        data_layout.setContentsMargins(8, 10, 8, 8)
        data_layout.setSpacing(8)

        mode_layout = QFormLayout()
        self.source_mode_combo = MenuWheelComboBox(group)
        for mode in SourceMode:
            self.source_mode_combo.addItem(mode.value, mode.value)
        mode_layout.addRow("Mode", self.source_mode_combo)
        data_layout.addLayout(mode_layout)

        self.dataset_section = QWidget(data_group)
        dataset_layout = QFormLayout(self.dataset_section)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(6)

        self.config_path_edit = QLineEdit(self.dataset_section)
        self.config_path_edit.setReadOnly(True)
        dataset_layout.addRow("Config", self.config_path_edit)

        self.edit_config_btn = QPushButton("Edit Config", self.dataset_section)
        self.reload_config_btn = QPushButton("Reload Config", self.dataset_section)
        dataset_layout.addRow("", self._row_widget(self.edit_config_btn, self.reload_config_btn))

        self.dataset_combo = MenuWheelComboBox(self.dataset_section)
        self.recording_combo = MenuWheelComboBox(self.dataset_section)
        dataset_layout.addRow("Dataset", self.dataset_combo)
        dataset_layout.addRow("Recording", self.recording_combo)
        data_layout.addWidget(self.dataset_section)

        self.manual_section = QWidget(data_group)
        manual_layout = QFormLayout(self.manual_section)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)

        self.manual_path_edit = QLineEdit(self.manual_section)
        self.manual_path_edit.setReadOnly(True)
        manual_layout.addRow("Media", self.manual_path_edit)
        self.open_video_btn = QPushButton("Open Video", self.manual_section)
        self.open_folder_btn = QPushButton("Open Image Folder", self.manual_section)
        manual_layout.addRow("", self._row_widget(self.open_video_btn, self.open_folder_btn))
        data_layout.addWidget(self.manual_section)

        source_path_form = QFormLayout()
        self.active_source_path_edit = QLineEdit(data_group)
        self.active_source_path_edit.setReadOnly(True)
        source_path_form.addRow("Active Path", self.active_source_path_edit)
        data_layout.addLayout(source_path_form)

        layout.addWidget(data_group)

        annotation_group = QGroupBox("Annotation CSV", group)
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.setContentsMargins(8, 10, 8, 8)
        annotation_layout.setSpacing(8)

        self.annotation_csv_hint_label = QLabel(
            "Load an existing annotation CSV for the current media source, or leave it empty to use the source-derived CSV. "
            "If the source has not been annotated yet, a blank CSV will be created automatically.",
            annotation_group,
        )
        self.annotation_csv_hint_label.setWordWrap(True)
        annotation_layout.addWidget(self.annotation_csv_hint_label)

        annotation_form = QFormLayout()
        self.annotation_csv_path_edit = QLineEdit(annotation_group)
        self.annotation_csv_path_edit.setPlaceholderText("Optional existing CSV path for this source")
        annotation_form.addRow("Load From", self.annotation_csv_path_edit)

        self.output_csv_edit = QLineEdit(annotation_group)
        self.output_csv_edit.setReadOnly(True)
        annotation_form.addRow("Active CSV", self.output_csv_edit)
        annotation_layout.addLayout(annotation_form)

        self.browse_annotation_csv_btn = QPushButton("Browse CSV", annotation_group)
        self.load_annotation_csv_btn = QPushButton("Load CSV", annotation_group)
        self.use_source_csv_btn = QPushButton("Use Source CSV", annotation_group)
        annotation_layout.addWidget(
            self._row_widget(self.browse_annotation_csv_btn, self.load_annotation_csv_btn, self.use_source_csv_btn)
        )

        self.open_result_folder_btn = QPushButton("Open Result Folder", annotation_group)
        self.save_csv_btn = QPushButton("Save CSV Now", annotation_group)
        annotation_layout.addWidget(self._row_widget(self.open_result_folder_btn, self.save_csv_btn))

        layout.addWidget(annotation_group)
        return group

    def _build_prompt_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Prompt", parent)
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.prompt_mode_combo = MenuWheelComboBox(group)
        for mode in PromptMode:
            self.prompt_mode_combo.addItem(mode.value, mode.value)
        layout.addRow("Mode", self.prompt_mode_combo)

        self.text_prompt_edit = QLineEdit(group)
        layout.addRow("Text Prompt", self.text_prompt_edit)

        self.prompt_label_edit = QLineEdit(group)
        layout.addRow("Prompt Label", self.prompt_label_edit)

        self.point_label_combo = MenuWheelComboBox(group)
        self.point_label_combo.addItem("Positive", 1)
        self.point_label_combo.addItem("Negative", 0)
        layout.addRow("Point Label", self.point_label_combo)

        self.prompt_hint_label = QLabel(group)
        self.prompt_hint_label.setWordWrap(True)
        layout.addRow("Hint", self.prompt_hint_label)

        self.copied_prompt_label = QLabel("No copied range prompt", group)
        self.copied_prompt_label.setWordWrap(True)
        layout.addRow("Range Prompt", self.copied_prompt_label)

        self.clear_prompt_btn = QPushButton("Clear Current Prompt", group)
        self.copy_prompt_btn = QPushButton("Copy Current Prompt To Range", group)
        layout.addRow("", self._row_widget(self.clear_prompt_btn, self.copy_prompt_btn))
        return group

    def _build_detection_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Detection Parameters", parent)
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.start_frame_spin = self._make_int_spinbox(0, 0, 0)
        self.end_frame_spin = self._make_int_spinbox(0, 0, 0)
        self.frame_step_spin = self._make_int_spinbox(1, 1, 999999)
        self.score_threshold_spin = self._make_float_spinbox(0.3, 0.0, 1.0, 0.01, 3)
        self.score_3d_threshold_spin = self._make_float_spinbox(0.1, 0.0, 1.0, 0.01, 3)
        self.nms_iou_spin = self._make_float_spinbox(0.8, 0.0, 1.0, 0.01, 3)

        self.use_actual_k_checkbox = QCheckBox("Use Actual Intrinsics", group)
        self.fx_spin = self._make_float_spinbox(0.0, 0.0, 1_000_000.0, 1.0, 6)
        self.fy_spin = self._make_float_spinbox(0.0, 0.0, 1_000_000.0, 1.0, 6)
        self.cx_spin = self._make_float_spinbox(0.0, 0.0, 1_000_000.0, 1.0, 6)
        self.cy_spin = self._make_float_spinbox(0.0, 0.0, 1_000_000.0, 1.0, 6)

        layout.addRow("Start Frame", self.start_frame_spin)
        layout.addRow("End Frame", self.end_frame_spin)
        layout.addRow("Frame Step", self.frame_step_spin)
        layout.addRow("Score Threshold", self.score_threshold_spin)
        layout.addRow("3D Score Threshold", self.score_3d_threshold_spin)
        layout.addRow("Cross-Category NMS IoU", self.nms_iou_spin)
        layout.addRow("", self.use_actual_k_checkbox)
        layout.addRow("fx", self.fx_spin)
        layout.addRow("fy", self.fy_spin)
        layout.addRow("cx", self.cx_spin)
        layout.addRow("cy", self.cy_spin)
        return group

    def _build_run_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Run", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.run_current_btn = QPushButton("Run Current Frame", group)
        self.run_range_btn = QPushButton("Run Selected Range", group)
        self.cancel_run_btn = QPushButton("Cancel", group)
        layout.addWidget(self._row_widget(self.run_current_btn, self.run_range_btn, self.cancel_run_btn))

        self.progress_bar = QProgressBar(group)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        return group

    def _build_postprocess_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Postprocessing", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.stage_label = QLabel("Stage: Detection stage", group)
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        self.start_postprocess_btn = QPushButton("Start Postprocessing", group)
        self.run_tracking_btn = QPushButton("Run Tracking", group)
        layout.addWidget(self._row_widget(self.start_postprocess_btn, self.run_tracking_btn))

        self.undo_change_btn = QPushButton("Undo Last Change", group)
        self.save_latest_btn = QPushButton("Save Latest", group)
        self.reset_to_raw_btn = QPushButton("Reset to Raw", group)
        layout.addWidget(self._row_widget(self.undo_change_btn, self.save_latest_btn, self.reset_to_raw_btn))

        filter_form = QFormLayout()
        self.filter_min_score_spin = self._make_float_spinbox(0.0, 0.0, 1.0, 0.01, 3)
        self.filter_min_score_3d_spin = self._make_float_spinbox(0.0, 0.0, 1.0, 0.01, 3)
        self.filter_max_center_z_spin = self._make_float_spinbox(0.0, 0.0, 100000.0, 0.5, 3)
        self.filter_max_range_xz_spin = self._make_float_spinbox(0.0, 0.0, 100000.0, 0.5, 3)
        filter_form.addRow("Min Score", self.filter_min_score_spin)
        filter_form.addRow("Min 3D Score", self.filter_min_score_3d_spin)
        filter_form.addRow("Max center_z", self.filter_max_center_z_spin)
        filter_form.addRow("Max range_xz", self.filter_max_range_xz_spin)
        layout.addLayout(filter_form)

        scope_form = QFormLayout()
        self.processing_scope_combo = MenuWheelComboBox(group)
        for scope in ProcessingScope:
            self.processing_scope_combo.addItem(scope.value, scope.value)
        scope_form.addRow("Scope", self.processing_scope_combo)
        layout.addLayout(scope_form)

        self.apply_filter_btn = QPushButton("Apply Filter", group)
        layout.addWidget(self.apply_filter_btn)
        return group

    def _build_outlier_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Outlier Manager", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.outlier_rule_table = QTableWidget(group)
        self.outlier_rule_table.setColumnCount(3)
        self.outlier_rule_table.setHorizontalHeaderLabels(["Enabled", "Rule", "Hits"])
        self.outlier_rule_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.outlier_rule_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.outlier_rule_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.outlier_rule_table.verticalHeader().setVisible(False)
        outlier_rule_header = self.outlier_rule_table.horizontalHeader()
        outlier_rule_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        outlier_rule_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        outlier_rule_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.outlier_rule_table)

        self.outlier_param_form = QFormLayout()
        layout.addLayout(self.outlier_param_form)

        self.refresh_outliers_btn = QPushButton("Refresh Outliers", group)
        self.fix_selected_outlier_btn = QPushButton("Fix Selected", group)
        self.fix_scope_outliers_btn = QPushButton("Fix Current Scope", group)
        self.fix_all_visible_outliers_btn = QPushButton("Fix All Visible", group)
        layout.addWidget(self._row_widget(self.refresh_outliers_btn, self.fix_selected_outlier_btn))
        layout.addWidget(self._row_widget(self.fix_scope_outliers_btn, self.fix_all_visible_outliers_btn))

        self.outlier_table = QTableWidget(group)
        self.outlier_table.setColumnCount(len(OUTLIER_TABLE_COLUMNS))
        self.outlier_table.setHorizontalHeaderLabels(["Frame", "Track", "Category", "Rule", "Severity", "Fix", "Message"])
        self.outlier_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.outlier_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.outlier_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.outlier_table.verticalHeader().setVisible(False)
        outlier_header = self.outlier_table.horizontalHeader()
        for index in range(len(OUTLIER_TABLE_COLUMNS) - 1):
            outlier_header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        outlier_header.setSectionResizeMode(len(OUTLIER_TABLE_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.outlier_table)
        return group

    def _build_bulk_ops_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Bulk Operations", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        selector_form = QFormLayout()
        self.bulk_operation_combo = MenuWheelComboBox(group)
        selector_form.addRow("Operation", self.bulk_operation_combo)
        layout.addLayout(selector_form)

        self.bulk_param_form = QFormLayout()
        layout.addLayout(self.bulk_param_form)

        self.apply_bulk_operation_btn = QPushButton("Apply To Scope", group)
        layout.addWidget(self.apply_bulk_operation_btn)
        return group

    def _build_track_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Track Manager", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.track_table = QTableWidget(group)
        self.track_table.setColumnCount(len(TRACK_TABLE_COLUMNS))
        self.track_table.setHorizontalHeaderLabels(
            ["Track ID", "Category", "Enabled", "First Frame", "Last Frame", "Status"]
        )
        self.track_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_table.verticalHeader().setVisible(False)
        track_header = self.track_table.horizontalHeader()
        for index in range(len(TRACK_TABLE_COLUMNS)):
            mode = QHeaderView.ResizeMode.Stretch if index == len(TRACK_TABLE_COLUMNS) - 1 else QHeaderView.ResizeMode.ResizeToContents
            track_header.setSectionResizeMode(index, mode)
        layout.addWidget(self.track_table)

        self.delete_track_btn = QPushButton("Delete Track", group)
        self.lock_track_btn = QPushButton("Lock Track", group)
        self.unlock_track_btn = QPushButton("Unlock Track", group)
        layout.addWidget(self._row_widget(self.delete_track_btn, self.lock_track_btn, self.unlock_track_btn))

        merge_row = QWidget(group)
        merge_layout = QHBoxLayout(merge_row)
        merge_layout.setContentsMargins(0, 0, 0, 0)
        merge_layout.setSpacing(6)
        merge_layout.addWidget(QLabel("Merge To", merge_row))
        self.merge_target_combo = MenuWheelComboBox(merge_row)
        merge_layout.addWidget(self.merge_target_combo, 1)
        self.merge_track_btn = QPushButton("Merge Selected -> Target", merge_row)
        merge_layout.addWidget(self.merge_track_btn)
        layout.addWidget(merge_row)
        return group

    def _build_results_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Current Frame Results", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        self.results_table = QTableWidget(group)
        self.results_table.setColumnCount(len(RESULTS_TABLE_COLUMNS))
        self.results_table.setHorizontalHeaderLabels([header for header, _, _, _ in RESULTS_TABLE_COLUMNS])
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        for index in range(len(RESULTS_TABLE_COLUMNS) - 1):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(RESULTS_TABLE_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table)

        self.results_hint_label = QLabel(
            "Double-click editable cells to update 3D boxes. "
            "Only checked rows are visualized on the left.",
            group,
        )
        self.results_hint_label.setWordWrap(True)
        layout.addWidget(self.results_hint_label)

        self.replace_selected_btn = QPushButton("Replace Selected With Current Box/Point Prompt", group)
        layout.addWidget(self.replace_selected_btn)
        return group

    def _build_info_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Info", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        self.info_label = QLabel(group)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info_label)
        return group

    def _connect_signals(self) -> None:
        self.canvas.boxCompleted.connect(self._on_prompt_box_completed)
        self.canvas.pointAdded.connect(self._on_prompt_point_added)

        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self.recording_combo.currentIndexChanged.connect(self._on_recording_changed)
        self.edit_config_btn.clicked.connect(self._edit_config_file)
        self.reload_config_btn.clicked.connect(self._reload_dataset_config)
        self.open_video_btn.clicked.connect(self._choose_video)
        self.open_folder_btn.clicked.connect(self._choose_image_folder)
        self.annotation_csv_path_edit.textChanged.connect(self._on_annotation_csv_path_changed)
        self.browse_annotation_csv_btn.clicked.connect(self._browse_annotation_csv)
        self.load_annotation_csv_btn.clicked.connect(self._load_selected_annotation_csv)
        self.use_source_csv_btn.clicked.connect(self._use_source_csv)
        self.open_result_folder_btn.clicked.connect(self._open_result_folder)
        self.save_csv_btn.clicked.connect(self._save_csv_now)

        self.prompt_mode_combo.currentIndexChanged.connect(self._on_prompt_mode_changed)
        self.point_label_combo.currentIndexChanged.connect(self._on_point_label_changed)
        self.clear_prompt_btn.clicked.connect(self._clear_current_prompt)
        self.copy_prompt_btn.clicked.connect(self._copy_prompt_to_range)

        self.use_actual_k_checkbox.toggled.connect(self._on_intrinsics_mode_changed)

        self.prev_frame_btn.clicked.connect(lambda: self._step_frame(-1))
        self.next_frame_btn.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_index_spin.valueChanged.connect(self._on_frame_spin_changed)

        self.run_current_btn.clicked.connect(self._run_current_frame)
        self.run_range_btn.clicked.connect(self._run_selected_range)
        self.cancel_run_btn.clicked.connect(self._cancel_run)

        self.start_postprocess_btn.clicked.connect(self._start_postprocessing_stage)
        self.run_tracking_btn.clicked.connect(self._run_tracking)
        self.undo_change_btn.clicked.connect(self._undo_last_change)
        self.save_latest_btn.clicked.connect(self._save_latest_now)
        self.reset_to_raw_btn.clicked.connect(self._reset_to_raw)
        self.apply_filter_btn.clicked.connect(self._apply_filter)
        self.processing_scope_combo.currentIndexChanged.connect(self._on_processing_scope_changed)

        self.outlier_rule_table.currentCellChanged.connect(self._on_outlier_rule_selection_changed)
        self.outlier_rule_table.itemChanged.connect(self._on_outlier_rule_item_changed)
        self.refresh_outliers_btn.clicked.connect(self._refresh_outlier_analysis_requested)
        self.fix_selected_outlier_btn.clicked.connect(self._fix_selected_outlier)
        self.fix_scope_outliers_btn.clicked.connect(self._fix_current_scope_outliers)
        self.fix_all_visible_outliers_btn.clicked.connect(self._fix_all_visible_outliers)
        self.outlier_table.currentCellChanged.connect(self._on_outlier_selection_changed)

        self.bulk_operation_combo.currentIndexChanged.connect(self._on_bulk_operation_changed)
        self.apply_bulk_operation_btn.clicked.connect(self._apply_selected_bulk_operation)

        self.track_table.currentCellChanged.connect(self._on_track_selection_changed)
        self.delete_track_btn.clicked.connect(self._delete_selected_track)
        self.lock_track_btn.clicked.connect(self._lock_selected_track)
        self.unlock_track_btn.clicked.connect(self._unlock_selected_track)
        self.merge_track_btn.clicked.connect(self._merge_selected_track)

        self.results_table.currentCellChanged.connect(self._on_results_selection_changed)
        self.results_table.itemChanged.connect(self._on_results_table_item_changed)
        self.replace_selected_btn.clicked.connect(self._replace_selected_detection)

    def _initialize_processing_controls(self) -> None:
        for operation in self.bulk_operation_registry.all():
            self.bulk_operation_combo.addItem(operation.display_name, operation.operation_id)
        self._refresh_outlier_rule_table()
        self._load_selected_outlier_rule_editor()
        self._load_selected_bulk_operation_editor()
        self._refresh_visible_outlier_table()
        self.frame_slider.clear_bookmarks()

    def current_source_mode(self) -> SourceMode:
        value = self.source_mode_combo.currentData() or self.source_mode_combo.currentText()
        return SourceMode(value)

    def current_prompt_mode(self) -> PromptMode:
        value = self.prompt_mode_combo.currentData() or self.prompt_mode_combo.currentText()
        return PromptMode(value)

    def _selected_processing_scope(self) -> ProcessingScope:
        value = self.processing_scope_combo.currentData() or self.processing_scope_combo.currentText()
        return ProcessingScope(value)

    def _selected_outlier_rule_id(self) -> str | None:
        row = self.outlier_rule_table.currentRow()
        if row < 0:
            return None
        item = self.outlier_rule_table.item(row, 1)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _selected_bulk_operation_id(self) -> str | None:
        value = self.bulk_operation_combo.currentData() or self.bulk_operation_combo.currentText()
        return str(value) if value else None

    def _enabled_outlier_rule_ids(self) -> list[str]:
        return [
            rule.rule_id
            for rule in self.outlier_registry.all()
            if self.outlier_rule_enabled.get(rule.rule_id, False)
        ]

    @staticmethod
    def _clear_form_layout(layout: QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            if child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    child_widget = child_item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def _populate_param_form(
        self,
        layout: QFormLayout,
        specs: Sequence[ParameterSpec],
        values: dict[str, float | int],
        widget_store: dict[str, QWidget],
    ) -> None:
        self._clear_form_layout(layout)
        widget_store.clear()
        for spec in specs:
            if spec.kind == "int":
                widget = NoWheelSpinBox(self)
                widget.setRange(int(spec.minimum), int(spec.maximum))
                widget.setSingleStep(max(1, int(spec.step)))
                widget.setValue(int(values.get(spec.key, spec.default)))
                widget.valueChanged.connect(
                    lambda value, key=spec.key, store=values: store.__setitem__(key, int(value))
                )
            else:
                widget = NoWheelDoubleSpinBox(self)
                widget.setRange(float(spec.minimum), float(spec.maximum))
                widget.setDecimals(int(spec.decimals))
                widget.setSingleStep(float(spec.step))
                widget.setValue(float(values.get(spec.key, spec.default)))
                widget.valueChanged.connect(
                    lambda value, key=spec.key, store=values: store.__setitem__(key, float(value))
                )
            widget_store[spec.key] = widget
            layout.addRow(spec.label, widget)

    def _load_selected_outlier_rule_editor(self) -> None:
        rule_id = self._selected_outlier_rule_id()
        if not rule_id:
            self._clear_form_layout(self.outlier_param_form)
            self._outlier_param_widgets.clear()
            return
        rule = self.outlier_registry.get(rule_id)
        values = self.outlier_rule_params.setdefault(rule_id, rule.default_params())
        self._populate_param_form(self.outlier_param_form, rule.param_specs, values, self._outlier_param_widgets)

    def _load_selected_bulk_operation_editor(self) -> None:
        operation_id = self._selected_bulk_operation_id()
        if not operation_id:
            self._clear_form_layout(self.bulk_param_form)
            self._bulk_param_widgets.clear()
            return
        operation = self.bulk_operation_registry.get(operation_id)
        values = self.bulk_operation_params.setdefault(operation_id, operation.default_params())
        self._populate_param_form(self.bulk_param_form, operation.param_specs, values, self._bulk_param_widgets)

    def _refresh_outlier_rule_table(self, preserve_rule_id: str | None = None) -> None:
        selected_rule_id = preserve_rule_id or self._selected_outlier_rule_id()
        hit_counts: dict[str, int] = {}
        for hit in self.outlier_hits_global:
            hit_counts[hit.rule_id] = hit_counts.get(hit.rule_id, 0) + 1

        self._outlier_rule_table_internal_change = True
        with QSignalBlocker(self.outlier_rule_table):
            rules = self.outlier_registry.all()
            self.outlier_rule_table.setRowCount(len(rules))
            for row_index, rule in enumerate(rules):
                background_color = self._outlier_background_for_rule(rule.rule_id)
                enabled_item = QTableWidgetItem()
                enabled_item.setData(Qt.ItemDataRole.UserRole, rule.rule_id)
                enabled_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                enabled_item.setCheckState(
                    Qt.CheckState.Checked if self.outlier_rule_enabled.get(rule.rule_id, False) else Qt.CheckState.Unchecked
                )
                if background_color is not None:
                    enabled_item.setBackground(background_color)
                self.outlier_rule_table.setItem(row_index, 0, enabled_item)

                name_item = QTableWidgetItem(rule.display_name)
                name_item.setData(Qt.ItemDataRole.UserRole, rule.rule_id)
                name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if background_color is not None:
                    name_item.setBackground(background_color)
                self.outlier_rule_table.setItem(row_index, 1, name_item)

                count_item = QTableWidgetItem(str(hit_counts.get(rule.rule_id, 0)))
                count_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if background_color is not None:
                    count_item.setBackground(background_color)
                self.outlier_rule_table.setItem(row_index, 2, count_item)

            selected_row = 0 if rules else -1
            if selected_rule_id:
                for idx, rule in enumerate(rules):
                    if rule.rule_id == selected_rule_id:
                        selected_row = idx
                        break
            if selected_row >= 0:
                self.outlier_rule_table.setCurrentCell(selected_row, 1)
            else:
                self.outlier_rule_table.clearSelection()
        self._outlier_rule_table_internal_change = False

    @staticmethod
    def _outlier_hit_key(hit: OutlierHit) -> str:
        return f"{hit.rule_id}:{hit.det_id}:{hit.frame_index}"

    def _selected_outlier_hit(self) -> OutlierHit | None:
        row = self.outlier_table.currentRow()
        if row < 0:
            return None
        item = self.outlier_table.item(row, 0)
        if item is None:
            return None
        hit = item.data(Qt.ItemDataRole.UserRole)
        return hit if isinstance(hit, OutlierHit) else None

    def _selected_outlier_key(self) -> str | None:
        hit = self._selected_outlier_hit()
        return self._outlier_hit_key(hit) if hit is not None else None

    def _scope_filtered_hits(self, scope: ProcessingScope | None = None, rule_id: str | None = None) -> list[OutlierHit]:
        if not self.outlier_hits_global:
            return []
        resolved_scope = ProcessingScope(scope or self._selected_processing_scope())
        if resolved_scope == ProcessingScope.CURRENT_FRAME:
            hits = self.outlier_hits_by_frame.get(int(self.current_frame_index), [])
        elif resolved_scope == ProcessingScope.SELECTED_TRACK:
            selected_track_id = str(self._selected_track_id() or "").strip()
            hits = self.outlier_hits_by_track_id.get(selected_track_id, []) if selected_track_id else []
        else:
            hits = self.outlier_hits_by_rule_id.get(rule_id, self.outlier_hits_global) if rule_id else self.outlier_hits_global
        if rule_id is None or resolved_scope == ProcessingScope.GLOBAL:
            return list(hits)
        return [hit for hit in hits if hit.rule_id == rule_id]

    def _refresh_visible_outlier_table(self, preserve_key: str | None = None) -> None:
        selected_key = preserve_key or self._selected_outlier_key()
        self.outlier_hits_visible = list(self.outlier_hits_global)
        with QSignalBlocker(self.outlier_table):
            self.outlier_table.setRowCount(len(self.outlier_hits_visible))
            for row_index, hit in enumerate(self.outlier_hits_visible):
                values = (
                    str(hit.frame_index),
                    hit.track_id,
                    hit.category,
                    hit.rule_id,
                    f"{float(hit.severity):.2f}",
                    "Yes" if hit.fixable else "No",
                    hit.message,
                )
                background_color = self._outlier_background_for_rule(hit.rule_id)
                for col_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, hit)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if background_color is not None:
                        item.setBackground(background_color)
                    self.outlier_table.setItem(row_index, col_index, item)
            selected_row = -1
            if selected_key:
                for index, hit in enumerate(self.outlier_hits_visible):
                    if self._outlier_hit_key(hit) == selected_key:
                        selected_row = index
                        break
            if selected_row >= 0:
                self.outlier_table.setCurrentCell(selected_row, 0)
            else:
                self.outlier_table.clearSelection()
        self._update_action_states()

    def _clear_outlier_state(self) -> None:
        self.outlier_hits_global = []
        self.outlier_hits_visible = []
        self.outlier_hits_by_det_id = {}
        self.outlier_hits_by_frame = {}
        self.outlier_hits_by_track_id = {}
        self.outlier_hits_by_rule_id = {}
        self.outlier_frames = set()
        self.frame_slider.clear_bookmarks()
        self._refresh_outlier_rule_table()
        self._refresh_visible_outlier_table()

    def _build_processing_context(self, require_reprojection: bool) -> ProcessingContext | None:
        reproject_callback = None
        if require_reprojection:
            if self.current_provider is None:
                self._show_warning("No Source", "Load a media source first.")
                return None
            reproject_callback = self._reproject_record_for_processing
        return ProcessingContext(
            records=self.records,
            current_frame_index=self.current_frame_index,
            selected_track_id=self._selected_track_id() or "",
            track_summaries=self.postprocess_session.build_track_summaries(self.records),
            reproject_record=reproject_callback,
        )

    def _reproject_record_for_processing(self, record: DetectionRecord) -> None:
        if self.current_provider is None:
            raise ValueError("No active media source.")
        intrinsics = self._intrinsics_for_record_projection(record)
        (
            record.box2d_x1,
            record.box2d_y1,
            record.box2d_x2,
            record.box2d_y2,
        ) = self.detector.project_record_to_box2d(
            record=record,
            intrinsics=intrinsics,
            image_shape=self.current_provider.frame_shape(record.frame_index),
        )

    def _refresh_outlier_analysis_requested(self) -> None:
        self._refresh_outlier_analysis(show_status=True)

    def _refresh_outlier_analysis(self, show_status: bool = False) -> None:
        if show_status:
            self.statusBar().showMessage("Refreshing outliers...")
        if not self._is_postprocessing_stage() or self.current_provider is None:
            self._clear_outlier_state()
            self._refresh_table(preserve_det_id=self._selected_det_id())
            self._update_action_states()
            self._refresh_info_panel()
            if show_status:
                self.statusBar().showMessage("Outlier refresh skipped: postprocessing is not active.", 5000)
            return
        enabled_rules = self._enabled_outlier_rule_ids()
        if not enabled_rules:
            self._clear_outlier_state()
            self._refresh_table(preserve_det_id=self._selected_det_id())
            self._update_action_states()
            self._refresh_info_panel()
            if show_status:
                self.statusBar().showMessage("Outlier refresh complete: no rules are enabled.", 5000)
            return
        context = self._build_processing_context(require_reprojection=False)
        if context is None:
            self._clear_outlier_state()
            if show_status:
                self.statusBar().showMessage("Outlier refresh failed: processing context is unavailable.", 5000)
            return
        preserve_rule_id = self._selected_outlier_rule_id()
        preserve_outlier_key = self._selected_outlier_key()
        hits = self.processing_engine.analyze_outliers(
            records=self.records,
            scope=OutlierScope.GLOBAL,
            enabled_rule_ids=enabled_rules,
            params_by_rule=self.outlier_rule_params,
            context=context,
        )
        self.outlier_hits_global = hits
        self.outlier_hits_by_det_id = {}
        self.outlier_hits_by_frame = {}
        self.outlier_hits_by_track_id = {}
        self.outlier_hits_by_rule_id = {}
        self.outlier_frames = set()
        for hit in hits:
            self.outlier_hits_by_det_id.setdefault(hit.det_id, []).append(hit)
            self.outlier_hits_by_frame.setdefault(int(hit.frame_index), []).append(hit)
            if str(hit.track_id).strip():
                self.outlier_hits_by_track_id.setdefault(str(hit.track_id).strip(), []).append(hit)
            self.outlier_hits_by_rule_id.setdefault(hit.rule_id, []).append(hit)
            self.outlier_frames.add(int(hit.frame_index))
        colored_bookmarks: dict[int, list[QColor]] = {}
        for hit in hits:
            color = self._outlier_rule_color(hit.rule_id)
            if color is None:
                continue
            bucket = colored_bookmarks.setdefault(int(hit.frame_index), [])
            if all(existing != color for existing in bucket):
                bucket.append(color)
        if colored_bookmarks:
            self.frame_slider.set_colored_bookmarks(colored_bookmarks)
        else:
            self.frame_slider.clear_bookmarks()
        self._refresh_outlier_rule_table(preserve_rule_id=preserve_rule_id)
        self._load_selected_outlier_rule_editor()
        self._refresh_visible_outlier_table(preserve_key=preserve_outlier_key)
        self._refresh_table(preserve_det_id=self._selected_det_id())
        self._update_action_states()
        self._refresh_info_panel()
        if show_status:
            self.statusBar().showMessage(
                f"Outlier refresh complete: {len(hits)} hits across {len(self.outlier_frames)} frame(s).",
                6000,
            )

    def _on_processing_scope_changed(self) -> None:
        self._update_action_states()

    def _on_outlier_rule_selection_changed(self) -> None:
        self._load_selected_outlier_rule_editor()
        self._update_action_states()

    def _on_outlier_rule_item_changed(self, item: QTableWidgetItem) -> None:
        if self._outlier_rule_table_internal_change or item.column() != 0:
            return
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        if not rule_id:
            return
        self.outlier_rule_enabled[str(rule_id)] = item.checkState() == Qt.CheckState.Checked
        self._refresh_outlier_analysis()

    def _on_bulk_operation_changed(self) -> None:
        self._load_selected_bulk_operation_editor()
        self._update_action_states()

    def _on_outlier_selection_changed(self) -> None:
        hit = self._selected_outlier_hit()
        if hit is None:
            self._update_action_states()
            return
        if self.current_provider is not None and int(hit.frame_index) != int(self.current_frame_index):
            self._set_current_frame(int(hit.frame_index))
        self._refresh_table(preserve_det_id=hit.det_id)
        self._update_action_states()

    def _fix_selected_outlier(self) -> None:
        hit = self._selected_outlier_hit()
        if hit is None:
            self._show_warning("No Outlier Selected", "Select one outlier first.")
            return
        self._apply_outlier_fixes([hit], empty_message="The selected outlier could not be fixed.")

    def _fix_current_scope_outliers(self) -> None:
        rule_id = self._selected_outlier_rule_id()
        if not rule_id:
            self._show_warning("No Rule Selected", "Select one outlier rule first.")
            return
        hits = self._scope_filtered_hits(rule_id=rule_id)
        self._apply_outlier_fixes(hits, empty_message="No fixable outliers were found for the current rule and scope.")

    def _fix_all_visible_outliers(self) -> None:
        self._apply_outlier_fixes(
            self.outlier_hits_visible,
            empty_message="No visible fixable outliers were found.",
        )

    def _apply_outlier_fixes(self, hits: Sequence[OutlierHit], empty_message: str) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Outlier fixing is only available in postprocessing.")
            return
        if not hits:
            self.statusBar().showMessage(empty_message, 5000)
            return
        context = self._build_processing_context(require_reprojection=True)
        if context is None:
            return
        before = clone_records(self.records)
        result = self.processing_engine.fix_hits(
            records=self.records,
            hits=hits,
            params_by_rule=self.outlier_rule_params,
            context=context,
        )
        if result.updated_count <= 0:
            self.statusBar().showMessage(empty_message, 5000)
            return
        self.postprocess_session.push_undo_snapshot(before)
        self._persist_active_records()
        self._refresh_current_frame_view()
        self._refresh_track_table(preserve_track_id=self._selected_track_id())
        self._refresh_outlier_analysis()
        self.statusBar().showMessage(result.message, 6000)

    def _apply_selected_bulk_operation(self) -> None:
        operation_id = self._selected_bulk_operation_id()
        if not operation_id:
            self._show_warning("No Operation Selected", "Select one bulk operation first.")
            return
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Bulk operations are only available in postprocessing.")
            return
        context = self._build_processing_context(require_reprojection=True)
        if context is None:
            return
        before = clone_records(self.records)
        result = self.processing_engine.apply_operation(
            operation_id=operation_id,
            records=self.records,
            scope=OperationScope(self._selected_processing_scope()),
            params=self.bulk_operation_params.get(operation_id),
            context=context,
        )
        if result.updated_count <= 0:
            self.statusBar().showMessage("The selected bulk operation made no changes.", 5000)
            return
        self.postprocess_session.push_undo_snapshot(before)
        self._persist_active_records()
        self._refresh_current_frame_view()
        self._refresh_track_table(preserve_track_id=self._selected_track_id())
        self._refresh_outlier_analysis()
        self.statusBar().showMessage(result.message, 6000)

    def _outlier_flags_for_record(self, record: DetectionRecord) -> str:
        hits = self.outlier_hits_by_det_id.get(record.det_id, [])
        if not hits:
            return ""
        return ", ".join(sorted({hit.rule_id for hit in hits}))

    def _outlier_color_for_record(self, record: DetectionRecord):
        hit = self._primary_outlier_hit_for_record(record)
        if hit is None:
            return None
        return self._outlier_background_for_rule(hit.rule_id)

    def _primary_outlier_hit_for_record(self, record: DetectionRecord) -> OutlierHit | None:
        hits = self.outlier_hits_by_det_id.get(record.det_id, [])
        if not hits:
            return None
        return max(
            hits,
            key=lambda hit: (float(hit.severity), -int(hit.frame_index), hit.rule_id),
        )

    def _outlier_rule_color(self, rule_id: str) -> QColor | None:
        color = OUTLIER_RULE_COLORS.get(rule_id)
        return QColor(color) if color is not None else None

    def _outlier_background_for_rule(self, rule_id: str) -> QColor | None:
        color = self._outlier_rule_color(rule_id)
        if color is None:
            return None
        color.setAlpha(72)
        return color

    def _apply_source_mode_ui(self) -> None:
        mode = self.current_source_mode()
        self.dataset_section.setVisible(mode == SourceMode.DATASET)
        self.manual_section.setVisible(mode in {SourceMode.VIDEO, SourceMode.IMAGE_FOLDER})
        self.open_video_btn.setVisible(mode == SourceMode.VIDEO)
        self.open_folder_btn.setVisible(mode == SourceMode.IMAGE_FOLDER)

    def _sync_prompt_ui(self) -> None:
        mode = self.current_prompt_mode()
        self.canvas.set_prompt_mode(mode)
        self.canvas.set_point_label(self.point_label_combo.currentData() or 1)
        text_visible = mode == PromptMode.TEXT
        point_visible = mode == PromptMode.POINT
        self.text_prompt_edit.setVisible(text_visible)
        self.point_label_combo.setVisible(point_visible)
        text_label = self.prompt_group.layout().labelForField(self.text_prompt_edit)
        point_label = self.prompt_group.layout().labelForField(self.point_label_combo)
        if text_label is not None:
            text_label.setVisible(text_visible)
        if point_label is not None:
            point_label.setVisible(point_visible)

        hint = "Text mode: separate categories with '.'"
        if mode in {PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE}:
            hint = "Drag on the left canvas to place a box prompt."
        elif mode == PromptMode.POINT:
            hint = "Left click adds a point, right click removes the last point."
        self.prompt_hint_label.setText(hint)
        self.canvas.set_prompt_box(self.current_prompt_box if mode in {PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE} else None)
        self.canvas.set_prompt_points(self.current_prompt_points if mode == PromptMode.POINT else [])
        self._update_action_states()

    def _load_dataset_config(self, initial: bool = False) -> None:
        try:
            self.dataset_config_store.ensure_exists()
            config = self.dataset_config_store.load()
        except Exception as exc:
            if not initial:
                QMessageBox.critical(self, "Config Error", str(exc))
            else:
                self.statusBar().showMessage(f"Failed to load dataset config: {exc}", 10000)
            return

        self.dataset_config = config
        self.current_result_dir = self.dataset_config_store.ensure_result_dir(config, on_date=date.today())
        self.config_path_edit.setText(str(self.dataset_config_store.path))

        previous_dataset = self.selected_dataset_id or self.dataset_combo.currentData() or ""
        previous_recording = self.selected_recording_id or self.recording_combo.currentData() or ""

        with QSignalBlocker(self.dataset_combo):
            self.dataset_combo.clear()
            for dataset in config.datasets:
                self.dataset_combo.addItem(dataset.display_name, dataset.id)
            index = self.dataset_combo.findData(previous_dataset)
            if index < 0 and self.dataset_combo.count():
                index = 0
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)

        self.selected_dataset_id = str(self.dataset_combo.currentData() or "")
        self._refresh_recordings(previous_recording)
        self._refresh_output_path_display()

        if self.current_source_mode() == SourceMode.DATASET and self.selected_dataset_id and self.selected_recording_id:
            self._load_selected_dataset_recording()

    def _refresh_recordings(self, preferred_recording: str = "") -> None:
        config = self.dataset_config
        dataset_id = str(self.dataset_combo.currentData() or "")
        self.selected_dataset_id = dataset_id
        recordings: list[str] = []
        if config and dataset_id:
            recordings = self.dataset_config_store.discover_recordings(config, dataset_id)

        with QSignalBlocker(self.recording_combo):
            self.recording_combo.clear()
            for recording in recordings:
                self.recording_combo.addItem(recording, recording)
            index = self.recording_combo.findData(preferred_recording)
            if index < 0 and self.recording_combo.count():
                index = 0
            if index >= 0:
                self.recording_combo.setCurrentIndex(index)
        self.selected_recording_id = str(self.recording_combo.currentData() or "")

    def _edit_config_file(self) -> None:
        path = self.dataset_config_store.ensure_exists()
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:
            QMessageBox.warning(self, "Open Config Failed", str(exc))

    def _reload_dataset_config(self) -> None:
        self._load_dataset_config(initial=False)
        self._refresh_info_panel()

    def _on_annotation_csv_path_changed(self) -> None:
        self._update_action_states()

    def _on_source_mode_changed(self) -> None:
        if self._source_ui_guard:
            return
        self._apply_source_mode_ui()
        mode = self.current_source_mode()
        if mode == SourceMode.DATASET:
            if self.selected_dataset_id and self.selected_recording_id:
                self._load_selected_dataset_recording()
        elif self.manual_source_path and self._manual_path_matches_mode(mode):
            self._load_manual_source(self.manual_source_path, mode)
        else:
            self._deactivate_source(clear_manual_path=False)

    def _on_dataset_changed(self) -> None:
        if self._source_ui_guard:
            return
        previous = self.selected_recording_id
        self.selected_dataset_id = str(self.dataset_combo.currentData() or "")
        self._refresh_recordings(previous)
        self._refresh_output_path_display()
        if self.current_source_mode() == SourceMode.DATASET and self.selected_recording_id:
            self._load_selected_dataset_recording()

    def _on_recording_changed(self) -> None:
        if self._source_ui_guard:
            return
        self.selected_recording_id = str(self.recording_combo.currentData() or "")
        self._refresh_output_path_display()
        if self.current_source_mode() == SourceMode.DATASET and self.selected_recording_id:
            self._load_selected_dataset_recording()

    def _load_selected_dataset_recording(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before switching source.")
            return
        config = self.dataset_config
        dataset_id = str(self.dataset_combo.currentData() or "")
        recording_id = str(self.recording_combo.currentData() or "")
        if not config or not dataset_id or not recording_id:
            return

        dataset = config.get_dataset(dataset_id)
        if dataset is None:
            self._show_warning("Dataset Error", f"Unknown dataset: {dataset_id}")
            return

        try:
            media_path = self.dataset_config_store.resolve_media_path(config, dataset_id, recording_id)
            provider = ImageFolderFrameProvider(media_path)
            output_path = self.dataset_config_store.seed_output_session_files(
                config=config,
                source_mode=SourceMode.DATASET,
                dataset_id=dataset_id,
                recording_id=recording_id,
                on_date=date.today(),
            )
        except Exception as exc:
            self._show_warning("Source Load Failed", str(exc))
            return

        self.selected_dataset_id = dataset_id
        self.selected_recording_id = recording_id
        self.use_actual_k_checkbox.setChecked(True)
        self.fx_spin.setValue(dataset.default_intrinsics.fx)
        self.fy_spin.setValue(dataset.default_intrinsics.fy)
        self.cx_spin.setValue(dataset.default_intrinsics.cx)
        self.cy_spin.setValue(dataset.default_intrinsics.cy)

        self._activate_provider(
            provider=provider,
            source_context=SourceContext(
                source_mode=SourceMode.DATASET,
                source_type="dataset_image_folder",
                dataset_id=dataset_id,
                recording_id=recording_id,
            ),
            output_path=output_path,
            active_source_path=media_path,
        )

    def _choose_video(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before switching source.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(self.manual_source_path.parent if self.manual_source_path else Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm)",
        )
        if path:
            self._load_manual_source(Path(path), SourceMode.VIDEO)

    def _choose_image_folder(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before switching source.")
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Image Folder",
            str(self.manual_source_path if self.manual_source_path and self.manual_source_path.is_dir() else Path.home()),
        )
        if folder:
            self._load_manual_source(Path(folder), SourceMode.IMAGE_FOLDER)

    def _load_manual_source(self, path: str | Path, mode: SourceMode) -> None:
        config = self.dataset_config
        source_path = Path(path).resolve()
        try:
            if mode == SourceMode.VIDEO:
                provider = VideoFrameProvider(source_path)
            elif mode == SourceMode.IMAGE_FOLDER:
                provider = ImageFolderFrameProvider(source_path)
            else:
                raise ValueError(f"Unsupported manual source mode: {mode}")
            if config is None:
                raise ValueError("Dataset config is not loaded.")
            output_path = self.dataset_config_store.seed_output_session_files(
                config=config,
                source_mode=mode,
                manual_source_path=source_path,
                on_date=date.today(),
            )
        except Exception as exc:
            self._show_warning("Source Load Failed", str(exc))
            return

        self.manual_source_path = source_path
        self.manual_path_edit.setText(str(source_path))
        self._activate_provider(
            provider=provider,
            source_context=SourceContext(source_mode=mode, source_type=provider.source_type),
            output_path=output_path,
            active_source_path=source_path,
        )

    @staticmethod
    def _ensure_blank_csv_exists(path: Path) -> None:
        resolved = Path(path).resolve()
        if resolved.exists():
            return
        CSVStore(resolved, backup_enabled=False).save_records([])

    def _activate_csv_session(
        self,
        output_path: Path,
        active_source_path: Path,
        *,
        selected_csv_path: Path | None = None,
        ensure_blank_source_csv: bool = False,
        reset_frame_index: bool = False,
    ) -> None:
        resolved_output_path = Path(output_path).resolve()
        resolved_source_path = Path(active_source_path).resolve()
        resolved_selected_csv = Path(selected_csv_path).resolve() if selected_csv_path is not None else None
        if ensure_blank_source_csv and resolved_selected_csv is None:
            self._ensure_blank_csv_exists(resolved_output_path)

        stage, active_csv_path, records = self.postprocess_session.activate(
            resolved_output_path,
            selected_csv_path=resolved_selected_csv,
        )
        self.current_source_output_csv_path = resolved_output_path
        self.current_annotation_csv_override_path = resolved_selected_csv
        self.current_raw_csv_path = self.postprocess_session.raw_path
        self.current_latest_csv_path = self.postprocess_session.latest_path
        self.current_output_csv_path = active_csv_path.resolve()
        self.current_result_dir = self.current_output_csv_path.parent
        self.current_active_source_path = resolved_source_path
        self.active_source_path_edit.setText(str(resolved_source_path))

        self.stage_label.setText(f"Stage: {stage.value}")
        self.records = records
        if reset_frame_index:
            self.current_frame_index = 0
        self._sync_frame_controls()
        self._load_csv_side_inputs()
        self._refresh_output_path_display()
        self._refresh_outlier_analysis()
        self._expand_results_panel_if_needed(force=bool(self.records))
        self._refresh_current_frame_view()
        self._refresh_track_table()
        self._refresh_info_panel()
        self._update_action_states()

    def _browse_annotation_csv(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before loading another CSV.")
            return
        initial_path = self.annotation_csv_path_edit.text().strip()
        if initial_path:
            start_dir = str(Path(initial_path).resolve().parent)
        elif self.current_result_dir is not None:
            start_dir = str(self.current_result_dir)
        else:
            start_dir = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Annotation CSV",
            start_dir,
            "CamLabel3D CSV (*.csv);;All Files (*)",
        )
        if path:
            self.annotation_csv_path_edit.setText(path)

    def _load_selected_annotation_csv(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before loading another CSV.")
            return
        if self.current_provider is None or self.current_active_source_path is None:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        if self.current_source_output_csv_path is None:
            self._show_warning("No Default CSV", "The current source does not have a resolved CSV path yet.")
            return
        raw_path = self.annotation_csv_path_edit.text().strip()
        if not raw_path:
            self._show_warning("No CSV Path", "Enter or browse for an existing CSV file first.")
            return
        selected_csv_path = Path(raw_path).resolve()
        if not selected_csv_path.exists() or not selected_csv_path.is_file():
            self._show_warning("CSV Not Found", f"Annotation CSV not found:\n{selected_csv_path}")
            return
        try:
            self._activate_csv_session(
                self.current_source_output_csv_path,
                self.current_active_source_path,
                selected_csv_path=selected_csv_path,
                ensure_blank_source_csv=False,
                reset_frame_index=False,
            )
        except Exception as exc:
            self._show_warning("CSV Load Failed", str(exc))
            return
        self.statusBar().showMessage(f"Loaded annotation CSV: {selected_csv_path}", 5000)

    def _use_source_csv(self) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Cancel the active run before switching CSV.")
            return
        if self.current_provider is None or self.current_active_source_path is None:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        if self.current_source_output_csv_path is None:
            self._show_warning("No Default CSV", "The current source does not have a resolved CSV path yet.")
            return
        try:
            self._activate_csv_session(
                self.current_source_output_csv_path,
                self.current_active_source_path,
                selected_csv_path=None,
                ensure_blank_source_csv=True,
                reset_frame_index=False,
            )
        except Exception as exc:
            self._show_warning("CSV Load Failed", str(exc))
            return
        self.statusBar().showMessage("Switched to the source-derived annotation CSV.", 5000)

    def _activate_provider(
        self,
        provider: ImageFolderFrameProvider | VideoFrameProvider,
        source_context: SourceContext,
        output_path: Path,
        active_source_path: Path,
    ) -> None:
        self._close_provider()
        self.current_provider = provider
        self.current_source_context = source_context
        self._activate_csv_session(
            output_path,
            active_source_path,
            selected_csv_path=None,
            ensure_blank_source_csv=True,
            reset_frame_index=True,
        )

    def _deactivate_source(self, clear_manual_path: bool) -> None:
        self._close_provider()
        self.postprocess_session.clear()
        self.current_provider = None
        self.current_source_context = None
        self.current_raw_csv_path = None
        self.current_latest_csv_path = None
        self.current_output_csv_path = None
        self.current_source_output_csv_path = None
        self.records = []
        self.current_frame_index = 0
        self.current_frame_det_ids = []
        self.current_track_ids = []
        self.current_active_source_path = None
        self.current_annotation_csv_override_path = None
        self.active_source_path_edit.clear()
        self.annotation_csv_path_edit.clear()
        self.output_csv_edit.clear()
        if clear_manual_path:
            self.manual_source_path = None
            self.manual_path_edit.clear()
        self.canvas.clear()
        self.stage_label.setText(f"Stage: {WorkflowStage.DETECTION.value}")
        self._sync_frame_controls()
        self._clear_outlier_state()
        self._refresh_table()
        self._refresh_track_table()
        self._refresh_frame_labels()
        self._refresh_info_panel()
        self._update_action_states()

    def _close_provider(self) -> None:
        if self.current_provider is not None:
            try:
                self.current_provider.close()
            except Exception:
                pass

    def _sync_frame_controls(self) -> None:
        frame_count = self.current_provider.frame_count if self.current_provider else 0
        maximum = max(0, frame_count - 1)
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_index_spin):
            self.frame_slider.setMaximum(maximum)
            self.frame_index_spin.setMaximum(maximum)
            self.start_frame_spin.setMaximum(maximum)
            self.end_frame_spin.setMaximum(maximum)
            self.frame_slider.setValue(min(self.current_frame_index, maximum))
            self.frame_index_spin.setValue(min(self.current_frame_index, maximum))
            if frame_count:
                self.end_frame_spin.setValue(maximum)
            else:
                self.start_frame_spin.setValue(0)
                self.end_frame_spin.setValue(0)
        self._refresh_frame_labels()

    def _refresh_frame_labels(self) -> None:
        if not self.current_provider:
            self.frame_info_label.setText("Frame: -- / --")
            self.time_info_label.setText("t = --")
            return
        total = self.current_provider.frame_count
        self.frame_info_label.setText(f"Frame: {self.current_frame_index + 1} / {total}")
        timestamp_ms = self.current_provider.get_timestamp_ms(self.current_frame_index)
        if timestamp_ms is None:
            self.time_info_label.setText("t = --")
        else:
            self.time_info_label.setText(f"t = {timestamp_ms / 1000.0:.3f}s")

    def _refresh_current_frame_view(self) -> None:
        self._refresh_frame_labels()
        self._refresh_table()
        self._refresh_preview()
        self._refresh_info_panel()

    def _current_stage(self) -> WorkflowStage:
        return self.postprocess_session.stage

    def _is_postprocessing_stage(self) -> bool:
        return self._current_stage() == WorkflowStage.POSTPROCESSING

    def _is_detection_stage(self) -> bool:
        return self._current_stage() == WorkflowStage.DETECTION

    def _is_tracking_running(self) -> bool:
        return bool(self.tracking_worker and self.tracking_worker.isRunning())

    def _is_background_task_running(self) -> bool:
        return self._is_detection_running() or self._is_tracking_running()

    def _refresh_track_table(self, preserve_track_id: str | None = None) -> None:
        summaries = self.postprocess_session.build_track_summaries(self.records)
        selected_track_id = preserve_track_id or self._selected_track_id()
        self.current_track_ids = [summary.track_id for summary in summaries]

        with QSignalBlocker(self.track_table), QSignalBlocker(self.merge_target_combo):
            self.track_table.setRowCount(len(summaries))
            self.merge_target_combo.clear()
            for row_index, summary in enumerate(summaries):
                for col_index, value in enumerate(
                    (
                        summary.track_id,
                        summary.category,
                        str(summary.enabled_count),
                        str(summary.first_frame),
                        str(summary.last_frame),
                        summary.status,
                    )
                ):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, summary.track_id)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.track_table.setItem(row_index, col_index, item)
                self.merge_target_combo.addItem(summary.track_id, summary.track_id)

            selected_row = -1
            if selected_track_id:
                for idx, track_id in enumerate(self.current_track_ids):
                    if track_id == selected_track_id:
                        selected_row = idx
                        break
            if selected_row >= 0:
                self.track_table.setCurrentCell(selected_row, 0)
            else:
                self.track_table.clearSelection()

            target_index = -1
            for index in range(self.merge_target_combo.count()):
                candidate = str(self.merge_target_combo.itemData(index) or "").strip()
                if candidate and candidate != (selected_track_id or ""):
                    target_index = index
                    break
            if target_index >= 0:
                self.merge_target_combo.setCurrentIndex(target_index)
        self._update_action_states()

    def _selected_track_id(self) -> str | None:
        row = self.track_table.currentRow()
        if row < 0:
            return None
        item = self.track_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _selected_track_summary(self) -> TrackSummary | None:
        track_id = self._selected_track_id()
        if not track_id:
            return None
        for summary in self.postprocess_session.build_track_summaries(self.records):
            if summary.track_id == track_id:
                return summary
        return None

    def _current_filter_config(self) -> FilterConfig:
        return FilterConfig(
            min_score=float(self.filter_min_score_spin.value()),
            min_score_3d=float(self.filter_min_score_3d_spin.value()),
            max_center_z=float(self.filter_max_center_z_spin.value()),
            max_range_xz=float(self.filter_max_range_xz_spin.value()),
        )

    def _on_track_selection_changed(self) -> None:
        selected_track_id = self._selected_track_id() or ""
        with QSignalBlocker(self.merge_target_combo):
            for index in range(self.merge_target_combo.count()):
                candidate = str(self.merge_target_combo.itemData(index) or "").strip()
                if candidate and candidate != selected_track_id:
                    self.merge_target_combo.setCurrentIndex(index)
                    break
        self._update_action_states()

    def _start_postprocessing_stage(self) -> None:
        if not self.current_provider:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        if self._is_postprocessing_stage():
            self._show_warning("Already Started", "This source is already in postprocessing mode.")
            return
        if not self.postprocess_session.can_start_postprocessing(self.records):
            self._show_warning("Raw Missing", "Run detection first so the raw detection CSV exists.")
            return
        try:
            active_path, records = self.postprocess_session.start_postprocessing(self.records)
        except Exception as exc:
            self._show_warning("Start Postprocessing Failed", str(exc))
            return

        self.current_latest_csv_path = self.postprocess_session.latest_path
        self.current_output_csv_path = active_path
        self.records = records
        self.stage_label.setText(f"Stage: {self._current_stage().value}")
        self._refresh_output_path_display()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table()
        self.statusBar().showMessage("Postprocessing started.", 5000)

    def _apply_filter(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Start postprocessing before applying filters.")
            return
        filter_config = self._current_filter_config()
        if not filter_config.has_active_threshold():
            self._show_warning("No Active Filter", "Set at least one filter threshold first.")
            return
        before = clone_records(self.records)
        disabled = self.postprocess_session.apply_filter(self.records, filter_config)
        if disabled <= 0:
            self.statusBar().showMessage("No detections matched the filter.", 5000)
            return
        self.postprocess_session.push_undo_snapshot(before)
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table()
        self.statusBar().showMessage(f"Disabled {disabled} detections.", 5000)

    def _run_tracking(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Start postprocessing before running tracking.")
            return
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "Another background task is already active.")
            return
        self.tracking_worker = TrackingWorker(records=clone_records(self.records), config=TrackingConfig(), parent=self)
        self.tracking_worker.progressChanged.connect(self._on_run_progress)
        self.tracking_worker.runCompleted.connect(self._on_tracking_completed)
        self.tracking_worker.runFailed.connect(self._on_tracking_failed)

        self._set_running_state(True)
        self._set_idle_progress("Tracking...")
        self.statusBar().showMessage("Tracking started.")
        self.tracking_worker.start()

    def _on_tracking_completed(self, payload: object) -> None:
        data = dict(payload)
        worker = self.tracking_worker
        self.tracking_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running_state(False)

        if data.get("canceled"):
            self.statusBar().showMessage("Tracking canceled.", 5000)
            self._set_idle_progress("Canceled")
            self._refresh_info_panel()
            return

        updated_records = list(data.get("records", []))
        self.postprocess_session.push_undo_snapshot(self.records)
        self.records = updated_records
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_table(preserve_det_id=self._selected_det_id())
        self._refresh_preview()
        self._refresh_track_table(preserve_track_id=self._selected_track_id())
        self._set_idle_progress("Done")
        self.statusBar().showMessage("Tracking finished.", 5000)
        self._refresh_info_panel()

    def _on_tracking_failed(self, error_message: str) -> None:
        worker = self.tracking_worker
        self.tracking_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running_state(False)
        self._set_idle_progress("Failed")
        self._show_warning("Tracking Failed", error_message)
        self._refresh_info_panel()

    def _undo_last_change(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Undo is only available in postprocessing.")
            return
        try:
            self.records = self.postprocess_session.undo()
        except Exception as exc:
            self._show_warning("Undo Failed", str(exc))
            return
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table()
        self.statusBar().showMessage("Reverted the last change.", 5000)

    def _save_latest_now(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Latest is only available in postprocessing.")
            return
        self._persist_active_records()
        self.statusBar().showMessage("Latest CSV saved.", 4000)

    def _reset_to_raw(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Start postprocessing before resetting to raw.")
            return
        before = clone_records(self.records)
        try:
            active_path, records = self.postprocess_session.reset_to_raw()
        except Exception as exc:
            self._show_warning("Reset Failed", str(exc))
            return
        self.postprocess_session.push_undo_snapshot(before)
        self.current_output_csv_path = active_path
        self.records = records
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table()
        self.statusBar().showMessage("Latest restored from raw detection.", 5000)

    def _delete_selected_track(self) -> None:
        self._mutate_selected_track(
            mutation=lambda track_id: self.postprocess_session.delete_track(self.records, track_id),
            success_format="Disabled {count} detections from track {track_id}.",
        )

    def _lock_selected_track(self) -> None:
        self._mutate_selected_track(
            mutation=lambda track_id: self.postprocess_session.lock_track(self.records, track_id),
            success_format="Locked {count} detections in track {track_id}.",
        )

    def _unlock_selected_track(self) -> None:
        self._mutate_selected_track(
            mutation=lambda track_id: self.postprocess_session.unlock_track(self.records, track_id),
            success_format="Unlocked {count} detections in track {track_id}.",
        )

    def _merge_selected_track(self) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Track editing is only available in postprocessing.")
            return
        source_track_id = self._selected_track_id()
        target_track_id = str(self.merge_target_combo.currentData() or "").strip()
        if not source_track_id:
            self._show_warning("No Track Selected", "Select the source track first.")
            return
        before = clone_records(self.records)
        try:
            updated = self.postprocess_session.merge_tracks(self.records, source_track_id, target_track_id)
        except Exception as exc:
            self._show_warning("Merge Failed", str(exc))
            return
        self.postprocess_session.push_undo_snapshot(before)
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table(preserve_track_id=target_track_id)
        self.statusBar().showMessage(
            f"Merged {updated} detections from track {source_track_id} into {target_track_id}.",
            5000,
        )

    def _mutate_selected_track(
        self,
        mutation,
        success_format: str,
    ) -> None:
        if not self._is_postprocessing_stage():
            self._show_warning("Postprocessing Required", "Track editing is only available in postprocessing.")
            return
        track_id = self._selected_track_id()
        if not track_id:
            self._show_warning("No Track Selected", "Select one track first.")
            return
        before = clone_records(self.records)
        try:
            count = int(mutation(track_id))
        except Exception as exc:
            self._show_warning("Track Update Failed", str(exc))
            return
        self.postprocess_session.push_undo_snapshot(before)
        self._persist_active_records()
        self._refresh_outlier_analysis()
        self._refresh_current_frame_view()
        self._refresh_track_table(preserve_track_id=track_id)
        self.statusBar().showMessage(success_format.format(count=count, track_id=track_id), 5000)

    def _persist_active_records(self) -> None:
        path = self.postprocess_session.save_records(self.records)
        self.current_output_csv_path = path
        if self._is_postprocessing_stage():
            self.current_latest_csv_path = path
        else:
            self.current_raw_csv_path = path
        self.output_csv_edit.setText(str(path))
        self.stage_label.setText(f"Stage: {self._current_stage().value}")

    def _refresh_preview(self) -> None:
        if not self.current_provider:
            self.canvas.clear()
            return

        try:
            frame_rgb = self.current_provider.get_frame(self.current_frame_index)
            frame_records = self._current_frame_records()
            prompt_spec = self._current_prompt_spec() if self._is_detection_stage() else None
            preview = self.detector.render_frame_preview(
                frame_rgb=frame_rgb,
                records=frame_records,
                prompt_spec=prompt_spec,
                highlight_det_id=self._selected_det_id(),
                intrinsics_override=self._preview_intrinsics_for_current_view(frame_rgb.shape[:2]),
            )
            self.canvas.set_image(preview)
            self.canvas.set_detection_boxes([])
            self.canvas.set_highlight_box(None)
            self.canvas.set_prompt_box(
                self.current_prompt_box
                if self._is_detection_stage() and self.current_prompt_mode() in {PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE}
                else None
            )
            self.canvas.set_prompt_points(
                self.current_prompt_points if self._is_detection_stage() and self.current_prompt_mode() == PromptMode.POINT else []
            )
        except Exception as exc:
            self.canvas.clear()
            self.statusBar().showMessage(f"Preview refresh failed: {exc}", 8000)

    def _refresh_table(self, preserve_det_id: str | None = None) -> None:
        frame_records = self._current_frame_records()
        selected_det_id = preserve_det_id or self._selected_det_id()
        self.current_frame_det_ids = [record.det_id for record in frame_records]
        allow_edit = self._is_postprocessing_stage()

        self._results_table_internal_change = True
        with QSignalBlocker(self.results_table):
            self.results_table.setRowCount(len(frame_records))
            for row_index, record in enumerate(frame_records):
                background_color = self._outlier_color_for_record(record)
                for col_index, (_, field_name, value_kind, editable) in enumerate(RESULTS_TABLE_COLUMNS):
                    item = QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, record.det_id)
                    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    if editable and value_kind != "bool" and allow_edit:
                        flags |= Qt.ItemFlag.ItemIsEditable
                    if value_kind == "bool":
                        if allow_edit:
                            flags |= Qt.ItemFlag.ItemIsUserCheckable
                        item.setFlags(flags)
                        item.setCheckState(Qt.CheckState.Checked if record.is_enabled else Qt.CheckState.Unchecked)
                    else:
                        item.setFlags(flags)
                        item.setText(self._format_result_field(record, field_name, value_kind))
                    if background_color is not None:
                        item.setBackground(background_color)
                    self.results_table.setItem(row_index, col_index, item)

            selected_row = -1
            if selected_det_id:
                for idx, det_id in enumerate(self.current_frame_det_ids):
                    if det_id == selected_det_id:
                        selected_row = idx
                        break
            if selected_row >= 0:
                self.results_table.setCurrentCell(selected_row, 0)
            else:
                self.results_table.clearSelection()
        self._results_table_internal_change = False
        if not self.current_provider:
            self.results_hint_label.setText("Load a dataset, video, or image folder to inspect detections.")
        elif not self.records:
            self.results_hint_label.setText("No detections are loaded yet. Run detection or open a source with saved results.")
        elif not frame_records:
            self.results_hint_label.setText("No detections exist in the current frame.")
        else:
            self.results_hint_label.setText(
                "Double-click editable cells to update 3D boxes. "
                "Only checked rows are visualized on the left."
            )
        self._sync_selection_inputs()

    def _current_frame_records(self) -> list[DetectionRecord]:
        records = [record for record in self.records if record.frame_index == self.current_frame_index]
        return sorted(records, key=lambda record: (-record.score, record.det_id))

    def _load_csv_side_inputs(self) -> None:
        self.stage_label.setText(f"Stage: {self._current_stage().value}")
        with QSignalBlocker(self.annotation_csv_path_edit):
            if self.current_annotation_csv_override_path is not None:
                self.annotation_csv_path_edit.setText(str(self.current_annotation_csv_override_path))
            else:
                self.annotation_csv_path_edit.clear()
        self._update_action_states()

    def _save_csv_now(self) -> None:
        if not self.current_provider:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        self._persist_active_records()
        self.statusBar().showMessage("CSV saved.", 4000)

    def _autosave(self) -> None:
        if self.current_provider is not None and self.current_output_csv_path is not None:
            self._persist_active_records()

    def _open_result_folder(self) -> None:
        target = self.current_result_dir
        if target is None and self.dataset_config is not None:
            target = self.dataset_config_store.ensure_result_dir(self.dataset_config, on_date=date.today())
        if target is None:
            self._show_warning("No Result Folder", "Result folder is not ready yet.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _on_prompt_mode_changed(self) -> None:
        self._sync_prompt_ui()
        self._refresh_current_frame_view()

    def _on_point_label_changed(self) -> None:
        self.canvas.set_point_label(self.point_label_combo.currentData() or 1)

    def _clear_current_prompt(self) -> None:
        self.current_prompt_box = None
        self.current_prompt_points = []
        self.text_prompt_edit.clear()
        self.prompt_label_edit.clear()
        self.canvas.set_prompt_box(None)
        self.canvas.set_prompt_points([])
        self._refresh_current_frame_view()

    def _copy_prompt_to_range(self) -> None:
        prompt = self._current_prompt_spec()
        if not prompt.is_valid():
            self._show_warning("Prompt Incomplete", "Finish the current prompt before copying it to the range.")
            return
        self.copied_range_prompt = prompt.clone()
        self.copied_prompt_label.setText(self._prompt_summary(self.copied_range_prompt))
        self.statusBar().showMessage("Range prompt updated.", 4000)

    def _on_prompt_box_completed(self, box: tuple[float, float, float, float]) -> None:
        self.current_prompt_box = tuple(float(value) for value in box)
        self.current_prompt_points = []
        self.canvas.set_prompt_box(self.current_prompt_box)
        self.canvas.set_prompt_points([])
        self._refresh_current_frame_view()

    def _on_prompt_point_added(self, point: PointPrompt | None) -> None:
        self.current_prompt_box = None
        if point is None:
            if self.current_prompt_points:
                self.current_prompt_points.pop()
        else:
            self.current_prompt_points.append(point)
        self.canvas.set_prompt_box(None)
        self.canvas.set_prompt_points(self.current_prompt_points)
        self._refresh_current_frame_view()

    def _on_intrinsics_mode_changed(self) -> None:
        enabled = self.use_actual_k_checkbox.isChecked()
        for widget in (self.fx_spin, self.fy_spin, self.cx_spin, self.cy_spin):
            widget.setEnabled(enabled)
        self._refresh_info_panel()

    def _step_frame(self, delta: int) -> None:
        if not self.current_provider:
            return
        next_index = min(max(self.current_frame_index + int(delta), 0), self.current_provider.frame_count - 1)
        self._set_current_frame(next_index)

    def _on_frame_slider_changed(self, value: int) -> None:
        self._set_current_frame(int(value))

    def _on_frame_spin_changed(self, value: int) -> None:
        self._set_current_frame(int(value))

    def _set_current_frame(self, frame_index: int) -> None:
        if not self.current_provider:
            return
        frame_index = min(max(int(frame_index), 0), self.current_provider.frame_count - 1)
        if frame_index == self.current_frame_index and self.current_provider is not None:
            with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_index_spin):
                self.frame_slider.setValue(frame_index)
                self.frame_index_spin.setValue(frame_index)
            return
        self.current_frame_index = frame_index
        with QSignalBlocker(self.frame_slider), QSignalBlocker(self.frame_index_spin):
            self.frame_slider.setValue(frame_index)
            self.frame_index_spin.setValue(frame_index)
        self._refresh_current_frame_view()
        self._update_action_states()

    def _run_current_frame(self) -> None:
        if not self.current_provider or not self.current_source_context:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        if not self._is_detection_stage():
            self._show_warning("Detection Disabled", "Detection is disabled after postprocessing starts.")
            return
        prompt = self._current_prompt_spec()
        if not prompt.is_valid():
            self._show_warning("Prompt Incomplete", "Finish the current prompt before running detection.")
            return
        self._start_detection(
            frame_indices=[self.current_frame_index],
            prompt_spec=prompt,
            replace_target_id=None,
        )

    def _run_selected_range(self) -> None:
        if not self.current_provider or not self.current_source_context:
            self._show_warning("No Source", "Load a dataset, video, or image folder first.")
            return
        if not self._is_detection_stage():
            self._show_warning("Detection Disabled", "Detection is disabled after postprocessing starts.")
            return
        prompt = self._range_prompt_spec()
        config = self._current_detection_config()
        frame_indices = config.clamped_range(self.current_provider.frame_count)
        if not frame_indices:
            self._show_warning("Invalid Range", "Adjust start and end frame first.")
            return
        self._start_detection(
            frame_indices=frame_indices,
            prompt_spec=prompt,
            replace_target_id=None,
        )

    def _replace_selected_detection(self) -> None:
        selected = self._selected_record()
        if not selected:
            self._show_warning("No Selection", "Select one detection to replace first.")
            return
        if not self._is_detection_stage():
            self._show_warning("Detection Disabled", "Replacement is disabled after postprocessing starts.")
            return
        prompt = self._current_prompt_spec()
        if prompt.mode not in {PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE, PromptMode.POINT}:
            self._show_warning("Unsupported Prompt", "Replacement requires a box or point prompt.")
            return
        if not prompt.is_valid():
            self._show_warning("Prompt Incomplete", "Finish the current box or point prompt first.")
            return
        self._start_detection(
            frame_indices=[self.current_frame_index],
            prompt_spec=prompt,
            replace_target_id=selected.det_id,
        )

    def _start_detection(
        self,
        frame_indices: list[int],
        prompt_spec: PromptSpec,
        replace_target_id: str | None,
    ) -> None:
        if self._is_background_task_running():
            self._show_warning("Run in Progress", "A detection run is already active.")
            return
        if not self.current_provider or not self.current_source_context:
            return

        try:
            config = self._current_detection_config()
        except Exception as exc:
            self._show_warning("Invalid Config", str(exc))
            return

        self.detection_worker = DetectionWorker(
            detector=self.detector,
            provider=self.current_provider,
            frame_indices=frame_indices,
            prompt_spec=prompt_spec,
            config=config,
            source_context=self.current_source_context,
            replace_target_id=replace_target_id,
            parent=self,
        )
        self.detection_worker.progressChanged.connect(self._on_run_progress)
        self.detection_worker.runCompleted.connect(self._on_run_completed)
        self.detection_worker.runFailed.connect(self._on_run_failed)

        self._set_running_state(True)
        self.model_status_text = "Running detection (loading model on demand)"
        self._refresh_info_panel()
        self.statusBar().showMessage("Detection started. Loading model on demand...")
        self.detection_worker.start()

    def _cancel_run(self) -> None:
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.cancel()
            self.statusBar().showMessage("Cancel requested...", 4000)
            return
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.cancel()
            self.statusBar().showMessage("Tracking cancel requested...", 4000)

    def _on_run_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(current, total))
        self.progress_bar.setFormat(f"{message} ({current}/{total})")
        self.statusBar().showMessage(message)

    def _on_run_completed(self, payload: object) -> None:
        data = dict(payload)
        worker = self.detection_worker
        self.detection_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running_state(False)

        if data.get("canceled"):
            self.model_status_text = "Model idle (GPU released)"
            self.statusBar().showMessage("Detection canceled.", 5000)
            self._set_idle_progress("Canceled")
            self._refresh_info_panel()
            return

        records = list(data.get("records", []))
        replace_target_id = data.get("replace_target_id")
        prompt_spec = data.get("prompt_spec")
        frame_indices = list(data.get("frame_indices", []))

        try:
            had_records = bool(self.records)
            if replace_target_id:
                self._apply_replacement_result(replace_target_id, records)
                preserved = replace_target_id
            else:
                self._merge_run_results(records, frame_indices, prompt_spec)
                preserved = records[0].det_id if records else self._selected_det_id()
            self._autosave()
            if records and (not had_records or not self._current_frame_records()):
                self._expand_results_panel_if_needed(force=True)
            else:
                self._expand_results_panel_if_needed(force=not had_records and bool(self.records))
            self._refresh_table(preserve_det_id=preserved)
            self._refresh_preview()
            self._refresh_track_table()
            self._set_idle_progress("Done")
            self.model_status_text = "Model idle (GPU released)"
            self.statusBar().showMessage("Detection finished.", 5000)
        except Exception as exc:
            self._show_warning("Result Merge Failed", str(exc))
            self._set_idle_progress("Failed")
            self.model_status_text = "Model idle (GPU released)"
        self._refresh_info_panel()

    def _on_run_failed(self, error_message: str) -> None:
        worker = self.detection_worker
        self.detection_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running_state(False)
        self._set_idle_progress("Failed")
        self.model_status_text = "Model idle (GPU released)"
        self._show_warning("Detection Failed", error_message)
        self._refresh_info_panel()

    def _merge_run_results(
        self,
        new_records: list[DetectionRecord],
        frame_indices: Iterable[int],
        prompt_spec: PromptSpec | None,
    ) -> None:
        del prompt_spec
        frame_set = {int(index) for index in frame_indices}
        self.records = [
            record
            for record in self.records
            if record.frame_index not in frame_set
        ]
        self.records.extend(new_records)

    def _apply_replacement_result(self, target_det_id: str, new_records: list[DetectionRecord]) -> None:
        target_index = next((idx for idx, record in enumerate(self.records) if record.det_id == target_det_id), -1)
        if target_index < 0:
            raise ValueError("The selected detection no longer exists.")
        if not new_records:
            raise ValueError("The replacement prompt produced no detection.")

        replacement = max(new_records, key=lambda record: record.score)
        original = self.records[target_index]
        replacement.det_id = original.det_id
        replacement.track_id = original.track_id
        replacement.track_status = original.track_status
        replacement.is_enabled = original.is_enabled
        self.records[target_index] = replacement

    def _selected_det_id(self) -> str | None:
        row = self.results_table.currentRow()
        if row < 0:
            return None
        item = self.results_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _selected_record(self) -> DetectionRecord | None:
        det_id = self._selected_det_id()
        if not det_id:
            return None
        return self._record_by_det_id(det_id)

    def _record_by_det_id(self, det_id: str) -> DetectionRecord | None:
        for record in self.records:
            if record.det_id == det_id:
                return record
        return None

    def _on_results_selection_changed(self) -> None:
        self._sync_selection_inputs()
        self._refresh_preview()

    def _sync_selection_inputs(self) -> None:
        self._update_action_states()

    def _on_results_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._results_table_internal_change:
            return
        if not self._is_postprocessing_stage():
            return
        det_id = item.data(Qt.ItemDataRole.UserRole)
        if not det_id:
            return
        record = self._record_by_det_id(str(det_id))
        if record is None:
            return
        column = item.column()
        if not (0 <= column < len(RESULTS_TABLE_COLUMNS)):
            return
        _, field_name, value_kind, editable = RESULTS_TABLE_COLUMNS[column]
        if not editable:
            return
        self._commit_results_item_change(record, item, field_name, value_kind)

    def _commit_results_item_change(
        self,
        record: DetectionRecord,
        item: QTableWidgetItem,
        field_name: str,
        value_kind: str,
    ) -> None:
        before = clone_records(self.records)
        if value_kind == "bool":
            old_value = bool(record.is_enabled)
            record.is_enabled = item.checkState() == Qt.CheckState.Checked
            if record.is_enabled == old_value:
                return
            self.postprocess_session.push_undo_snapshot(before)
            self._autosave()
            self._refresh_outlier_analysis()
            self._refresh_preview()
            self._refresh_track_table()
            self._refresh_info_panel()
            return
        new_text = item.text().strip()
        old_value = getattr(record, field_name)
        old_track_status = record.track_status
        try:
            if value_kind == "float":
                parsed_value = float(new_text)
                if not np.isfinite(parsed_value):
                    raise ValueError("Value must be finite.")
                if field_name in {"size_w", "size_l", "size_h"} and parsed_value <= 0:
                    raise ValueError("3D size values must be greater than 0.")
                setattr(record, field_name, parsed_value)
                if field_name in GEOMETRY_RESULT_FIELDS:
                    intrinsics = self._intrinsics_for_record_projection(record)
                    if self.current_provider is None:
                        raise ValueError("No active media source.")
                    (
                        record.box2d_x1,
                        record.box2d_y1,
                        record.box2d_x2,
                        record.box2d_y2,
                    ) = self.detector.project_record_to_box2d(
                        record=record,
                        intrinsics=intrinsics,
                        image_shape=self.current_provider.frame_shape(record.frame_index),
                    )
            else:
                if field_name == "category" and not new_text:
                    raise ValueError("Category cannot be empty.")
                if field_name == "track_id" and record.track_status == "locked":
                    raise ValueError("Unlock this track before editing its track ID.")
                setattr(record, field_name, new_text)
                if field_name == "track_id":
                    record.track_status = "manual" if new_text else ""
        except Exception as exc:
            setattr(record, field_name, old_value)
            record.track_status = old_track_status
            self._results_table_internal_change = True
            with QSignalBlocker(self.results_table):
                if value_kind == "bool":
                    item.setCheckState(Qt.CheckState.Checked if bool(old_value) else Qt.CheckState.Unchecked)
                else:
                    item.setText(self._format_result_field(record, field_name, value_kind))
            self._results_table_internal_change = False
            self.statusBar().showMessage(f"Edit rejected: {exc}", 6000)
            return

        if getattr(record, field_name) == old_value and record.track_status == old_track_status:
            self._results_table_internal_change = True
            with QSignalBlocker(self.results_table):
                item.setText(self._format_result_field(record, field_name, value_kind))
            self._results_table_internal_change = False
            return

        self._results_table_internal_change = True
        with QSignalBlocker(self.results_table):
            item.setText(self._format_result_field(record, field_name, value_kind))
        self._results_table_internal_change = False
        self.postprocess_session.push_undo_snapshot(before)
        self._autosave()
        self._refresh_outlier_analysis()
        self._refresh_preview()
        self._refresh_track_table()
        self._refresh_info_panel()

    def _format_result_field(self, record: DetectionRecord, field_name: str, value_kind: str) -> str:
        if value_kind == "computed" and field_name == "_outlier_flags":
            return self._outlier_flags_for_record(record)
        value = getattr(record, field_name)
        if value_kind == "score":
            return f"{float(value):.3f}"
        if value_kind == "float":
            return f"{float(value):.6f}"
        return str(value)

    def _preview_intrinsics_for_current_view(self, image_shape: tuple[int, int]) -> np.ndarray | None:
        if self.use_actual_k_checkbox.isChecked():
            try:
                return self._current_detection_config().to_intrinsics_matrix()
            except Exception:
                pass
        frame_records = self._current_frame_records()
        if frame_records:
            return frame_records[0].intrinsics_for_preview(image_shape)
        return None

    def _intrinsics_for_record_projection(self, record: DetectionRecord) -> np.ndarray:
        if self.use_actual_k_checkbox.isChecked():
            intrinsics = self._current_detection_config().to_intrinsics_matrix()
            if intrinsics is not None:
                return intrinsics
        if None not in (record.pred_fx, record.pred_fy, record.pred_cx, record.pred_cy):
            return np.array(
                [
                    [float(record.pred_fx), 0.0, float(record.pred_cx)],
                    [0.0, float(record.pred_fy), float(record.pred_cy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        raise ValueError("3D editing requires actual intrinsics enabled for this session.")

    def _current_prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            mode=self.current_prompt_mode(),
            text_prompt=self.text_prompt_edit.text().strip(),
            prompt_label=self.prompt_label_edit.text().strip(),
            box=self.current_prompt_box,
            points=list(self.current_prompt_points),
        )

    def _range_prompt_spec(self) -> PromptSpec:
        prompt = self._current_prompt_spec()
        if prompt.mode == PromptMode.TEXT:
            return prompt
        if self.copied_range_prompt and self.copied_range_prompt.is_valid():
            return self.copied_range_prompt.clone()
        raise ValueError("Box/Point range detection requires 'Copy Current Prompt To Range' first.")

    def _current_detection_config(self) -> DetectionConfig:
        config = DetectionConfig(
            start_frame=int(self.start_frame_spin.value()),
            end_frame=int(self.end_frame_spin.value()),
            frame_step=int(self.frame_step_spin.value()),
            score_threshold=float(self.score_threshold_spin.value()),
            score_3d_threshold=float(self.score_3d_threshold_spin.value()),
            cross_category_nms_iou=float(self.nms_iou_spin.value()),
            use_actual_intrinsics=self.use_actual_k_checkbox.isChecked(),
            fx=float(self.fx_spin.value()) if self.use_actual_k_checkbox.isChecked() else None,
            fy=float(self.fy_spin.value()) if self.use_actual_k_checkbox.isChecked() else None,
            cx=float(self.cx_spin.value()) if self.use_actual_k_checkbox.isChecked() else None,
            cy=float(self.cy_spin.value()) if self.use_actual_k_checkbox.isChecked() else None,
        )
        config.to_intrinsics_matrix()
        return config

    def _prompt_summary(self, prompt: PromptSpec) -> str:
        mode = PromptMode(prompt.mode)
        if mode == PromptMode.TEXT:
            return f"{mode.value}: {', '.join(prompt.parsed_texts())}"
        if mode in {PromptMode.BOX_MULTI, PromptMode.BOX_SINGLE}:
            return f"{mode.value}: {prompt.box}"
        return f"{mode.value}: {len(prompt.points)} point(s)"

    def _refresh_output_path_display(self) -> None:
        if self.current_output_csv_path is not None:
            self.output_csv_edit.setText(str(self.current_output_csv_path))
            return
        config = self.dataset_config
        if config is None:
            self.output_csv_edit.clear()
            return
        mode = self.current_source_mode()
        try:
            if mode == SourceMode.DATASET and self.selected_dataset_id and self.selected_recording_id:
                path = self.dataset_config_store.resolve_output_csv_path(
                    config=config,
                    source_mode=mode,
                    dataset_id=self.selected_dataset_id,
                    recording_id=self.selected_recording_id,
                    on_date=date.today(),
                )
            elif mode in {SourceMode.VIDEO, SourceMode.IMAGE_FOLDER} and self.manual_source_path and self._manual_path_matches_mode(mode):
                path = self.dataset_config_store.resolve_output_csv_path(
                    config=config,
                    source_mode=mode,
                    manual_source_path=self.manual_source_path,
                    on_date=date.today(),
                )
            else:
                path = None
        except Exception:
            path = None
        self.output_csv_edit.setText(str(path) if path else "")

    def _refresh_info_panel(self) -> None:
        lines = [f"Model: {self.model_status_text}"]
        lines.append(f"Stage: {self._current_stage().value}")
        if self.current_source_context is None:
            lines.append("Source: none")
        else:
            lines.append(f"Source Mode: {self.current_source_context.source_mode.value}")
            lines.append(f"Source Type: {self.current_source_context.source_type}")
            if self.current_source_context.dataset_id:
                lines.append(f"Dataset: {self.current_source_context.dataset_id}")
            if self.current_source_context.recording_id:
                lines.append(f"Recording: {self.current_source_context.recording_id}")
        if self.current_provider is not None:
            lines.append(f"Media Path: {self.current_provider.path}")
            lines.append(f"Frames: {self.current_provider.frame_count}")
            if self.current_provider.fps:
                lines.append(f"FPS: {self.current_provider.fps:.3f}")
        if self.current_raw_csv_path is not None:
            lines.append(f"Raw CSV: {self.current_raw_csv_path}")
        if self.current_latest_csv_path is not None:
            lines.append(f"Latest CSV: {self.current_latest_csv_path}")
        if self.current_output_csv_path is not None:
            lines.append(f"Active CSV: {self.current_output_csv_path}")
        if self.current_result_dir is not None:
            lines.append(f"Result Folder: {self.current_result_dir}")
        lines.append(f"Detections Loaded: {len(self.records)}")
        lines.append(f"Enabled Detections: {sum(1 for record in self.records if record.is_enabled)}")
        lines.append(f"Outlier Hits: {len(self.outlier_hits_global)}")
        lines.append(f"Outlier Frames: {len(self.outlier_frames)}")
        lines.append(
            f"Intrinsics Mode: {'actual' if self.use_actual_k_checkbox.isChecked() else 'predicted'}"
        )
        self.info_label.setText("\n".join(lines))

    def _update_action_states(self) -> None:
        has_source = self.current_provider is not None
        has_selection = self._selected_record() is not None
        has_track_selection = self._selected_track_id() is not None
        detection_running = self._is_detection_running()
        tracking_running = self._is_tracking_running()
        running = self._controls_locked or detection_running or tracking_running
        postprocessing = self._is_postprocessing_stage()
        detection_stage = self._is_detection_stage()

        self.navigation_widget.setEnabled(has_source and not detection_running)
        self.prompt_group.setEnabled(has_source and detection_stage and not running)
        self.detection_params_group.setEnabled(has_source and detection_stage and not running)
        self.source_group.setEnabled(not running)
        self.results_group.setEnabled(has_source and not running)
        self.postprocess_group.setEnabled(has_source)
        self.outlier_group.setEnabled(has_source and postprocessing)
        self.bulk_ops_group.setEnabled(has_source and postprocessing)
        self.track_group.setEnabled(has_source and postprocessing)

        self.run_current_btn.setEnabled(has_source and detection_stage and not running)
        self.run_range_btn.setEnabled(has_source and detection_stage and not running)
        self.cancel_run_btn.setEnabled(detection_running or tracking_running)
        self.annotation_csv_path_edit.setEnabled(not running)
        self.browse_annotation_csv_btn.setEnabled(not running)
        self.load_annotation_csv_btn.setEnabled(
            has_source and not running and bool(self.annotation_csv_path_edit.text().strip())
        )
        self.use_source_csv_btn.setEnabled(
            has_source and not running and self.current_source_output_csv_path is not None
        )
        self.save_csv_btn.setEnabled(has_source and not running)
        self.open_result_folder_btn.setEnabled(self.current_result_dir is not None)

        self.replace_selected_btn.setEnabled(has_selection and detection_stage and not running)

        self.start_postprocess_btn.setEnabled(
            has_source and detection_stage and not running and self.postprocess_session.can_start_postprocessing(self.records)
        )
        self.run_tracking_btn.setEnabled(has_source and postprocessing and not running)
        self.undo_change_btn.setEnabled(has_source and postprocessing and not running and self.postprocess_session.has_undo())
        self.save_latest_btn.setEnabled(has_source and postprocessing and not running)
        self.reset_to_raw_btn.setEnabled(
            has_source and postprocessing and not running and self.current_raw_csv_path is not None and self.current_raw_csv_path.exists()
        )
        self.apply_filter_btn.setEnabled(has_source and postprocessing and not running)
        self.processing_scope_combo.setEnabled(has_source and postprocessing and not running)

        self.outlier_rule_table.setEnabled(has_source and postprocessing and not running)
        self.refresh_outliers_btn.setEnabled(has_source and postprocessing and not running)
        selected_hit = self._selected_outlier_hit()
        self.fix_selected_outlier_btn.setEnabled(
            has_source and postprocessing and not running and selected_hit is not None and bool(selected_hit.fixable)
        )
        self.fix_scope_outliers_btn.setEnabled(
            has_source
            and postprocessing
            and not running
            and self._selected_outlier_rule_id() is not None
            and bool(self._scope_filtered_hits(rule_id=self._selected_outlier_rule_id()))
        )
        self.fix_all_visible_outliers_btn.setEnabled(
            has_source and postprocessing and not running and bool(self.outlier_hits_visible)
        )
        self.outlier_table.setEnabled(has_source and postprocessing and not running)

        operation_id = self._selected_bulk_operation_id()
        self.bulk_operation_combo.setEnabled(has_source and postprocessing and not running)
        self.apply_bulk_operation_btn.setEnabled(
            has_source and postprocessing and not running and bool(operation_id)
        )

        self.track_table.setEnabled(has_source and postprocessing and not running)
        self.merge_target_combo.setEnabled(has_source and postprocessing and not running)
        self.delete_track_btn.setEnabled(has_track_selection and postprocessing and not running)
        self.lock_track_btn.setEnabled(has_track_selection and postprocessing and not running)
        self.unlock_track_btn.setEnabled(has_track_selection and postprocessing and not running)
        selected_track_id = self._selected_track_id() or ""
        merge_target = str(self.merge_target_combo.currentData() or "").strip()
        self.merge_track_btn.setEnabled(
            has_track_selection and postprocessing and not running and bool(merge_target) and merge_target != selected_track_id
        )

    def _set_running_state(self, running: bool) -> None:
        self._controls_locked = bool(running)
        self._update_action_states()

    def _set_idle_progress(self, label: str = "Idle") -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(label)

    def _manual_path_matches_mode(self, mode: SourceMode) -> bool:
        if self.manual_source_path is None:
            return False
        if mode == SourceMode.VIDEO:
            return self.manual_source_path.is_file()
        if mode == SourceMode.IMAGE_FOLDER:
            return self.manual_source_path.is_dir()
        return False

    def _is_detection_running(self) -> bool:
        return bool(self.detection_worker and self.detection_worker.isRunning())

    def _row_widget(self, first: QWidget, second: QWidget, third: QWidget | None = None, stretch_first: bool = False) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(first, 1 if stretch_first else 0)
        layout.addWidget(second)
        if third is not None:
            layout.addWidget(third)
        return container

    @staticmethod
    def _make_int_spinbox(value: int, minimum: int, maximum: int) -> NoWheelSpinBox:
        widget = NoWheelSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _make_float_spinbox(
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
    ) -> NoWheelDoubleSpinBox:
        widget = NoWheelDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.cancel()
            self.detection_worker.wait()
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.cancel()
            self.tracking_worker.wait()
        self.detector.release_models()
        self._close_provider()
        super().closeEvent(event)
