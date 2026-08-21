"""
Auto-Annotate (Zero-Shot) Dialog - build a labelled YOLO dataset from text
prompts using Grounding DINO, then hand the result off to the existing
dataset importer to merge it into the current project.

This dialog only talks to core.zero_shot_annotator (guarded import inside
that module) and core.dataset_importer (already used by the FiftyOne/COCO
import feature) - no existing module is modified to support it.
"""
import time
from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QLineEdit, QFileDialog, QTableWidget,
                             QDoubleSpinBox, QCheckBox, QGroupBox, QProgressBar,
                             QTextEdit, QMessageBox, QHeaderView)
from PyQt6.QtCore import QThread, pyqtSignal

from core.zero_shot_annotator import (ZeroShotAnnotator, is_available,
                                      get_availability_message)
from core.logger import get_logger

logger = get_logger(__name__)


class _AnnotateWorker(QThread):
    """Runs ZeroShotAnnotator off the UI thread."""

    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ontology, source_dir, output_dir,
                box_threshold, text_threshold,
                use_nms, nms_threshold, class_agnostic_nms, parent=None):
        super().__init__(parent)
        self.ontology = ontology
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.class_agnostic_nms = class_agnostic_nms
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self.progress.emit(
                0, 0,
                "Loading Grounding DINO model (first run may download weights)..."
            )
            annotator = ZeroShotAnnotator(
                self.ontology,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            stats = annotator.run(
                self.source_dir, self.output_dir,
                use_nms=self.use_nms,
                nms_threshold=self.nms_threshold,
                class_agnostic_nms=self.class_agnostic_nms,
                progress_callback=lambda cur, total, msg:
                    self.progress.emit(cur, total, msg),
                should_stop=lambda: self._stop_requested,
            )
            self.finished_ok.emit(stats)
        except Exception as e:
            logger.error(f"Auto-annotation worker failed: {e}", exc_info=True)
            self.failed.emit(str(e))


class AutoAnnotateDialog(QDialog):
    """Dialog for zero-shot dataset auto-annotation + hand-off to project."""

    # Emitted after a successful "Import into Current Project" with the
    # merged class list, so main_window can refresh its widgets the same
    # way it does for the COCO/YOLO importer.
    dataset_imported = pyqtSignal(list)

    def __init__(self, project_path: Path, existing_classes, parent=None):
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.existing_classes = list(existing_classes)
        self.worker = None
        self._last_stats = None
        self._last_output_dir = None

        self.default_source = self.project_path / 'input_images'
        self.default_output = self.project_path / 'yolo_dataset'
        # Requirement: auto-create the intake/output folders up front.
        self.default_source.mkdir(parents=True, exist_ok=True)
        self.default_output.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("Auto-Annotate (Zero-Shot)")
        self.setMinimumSize(760, 620)
        self._init_ui()
        self._refresh_image_count()

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        header = QLabel("Auto-Annotate with Grounding DINO")
        header.setStyleSheet("font-size: 17px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel(
            "Describe objects in plain English; every image in the source "
            "folder gets auto-labelled, then can be imported into this "
            "project as a YOLO dataset."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8891a0; font-size: 12px;")
        layout.addWidget(desc)

        # Availability banner
        self.avail_label = QLabel()
        self.avail_label.setWordWrap(True)
        self.avail_label.setStyleSheet(
            "padding: 8px 12px; border-radius: 6px; font-size: 12px;"
        )
        layout.addWidget(self.avail_label)
        self._update_availability_banner()

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._build_setup_column(), 3)
        body.addLayout(self._build_run_column(), 2)
        layout.addLayout(body, 1)

        self.setLayout(layout)

    def _update_availability_banner(self):
        if is_available():
            self.avail_label.setText("✓ " + get_availability_message())
            self.avail_label.setStyleSheet(
                "background-color: rgba(45,125,70,0.18); color: #a6e8bb; "
                "padding: 8px 12px; border-radius: 6px; font-size: 12px;"
            )
        else:
            self.avail_label.setText("⚠ " + get_availability_message())
            self.avail_label.setStyleSheet(
                "background-color: rgba(224,168,62,0.15); color: #f0cd8e; "
                "padding: 8px 12px; border-radius: 6px; font-size: 12px;"
            )

    # ---- left column: setup -------------------------------------------
    def _build_setup_column(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        # Folders
        folders_group = QGroupBox("Folders")
        f_layout = QVBoxLayout()

        f_layout.addWidget(QLabel("Source (input_images/):"))
        src_row = QHBoxLayout()
        self.source_edit = QLineEdit(str(self.default_source))
        src_row.addWidget(self.source_edit, 1)
        btn_src = QPushButton("Browse...")
        btn_src.clicked.connect(self._on_browse_source)
        src_row.addWidget(btn_src)
        f_layout.addLayout(src_row)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #8891a0; font-size: 11px;")
        f_layout.addWidget(self.count_label)

        f_layout.addWidget(QLabel("Output (yolo_dataset/):"))
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(self.default_output))
        out_row.addWidget(self.output_edit, 1)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn_out)
        f_layout.addLayout(out_row)

        folders_group.setLayout(f_layout)
        col.addWidget(folders_group)

        # Ontology
        ont_group = QGroupBox("Ontology - Caption -> YOLO Class")
        ont_layout = QVBoxLayout()

        self.ont_table = QTableWidget(0, 3)
        self.ont_table.setHorizontalHeaderLabels(
            ["Caption (prompt)", "Class name", ""])
        self.ont_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.ont_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.ont_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self.ont_table.verticalHeader().setVisible(False)
        self.ont_table.setMinimumHeight(140)
        ont_layout.addWidget(self.ont_table)
        self._add_ontology_row()

        btn_add_row = QPushButton("+ Add Class")
        btn_add_row.clicked.connect(lambda: self._add_ontology_row())
        ont_layout.addWidget(btn_add_row)

        ont_group.setLayout(ont_layout)
        col.addWidget(ont_group)

        # Thresholds
        thresh_group = QGroupBox("Thresholds")
        t_layout = QVBoxLayout()

        box_row = QHBoxLayout()
        box_row.addWidget(QLabel("Box Threshold:"))
        self.box_threshold_spin = QDoubleSpinBox()
        self.box_threshold_spin.setRange(0.0, 1.0)
        self.box_threshold_spin.setSingleStep(0.05)
        self.box_threshold_spin.setDecimals(2)
        self.box_threshold_spin.setValue(ZeroShotAnnotator.DEFAULT_BOX_THRESHOLD)
        self.box_threshold_spin.setToolTip(
            "Higher = fewer false positives but may miss real objects.\n"
            "Lower = catches more, but more noise (rely on NMS + review).")
        box_row.addWidget(self.box_threshold_spin)
        t_layout.addLayout(box_row)

        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Text Threshold:"))
        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setRange(0.0, 1.0)
        self.text_threshold_spin.setSingleStep(0.05)
        self.text_threshold_spin.setDecimals(2)
        self.text_threshold_spin.setValue(ZeroShotAnnotator.DEFAULT_TEXT_THRESHOLD)
        self.text_threshold_spin.setToolTip(
            "Confidence that the caption matches what's seen in the image.")
        text_row.addWidget(self.text_threshold_spin)
        t_layout.addLayout(text_row)

        thresh_group.setLayout(t_layout)
        col.addWidget(thresh_group)

        # NMS
        nms_group = QGroupBox("NMS - Remove Overlapping Duplicate Boxes")
        n_layout = QVBoxLayout()

        self.nms_check = QCheckBox("Enable NMS")
        self.nms_check.setChecked(True)
        self.nms_check.setToolTip(
            "Merges multiple boxes drawn on the same object (common when "
            "objects sit close together) into a single best box.")
        n_layout.addWidget(self.nms_check)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU Threshold:"))
        self.nms_iou_spin = QDoubleSpinBox()
        self.nms_iou_spin.setRange(0.0, 1.0)
        self.nms_iou_spin.setSingleStep(0.05)
        self.nms_iou_spin.setDecimals(2)
        self.nms_iou_spin.setValue(ZeroShotAnnotator.DEFAULT_NMS_THRESHOLD)
        iou_row.addWidget(self.nms_iou_spin)
        n_layout.addLayout(iou_row)

        self.class_agnostic_check = QCheckBox(
            "Class-agnostic (also suppress overlaps across different classes)")
        self.class_agnostic_check.setChecked(True)
        n_layout.addWidget(self.class_agnostic_check)

        nms_group.setLayout(n_layout)
        col.addWidget(nms_group)

        col.addStretch()
        return col

    # ---- right column: run ---------------------------------------------
    def _build_run_column(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        run_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Auto-Annotate")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: #ffffff; "
            "border: none; border-radius: 8px; font-weight: 600; }"
            "QPushButton:hover { background-color: #339952; }"
            "QPushButton:disabled { background-color: #3a3f4a; color: #6c7280; }"
        )
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setEnabled(is_available())
        if not is_available():
            self.btn_start.setToolTip(get_availability_message())
        run_row.addWidget(self.btn_start, 1)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        run_row.addWidget(self.btn_stop)
        col.addLayout(run_row)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #8891a0; font-size: 11px;")
        col.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        col.addWidget(self.progress_bar)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 12px;")
        col.addWidget(self.summary_label)

        col.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: 'Courier New', monospace; font-size: 11px;")
        col.addWidget(self.log_text, 1)

        foot_row = QHBoxLayout()
        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.clicked.connect(self._on_open_output_folder)
        foot_row.addWidget(self.btn_open_output)

        self.btn_import = QPushButton("Import into Current Project")
        self.btn_import.setEnabled(False)
        self.btn_import.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: #ffffff; "
            "border: none; border-radius: 8px; font-weight: 600; padding: 6px 10px; }"
            "QPushButton:hover { background-color: #1e88e5; }"
            "QPushButton:disabled { background-color: #3a3f4a; color: #6c7280; }"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        foot_row.addWidget(self.btn_import)
        col.addLayout(foot_row)

        return col

    # ------------------------------------------------------------- ontology
    def _add_ontology_row(self, caption="", class_name=""):
        row = self.ont_table.rowCount()
        self.ont_table.insertRow(row)

        caption_edit = QLineEdit(caption)
        caption_edit.setPlaceholderText("e.g. iced coffee cup")
        self.ont_table.setCellWidget(row, 0, caption_edit)

        class_edit = QLineEdit(class_name)
        class_edit.setPlaceholderText("e.g. cup")
        self.ont_table.setCellWidget(row, 1, class_edit)

        btn_del = QPushButton("×")
        btn_del.setFixedWidth(28)
        btn_del.setToolTip("Remove this class")
        btn_del.clicked.connect(lambda: self._remove_ontology_row(btn_del))
        self.ont_table.setCellWidget(row, 2, btn_del)

    def _remove_ontology_row(self, button: QPushButton):
        for row in range(self.ont_table.rowCount()):
            if self.ont_table.cellWidget(row, 2) is button:
                self.ont_table.removeRow(row)
                return
        if self.ont_table.rowCount() == 0:
            self._add_ontology_row()

    def _get_ontology(self) -> dict:
        ontology = {}
        for row in range(self.ont_table.rowCount()):
            caption_widget = self.ont_table.cellWidget(row, 0)
            class_widget = self.ont_table.cellWidget(row, 1)
            if caption_widget is None or class_widget is None:
                continue
            caption = caption_widget.text().strip()
            class_name = class_widget.text().strip()
            if caption and class_name:
                ontology[caption] = class_name
        return ontology

    # ------------------------------------------------------------- folders
    def _on_browse_source(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", self.source_edit.text())
        if path:
            self.source_edit.setText(path)
            self._refresh_image_count()

    def _on_browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _refresh_image_count(self):
        from core.zero_shot_annotator import DEFAULT_EXTENSIONS

        source = Path(self.source_edit.text())
        if not source.exists():
            self.count_label.setText("Folder does not exist yet")
            return

        counts = {}
        for ext in DEFAULT_EXTENSIONS:
            n = len(list(source.glob(f'*{ext}'))) + len(list(source.glob(f'*{ext.upper()}')))
            counts[ext] = n
        total = sum(counts.values())
        parts = "  ".join(f"{ext} {n}" for ext, n in counts.items())
        self.count_label.setText(f"{total} image(s) found  ({parts})")

    # ------------------------------------------------------------- run
    def _on_start(self):
        if not is_available():
            QMessageBox.warning(self, "Not Installed", get_availability_message())
            return

        ontology = self._get_ontology()
        if not ontology:
            QMessageBox.warning(
                self, "No Classes",
                "Add at least one Caption -> Class Name row before starting.")
            return

        source_dir = Path(self.source_edit.text())
        if not source_dir.exists():
            QMessageBox.warning(self, "Source Not Found",
                                f"Folder does not exist:\n{source_dir}")
            return

        output_dir = Path(self.output_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_text.clear()
        self.summary_label.setText("")
        self.btn_import.setEnabled(False)
        self._last_stats = None
        self._last_output_dir = None

        self._set_running_state(True)

        self.worker = _AnnotateWorker(
            ontology=ontology,
            source_dir=source_dir,
            output_dir=output_dir,
            box_threshold=self.box_threshold_spin.value(),
            text_threshold=self.text_threshold_spin.value(),
            use_nms=self.nms_check.isChecked(),
            nms_threshold=self.nms_iou_spin.value(),
            class_agnostic_nms=self.class_agnostic_check.isChecked(),
            parent=self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.status_label.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def _set_running_state(self, running: bool):
        self.btn_start.setEnabled(not running and is_available())
        self.btn_stop.setEnabled(running)
        for w in (self.source_edit, self.output_edit, self.ont_table,
                 self.box_threshold_spin, self.text_threshold_spin,
                 self.nms_check, self.nms_iou_spin, self.class_agnostic_check):
            w.setEnabled(not running)

    def _on_progress(self, current, total, message):
        if total == 0:
            self.progress_bar.setRange(0, 0)  # busy/indeterminate
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.status_label.setText(f"Processing {current} / {total}")
        self._append_log(message)

    def _append_log(self, text):
        ts = time.strftime('%H:%M:%S')
        self.log_text.append(f"{ts}  {text}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, stats):
        self._set_running_state(False)
        self.progress_bar.setRange(0, max(1, stats.get('images_found', 1)))
        self.progress_bar.setValue(stats.get('processed', 0))
        self.status_label.setText("Done")

        self._last_stats = stats
        self._last_output_dir = Path(stats['output_dir'])

        self.summary_label.setText(
            f"<b>{stats['annotated']}</b> images annotated &nbsp;|&nbsp; "
            f"<b>{stats['total_objects']}</b> objects &nbsp;|&nbsp; "
            f"<b>{stats['skipped_no_detection']}</b> no detection &nbsp;|&nbsp; "
            f"<b>{stats['skipped_error']}</b> errors &nbsp;|&nbsp; "
            f"classes: {', '.join(stats['classes'])}"
        )

        self.btn_import.setEnabled(stats['annotated'] > 0)

        if stats['errors']:
            self._append_log(f"-- {len(stats['errors'])} file(s) skipped due to errors --")

        QMessageBox.information(
            self, "Auto-Annotation Complete",
            f"Images annotated: {stats['annotated']} / {stats['images_found']}\n"
            f"Objects: {stats['total_objects']}\n"
            f"No detection: {stats['skipped_no_detection']}\n"
            f"Errors: {stats['skipped_error']}\n\n"
            f"Output: {stats['output_dir']}"
        )

    def _on_failed(self, error_message):
        self._set_running_state(False)
        self.status_label.setText("Failed")
        self._append_log(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Auto-Annotation Failed", error_message)

    # ------------------------------------------------------------- footer
    def _on_open_output_folder(self):
        import subprocess
        import sys

        folder = Path(self.output_edit.text())
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', str(folder)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(folder)])
            else:
                subprocess.Popen(['xdg-open', str(folder)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open folder:\n{e}")

    def _on_import_clicked(self):
        if self._last_output_dir is None:
            return

        from core.dataset_importer import DatasetImporter

        try:
            importer = DatasetImporter(self.project_path, self.existing_classes)
            stats = importer.import_dataset(self._last_output_dir, fmt='yolo')
        except Exception as e:
            QMessageBox.critical(self, "Import Failed",
                                 f"Failed to import into project:\n{e}")
            return

        self.dataset_imported.emit(stats['classes'])
        self.btn_import.setEnabled(False)

        QMessageBox.information(
            self, "Import Complete",
            f"Images imported: {stats['imported']}\n"
            f"Annotations: {stats['annotations']}\n\n"
            f"Project classes: {', '.join(stats['classes'])}"
        )

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        super().closeEvent(event)
