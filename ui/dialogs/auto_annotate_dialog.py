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
                             QTextEdit, QMessageBox, QHeaderView, QScrollArea,
                             QWidget, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core.zero_shot_annotator import (ZeroShotAnnotator, is_available,
                                      get_availability_message)
from ui.widgets.training_widget import CollapsibleSection
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
        # A plain QDialog has no minimize/maximize buttons; add them so the
        # window can be resized to full screen when the lists get long.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        # Roomy enough that nothing is clipped, but the setup column also
        # scrolls so the dialog still works on a short screen.
        self.setMinimumSize(900, 560)
        self.resize(1080, 780)
        self.setSizeGripEnabled(True)
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

        # The setup column is tall; put it in a scroll area so groups are
        # never squeezed into each other on a small window.
        setup_container = QWidget()
        setup_container.setLayout(self._build_setup_column())
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # No sideways scrolling: the content must fit the viewport width,
        # otherwise the right-hand column of the ontology table (the remove
        # button) ends up pushed outside the visible area.
        setup_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        setup_scroll.setWidget(setup_container)
        setup_scroll.setMinimumWidth(420)
        body.addWidget(setup_scroll, 3)

        run_container = QWidget()
        run_container.setLayout(self._build_run_column())
        run_container.setMinimumWidth(320)
        body.addWidget(run_container, 2)

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
        # Keep the count in step when the path is typed, not just browsed
        self.source_edit.textChanged.connect(self._refresh_image_count)
        src_row.addWidget(self.source_edit, 1)
        btn_src = QPushButton("Browse...")
        btn_src.clicked.connect(self._on_browse_source)
        src_row.addWidget(btn_src)
        f_layout.addLayout(src_row)

        # Count + manual refresh: images are often dropped into the folder
        # while this dialog is already open, and the path itself doesn't
        # change in that case, so there is nothing to react to.
        count_row = QHBoxLayout()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #8891a0; font-size: 11px;")
        count_row.addWidget(self.count_label, 1)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setToolTip("Re-scan the source folder for images")
        btn_refresh.clicked.connect(self._refresh_image_count)
        count_row.addWidget(btn_refresh)
        f_layout.addLayout(count_row)

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
            ["Caption (prompt)", "Class name", "Remove"])
        self.ont_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.ont_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.ont_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed)
        self.ont_table.setColumnWidth(2, 72)
        self.ont_table.horizontalHeader().setStretchLastSection(False)
        # The two stretch columns shrink to fit, so the Remove column is
        # always reachable instead of being scrolled out of view.
        self.ont_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ont_table.verticalHeader().setVisible(False)
        # Fixed row height matching ROW_WIDGET_HEIGHT (plus padding) so the
        # fields and the Remove button share one baseline.
        self.ont_table.verticalHeader().setDefaultSectionSize(
            self.ROW_WIDGET_HEIGHT + 6)
        self.ont_table.setMinimumHeight(180)
        ont_layout.addWidget(self.ont_table)
        self._add_ontology_row()

        btn_add_row = QPushButton("+ Add Class")
        btn_add_row.clicked.connect(lambda: self._add_ontology_row())
        ont_layout.addWidget(btn_add_row)

        ont_group.setLayout(ont_layout)
        col.addWidget(ont_group)

        # Thresholds - collapsible, so the ontology table can have the room
        # when these are left at their defaults.
        thresh_section = CollapsibleSection("Thresholds")

        self.box_threshold_spin = QDoubleSpinBox()
        self.box_threshold_spin.setRange(0.0, 1.0)
        self.box_threshold_spin.setSingleStep(0.05)
        self.box_threshold_spin.setDecimals(2)
        self.box_threshold_spin.setValue(ZeroShotAnnotator.DEFAULT_BOX_THRESHOLD)
        self.box_threshold_spin.setToolTip(
            "Higher = fewer false positives but may miss real objects.\n"
            "Lower = catches more, but more noise (rely on NMS + review).")
        thresh_section.add_row("Box Threshold:", self.box_threshold_spin)

        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setRange(0.0, 1.0)
        self.text_threshold_spin.setSingleStep(0.05)
        self.text_threshold_spin.setDecimals(2)
        self.text_threshold_spin.setValue(ZeroShotAnnotator.DEFAULT_TEXT_THRESHOLD)
        self.text_threshold_spin.setToolTip(
            "Confidence that the caption matches what's seen in the image.")
        thresh_section.add_row("Text Threshold:", self.text_threshold_spin)

        self._expand(thresh_section)
        col.addWidget(thresh_section)

        # NMS - also collapsible
        nms_section = CollapsibleSection("NMS - Remove Overlapping Duplicate Boxes")

        self.nms_check = QCheckBox("Enable NMS")
        self.nms_check.setChecked(True)
        self.nms_check.setToolTip(
            "Merges multiple boxes drawn on the same object (common when "
            "objects sit close together) into a single best box.")
        nms_section.add_row("", self.nms_check)

        self.nms_iou_spin = QDoubleSpinBox()
        self.nms_iou_spin.setRange(0.0, 1.0)
        self.nms_iou_spin.setSingleStep(0.05)
        self.nms_iou_spin.setDecimals(2)
        self.nms_iou_spin.setValue(ZeroShotAnnotator.DEFAULT_NMS_THRESHOLD)
        nms_section.add_row("IoU Threshold:", self.nms_iou_spin)

        self.class_agnostic_check = QCheckBox(
            "Class-agnostic (also across classes)")
        self.class_agnostic_check.setChecked(True)
        self.class_agnostic_check.setToolTip(
            "Also suppress overlapping boxes that belong to different "
            "classes, not just duplicates of the same class.")
        nms_section.add_row("", self.class_agnostic_check)

        self._expand(nms_section)
        col.addWidget(nms_section)

        col.addStretch()
        return col

    @staticmethod
    def _expand(section: CollapsibleSection):
        """Open a CollapsibleSection (they start collapsed by default)."""
        section.toggle_button.setChecked(True)
        section._on_toggle(True)

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

        # Stacked, not side by side: the labels are long enough that a
        # single row clipped them ("pen Output Folde") on a narrow dialog.
        foot_row = QVBoxLayout()
        foot_row.setSpacing(6)
        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.setMinimumHeight(32)
        self.btn_open_output.clicked.connect(self._on_open_output_folder)
        foot_row.addWidget(self.btn_open_output)

        self.btn_import = QPushButton("Import into Current Project")
        self.btn_import.setMinimumHeight(34)
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
    # Every widget in an ontology row uses this height so the text fields
    # and the Remove button line up instead of sitting at different offsets.
    ROW_WIDGET_HEIGHT = 28

    def _add_ontology_row(self, caption="", class_name=""):
        row = self.ont_table.rowCount()
        self.ont_table.insertRow(row)

        caption_edit = QLineEdit(caption)
        caption_edit.setPlaceholderText("e.g. iced coffee cup")
        caption_edit.setMinimumHeight(self.ROW_WIDGET_HEIGHT)
        self.ont_table.setCellWidget(row, 0, caption_edit)

        class_edit = QLineEdit(class_name)
        class_edit.setPlaceholderText("e.g. cup")
        class_edit.setMinimumHeight(self.ROW_WIDGET_HEIGHT)
        self.ont_table.setCellWidget(row, 1, class_edit)

        btn_del = QPushButton("Remove")
        btn_del.setToolTip("Remove this class")
        btn_del.setMinimumHeight(self.ROW_WIDGET_HEIGHT)
        btn_del.clicked.connect(lambda: self._remove_ontology_row(btn_del))
        self.ont_table.setCellWidget(row, 2, btn_del)

    def _remove_ontology_row(self, button: QPushButton):
        for row in range(self.ont_table.rowCount()):
            if self.ont_table.cellWidget(row, 2) is button:
                self.ont_table.removeRow(row)
                break
        # Always leave one editable row behind so the table is never empty
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

    @staticmethod
    def _count_source_images(source: Path) -> int:
        """Total images the annotator would pick up from `source`."""
        from core.zero_shot_annotator import DEFAULT_EXTENSIONS

        if not source.exists():
            return 0
        seen = set()
        for ext in DEFAULT_EXTENSIONS:
            for pattern in (f'*{ext}', f'*{ext.upper()}'):
                seen.update(p for p in source.glob(pattern) if p.is_file())
        return len(seen)

    def _refresh_image_count(self):
        from core.zero_shot_annotator import DEFAULT_EXTENSIONS

        source = Path(self.source_edit.text())
        if not source.exists():
            self.count_label.setText("Folder does not exist yet")
            return

        counts = {}
        for ext in DEFAULT_EXTENSIONS:
            found = set()
            for pattern in (f'*{ext}', f'*{ext.upper()}'):
                found.update(p for p in source.glob(pattern) if p.is_file())
            counts[ext] = len(found)
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

        # Check for images before loading the model - the first run
        # downloads GroundingDINO weights, so don't spend that on an
        # empty folder.
        if self._count_source_images(source_dir) == 0:
            QMessageBox.warning(
                self, "No Images",
                f"No .jpg / .jpeg / .png images found in:\n{source_dir}\n\n"
                "Copy the images you want to annotate into this folder first.")
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
        # Leave the indeterminate/busy state the model-loading phase set,
        # otherwise the bar keeps animating forever after a failure.
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
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
            # Disconnect first: the worker can still be mid-image (or
            # downloading weights) when the wait times out, and a signal
            # arriving after the dialog is destroyed would crash.
            try:
                self.worker.progress.disconnect()
                self.worker.finished_ok.disconnect()
                self.worker.failed.disconnect()
            except TypeError:
                pass  # already disconnected
            self.worker.request_stop()
            self.worker.wait(5000)
        super().closeEvent(event)
