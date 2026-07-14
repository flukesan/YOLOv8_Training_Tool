"""
Import Dataset Dialog - import COCO / YOLO (FiftyOne export) datasets
"""
from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QLineEdit, QFileDialog, QListWidget,
                             QListWidgetItem, QCheckBox, QGroupBox,
                             QMessageBox)
from PyQt6.QtCore import Qt

from core.dataset_importer import DatasetImporter, detect_format


class ImportDatasetDialog(QDialog):
    """Pick a dataset folder (COCO json / YOLO export), choose classes to
    import, and configure filename prefix."""

    def __init__(self, project_path, existing_classes, parent=None):
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.existing_classes = list(existing_classes)
        self._scan_info = None

        self.setWindowTitle("Import Dataset (COCO / YOLO)")
        self.setMinimumWidth(520)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        header = QLabel("Import External Dataset")
        header.setStyleSheet("font-size: 17px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel(
            "Supports COCO (annotations .json) and YOLO folder exports\n"
            "(e.g. FiftyOne YOLOv5Dataset: images/ + labels/ + dataset.yaml)."
        )
        desc.setStyleSheet("color: #8891a0; font-size: 12px;")
        layout.addWidget(desc)

        # Source folder row
        src_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select dataset folder...")
        self.path_edit.setReadOnly(True)
        src_row.addWidget(self.path_edit, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse)
        src_row.addWidget(btn_browse)
        layout.addLayout(src_row)

        # Scan summary
        self.summary_label = QLabel("No folder selected")
        self.summary_label.setStyleSheet("color: #8891a0; font-size: 12px;")
        layout.addWidget(self.summary_label)

        # Class selection
        class_group = QGroupBox("Classes to Import")
        class_layout = QVBoxLayout()
        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(160)
        class_layout.addWidget(self.class_list)

        toggle_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        toggle_row.addWidget(btn_all)
        toggle_row.addWidget(btn_none)
        toggle_row.addStretch()
        class_layout.addLayout(toggle_row)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Options
        opt_group = QGroupBox("Options")
        opt_layout = QVBoxLayout()

        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Filename prefix:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("e.g. cam1_  (optional)")
        prefix_row.addWidget(self.prefix_edit, 1)
        opt_layout.addLayout(prefix_row)

        self.include_unlabeled_check = QCheckBox(
            "Include images without annotations (background images)")
        opt_layout.addWidget(self.include_unlabeled_check)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self.btn_import = QPushButton("Import")
        self.btn_import.setEnabled(False)
        self.btn_import.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 10px 24px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #339952; }"
            "QPushButton:disabled { background-color: #3a3f4a; color: #6c7280; }"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        btn_row.addWidget(self.btn_import)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    # ------------------------------------------------------------------ scan
    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not path:
            return
        self.path_edit.setText(path)
        self._scan(Path(path))

    def _scan(self, source_dir: Path):
        self.class_list.clear()
        self._scan_info = None
        self.btn_import.setEnabled(False)

        fmt = detect_format(source_dir)
        if fmt is None:
            self.summary_label.setText(
                "❌ Not recognised. Select the folder that contains "
                "labels.json (COCO) or dataset.yaml / a labels folder (YOLO) "
                "- not the images-only subfolder.")
            self.summary_label.setWordWrap(True)
            self.summary_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            return

        try:
            importer = DatasetImporter(self.project_path, self.existing_classes)
            info = importer.scan(source_dir, fmt)
        except Exception as e:
            self.summary_label.setText(f"❌ Scan failed: {e}")
            self.summary_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            return

        self._scan_info = info
        self.summary_label.setText(
            f"✅ Format: {info['format'].upper()}  |  "
            f"{info['image_count']} images  |  "
            f"{info['annotation_count']} annotations  |  "
            f"{len(info['classes'])} classes"
        )
        self.summary_label.setStyleSheet("color: #4CAF50; font-size: 12px;")

        existing_lower = {c.lower() for c in self.existing_classes}
        for name in info['classes']:
            tag = "existing" if name.lower() in existing_lower else "new"
            item = QListWidgetItem(f"{name}   ({tag})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.class_list.addItem(item)

        self.btn_import.setEnabled(True)

    def _set_all_checked(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(state)

    # ---------------------------------------------------------------- accept
    def _on_import_clicked(self):
        if self._scan_info is None:
            return
        if not self.get_selected_classes():
            QMessageBox.warning(self, "No Classes",
                                "Select at least one class to import.")
            return
        self.accept()

    def get_import_config(self) -> dict:
        """Selected options for the import (valid after accept())."""
        return {
            'source_dir': Path(self.path_edit.text()),
            'format': self._scan_info['format'] if self._scan_info else None,
            'selected_classes': self.get_selected_classes(),
            'filename_prefix': self.prefix_edit.text().strip(),
            'include_unlabeled': self.include_unlabeled_check.isChecked(),
        }

    def get_selected_classes(self):
        selected = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))
        return selected
