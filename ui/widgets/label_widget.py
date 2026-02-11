"""
Label Widget - annotation tools panel with modern UI
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal, Qt


class LabelWidget(QWidget):
    """Widget for managing labels and annotations"""

    delete_annotation = pyqtSignal(int)
    annotation_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Title row with count
        title_row = QHBoxLayout()
        title = QLabel("Annotations")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        title_row.addWidget(title)

        self.count_label = QLabel("0 annotations")
        self.count_label.setStyleSheet("color: #8891a0; font-size: 12px;")
        title_row.addStretch()
        title_row.addWidget(self.count_label)
        layout.addLayout(title_row)

        # Annotation list
        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self._on_annotation_selected)
        layout.addWidget(self.annotation_list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 8px 16px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #d94435; }"
        )
        self.btn_delete.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def update_annotations(self, annotations):
        """Update annotation list"""
        self.annotations = annotations
        self.annotation_list.clear()

        for i, ann in enumerate(annotations):
            if hasattr(ann, 'annotation_type'):
                if ann.annotation_type == 'box':
                    self.annotation_list.addItem(
                        f"  [{i+1}]  Box - Class {ann.class_id}"
                    )
                elif ann.annotation_type == 'polygon':
                    num_points = len(ann.points) if hasattr(ann, 'points') else 0
                    self.annotation_list.addItem(
                        f"  [{i+1}]  Polygon - Class {ann.class_id} ({num_points} pts)"
                    )
            else:
                self.annotation_list.addItem(
                    f"  [{i+1}]  Annotation - Class {ann.class_id}"
                )

        self.count_label.setText(f"{len(annotations)} annotations")

    def _on_annotation_selected(self, item):
        """Handle annotation selection"""
        index = self.annotation_list.row(item)
        self.annotation_selected.emit(index)

    def _on_delete(self):
        """Handle delete button"""
        current_row = self.annotation_list.currentRow()
        if current_row >= 0:
            self.delete_annotation.emit(current_row)
