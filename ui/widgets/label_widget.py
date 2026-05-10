"""
Label Widget - annotation tools panel with modern UI
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                             QPushButton, QLabel, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QBrush, QIcon, QPixmap, QPainter


class LabelWidget(QWidget):
    """Widget for managing labels and annotations"""

    delete_annotation = pyqtSignal(int)
    annotation_selected = pyqtSignal(int)
    paste_from_previous = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = []
        self.class_names = []
        self.class_colors = {}
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
        self.annotation_list.setStyleSheet(
            "QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #2d313a; }"
            "QListWidget::item:selected { background-color: #2d7d46; color: #ffffff; }"
            "QListWidget::item:hover { background-color: #2a2e38; }"
        )
        self.annotation_list.currentRowChanged.connect(self._on_annotation_selected)
        layout.addWidget(self.annotation_list)

        # Paste from previous button
        self.btn_paste_prev = QPushButton("Paste from previous image")
        self.btn_paste_prev.setStyleSheet(
            "QPushButton { background-color: #2d6a4f; color: #ffffff; "
            "border: none; border-radius: 6px; padding: 5px 10px; "
            "font-size: 11px; }"
            "QPushButton:hover { background-color: #40916c; }"
            "QPushButton:disabled { background-color: #3a3f4a; color: #6c7280; }"
        )
        self.btn_paste_prev.setEnabled(False)
        self.btn_paste_prev.clicked.connect(lambda: self.paste_from_previous.emit())
        layout.addWidget(self.btn_paste_prev)

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

    def set_classes(self, class_names, class_colors):
        """Set class info for display"""
        self.class_names = class_names
        self.class_colors = class_colors

    def _make_color_icon(self, color_tuple, size=14):
        """Create a small colored square icon"""
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(*color_tuple)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, size, size, 3, 3)
        painter.end()
        return QIcon(pm)

    def update_annotations(self, annotations):
        """Update annotation list with class names and color indicators"""
        self.annotations = annotations
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()

        for i, ann in enumerate(annotations):
            class_id = ann.class_id if hasattr(ann, 'class_id') else 0
            color = self.class_colors.get(class_id, (255, 255, 255))

            # Determine class name
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class {class_id}"

            # Determine annotation type
            ann_type = "Annotation"
            extra = ""
            if hasattr(ann, 'annotation_type'):
                if ann.annotation_type == 'box':
                    ann_type = "Box"
                elif ann.annotation_type == 'polygon':
                    ann_type = "Polygon"
                    num_pts = len(ann.points) if hasattr(ann, 'points') else 0
                    extra = f" ({num_pts} pts)"

            text = f"  [{i+1}]  {ann_type} - {class_name}{extra}"
            item = QListWidgetItem(text)
            item.setIcon(self._make_color_icon(color))
            self.annotation_list.addItem(item)

        self.annotation_list.blockSignals(False)
        self.count_label.setText(f"{len(annotations)} annotations")

    def select_annotation(self, index: int):
        """Programmatically select/highlight an annotation row"""
        self.annotation_list.blockSignals(True)
        if 0 <= index < self.annotation_list.count():
            self.annotation_list.setCurrentRow(index)
        else:
            self.annotation_list.setCurrentRow(-1)
        self.annotation_list.blockSignals(False)

    def _on_annotation_selected(self, row):
        """Handle annotation selection from list"""
        if row >= 0:
            self.annotation_selected.emit(row)

    def _on_delete(self):
        """Handle delete button"""
        current_row = self.annotation_list.currentRow()
        if current_row >= 0:
            self.delete_annotation.emit(current_row)
