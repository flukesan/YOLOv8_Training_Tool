"""
Class Manager Widget - manage object classes with modern UI
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout, QInputDialog, QListWidgetItem)
from PyQt6.QtCore import pyqtSignal, Qt


class ClassManagerWidget(QWidget):
    """Widget for managing classes"""

    class_added = pyqtSignal(str)
    class_deleted = pyqtSignal(int)
    class_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Title row with count
        title_row = QHBoxLayout()
        title = QLabel("Classes")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        title_row.addWidget(title)

        self.count_label = QLabel("0 classes")
        self.count_label.setStyleSheet("color: #8891a0; font-size: 12px;")
        title_row.addStretch()
        title_row.addWidget(self.count_label)
        layout.addLayout(title_row)

        # Class list
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self._on_class_selected)
        self.class_list.setMinimumHeight(80)
        layout.addWidget(self.class_list)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_add = QPushButton("+ Add Class")
        self.btn_add.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 8px 16px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #339952; }"
        )
        self.btn_add.clicked.connect(self._on_add)
        btn_layout.addWidget(self.btn_add)

        self.btn_delete = QPushButton("Delete")
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

    def set_classes(self, classes):
        """Set class list with color indicators"""
        self.classes = classes
        self.class_list.clear()

        for i, cls in enumerate(classes):
            item = QListWidgetItem(f"  [{i}]  {cls}")
            item.setForeground(Qt.GlobalColor.white)
            self.class_list.addItem(item)

        self.count_label.setText(f"{len(classes)} classes")

    def _on_add(self):
        """Add new class"""
        text, ok = QInputDialog.getText(self, 'Add Class', 'Enter class name:')
        if ok and text.strip():
            self.class_added.emit(text.strip())

    def _on_delete(self):
        """Delete selected class"""
        current_row = self.class_list.currentRow()
        if current_row >= 0:
            self.class_deleted.emit(current_row)

    def _on_class_selected(self, item):
        """Handle class selection"""
        index = self.class_list.row(item)
        self.class_selected.emit(index)
