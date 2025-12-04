"""
Class Manager Widget - manage object classes
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout, QInputDialog)
from PyQt6.QtCore import pyqtSignal


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

        # Title
        title = QLabel("Classes")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Class list
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self._on_class_selected)
        layout.addWidget(self.class_list)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)

        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def set_classes(self, classes):
        """Set class list"""
        self.classes = classes
        self.class_list.clear()
        for i, cls in enumerate(classes):
            self.class_list.addItem(f"{i}: {cls}")

    def _on_add(self):
        """Add new class"""
        text, ok = QInputDialog.getText(self, 'Add Class', 'Class name:')
        if ok and text:
            self.class_added.emit(text)

    def _on_delete(self):
        """Delete selected class"""
        current_row = self.class_list.currentRow()
        if current_row >= 0:
            self.class_deleted.emit(current_row)

    def _on_class_selected(self, item):
        """Handle class selection"""
        index = self.class_list.row(item)
        self.class_selected.emit(index)
