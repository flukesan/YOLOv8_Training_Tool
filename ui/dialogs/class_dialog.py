"""
Class Dialog
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QPushButton,
                             QHBoxLayout, QLabel)


class ClassDialog(QDialog):
    """Dialog for adding/editing class"""

    def __init__(self, class_name="", parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Class")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Class name
        label = QLabel("Class Name:")
        layout.addWidget(label)

        self.name_edit = QLineEdit()
        self.name_edit.setText(self.class_name)
        layout.addWidget(self.name_edit)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_class_name(self):
        """Get class name"""
        return self.name_edit.text()
