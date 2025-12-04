"""
New Project Dialog
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QPushButton, QFileDialog, QHBoxLayout)
from pathlib import Path


class NewProjectDialog(QDialog):
    """Dialog for creating new project"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_name = ""
        self.project_path = ""
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("New Project")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()
        form = QFormLayout()

        # Project name
        self.name_edit = QLineEdit()
        form.addRow("Project Name:", self.name_edit)

        # Project location
        location_layout = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_location)
        location_layout.addWidget(self.location_edit)
        location_layout.addWidget(self.btn_browse)
        form.addRow("Location:", location_layout)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_create = QPushButton("Create")
        self.btn_create.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_create)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse_location(self):
        """Browse for project location"""
        path = QFileDialog.getExistingDirectory(self, "Select Project Location")
        if path:
            self.location_edit.setText(path)

    def get_project_info(self):
        """Get project information"""
        return {
            'name': self.name_edit.text(),
            'path': Path(self.location_edit.text()) / self.name_edit.text()
        }
