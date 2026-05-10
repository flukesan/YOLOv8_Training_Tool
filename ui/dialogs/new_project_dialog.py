"""
New Project Dialog - modern design
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QPushButton, QFileDialog, QHBoxLayout, QLabel)
from pathlib import Path
from config.settings import Settings


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
        self.setMinimumWidth(500)

        layout = QVBoxLayout()
        layout.setSpacing(16)

        # Header
        header = QLabel("Create New Project")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel("Set up a new YOLOv8 training project.")
        desc.setStyleSheet("color: #8891a0; font-size: 13px;")
        layout.addWidget(desc)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Quality_Inspection_v1")
        form.addRow("Project Name:", self.name_edit)

        # Location with browse
        location_layout = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setText(str(Settings.DEFAULT_PROJECTS_DIR))
        self.location_edit.setPlaceholderText("Select project folder...")
        location_layout.addWidget(self.location_edit)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_location)
        location_layout.addWidget(self.btn_browse)
        form.addRow("Location:", location_layout)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_create = QPushButton("Create Project")
        self.btn_create.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 10px 24px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #339952; }"
        )
        self.btn_create.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_create)

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
