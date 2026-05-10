"""
Export Dialog - modern design
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QCheckBox, QPushButton,
                             QHBoxLayout, QLabel, QGroupBox)


class ExportDialog(QDialog):
    """Dialog for exporting model"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Export Model")
        self.setMinimumWidth(420)

        layout = QVBoxLayout()
        layout.setSpacing(16)

        # Header
        header = QLabel("Export Trained Model")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel("Select the formats to export your trained model.")
        desc.setStyleSheet("color: #8891a0; font-size: 13px;")
        layout.addWidget(desc)

        # Format checkboxes
        format_group = QGroupBox("Export Formats")
        format_layout = QVBoxLayout()
        format_layout.setSpacing(8)

        self.cb_pt = QCheckBox("PyTorch (.pt) - Native format, fastest")
        self.cb_pt.setChecked(True)
        format_layout.addWidget(self.cb_pt)

        # Separator
        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #3d4250;")
        format_layout.addWidget(sep)

        converted_label = QLabel("Deployment Formats:")
        converted_label.setStyleSheet(
            "font-weight: 600; font-size: 12px; color: #8891a0;"
        )
        format_layout.addWidget(converted_label)

        self.cb_onnx = QCheckBox("ONNX - Cross-platform (recommended)")
        self.cb_tflite = QCheckBox("TensorFlow Lite - Mobile / Embedded")
        self.cb_torchscript = QCheckBox("TorchScript - PyTorch optimized")
        self.cb_coreml = QCheckBox("CoreML - Apple devices")

        format_layout.addWidget(self.cb_onnx)
        format_layout.addWidget(self.cb_tflite)
        format_layout.addWidget(self.cb_torchscript)
        format_layout.addWidget(self.cb_coreml)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Info
        info = QLabel(
            "PyTorch format preserves full model capabilities.\n"
            "ONNX is recommended for cross-platform deployment."
        )
        info.setStyleSheet("color: #8891a0; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_export = QPushButton("Export")
        self.btn_export.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 10px 24px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #339952; }"
        )
        self.btn_export.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_export)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_selected_formats(self):
        """Get selected export formats"""
        formats = []
        if self.cb_pt.isChecked():
            formats.append('pt')
        if self.cb_onnx.isChecked():
            formats.append('onnx')
        if self.cb_tflite.isChecked():
            formats.append('tflite')
        if self.cb_torchscript.isChecked():
            formats.append('torchscript')
        if self.cb_coreml.isChecked():
            formats.append('coreml')
        return formats
