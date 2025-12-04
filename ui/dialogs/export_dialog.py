"""
Export Dialog
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
        self.setMinimumWidth(350)

        layout = QVBoxLayout()

        # Title
        label = QLabel("Select export formats:")
        layout.addWidget(label)

        # Format checkboxes
        format_group = QGroupBox("Export Formats")
        format_layout = QVBoxLayout()

        # Native PyTorch format
        self.cb_pt = QCheckBox("PyTorch (.pt) - Native format")
        self.cb_pt.setChecked(True)  # Default checked
        format_layout.addWidget(self.cb_pt)

        # Separator
        separator = QLabel("─" * 40)
        separator.setStyleSheet("color: gray;")
        format_layout.addWidget(separator)

        # Converted formats
        converted_label = QLabel("Converted Formats:")
        converted_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        format_layout.addWidget(converted_label)

        self.cb_onnx = QCheckBox("ONNX - Cross-platform format")
        self.cb_tflite = QCheckBox("TensorFlow Lite - Mobile/Embedded")
        self.cb_torchscript = QCheckBox("TorchScript - PyTorch optimized")
        self.cb_coreml = QCheckBox("CoreML - Apple devices")

        format_layout.addWidget(self.cb_onnx)
        format_layout.addWidget(self.cb_tflite)
        format_layout.addWidget(self.cb_torchscript)
        format_layout.addWidget(self.cb_coreml)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Info label
        info_label = QLabel("Note: PyTorch (.pt) format is the fastest option\nand preserves full model capabilities.")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_cancel)
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
