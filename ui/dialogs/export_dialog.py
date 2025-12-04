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
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Title
        label = QLabel("Select export formats:")
        layout.addWidget(label)

        # Format checkboxes
        format_group = QGroupBox("Formats")
        format_layout = QVBoxLayout()

        self.cb_onnx = QCheckBox("ONNX")
        self.cb_tflite = QCheckBox("TensorFlow Lite")
        self.cb_torchscript = QCheckBox("TorchScript")
        self.cb_coreml = QCheckBox("CoreML")

        format_layout.addWidget(self.cb_onnx)
        format_layout.addWidget(self.cb_tflite)
        format_layout.addWidget(self.cb_torchscript)
        format_layout.addWidget(self.cb_coreml)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

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
        if self.cb_onnx.isChecked():
            formats.append('onnx')
        if self.cb_tflite.isChecked():
            formats.append('tflite')
        if self.cb_torchscript.isChecked():
            formats.append('torchscript')
        if self.cb_coreml.isChecked():
            formats.append('coreml')
        return formats
