"""
Split Dataset Dialog - configure train/val/test split ratios
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSpinBox, QFormLayout, QMessageBox)
from PyQt6.QtCore import Qt


class SplitDatasetDialog(QDialog):
    """Dialog for configuring dataset split ratios"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split Dataset")
        self.setModal(True)
        self.setMinimumWidth(350)

        self.train_ratio = 70
        self.val_ratio = 20
        self.test_ratio = 10

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Set the split ratios for train/val/test sets:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Form layout for ratios
        form_layout = QFormLayout()

        # Train ratio
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        self.train_spin.valueChanged.connect(self._on_ratio_changed)
        form_layout.addRow("Train:", self.train_spin)

        # Val ratio
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix("%")
        self.val_spin.valueChanged.connect(self._on_ratio_changed)
        form_layout.addRow("Val:", self.val_spin)

        # Test ratio
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(10)
        self.test_spin.setSuffix("%")
        self.test_spin.valueChanged.connect(self._on_ratio_changed)
        form_layout.addRow("Test:", self.test_spin)

        layout.addLayout(form_layout)

        # Total label
        self.total_label = QLabel("Total: 100%")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.total_label)

        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.warning_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _on_ratio_changed(self):
        """Handle ratio change"""
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        self.total_label.setText(f"Total: {total}%")

        if total != 100:
            self.warning_label.setText("⚠ Total must equal 100%")
            self.btn_ok.setEnabled(False)
        else:
            self.warning_label.setText("")
            self.btn_ok.setEnabled(True)

    def _on_ok(self):
        """Validate and accept"""
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()

        if total != 100:
            QMessageBox.warning(
                self,
                "Invalid Ratios",
                "The sum of train, val, and test ratios must equal 100%"
            )
            return

        if self.train_spin.value() == 0:
            QMessageBox.warning(
                self,
                "Invalid Ratios",
                "Train ratio must be greater than 0%"
            )
            return

        self.train_ratio = self.train_spin.value()
        self.val_ratio = self.val_spin.value()
        self.test_ratio = self.test_spin.value()

        self.accept()

    def get_ratios(self):
        """Get selected ratios as dictionary"""
        return {
            'train': self.train_ratio / 100.0,
            'val': self.val_ratio / 100.0,
            'test': self.test_ratio / 100.0
        }
