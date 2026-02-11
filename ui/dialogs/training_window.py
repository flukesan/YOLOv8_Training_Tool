"""
Training Window - Separate window for Training Configuration and Metrics
Moved out of main window to maximize annotation/image viewer space.
"""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                             QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt

from ui.widgets.training_widget import TrainingWidget
from ui.widgets.metrics_widget import MetricsWidget


class TrainingWindow(QWidget):
    """Separate floating window for Training Config + Metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training - Configuration & Metrics")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(900, 500)
        self.resize(1000, 600)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QLabel("Training Configuration & Metrics")
        header.setStyleSheet(
            "color: #ffffff; font-size: 16px; font-weight: 700; "
            "padding: 8px; background-color: #15171c; "
            "border-bottom: 2px solid #2d7d46; border-radius: 4px;"
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Splitter: Training Config (left) | Metrics (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.training_widget = TrainingWidget()
        splitter.addWidget(self.training_widget)

        self.metrics_widget = MetricsWidget()
        splitter.addWidget(self.metrics_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        # Dark theme styling
        self.setStyleSheet(
            "TrainingWindow { background-color: #1e2128; }"
        )

    def closeEvent(self, event):
        """Hide instead of close so state is preserved"""
        self.hide()
        event.ignore()
