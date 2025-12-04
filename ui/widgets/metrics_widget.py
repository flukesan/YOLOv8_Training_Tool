"""
Metrics Widget - display training metrics
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
import json


class MetricsWidget(QWidget):
    """Widget for displaying training metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Training Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Epoch info
        self.epoch_label = QLabel("Epoch: 0/0")
        layout.addWidget(self.epoch_label)

        # Progress
        self.progress_label = QLabel("Progress: 0%")
        layout.addWidget(self.progress_label)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        layout.addWidget(self.metrics_text)

        layout.addStretch()
        self.setLayout(layout)

    def update_metrics(self, metrics: dict):
        """Update displayed metrics"""
        if 'epoch' in metrics and 'total_epochs' in metrics:
            self.epoch_label.setText(
                f"Epoch: {metrics['epoch']}/{metrics['total_epochs']}"
            )

        if 'progress' in metrics:
            self.progress_label.setText(
                f"Progress: {metrics['progress']:.1f}%"
            )

        # Display all metrics
        metrics_text = json.dumps(metrics, indent=2)
        self.metrics_text.setPlainText(metrics_text)
