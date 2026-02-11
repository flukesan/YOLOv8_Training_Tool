"""
Metrics Widget - Enhanced display for training metrics with visual progress
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QGroupBox, QGridLayout, QFrame,
                             QScrollArea)
from PyQt6.QtCore import Qt
import time


class MetricCard(QFrame):
    """A card widget for displaying a single metric with label and value"""

    def __init__(self, name: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.name = name
        self.unit = unit
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "MetricCard { border: 1px solid #444; border-radius: 4px; "
            "padding: 4px; }"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)

        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ddd;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        self.setLayout(layout)

    def set_value(self, value: float, color: str = None):
        """Update the displayed value"""
        if self.unit == "%":
            text = f"{value:.1f}%"
        elif self.unit == "loss":
            text = f"{value:.4f}"
        else:
            text = f"{value:.4f}"

        self.value_label.setText(text)

        if color:
            self.value_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {color};"
            )

    def clear(self):
        """Reset to default state"""
        self.value_label.setText("--")
        self.value_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ddd;"
        )


class MetricsWidget(QWidget):
    """Widget for displaying training metrics with visual progress indicators"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_start_time = None
        self._epoch_times = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Title
        title = QLabel("Training Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # === Progress Section ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(4)

        # Epoch progress bar
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setTextVisible(True)
        self.epoch_progress.setFormat("Epoch 0/0")
        self.epoch_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; "
            "text-align: center; background: #333; height: 20px; }"
            "QProgressBar::chunk { background-color: #4CAF50; }"
        )
        progress_layout.addWidget(self.epoch_progress)

        # Time info row
        time_row = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: --")
        self.elapsed_label.setStyleSheet("color: #aaa; font-size: 11px;")
        time_row.addWidget(self.elapsed_label)

        time_row.addStretch()

        self.eta_label = QLabel("ETA: --")
        self.eta_label.setStyleSheet("color: #aaa; font-size: 11px;")
        time_row.addWidget(self.eta_label)

        progress_layout.addLayout(time_row)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # === Current Metrics Section ===
        metrics_group = QGroupBox("Current Epoch Metrics")
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(6)

        # Loss metrics
        self.loss_card = MetricCard("Train Loss", "loss")
        metrics_layout.addWidget(self.loss_card, 0, 0)

        self.val_loss_card = MetricCard("Val Loss", "loss")
        metrics_layout.addWidget(self.val_loss_card, 0, 1)

        # Detection metrics
        self.precision_card = MetricCard("Precision", "%")
        metrics_layout.addWidget(self.precision_card, 1, 0)

        self.recall_card = MetricCard("Recall", "%")
        metrics_layout.addWidget(self.recall_card, 1, 1)

        # mAP metrics
        self.map50_card = MetricCard("mAP@50", "%")
        metrics_layout.addWidget(self.map50_card, 2, 0)

        self.map50_95_card = MetricCard("mAP@50-95", "%")
        metrics_layout.addWidget(self.map50_95_card, 2, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # === Best Metrics Section ===
        best_group = QGroupBox("Best Results")
        best_layout = QGridLayout()
        best_layout.setSpacing(6)

        self.best_map50_card = MetricCard("Best mAP@50", "%")
        best_layout.addWidget(self.best_map50_card, 0, 0)

        self.best_map50_95_card = MetricCard("Best mAP@50-95", "%")
        best_layout.addWidget(self.best_map50_95_card, 0, 1)

        best_group.setLayout(best_layout)
        layout.addWidget(best_group)

        # === Status label ===
        self.status_label = QLabel("Waiting for training to start...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

        # Track best values
        self._best_map50 = 0.0
        self._best_map50_95 = 0.0

    def update_metrics(self, metrics: dict):
        """Update displayed metrics from training callback data"""
        now = time.time()

        # Track training start
        if self._training_start_time is None:
            self._training_start_time = now

        # Update epoch progress
        epoch = metrics.get('epoch', 0)
        total_epochs = metrics.get('total_epochs', 0)

        if total_epochs > 0:
            progress = int((epoch / total_epochs) * 100)
            self.epoch_progress.setValue(progress)
            self.epoch_progress.setFormat(
                f"Epoch {epoch}/{total_epochs} ({progress}%)"
            )

        # Calculate time info
        elapsed = metrics.get('elapsed_time', 0)
        self.elapsed_label.setText(f"Elapsed: {self._format_time(elapsed)}")

        # ETA calculation
        if epoch > 0 and total_epochs > 0:
            avg_epoch_time = elapsed / epoch
            remaining_epochs = total_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            self.eta_label.setText(f"ETA: {self._format_time(eta_seconds)}")
        else:
            self.eta_label.setText("ETA: calculating...")

        # Update individual metric cards
        epoch_metrics = metrics.get('metrics', {})

        # Train loss
        train_losses = epoch_metrics.get('train_loss', [])
        if train_losses:
            loss_val = train_losses[-1]
            # Color: green if low, yellow if medium, red if high
            color = self._loss_color(loss_val)
            self.loss_card.set_value(loss_val, color)

        # Val loss
        val_losses = epoch_metrics.get('val_loss', [])
        if val_losses:
            val_loss_val = val_losses[-1]
            color = self._loss_color(val_loss_val)
            self.val_loss_card.set_value(val_loss_val, color)

        # Precision
        precisions = epoch_metrics.get('precision', [])
        if precisions:
            p_val = precisions[-1] * 100  # Convert to percentage
            color = self._metric_color(p_val)
            self.precision_card.set_value(p_val, color)

        # Recall
        recalls = epoch_metrics.get('recall', [])
        if recalls:
            r_val = recalls[-1] * 100
            color = self._metric_color(r_val)
            self.recall_card.set_value(r_val, color)

        # mAP50
        map50s = epoch_metrics.get('mAP50', [])
        if map50s:
            map50_val = map50s[-1] * 100
            color = self._metric_color(map50_val)
            self.map50_card.set_value(map50_val, color)

            # Track best
            if map50_val > self._best_map50:
                self._best_map50 = map50_val
                self.best_map50_card.set_value(map50_val, "#4CAF50")

        # mAP50-95
        map50_95s = epoch_metrics.get('mAP50-95', [])
        if map50_95s:
            map50_95_val = map50_95s[-1] * 100
            color = self._metric_color(map50_95_val)
            self.map50_95_card.set_value(map50_95_val, color)

            # Track best
            if map50_95_val > self._best_map50_95:
                self._best_map50_95 = map50_95_val
                self.best_map50_95_card.set_value(map50_95_val, "#4CAF50")

        # Update status
        status = metrics.get('status', 'running')
        if status == 'running':
            self.status_label.setText(f"Training in progress - Epoch {epoch}/{total_epochs}")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        elif status == 'paused':
            self.status_label.setText("Training paused")
            self.status_label.setStyleSheet("color: #FF9800; font-size: 11px;")
        elif status == 'completed':
            self.status_label.setText("Training completed successfully!")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
        elif status == 'failed':
            self.status_label.setText("Training failed")
            self.status_label.setStyleSheet("color: #f44336; font-size: 11px;")

    def reset(self):
        """Reset all metrics to default state"""
        self._training_start_time = None
        self._epoch_times = []
        self._best_map50 = 0.0
        self._best_map50_95 = 0.0

        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Epoch 0/0")
        self.elapsed_label.setText("Elapsed: --")
        self.eta_label.setText("ETA: --")

        self.loss_card.clear()
        self.val_loss_card.clear()
        self.precision_card.clear()
        self.recall_card.clear()
        self.map50_card.clear()
        self.map50_95_card.clear()
        self.best_map50_card.clear()
        self.best_map50_95_card.clear()

        self.status_label.setText("Waiting for training to start...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds <= 0:
            return "--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    @staticmethod
    def _metric_color(value: float) -> str:
        """Get color based on metric value (0-100 scale)"""
        if value >= 80:
            return "#4CAF50"  # Green - excellent
        elif value >= 60:
            return "#8BC34A"  # Light green - good
        elif value >= 40:
            return "#FF9800"  # Orange - fair
        elif value >= 20:
            return "#FF5722"  # Deep orange - poor
        else:
            return "#f44336"  # Red - very poor

    @staticmethod
    def _loss_color(value: float) -> str:
        """Get color based on loss value (lower is better)"""
        if value < 0.5:
            return "#4CAF50"  # Green
        elif value < 1.0:
            return "#8BC34A"  # Light green
        elif value < 2.0:
            return "#FF9800"  # Orange
        else:
            return "#f44336"  # Red
