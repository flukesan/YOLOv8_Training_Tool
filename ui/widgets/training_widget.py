"""
Training Widget - training controls
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QComboBox)
from PyQt6.QtCore import pyqtSignal
from config.settings import Settings


class TrainingWidget(QWidget):
    """Widget for training controls"""

    start_training = pyqtSignal(dict)
    stop_training = pyqtSignal()
    pause_training = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Training Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()

        # Get default values from Settings
        default_params = Settings.DEFAULT_TRAIN_PARAMS

        # Model selection
        self.model_combo = QComboBox()
        model_descriptions = {
            'YOLOv8n': '⚡ Fastest - For real-time detection',
            'YOLOv8s': '⚖️ Balanced - Recommended for Go/NoGo',
            'YOLOv8m': '🎯 Accurate - For small defects',
            'YOLOv8l': '💪 Very Accurate - High precision',
            'YOLOv8x': '🏆 Most Accurate - Best quality'
        }
        for model_name, model_file in Settings.YOLO_MODELS.items():
            display_text = f"{model_name} - {model_descriptions.get(model_name, '')}"
            self.model_combo.addItem(display_text, model_file)
        # Set default to YOLOv8s (index 1)
        self.model_combo.setCurrentIndex(1)
        self.model_combo.setToolTip(
            "Choose model size:\n"
            "• YOLOv8n: Fastest, lowest accuracy\n"
            "• YOLOv8s: Balanced (Recommended)\n"
            "• YOLOv8m: High accuracy\n"
            "• YOLOv8l: Very high accuracy\n"
            "• YOLOv8x: Maximum accuracy, slowest"
        )
        param_layout.addRow("Model:", self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(default_params.get('epochs', 100))
        param_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(default_params.get('batch', 16))
        param_layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(default_params.get('lr0', 0.01))
        self.lr_spin.setDecimals(4)
        param_layout.addRow("Learning Rate:", self.lr_spin)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Control buttons
        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self._on_start)
        layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_training.emit)
        self.btn_pause.setEnabled(False)
        layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_training.emit)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def _on_start(self):
        """Handle start training"""
        config = {
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),  # Changed from 'batch_size' to 'batch'
            'lr0': self.lr_spin.value(),
            'model': self.model_combo.currentData()  # Get selected model file
        }
        self.start_training.emit(config)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def set_status(self, status: str):
        """Update status label"""
        self.status_label.setText(status)

    def training_finished(self):
        """Reset UI after training"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
