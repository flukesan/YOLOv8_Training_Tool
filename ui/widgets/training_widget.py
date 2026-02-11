"""
Training Widget - Enhanced training controls with better usability
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSpinBox, QDoubleSpinBox, QFormLayout,
                             QGroupBox, QComboBox, QProgressBar, QCheckBox,
                             QScrollArea, QFrame, QToolButton, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from config.settings import Settings


class CollapsibleSection(QWidget):
    """A collapsible section widget for grouping advanced options"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self._on_toggle)

        self.content_widget = QWidget()
        self.content_layout = QFormLayout()
        self.content_widget.setLayout(self.content_layout)
        self.content_widget.setVisible(False)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)
        self.setLayout(layout)

    def _on_toggle(self, checked):
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.content_widget.setVisible(checked)

    def add_row(self, label: str, widget):
        self.content_layout.addRow(label, widget)


class TrainingWidget(QWidget):
    """Widget for training controls with enhanced usability"""

    start_training = pyqtSignal(dict)
    stop_training = pyqtSignal()
    pause_training = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._detected_device = self._detect_device()
        self.init_ui()

    def _detect_device(self) -> dict:
        """Detect available compute devices"""
        info = {
            'has_cuda': False,
            'cuda_name': '',
            'cuda_count': 0,
            'cuda_memory': '',
            'recommended': 'cpu'
        }
        try:
            import torch
            if torch.cuda.is_available():
                info['has_cuda'] = True
                info['cuda_count'] = torch.cuda.device_count()
                info['cuda_name'] = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                info['cuda_memory'] = f"{mem_gb:.1f} GB"
                info['recommended'] = '0'
        except ImportError:
            pass
        return info

    def init_ui(self):
        """Initialize UI"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Title
        title = QLabel("Training Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # === Device Info Banner ===
        self._create_device_banner(layout)

        # === Basic Parameters (always visible) ===
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout()
        basic_layout.setSpacing(6)

        # Model selection
        self.model_combo = QComboBox()
        model_descriptions = {
            'YOLOv8n': 'Nano - Fastest, for real-time',
            'YOLOv8s': 'Small - Balanced (Recommended)',
            'YOLOv8m': 'Medium - Higher accuracy',
            'YOLOv8l': 'Large - Very accurate',
            'YOLOv8x': 'XLarge - Best accuracy'
        }
        for model_name, model_file in Settings.YOLO_MODELS.items():
            display_text = f"{model_name} - {model_descriptions.get(model_name, '')}"
            self.model_combo.addItem(display_text, model_file)
        self.model_combo.setCurrentIndex(1)  # Default: YOLOv8s
        self.model_combo.setToolTip(
            "Choose model size:\n"
            "  YOLOv8n: ~3.2M params, fastest\n"
            "  YOLOv8s: ~11.2M params, balanced\n"
            "  YOLOv8m: ~25.9M params, accurate\n"
            "  YOLOv8l: ~43.7M params, very accurate\n"
            "  YOLOv8x: ~68.2M params, max accuracy"
        )
        basic_layout.addRow("Model:", self.model_combo)

        # Image size
        self.imgsz_combo = QComboBox()
        img_sizes = [
            ('320 - Fast, low detail', 320),
            ('416 - Balanced speed/detail', 416),
            ('640 - Default (Recommended)', 640),
            ('800 - High detail', 800),
            ('1024 - Very high detail', 1024),
            ('1280 - Maximum detail', 1280),
        ]
        for label, size in img_sizes:
            self.imgsz_combo.addItem(label, size)
        self.imgsz_combo.setCurrentIndex(2)  # Default: 640
        self.imgsz_combo.setToolTip(
            "Input image size for training.\n"
            "Larger = more detail but slower training and more GPU memory.\n"
            "640 is the standard for most use cases."
        )
        basic_layout.addRow("Image Size:", self.imgsz_combo)

        # Epochs
        default_params = Settings.DEFAULT_TRAIN_PARAMS
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(default_params.get('epochs', 100))
        self.epochs_spin.setToolTip(
            "Number of training epochs.\n"
            "More epochs = longer training but potentially better results.\n"
            "Early stopping will prevent overfitting."
        )
        basic_layout.addRow("Epochs:", self.epochs_spin)

        # Batch size with auto-detect recommendation
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 128)
        self.batch_spin.setValue(default_params.get('batch', 16))
        self.batch_spin.setSpecialValueText("Auto (-1)")
        self.batch_spin.setToolTip(
            "Batch size per iteration.\n"
            "Set to -1 for auto-detection based on GPU memory.\n"
            "Larger batch = faster training but more GPU memory.\n"
            "Common values: 8, 16, 32"
        )
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.batch_spin)
        self.batch_auto_btn = QPushButton("Auto")
        self.batch_auto_btn.setMaximumWidth(50)
        self.batch_auto_btn.setToolTip("Set to auto-detect optimal batch size")
        self.batch_auto_btn.clicked.connect(lambda: self.batch_spin.setValue(-1))
        batch_layout.addWidget(self.batch_auto_btn)
        basic_layout.addRow("Batch Size:", batch_layout)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # === Augmentation Preset ===
        aug_group = QGroupBox("Data Augmentation")
        aug_layout = QFormLayout()

        self.aug_combo = QComboBox()
        aug_presets = [
            ('None - No augmentation', 'none'),
            ('Light - Stable environments, large datasets', 'light'),
            ('Medium - Balanced (Recommended)', 'medium'),
            ('Heavy - Small datasets, varied conditions', 'heavy'),
            ('Industrial - Optimized for Go/NoGo inspection', 'industrial'),
        ]
        for label, key in aug_presets:
            self.aug_combo.addItem(label, key)
        self.aug_combo.setCurrentIndex(2)  # Default: medium
        self.aug_combo.setToolTip(
            "Data augmentation helps prevent overfitting:\n"
            "  None: No augmentation (use only if dataset is very large)\n"
            "  Light: Minimal transforms for controlled environments\n"
            "  Medium: Standard augmentation (recommended)\n"
            "  Heavy: Aggressive augmentation for small/varied datasets\n"
            "  Industrial: Optimized for factory inspection tasks"
        )
        aug_layout.addRow("Preset:", self.aug_combo)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

        # === Training Strategy (collapsible) ===
        strategy_section = CollapsibleSection("Training Strategy")

        # Optimizer
        self.optimizer_combo = QComboBox()
        optimizers = [
            ('SGD - Standard (Recommended)', 'SGD'),
            ('Adam - Fast convergence', 'Adam'),
            ('AdamW - Adam with weight decay', 'AdamW'),
            ('RMSProp - Adaptive learning rate', 'RMSProp'),
        ]
        for label, value in optimizers:
            self.optimizer_combo.addItem(label, value)
        self.optimizer_combo.setToolTip(
            "Optimizer algorithm:\n"
            "  SGD: Most stable, recommended for most tasks\n"
            "  Adam: Faster convergence, good for fine-tuning\n"
            "  AdamW: Adam with better weight decay handling"
        )
        strategy_section.add_row("Optimizer:", self.optimizer_combo)

        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(default_params.get('lr0', 0.01))
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setToolTip(
            "Initial learning rate.\n"
            "Default 0.01 works well for most cases.\n"
            "Lower values (0.001) for fine-tuning pre-trained models."
        )
        strategy_section.add_row("Learning Rate:", self.lr_spin)

        # Patience (early stopping)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 200)
        self.patience_spin.setValue(default_params.get('patience', 50))
        self.patience_spin.setSpecialValueText("Disabled (0)")
        self.patience_spin.setToolTip(
            "Early stopping patience.\n"
            "Training stops if no improvement for this many epochs.\n"
            "Set to 0 to disable early stopping.\n"
            "Default 50 is good for most tasks."
        )
        strategy_section.add_row("Early Stop Patience:", self.patience_spin)

        # Cosine LR
        self.cos_lr_check = QCheckBox("Use cosine learning rate schedule")
        self.cos_lr_check.setToolTip(
            "Smoothly decrease learning rate using cosine schedule.\n"
            "Often improves final model quality."
        )
        strategy_section.add_row("", self.cos_lr_check)

        layout.addWidget(strategy_section)

        # === Advanced Options (collapsible) ===
        advanced_section = CollapsibleSection("Advanced Options")

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItem(
            f"Auto ({self._detected_device['recommended']})", ''
        )
        if self._detected_device['has_cuda']:
            for i in range(self._detected_device['cuda_count']):
                try:
                    import torch
                    name = torch.cuda.get_device_name(i)
                    self.device_combo.addItem(f"GPU {i}: {name}", str(i))
                except Exception:
                    self.device_combo.addItem(f"GPU {i}", str(i))
        self.device_combo.addItem("CPU (slow)", 'cpu')
        self.device_combo.setToolTip(
            "Compute device for training.\n"
            "GPU is strongly recommended for reasonable training speed."
        )
        advanced_section.add_row("Device:", self.device_combo)

        # Workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(default_params.get('workers', 8))
        self.workers_spin.setToolTip(
            "Number of data loading workers.\n"
            "More workers = faster data loading.\n"
            "Set to 0 if you experience data loading issues."
        )
        advanced_section.add_row("Data Workers:", self.workers_spin)

        # Cache
        self.cache_combo = QComboBox()
        cache_options = [
            ('None - Load from disk each time', 'none'),
            ('RAM - Cache in memory (fastest, needs RAM)', 'ram'),
            ('Disk - Cache on disk', 'disk'),
        ]
        for label, value in cache_options:
            self.cache_combo.addItem(label, value)
        self.cache_combo.setToolTip(
            "Dataset caching:\n"
            "  None: Read from disk (default, lowest memory)\n"
            "  RAM: Cache in memory (fast, but needs lots of RAM)\n"
            "  Disk: Cache to disk (balance)"
        )
        advanced_section.add_row("Cache:", self.cache_combo)

        # AMP
        self.amp_check = QCheckBox("Mixed Precision Training (AMP)")
        self.amp_check.setChecked(default_params.get('amp', True))
        self.amp_check.setToolTip(
            "Automatic Mixed Precision.\n"
            "Speeds up training and reduces GPU memory usage.\n"
            "Recommended to keep enabled."
        )
        advanced_section.add_row("", self.amp_check)

        # Multi-scale
        self.multiscale_check = QCheckBox("Multi-scale Training")
        self.multiscale_check.setToolTip(
            "Vary input image size during training.\n"
            "Can improve model robustness at different scales."
        )
        advanced_section.add_row("", self.multiscale_check)

        # Freeze backbone
        self.freeze_spin = QSpinBox()
        self.freeze_spin.setRange(0, 24)
        self.freeze_spin.setValue(0)
        self.freeze_spin.setSpecialValueText("None (0)")
        self.freeze_spin.setToolTip(
            "Number of backbone layers to freeze.\n"
            "Useful for transfer learning / fine-tuning.\n"
            "Set to 0 to train all layers (default).\n"
            "Set to 10 to freeze the backbone (faster training)."
        )
        advanced_section.add_row("Freeze Layers:", self.freeze_spin)

        # Seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(default_params.get('seed', 0))
        self.seed_spin.setToolTip("Random seed for reproducibility.")
        advanced_section.add_row("Random Seed:", self.seed_spin)

        layout.addWidget(advanced_section)

        # === Progress Section ===
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to train")
        self.status_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.status_label)

        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888; font-size: 11px;")
        progress_layout.addWidget(self.eta_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # === Control Buttons ===
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("Start Training")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a9a58; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(
            "QPushButton { background-color: #c8a02a; color: white; "
            "padding: 8px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #dab52f; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)
        btn_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #a83232; color: white; "
            "padding: 8px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #c43c3c; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.btn_stop.clicked.connect(self.stop_training.emit)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)

        layout.addStretch()
        container.setLayout(layout)
        scroll.setWidget(container)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

        self._is_paused = False

    def _create_device_banner(self, layout):
        """Create device info banner at top"""
        banner = QFrame()
        banner.setFrameShape(QFrame.Shape.StyledPanel)
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(8, 4, 8, 4)

        if self._detected_device['has_cuda']:
            icon_label = QLabel("GPU")
            icon_label.setStyleSheet(
                "color: #4CAF50; font-weight: bold; font-size: 11px;"
            )
            banner_layout.addWidget(icon_label)
            info = (
                f"{self._detected_device['cuda_name']} "
                f"({self._detected_device['cuda_memory']})"
            )
            info_label = QLabel(info)
            info_label.setStyleSheet("font-size: 11px; color: #aaa;")
            banner_layout.addWidget(info_label)
        else:
            icon_label = QLabel("CPU Only")
            icon_label.setStyleSheet(
                "color: #FF9800; font-weight: bold; font-size: 11px;"
            )
            banner_layout.addWidget(icon_label)
            info_label = QLabel("No GPU detected. Training will be slow.")
            info_label.setStyleSheet("font-size: 11px; color: #aaa;")
            banner_layout.addWidget(info_label)

        banner_layout.addStretch()
        banner.setLayout(banner_layout)
        layout.addWidget(banner)

    def _set_config_enabled(self, enabled: bool):
        """Enable/disable all configuration controls during training"""
        controls = [
            self.model_combo, self.imgsz_combo, self.epochs_spin,
            self.batch_spin, self.batch_auto_btn, self.aug_combo,
            self.optimizer_combo, self.lr_spin, self.patience_spin,
            self.cos_lr_check, self.device_combo, self.workers_spin,
            self.cache_combo, self.amp_check, self.multiscale_check,
            self.freeze_spin, self.seed_spin,
        ]
        for ctrl in controls:
            ctrl.setEnabled(enabled)

    def _on_start(self):
        """Handle start training - collect all config"""
        config = self.get_training_config()
        self.start_training.emit(config)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._set_config_enabled(False)
        self._is_paused = False
        self.progress_bar.setFormat("Starting...")
        self.status_label.setText("Initializing training...")
        self.status_label.setStyleSheet("color: #4CAF50;")

    def _on_pause(self):
        """Toggle pause/resume"""
        if self._is_paused:
            self.pause_training.emit()
            self.btn_pause.setText("Pause")
            self.status_label.setText("Training resumed")
            self.status_label.setStyleSheet("color: #4CAF50;")
            self._is_paused = False
        else:
            self.pause_training.emit()
            self.btn_pause.setText("Resume")
            self.status_label.setText("Training paused")
            self.status_label.setStyleSheet("color: #FF9800;")
            self._is_paused = True

    def get_training_config(self) -> dict:
        """Collect all training configuration into a dictionary"""
        config = {
            'model': self.model_combo.currentData(),
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),
            'imgsz': self.imgsz_combo.currentData(),
            'lr0': self.lr_spin.value(),
            'optimizer': self.optimizer_combo.currentData(),
            'patience': self.patience_spin.value(),
            'cos_lr': self.cos_lr_check.isChecked(),
            'amp': self.amp_check.isChecked(),
            'multi_scale': self.multiscale_check.isChecked(),
            'workers': self.workers_spin.value(),
            'seed': self.seed_spin.value(),
        }

        # Device
        device = self.device_combo.currentData()
        if device:
            config['device'] = device

        # Cache
        cache_val = self.cache_combo.currentData()
        if cache_val != 'none':
            config['cache'] = cache_val

        # Freeze
        freeze_val = self.freeze_spin.value()
        if freeze_val > 0:
            config['freeze'] = freeze_val

        # Augmentation preset
        aug_key = self.aug_combo.currentData()
        if aug_key != 'none' and aug_key in Settings.AUGMENTATION_PRESETS:
            config.update(Settings.AUGMENTATION_PRESETS[aug_key])

        return config

    def set_training_config(self, config: dict):
        """Load training configuration into UI widgets"""
        if not config:
            return

        # Model
        model = config.get('model', 'yolov8s.pt')
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model:
                self.model_combo.setCurrentIndex(i)
                break

        # Image size
        imgsz = config.get('imgsz', 640)
        for i in range(self.imgsz_combo.count()):
            if self.imgsz_combo.itemData(i) == imgsz:
                self.imgsz_combo.setCurrentIndex(i)
                break

        # Simple spin/check values
        if 'epochs' in config:
            self.epochs_spin.setValue(config['epochs'])
        if 'batch' in config:
            self.batch_spin.setValue(config['batch'])
        if 'lr0' in config:
            self.lr_spin.setValue(config['lr0'])
        if 'patience' in config:
            self.patience_spin.setValue(config['patience'])
        if 'cos_lr' in config:
            self.cos_lr_check.setChecked(config['cos_lr'])
        if 'amp' in config:
            self.amp_check.setChecked(config['amp'])
        if 'multi_scale' in config:
            self.multiscale_check.setChecked(config['multi_scale'])
        if 'workers' in config:
            self.workers_spin.setValue(config['workers'])
        if 'seed' in config:
            self.seed_spin.setValue(config['seed'])

        # Optimizer
        optimizer = config.get('optimizer', 'SGD')
        for i in range(self.optimizer_combo.count()):
            if self.optimizer_combo.itemData(i) == optimizer:
                self.optimizer_combo.setCurrentIndex(i)
                break

        # Device
        device = config.get('device', '')
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == device:
                self.device_combo.setCurrentIndex(i)
                break

        # Cache
        cache_val = config.get('cache', 'none') or 'none'
        for i in range(self.cache_combo.count()):
            if self.cache_combo.itemData(i) == cache_val:
                self.cache_combo.setCurrentIndex(i)
                break

        # Freeze
        if 'freeze' in config and config['freeze']:
            self.freeze_spin.setValue(config['freeze'])

    def set_status(self, status: str):
        """Update status label"""
        self.status_label.setText(status)

    def update_progress(self, epoch: int, total_epochs: int, eta_str: str = ""):
        """Update progress bar and ETA"""
        progress = int((epoch / max(total_epochs, 1)) * 100)
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"Epoch {epoch}/{total_epochs} ({progress}%)")
        if eta_str:
            self.eta_label.setText(f"ETA: {eta_str}")

    def training_finished(self):
        """Reset UI after training"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._is_paused = False
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete")
        self.status_label.setText("Training finished")
        self.status_label.setStyleSheet("color: #4CAF50;")
        self.eta_label.setText("")

    def training_failed(self):
        """Reset UI after training failure"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self._set_config_enabled(True)
        self._is_paused = False
        self.progress_bar.setFormat("Failed")
        self.status_label.setText("Training failed - check logs")
        self.status_label.setStyleSheet("color: #f44336;")
        self.eta_label.setText("")
