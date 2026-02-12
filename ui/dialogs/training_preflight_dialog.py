"""
Training Pre-flight Check Dialog - validates dataset readiness before training
Automatically splits dataset if not already split.
"""
from pathlib import Path
from typing import List, Dict
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QFrame, QScrollArea, QWidget)
from PyQt6.QtCore import Qt
from config.settings import Settings


class CheckItem(QFrame):
    """A single check item showing pass/fail status"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)

        self.icon_label = QLabel()
        self.icon_label.setFixedWidth(20)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel(label)
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.detail_label)

        self.setLayout(layout)
        self.set_pending()

    def set_pass(self, detail: str = ""):
        self.icon_label.setText("[OK]")
        self.icon_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
        self.text_label.setStyleSheet("color: #ddd;")
        self.detail_label.setText(detail)
        self.detail_label.setStyleSheet("color: #aaa; font-size: 11px;")

    def set_auto(self, detail: str = ""):
        """Set auto-completed status (action was taken automatically)"""
        self.icon_label.setText("[OK]")
        self.icon_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 11px;")
        self.text_label.setStyleSheet("color: #2196F3;")
        self.detail_label.setText(detail)
        self.detail_label.setStyleSheet("color: #2196F3; font-size: 11px;")

    def set_fail(self, detail: str = ""):
        self.icon_label.setText("[X]")
        self.icon_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 11px;")
        self.text_label.setStyleSheet("color: #f44336;")
        self.detail_label.setText(detail)
        self.detail_label.setStyleSheet("color: #f44336; font-size: 11px;")

    def set_warning(self, detail: str = ""):
        self.icon_label.setText("[!]")
        self.icon_label.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 11px;")
        self.text_label.setStyleSheet("color: #FF9800;")
        self.detail_label.setText(detail)
        self.detail_label.setStyleSheet("color: #FF9800; font-size: 11px;")

    def set_pending(self):
        self.icon_label.setText("[ ]")
        self.icon_label.setStyleSheet("color: #888; font-size: 11px;")
        self.text_label.setStyleSheet("color: #888;")
        self.detail_label.setText("")


class TrainingPreflightDialog(QDialog):
    """Dialog that validates dataset readiness before training starts.
    Automatically splits dataset if not already split."""

    def __init__(self, project_path: Path, classes: list,
                 dataset_manager=None, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.classes = classes
        self.dataset_manager = dataset_manager
        self.all_passed = False
        self._auto_split_performed = False

        self.setWindowTitle("Pre-Training Checklist")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.init_ui()
        self.run_checks()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Header
        header = QLabel("Pre-Training Validation")
        header.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(header)

        desc = QLabel(
            "Checking that your dataset and configuration are ready for training. "
            "Dataset will be split automatically if needed."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Checks container
        checks_group = QGroupBox("Validation Checks")
        checks_layout = QVBoxLayout()
        checks_layout.setSpacing(4)

        self.check_project = CheckItem("Project is open and valid")
        checks_layout.addWidget(self.check_project)

        self.check_classes = CheckItem("Classes are defined")
        checks_layout.addWidget(self.check_classes)

        self.check_images = CheckItem("Images exist in project")
        checks_layout.addWidget(self.check_images)

        self.check_split = CheckItem("Dataset is split (train/val)")
        checks_layout.addWidget(self.check_split)

        self.check_train_images = CheckItem("Training set has images")
        checks_layout.addWidget(self.check_train_images)

        self.check_val_images = CheckItem("Validation set has images")
        checks_layout.addWidget(self.check_val_images)

        self.check_annotations = CheckItem("Annotations exist in training set")
        checks_layout.addWidget(self.check_annotations)

        self.check_data_yaml = CheckItem("data.yaml configuration exists")
        checks_layout.addWidget(self.check_data_yaml)

        self.check_gpu = CheckItem("Compute device available")
        checks_layout.addWidget(self.check_gpu)

        checks_group.setLayout(checks_layout)
        layout.addWidget(checks_group)

        # Auto-split info banner (hidden by default)
        self.auto_split_banner = QFrame()
        self.auto_split_banner.setStyleSheet(
            "background-color: #1a2a3d; border: 1px solid #2196F3; "
            "border-radius: 4px; padding: 4px;"
        )
        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self.auto_split_label = QLabel("")
        self.auto_split_label.setWordWrap(True)
        self.auto_split_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        banner_layout.addWidget(self.auto_split_label)
        self.auto_split_banner.setLayout(banner_layout)
        self.auto_split_banner.setVisible(False)
        layout.addWidget(self.auto_split_banner)

        # Summary
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "font-size: 12px; padding: 8px; border-radius: 4px;"
        )
        layout.addWidget(self.summary_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_start = QPushButton("Start Training")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a9a58; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.btn_start.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _needs_auto_split(self) -> bool:
        """Check if dataset needs to be auto-split"""
        if not self.project_path:
            return False

        train_img_dir = self.project_path / 'train' / 'images'
        val_img_dir = self.project_path / 'val' / 'images'

        has_train = (train_img_dir.exists() and
                     any(self._iter_images(train_img_dir)))
        has_val = (val_img_dir.exists() and
                   any(self._iter_images(val_img_dir)))

        return not (has_train and has_val)

    def _has_source_images(self) -> int:
        """Check if source images directory has images"""
        images_dir = self.project_path / 'images'
        if not images_dir.exists():
            return 0
        return sum(1 for _ in self._iter_images(images_dir))

    def _perform_auto_split(self) -> dict:
        """Automatically split dataset with default ratios"""
        if not self.dataset_manager:
            return {}

        try:
            self.dataset_manager.refresh_images_list()
            if not self.dataset_manager.images_list:
                return {}

            ratios = Settings.DATASET_SPLIT  # 70/20/10
            stats = self.dataset_manager.split_dataset(ratios)
            self._auto_split_performed = True
            return stats
        except Exception as e:
            print(f"Auto-split failed: {e}")
            return {}

    @staticmethod
    def _iter_images(directory: Path):
        """Iterate over image files in a directory"""
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                yield f

    def run_checks(self):
        """Run all validation checks, auto-splitting if needed"""
        errors = 0
        warnings = 0

        # 1. Check project path
        if self.project_path and self.project_path.exists():
            self.check_project.set_pass(str(self.project_path.name))
        else:
            self.check_project.set_fail("No project open")
            errors += 1

        # 2. Check classes
        if self.classes and len(self.classes) > 0:
            self.check_classes.set_pass(f"{len(self.classes)} classes")
        else:
            self.check_classes.set_fail("No classes defined")
            errors += 1

        # 3. Check images in project
        source_img_count = self._has_source_images()
        if source_img_count > 0:
            self.check_images.set_pass(f"{source_img_count} images")
        else:
            self.check_images.set_fail("No images found in project")
            errors += 1

        # 4. Auto-split if needed and possible
        if self._needs_auto_split() and source_img_count > 0 and self.dataset_manager:
            split_stats = self._perform_auto_split()
            if split_stats:
                train_n = split_stats.get('train', 0)
                val_n = split_stats.get('val', 0)
                test_n = split_stats.get('test', 0)
                ratios = Settings.DATASET_SPLIT
                self.check_split.set_auto(
                    f"Auto-split: Train {train_n} / Val {val_n} / Test {test_n} "
                    f"({int(ratios['train']*100)}/{int(ratios['val']*100)}/{int(ratios['test']*100)}%)"
                )
                # Show auto-split banner
                self.auto_split_banner.setVisible(True)
                self.auto_split_label.setText(
                    f"Dataset was automatically split into "
                    f"Train ({train_n}), Val ({val_n}), Test ({test_n}) "
                    f"using default {int(ratios['train']*100)}/{int(ratios['val']*100)}/{int(ratios['test']*100)}% ratio. "
                    f"You can customize this ratio via Dataset > Split Dataset."
                )
            else:
                self.check_split.set_fail("Auto-split failed - try Dataset > Split Dataset manually")
                errors += 1
        elif not self._needs_auto_split():
            self.check_split.set_pass("Train and Val sets exist")
        else:
            self.check_split.set_fail("No images to split")
            errors += 1

        # 5. Check training images count
        train_img_dir = self.project_path / 'train' / 'images' if self.project_path else None
        if train_img_dir and train_img_dir.exists():
            train_count = sum(1 for _ in self._iter_images(train_img_dir))
            if train_count >= 10:
                self.check_train_images.set_pass(f"{train_count} images")
            elif train_count > 0:
                self.check_train_images.set_warning(
                    f"Only {train_count} images (recommend 50+)"
                )
                warnings += 1
            else:
                self.check_train_images.set_fail("No training images")
                errors += 1
        else:
            self.check_train_images.set_fail("No training directory")
            errors += 1

        # 6. Check validation images count
        val_img_dir = self.project_path / 'val' / 'images' if self.project_path else None
        if val_img_dir and val_img_dir.exists():
            val_count = sum(1 for _ in self._iter_images(val_img_dir))
            if val_count >= 5:
                self.check_val_images.set_pass(f"{val_count} images")
            elif val_count > 0:
                self.check_val_images.set_warning(
                    f"Only {val_count} images (recommend 10+)"
                )
                warnings += 1
            else:
                self.check_val_images.set_fail("No validation images")
                errors += 1
        else:
            self.check_val_images.set_fail("No validation directory")
            errors += 1

        # 7. Check annotations
        train_lbl_dir = self.project_path / 'train' / 'labels' if self.project_path else None
        if train_lbl_dir and train_lbl_dir.exists():
            label_count = len(list(train_lbl_dir.glob('*.txt')))
            if label_count > 0:
                self.check_annotations.set_pass(f"{label_count} annotation files")
            else:
                self.check_annotations.set_warning(
                    "No annotations yet - model will train without labels"
                )
                warnings += 1
        else:
            self.check_annotations.set_warning(
                "No annotations yet - model will train without labels"
            )
            warnings += 1

        # 8. Check data.yaml
        config_yaml = self.project_path / 'config.yaml' if self.project_path else None
        if config_yaml and config_yaml.exists():
            self.check_data_yaml.set_pass("config.yaml found")
        else:
            self.check_data_yaml.set_warning(
                "Will be created automatically"
            )
            warnings += 1

        # 9. Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.check_gpu.set_pass(f"{gpu_name} ({mem:.1f} GB)")
            else:
                self.check_gpu.set_warning("No GPU - will use CPU (slow)")
                warnings += 1
        except ImportError:
            self.check_gpu.set_warning("PyTorch not found - install required")
            warnings += 1

        # Summary
        if errors == 0:
            self.all_passed = True
            self.btn_start.setEnabled(True)
            if self._auto_split_performed:
                self.summary_label.setText(
                    "Dataset was automatically prepared! "
                    "All checks passed. Ready to start training."
                )
                self.summary_label.setStyleSheet(
                    "color: #2196F3; font-size: 12px; padding: 8px; "
                    "background-color: #1a2a3d; border-radius: 4px;"
                )
            elif warnings > 0:
                self.summary_label.setText(
                    f"All required checks passed with {warnings} warning(s). "
                    "You can start training."
                )
                self.summary_label.setStyleSheet(
                    "color: #FF9800; font-size: 12px; padding: 8px; "
                    "background-color: #3d3520; border-radius: 4px;"
                )
            else:
                self.summary_label.setText(
                    "All checks passed! Your dataset is ready for training."
                )
                self.summary_label.setStyleSheet(
                    "color: #4CAF50; font-size: 12px; padding: 8px; "
                    "background-color: #1a3320; border-radius: 4px;"
                )
        else:
            self.all_passed = False
            self.btn_start.setEnabled(False)
            self.summary_label.setText(
                f"{errors} check(s) failed. Please fix the issues above before training."
            )
            self.summary_label.setStyleSheet(
                "color: #f44336; font-size: 12px; padding: 8px; "
                "background-color: #3d1a1a; border-radius: 4px;"
            )
