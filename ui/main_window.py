"""
Main Window - Modern application interface designed for production workers
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QMenuBar, QFileDialog, QMessageBox, QSplitter,
                             QStatusBar, QLabel, QFrame, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path

from ui.widgets.image_viewer import ImageViewer
from ui.widgets.label_widget import LabelWidget
from ui.widgets.class_manager import ClassManagerWidget
from ui.widgets.dataset_widget import DatasetWidget
from ui.widgets.training_widget import TrainingWidget
from ui.widgets.metrics_widget import MetricsWidget
from ui.dialogs.new_project_dialog import NewProjectDialog
from ui.dialogs.export_dialog import ExportDialog
from ui.dialogs.split_dataset_dialog import SplitDatasetDialog
from ui.dialogs.training_results_dialog import TrainingResultsDialog
from ui.dialogs.dataset_statistics_dialog import DatasetStatisticsDialog
from ui.dialogs.model_testing_dialog import ModelTestingDialog
from ui.dialogs.training_preflight_dialog import TrainingPreflightDialog

from core.dataset_manager import DatasetManager
from core.label_manager import LabelManager, BoundingBox, Polygon
from core.model_trainer import ModelTrainer
from core.export_manager import ExportManager
from config.settings import Settings


class WorkflowStep(QFrame):
    """Visual step indicator for the workflow"""

    def __init__(self, number: str, title: str, parent=None):
        super().__init__(parent)
        self._active = False

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        # Step number badge
        self.badge = QLabel(number)
        self.badge.setFixedSize(28, 28)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(
            "background-color: #2d313a; color: #8891a0; "
            "border-radius: 14px; font-weight: bold; font-size: 13px;"
        )
        layout.addWidget(self.badge)

        # Step title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #8891a0; font-size: 13px;")
        layout.addWidget(self.title_label)

        layout.addStretch()
        self.setLayout(layout)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.badge.setStyleSheet(
                "background-color: #2d7d46; color: #ffffff; "
                "border-radius: 14px; font-weight: bold; font-size: 13px;"
            )
            self.title_label.setStyleSheet(
                "color: #ffffff; font-size: 13px; font-weight: 600;"
            )
        else:
            self.badge.setStyleSheet(
                "background-color: #2d313a; color: #8891a0; "
                "border-radius: 14px; font-weight: bold; font-size: 13px;"
            )
            self.title_label.setStyleSheet("color: #8891a0; font-size: 13px;")

    def set_completed(self):
        self.badge.setText("OK")
        self.badge.setStyleSheet(
            "background-color: #1a3320; color: #4CAF50; "
            "border-radius: 14px; font-weight: bold; font-size: 10px;"
        )
        self.title_label.setStyleSheet(
            "color: #4CAF50; font-size: 13px;"
        )


class MainWindow(QMainWindow):
    """Main application window with modern UI"""

    training_epoch_update = pyqtSignal(dict)
    training_started = pyqtSignal()
    training_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.project_path = None
        self.dataset_manager = None
        self.label_manager = None
        self.model_trainer = None
        self.current_image = None
        self.classes = []

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        self.setGeometry(50, 50,
                        Settings.UI_SETTINGS['window_width'],
                        Settings.UI_SETTINGS['window_height'])

        # Create menu bar
        self.create_menu_bar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === Header Bar ===
        header = self._create_header()
        main_layout.addWidget(header)

        # === Workflow Steps ===
        workflow_bar = self._create_workflow_bar()
        main_layout.addWidget(workflow_bar)

        # === Content Area ===
        content = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        # Left: Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.box_added.connect(self.on_box_added)
        self.image_viewer.polygon_added.connect(self.on_polygon_added)

        # Right panel with tabs
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(380)

        # Tab 1: Dataset & Classes
        data_tab = QWidget()
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)

        self.class_manager = ClassManagerWidget()
        self.class_manager.class_added.connect(self.on_class_added)
        self.class_manager.class_deleted.connect(self.on_class_deleted)
        self.class_manager.class_selected.connect(self.on_class_selected)
        data_layout.addWidget(self.class_manager)

        self.dataset_widget = DatasetWidget()
        self.dataset_widget.image_selected.connect(self.load_image)
        self.dataset_widget.images_deleted.connect(self.on_images_deleted)
        data_layout.addWidget(self.dataset_widget)

        data_tab.setLayout(data_layout)
        right_panel.addTab(data_tab, "Dataset")

        # Tab 2: Annotations
        ann_tab = QWidget()
        ann_layout = QVBoxLayout()

        self.label_widget = LabelWidget()
        self.label_widget.delete_annotation.connect(self.on_delete_annotation)
        ann_layout.addWidget(self.label_widget)

        ann_tab.setLayout(ann_layout)
        right_panel.addTab(ann_tab, "Annotations")

        # Bottom panel: Training (tabs)
        bottom_panel = QTabWidget()

        # Training config tab
        train_tab = QWidget()
        train_layout = QHBoxLayout()
        train_layout.setSpacing(8)

        self.training_widget = TrainingWidget()
        self.training_widget.start_training.connect(self.on_start_training)
        self.training_widget.stop_training.connect(self.on_stop_training)
        self.training_widget.pause_training.connect(self.on_pause_training)
        train_layout.addWidget(self.training_widget, 1)

        self.metrics_widget = MetricsWidget()
        train_layout.addWidget(self.metrics_widget, 1)

        train_tab.setLayout(train_layout)
        bottom_panel.addTab(train_tab, "Training")

        # Connect training signals
        self.training_epoch_update.connect(self.update_training_metrics)
        self.training_started.connect(self.on_training_started)
        self.training_finished.connect(self.on_training_finished)

        # Splitters
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(self.image_viewer)
        v_splitter.addWidget(bottom_panel)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.addWidget(v_splitter)
        h_splitter.addWidget(right_panel)
        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 1)

        content_layout.addWidget(h_splitter)
        content.setLayout(content_layout)
        main_layout.addWidget(content, 1)

        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready - Create or open a project to get started")

    def _create_header(self) -> QWidget:
        """Create the modern header bar"""
        header = QWidget()
        header.setStyleSheet(
            "background-color: #15171c; border-bottom: 2px solid #2d7d46;"
        )
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(16, 8, 16, 8)

        # App title
        title = QLabel(Settings.APP_NAME)
        title.setStyleSheet(
            "color: #ffffff; font-size: 18px; font-weight: 700; "
            "background-color: transparent;"
        )
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Project name indicator
        self.project_label = QLabel("No project open")
        self.project_label.setStyleSheet(
            "color: #8891a0; font-size: 13px; background-color: transparent;"
        )
        header_layout.addWidget(self.project_label)

        header.setLayout(header_layout)
        header.setFixedHeight(48)
        return header

    def _create_workflow_bar(self) -> QWidget:
        """Create the workflow step indicator"""
        bar = QWidget()
        bar.setStyleSheet(
            "background-color: #1e2128; border-bottom: 1px solid #2d313a;"
        )
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        self.step_project = WorkflowStep("1", "Project")
        self.step_project.mousePressEvent = lambda e: self.new_project()
        layout.addWidget(self.step_project)

        sep1 = QLabel("-->")
        sep1.setStyleSheet("color: #3d4250; background-color: transparent;")
        layout.addWidget(sep1)

        self.step_data = WorkflowStep("2", "Import Data")
        self.step_data.mousePressEvent = lambda e: self.import_images()
        layout.addWidget(self.step_data)

        sep2 = QLabel("-->")
        sep2.setStyleSheet("color: #3d4250; background-color: transparent;")
        layout.addWidget(sep2)

        self.step_annotate = WorkflowStep("3", "Annotate")
        layout.addWidget(self.step_annotate)

        sep3 = QLabel("-->")
        sep3.setStyleSheet("color: #3d4250; background-color: transparent;")
        layout.addWidget(sep3)

        self.step_train = WorkflowStep("4", "Train Model")
        layout.addWidget(self.step_train)

        sep4 = QLabel("-->")
        sep4.setStyleSheet("color: #3d4250; background-color: transparent;")
        layout.addWidget(sep4)

        self.step_export = WorkflowStep("5", "Export")
        self.step_export.mousePressEvent = lambda e: self.export_model()
        layout.addWidget(self.step_export)

        layout.addStretch()
        bar.setLayout(layout)
        bar.setFixedHeight(44)
        return bar

    def _update_workflow_steps(self):
        """Update workflow step indicators based on current project state"""
        for step in [self.step_project, self.step_data, self.step_annotate,
                     self.step_train, self.step_export]:
            step.set_active(False)

        if not self.project_path:
            self.step_project.set_active(True)
            return

        self.step_project.set_completed()

        images_dir = self.project_path / 'images'
        has_images = images_dir.exists() and any(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        )

        if has_images:
            self.step_data.set_completed()
        else:
            self.step_data.set_active(True)
            return

        labels_dir = self.project_path / 'labels'
        has_labels = labels_dir.exists() and any(labels_dir.glob('*.txt'))

        if has_labels:
            self.step_annotate.set_completed()
            self.step_train.set_active(True)
        else:
            self.step_annotate.set_active(True)

        runs_dir = self.project_path / 'runs' / 'train'
        if runs_dir.exists() and any(runs_dir.iterdir()):
            self.step_train.set_completed()
            self.step_export.set_active(True)

    def _update_status(self, message: str):
        """Update status bar"""
        self.status_bar.showMessage(message)

    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Project", self.new_project)
        file_menu.addAction("Open Project", self.open_project)
        file_menu.addAction("Save", self.save_project)
        file_menu.addSeparator()
        file_menu.addAction("Import Images", self.import_images)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        dataset_menu = menubar.addMenu("Dataset")
        dataset_menu.addAction("Split Dataset", self.split_dataset)
        dataset_menu.addAction("Statistics", self.show_statistics)

        training_menu = menubar.addMenu("Training")
        training_menu.addAction("Start Training", self.on_start_training)
        training_menu.addAction("View Results", self.view_training_results)
        training_menu.addSeparator()
        training_menu.addAction("Test Model", self.test_model)
        training_menu.addAction("Export Model", self.export_model)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def new_project(self):
        """Create new project"""
        dialog = NewProjectDialog(self)
        if dialog.exec():
            info = dialog.get_project_info()
            self.project_path = info['path']
            Settings.create_project_structure(self.project_path)
            self.dataset_manager = DatasetManager(self.project_path)
            self.label_manager = LabelManager(self.project_path)
            self.model_trainer = ModelTrainer(self.project_path)
            self.project_label.setText(f"Project: {info['name']}")
            self.project_label.setStyleSheet(
                "color: #4CAF50; font-size: 13px; font-weight: 500; "
                "background-color: transparent;"
            )
            self._update_workflow_steps()
            self._update_status(f"Project created: {info['name']}")

    def open_project(self):
        """Open existing project"""
        path = QFileDialog.getExistingDirectory(self, "Open Project")
        if path:
            self.project_path = Path(path)
            self.dataset_manager = DatasetManager(self.project_path)
            self.label_manager = LabelManager(self.project_path)
            self.model_trainer = ModelTrainer(self.project_path)
            self.project_label.setText(f"Project: {self.project_path.name}")
            self.project_label.setStyleSheet(
                "color: #4CAF50; font-size: 13px; font-weight: 500; "
                "background-color: transparent;"
            )
            self.load_project_data()
            self._update_workflow_steps()

    def save_project(self):
        """Save current project"""
        if self.current_image and self.label_manager:
            label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
            annotations = self.image_viewer.annotations
            self.label_manager.save_annotations(label_path, annotations)
            self._update_status("Saved")

    def import_images(self):
        """Import images to project"""
        if not self.dataset_manager:
            QMessageBox.warning(self, "Warning", "Please create or open a project first")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if files:
            stats = self.dataset_manager.import_images(files)
            QMessageBox.information(
                self, "Import Complete",
                f"Imported: {stats['imported']}\nSkipped: {stats['skipped']}"
            )
            self.refresh_dataset()
            self._update_workflow_steps()

    def load_image(self, image_path: str):
        """Load image in viewer"""
        self.current_image = Path(image_path)
        self.image_viewer.load_image(self.current_image)

        if self.label_manager:
            label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
            annotations = self.label_manager.load_annotations(label_path)
            self.image_viewer.set_annotations(annotations)
            self.label_widget.update_annotations(annotations)

    def on_images_deleted(self, image_paths):
        """Handle deletion of images"""
        import os
        if not image_paths:
            return

        deleted_count = 0
        error_count = 0

        for image_path in image_paths:
            try:
                image_path = Path(image_path)
                if image_path.exists():
                    os.remove(image_path)
                label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
                if label_path.exists():
                    os.remove(label_path)
                deleted_count += 1
                if self.current_image and self.current_image == image_path:
                    self.image_viewer.clear_image()
                    self.current_image = None
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")
                error_count += 1

        self.refresh_dataset()
        self._update_workflow_steps()

        if error_count == 0:
            self._update_status(f"Deleted {deleted_count} image(s)")
        else:
            QMessageBox.warning(
                self, "Delete Complete",
                f"Deleted: {deleted_count}\nErrors: {error_count}"
            )

    def on_box_added(self, x1, y1, x2, y2, class_id):
        """Handle new bounding box"""
        if not self.label_manager or not self.current_image:
            return
        img_width, img_height = self.image_viewer.pixmap.width(), self.image_viewer.pixmap.height()
        bbox = BoundingBox.from_absolute(class_id, x1, y1, x2, y2, img_width, img_height)
        label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
        self.label_manager.add_annotation(label_path, bbox)
        annotations = self.label_manager.load_annotations(label_path)
        self.image_viewer.set_annotations(annotations)
        self.label_widget.update_annotations(annotations)
        self._update_workflow_steps()

    def on_polygon_added(self, points, class_id):
        """Handle new polygon"""
        if not self.label_manager or not self.current_image:
            return
        img_width, img_height = self.image_viewer.pixmap.width(), self.image_viewer.pixmap.height()
        polygon = Polygon.from_absolute(class_id, points, img_width, img_height)
        label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
        self.label_manager.add_annotation(label_path, polygon)
        annotations = self.label_manager.load_annotations(label_path)
        self.image_viewer.set_annotations(annotations)
        self.label_widget.update_annotations(annotations)
        self._update_workflow_steps()

    def on_delete_annotation(self, index):
        """Delete annotation"""
        if not self.label_manager or not self.current_image:
            return
        label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
        self.label_manager.delete_annotation(label_path, index)
        annotations = self.label_manager.load_annotations(label_path)
        self.image_viewer.set_annotations(annotations)
        self.label_widget.update_annotations(annotations)

    def on_class_added(self, class_name):
        """Add new class"""
        self.classes.append(class_name)
        self.class_manager.set_classes(self.classes)
        if self.label_manager:
            self.label_manager.set_classes(self.classes)
            self.image_viewer.set_classes(self.classes, self.label_manager.class_colors)
        self.save_classes_to_config()

    def on_class_deleted(self, index):
        """Delete class"""
        if 0 <= index < len(self.classes):
            del self.classes[index]
            self.class_manager.set_classes(self.classes)
            self.save_classes_to_config()

    def save_classes_to_config(self):
        """Save classes to project config.yaml"""
        if not self.dataset_manager or not self.classes:
            return
        try:
            self.dataset_manager.create_data_yaml(self.classes)
        except Exception as e:
            print(f"Error saving classes to config: {e}")

    def on_class_selected(self, class_id):
        """Class selected"""
        self.image_viewer.set_current_class(class_id)

    def split_dataset(self):
        """Split dataset into train/val/test"""
        if not self.dataset_manager:
            QMessageBox.warning(self, "Warning", "Please create or open a project first")
            return
        self.dataset_manager.refresh_images_list()
        if not self.dataset_manager.images_list:
            QMessageBox.warning(self, "No Images",
                              "No images found.\nPlease import images first.")
            return

        dialog = SplitDatasetDialog(self)
        if dialog.exec() != SplitDatasetDialog.DialogCode.Accepted:
            return

        ratios = dialog.get_ratios()
        try:
            stats = self.dataset_manager.split_dataset(ratios)
            total_images = sum(stats.values())
            QMessageBox.information(
                self, "Split Complete",
                f"Successfully split {total_images} images:\n\n"
                f"Train: {stats.get('train', 0)} ({int(ratios['train']*100)}%)\n"
                f"Val: {stats.get('val', 0)} ({int(ratios['val']*100)}%)\n"
                f"Test: {stats.get('test', 0)} ({int(ratios['test']*100)}%)"
            )
            self.refresh_dataset()
        except Exception as e:
            QMessageBox.critical(self, "Split Failed", f"Failed:\n{str(e)}")

    def show_statistics(self):
        """Show dataset statistics"""
        if not self.dataset_manager or not self.project_path:
            QMessageBox.warning(self, "No Project",
                              "Please create or open a project first.")
            return
        stats = self.dataset_manager.get_dataset_statistics(self.classes)
        dialog = DatasetStatisticsDialog(stats, self.project_path, self)
        dialog.exec()

    def on_start_training(self, config=None):
        """Start training with pre-flight validation"""
        if not self.model_trainer or not self.dataset_manager:
            QMessageBox.warning(self, "Warning",
                              "Please create or open a project first")
            return

        if not self.classes:
            QMessageBox.warning(self, "No Classes",
                              "Please add classes before training.\n\n"
                              "Use the Dataset tab to add object classes.")
            return

        # Run pre-flight check (auto-splits if needed)
        preflight = TrainingPreflightDialog(
            self.project_path, self.classes, self.dataset_manager, self
        )
        if preflight.exec() != TrainingPreflightDialog.DialogCode.Accepted:
            return

        try:
            data_yaml = self.dataset_manager.create_data_yaml(self.classes)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to create data.yaml:\n{str(e)}")
            return

        if config is None or not isinstance(config, dict):
            config = self.training_widget.get_training_config()

        try:
            Settings.save_training_config(config, self.project_path)
        except Exception as e:
            print(f"Warning: Could not save training config: {e}")

        self.model_trainer.register_callback('on_train_start', self._on_train_start_cb)
        self.model_trainer.register_callback('on_epoch_end', self._on_epoch_end_cb)
        self.model_trainer.register_callback('on_train_end', self._on_train_end_cb)
        self.model_trainer.register_callback('on_train_error', self._on_train_error_cb)

        self.metrics_widget.reset()

        try:
            model_name = config.get('model', 'yolov8s.pt')
            train_config = {k: v for k, v in config.items() if k != 'model'}
            self.model_trainer.start_training(
                train_config, data_yaml, model_name=model_name
            )
            self.step_train.set_active(True)
            self._update_status(f"Training started with {model_name}...")
        except Exception as e:
            self.training_widget.training_failed()
            QMessageBox.critical(self, "Training Error",
                               f"Failed to start training:\n{str(e)}")

    def _on_train_start_cb(self, session):
        self.training_started.emit()

    def _on_epoch_end_cb(self, session, metrics):
        current_metrics = self.model_trainer.get_current_metrics()
        self.training_epoch_update.emit(current_metrics)

    def _on_train_end_cb(self, session, results):
        self.training_finished.emit()

    def _on_train_error_cb(self, error_msg):
        self.training_finished.emit()

    def on_training_started(self):
        self.training_widget.set_status("Training...")
        self._update_status("Training in progress...")

    def on_training_finished(self):
        if self.model_trainer and self.model_trainer.current_session:
            status = self.model_trainer.current_session.status

            if status == 'completed':
                self.training_widget.training_finished()
                self._update_status("Training completed!")
                self._update_workflow_steps()

                best_path = self.model_trainer.get_best_weights_path()
                best_metrics = self.model_trainer.current_session.best_metrics

                msg = "Model training completed successfully!\n\n"
                if best_path:
                    msg += f"Best weights saved to:\n{best_path}\n\n"
                if best_metrics:
                    map50 = best_metrics.get('metrics/mAP50(B)', 0)
                    map50_95 = best_metrics.get('metrics/mAP50-95(B)', 0)
                    if map50 or map50_95:
                        msg += f"Best mAP@50: {map50:.3f}\n"
                        msg += f"Best mAP@50-95: {map50_95:.3f}\n\n"
                msg += "Use Training > View Results for charts.\n"
                msg += "Use Training > Test Model to test on new images."
                QMessageBox.information(self, "Training Complete", msg)

            elif status == 'failed':
                self.training_widget.training_failed()
                self._update_status("Training failed")
                QMessageBox.warning(
                    self, "Training Failed",
                    "Training failed. Check console for details.\n\n"
                    "Common causes:\n"
                    "- Not enough GPU memory (smaller batch/model)\n"
                    "- Invalid annotations\n"
                    "- Missing dependencies"
                )
            else:
                self.training_widget.training_finished()
                self._update_status(f"Training {status}")
        else:
            self.training_widget.training_finished()

    def update_training_metrics(self, metrics):
        """Update training metrics display"""
        self.metrics_widget.update_metrics(metrics)

        epoch = metrics.get('epoch', 0)
        total_epochs = metrics.get('total_epochs', 0)
        elapsed = metrics.get('elapsed_time', 0)

        eta_str = ""
        if epoch > 0 and total_epochs > 0:
            avg_epoch_time = elapsed / epoch
            remaining = avg_epoch_time * (total_epochs - epoch)
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            secs = int(remaining % 60)
            if hours > 0:
                eta_str = f"{hours}h {minutes}m"
            elif minutes > 0:
                eta_str = f"{minutes}m {secs}s"
            else:
                eta_str = f"{secs}s"

        self.training_widget.update_progress(epoch, total_epochs, eta_str)

        if epoch and total_epochs:
            self._update_status(f"Training - Epoch {epoch}/{total_epochs}")

    def on_stop_training(self):
        if self.model_trainer:
            reply = QMessageBox.question(
                self, "Stop Training",
                "Are you sure you want to stop training?\n\n"
                "Model weights from the last epoch will be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.model_trainer.stop_training()
                self._update_status("Stopping training...")

    def on_pause_training(self):
        if self.model_trainer and self.model_trainer.current_session:
            if self.model_trainer.current_session.status == 'paused':
                self.model_trainer.resume_training()
                self._update_status("Training resumed")
            else:
                self.model_trainer.pause_training()
                self._update_status("Training paused")

    def view_training_results(self):
        if not self.project_path:
            QMessageBox.warning(self, "No Project",
                              "Please create or open a project first.")
            return
        runs_dir = self.project_path / 'runs' / 'train'
        if not runs_dir.exists():
            QMessageBox.warning(self, "No Results",
                              "No training results found.\nTrain a model first.")
            return

        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)
        if not runs:
            QMessageBox.warning(self, "No Results", "No training runs found.")
            return
        try:
            dialog = TrainingResultsDialog(runs[0], self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"Failed to load results:\n{str(e)}")

    def test_model(self):
        if not self.project_path:
            QMessageBox.warning(self, "No Project",
                              "Please create or open a project first.")
            return
        dialog = ModelTestingDialog(self.project_path, self)
        dialog.exec()

    def export_model(self):
        if not self.project_path:
            QMessageBox.warning(self, "No Project",
                              "Please create or open a project first.")
            return

        if not self.model_trainer:
            QMessageBox.warning(self, "No Model",
                              "Please train a model first.")
            return

        best_weights = self.model_trainer.get_best_weights_path()
        if not best_weights or not best_weights.exists():
            QMessageBox.warning(self, "No Model",
                              "No trained model found.\nPlease train a model first.")
            return

        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            formats = dialog.get_selected_formats()
            if not formats:
                return

            try:
                import shutil
                import time

                self._update_status("Exporting model...")
                results = {}

                if 'pt' in formats:
                    try:
                        models_dir = self.project_path / 'models'
                        models_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = models_dir / f"best_{int(time.time())}.pt"
                        shutil.copy2(best_weights, dest_path)
                        results['pt'] = {'success': True, 'path': str(dest_path)}
                    except Exception as e:
                        results['pt'] = {'success': False, 'error': str(e)}

                other_formats = [f for f in formats if f != 'pt']
                if other_formats:
                    export_mgr = ExportManager(best_weights)
                    results.update(export_mgr.export_multiple(other_formats))

                success_count = sum(1 for r in results.values() if r.get('success'))
                failed_count = len(results) - success_count

                if failed_count == 0:
                    msg = f"Exported to {success_count} format(s):\n\n"
                    for fmt, r in results.items():
                        if r.get('success'):
                            msg += f"  {fmt.upper()}: {r.get('path', 'N/A')}\n"
                    QMessageBox.information(self, "Export Complete", msg)
                else:
                    msg = f"Export done with {failed_count} error(s):\n\n"
                    for fmt, r in results.items():
                        s = r.get('path', 'N/A') if r.get('success') else f"FAILED"
                        msg += f"  {fmt.upper()}: {s}\n"
                    QMessageBox.warning(self, "Export", msg)

                self._update_status("Export completed")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed",
                                   f"Failed to export:\n{str(e)}")

    def refresh_dataset(self):
        if self.dataset_manager:
            self.dataset_manager.refresh_images_list()
            self.dataset_widget.set_images(self.dataset_manager.images_list)

    def load_project_data(self):
        self.load_classes_from_config()
        self._load_saved_training_config()
        self.refresh_dataset()
        self._update_status(f"Loaded project: {self.project_path.name}")

    def _load_saved_training_config(self):
        if not self.project_path:
            return
        try:
            saved_config = Settings.load_training_config(self.project_path)
            if saved_config:
                self.training_widget.set_training_config(saved_config)
        except Exception as e:
            print(f"Could not load training config: {e}")

    def load_classes_from_config(self):
        import yaml
        if not self.project_path:
            return
        config_path = self.project_path / 'config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                if config and 'names' in config:
                    self.classes = config['names']
                    self.class_manager.set_classes(self.classes)
                    if self.label_manager:
                        self.label_manager.set_classes(self.classes)
                        self.image_viewer.set_classes(
                            self.classes, self.label_manager.class_colors
                        )
            except Exception as e:
                print(f"Error loading classes from config: {e}")

    def show_about(self):
        QMessageBox.about(
            self, "About",
            f"{Settings.APP_NAME} v{Settings.APP_VERSION}\n\n"
            "YOLOv8 Object Detection Training Tool\n"
            "Designed for industrial quality inspection"
        )
