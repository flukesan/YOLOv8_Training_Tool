"""
Main Window - Application main interface
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QMenuBar, QFileDialog, QMessageBox, QSplitter,
                             QStatusBar)
from PyQt6.QtCore import Qt, pyqtSignal
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

from core.dataset_manager import DatasetManager
from core.label_manager import LabelManager, BoundingBox, Polygon
from core.model_trainer import ModelTrainer
from core.export_manager import ExportManager
from config.settings import Settings


class MainWindow(QMainWindow):
    """Main application window"""

    # Signals for training updates (thread-safe UI updates)
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
        self.setWindowTitle(Settings.APP_NAME)
        self.setGeometry(100, 100,
                        Settings.UI_SETTINGS['window_width'],
                        Settings.UI_SETTINGS['window_height'])

        # Create menu bar
        self.create_menu_bar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()

        # Left panel - Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.box_added.connect(self.on_box_added)
        self.image_viewer.polygon_added.connect(self.on_polygon_added)

        # Right panel - Tools
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Class manager
        self.class_manager = ClassManagerWidget()
        self.class_manager.class_added.connect(self.on_class_added)
        self.class_manager.class_deleted.connect(self.on_class_deleted)
        self.class_manager.class_selected.connect(self.on_class_selected)
        right_layout.addWidget(self.class_manager)

        # Dataset browser
        self.dataset_widget = DatasetWidget()
        self.dataset_widget.image_selected.connect(self.load_image)
        self.dataset_widget.images_deleted.connect(self.on_images_deleted)
        right_layout.addWidget(self.dataset_widget)

        # Label widget
        self.label_widget = LabelWidget()
        self.label_widget.delete_annotation.connect(self.on_delete_annotation)
        right_layout.addWidget(self.label_widget)

        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(300)

        # Bottom panel - Training
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout()

        self.training_widget = TrainingWidget()
        self.training_widget.start_training.connect(self.on_start_training)
        self.training_widget.stop_training.connect(self.on_stop_training)
        bottom_layout.addWidget(self.training_widget)

        self.metrics_widget = MetricsWidget()
        bottom_layout.addWidget(self.metrics_widget)

        # Connect training signals to update UI
        self.training_epoch_update.connect(self.update_training_metrics)
        self.training_started.connect(self.on_training_started)
        self.training_finished.connect(self.on_training_finished)

        bottom_panel.setLayout(bottom_layout)

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

        main_layout.addWidget(h_splitter)
        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Project", self.new_project)
        file_menu.addAction("Open Project", self.open_project)
        file_menu.addAction("Save", self.save_project)
        file_menu.addSeparator()
        file_menu.addAction("Import Images", self.import_images)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Dataset menu
        dataset_menu = menubar.addMenu("Dataset")
        dataset_menu.addAction("Split Dataset", self.split_dataset)
        dataset_menu.addAction("Statistics", self.show_statistics)

        # Training menu
        training_menu = menubar.addMenu("Training")
        training_menu.addAction("Start Training", self.on_start_training)
        training_menu.addAction("View Results", self.view_training_results)
        training_menu.addSeparator()
        training_menu.addAction("Test Model", self.test_model)
        training_menu.addAction("Export Model", self.export_model)

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def new_project(self):
        """Create new project"""
        dialog = NewProjectDialog(self)
        if dialog.exec():
            info = dialog.get_project_info()
            self.project_path = info['path']

            # Create project structure
            Settings.create_project_structure(self.project_path)

            # Initialize managers
            self.dataset_manager = DatasetManager(self.project_path)
            self.label_manager = LabelManager(self.project_path)
            self.model_trainer = ModelTrainer(self.project_path)

            self.status_bar.showMessage(f"Project created: {info['name']}")

    def open_project(self):
        """Open existing project"""
        path = QFileDialog.getExistingDirectory(self, "Open Project")
        if path:
            self.project_path = Path(path)
            self.dataset_manager = DatasetManager(self.project_path)
            self.label_manager = LabelManager(self.project_path)
            self.model_trainer = ModelTrainer(self.project_path)

            # Load project data
            self.load_project_data()

    def save_project(self):
        """Save current project"""
        if self.current_image and self.label_manager:
            label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
            annotations = self.image_viewer.annotations
            self.label_manager.save_annotations(label_path, annotations)
            self.status_bar.showMessage("Saved")

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

    def load_image(self, image_path: str):
        """Load image in viewer"""
        self.current_image = Path(image_path)
        self.image_viewer.load_image(self.current_image)

        # Load annotations
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

                # Delete image file
                if image_path.exists():
                    os.remove(image_path)

                # Delete corresponding label file
                label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
                if label_path.exists():
                    os.remove(label_path)

                deleted_count += 1

                # Clear viewer if current image was deleted
                if self.current_image and self.current_image == image_path:
                    self.image_viewer.clear_image()
                    self.current_image = None

            except Exception as e:
                print(f"Error deleting {image_path}: {e}")
                error_count += 1

        # Refresh dataset
        self.refresh_dataset()

        # Show result message
        if error_count == 0:
            self.status_bar.showMessage(f"Deleted {deleted_count} image(s)")
        else:
            QMessageBox.warning(
                self,
                "Delete Complete",
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

        # Reload annotations
        annotations = self.label_manager.load_annotations(label_path)
        self.image_viewer.set_annotations(annotations)
        self.label_widget.update_annotations(annotations)

    def on_polygon_added(self, points, class_id):
        """Handle new polygon"""
        if not self.label_manager or not self.current_image:
            return

        img_width, img_height = self.image_viewer.pixmap.width(), self.image_viewer.pixmap.height()
        polygon = Polygon.from_absolute(class_id, points, img_width, img_height)

        label_path = self.current_image.parent.parent / 'labels' / f"{self.current_image.stem}.txt"
        self.label_manager.add_annotation(label_path, polygon)

        # Reload annotations
        annotations = self.label_manager.load_annotations(label_path)
        self.image_viewer.set_annotations(annotations)
        self.label_widget.update_annotations(annotations)

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

        # Auto-save classes to config.yaml
        self.save_classes_to_config()

    def on_class_deleted(self, index):
        """Delete class"""
        if 0 <= index < len(self.classes):
            del self.classes[index]
            self.class_manager.set_classes(self.classes)

            # Auto-save classes to config.yaml
            self.save_classes_to_config()

    def save_classes_to_config(self):
        """Save classes to project config.yaml"""
        if not self.dataset_manager or not self.classes:
            return

        try:
            # Create/update data.yaml with current classes
            self.dataset_manager.create_data_yaml(self.classes)
            print(f"Saved {len(self.classes)} classes to config.yaml")
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

        # Check if there are images to split
        self.dataset_manager.refresh_images_list()
        if not self.dataset_manager.images_list:
            QMessageBox.warning(
                self,
                "No Images",
                "No images found in the project.\n\n"
                "Please import images first using File → Import Images"
            )
            return

        # Show split configuration dialog
        dialog = SplitDatasetDialog(self)
        if dialog.exec() != SplitDatasetDialog.DialogCode.Accepted:
            return

        # Get selected ratios
        ratios = dialog.get_ratios()

        # Perform split
        try:
            stats = self.dataset_manager.split_dataset(ratios)

            # Show success message
            total_images = sum(stats.values())
            QMessageBox.information(
                self, "Split Complete",
                f"Successfully split {total_images} images:\n\n"
                f"Train: {stats.get('train', 0)} ({int(ratios['train']*100)}%)\n"
                f"Val: {stats.get('val', 0)} ({int(ratios['val']*100)}%)\n"
                f"Test: {stats.get('test', 0)} ({int(ratios['test']*100)}%)\n\n"
                f"Images have been copied to train/val/test folders."
            )

            # Refresh dataset view
            self.refresh_dataset()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Split Failed",
                f"Failed to split dataset:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def show_statistics(self):
        """Show dataset statistics"""
        if not self.dataset_manager:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first to view statistics."
            )
            return

        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first to view statistics."
            )
            return

        # Get statistics with class names
        stats = self.dataset_manager.get_dataset_statistics(self.classes)

        # Show comprehensive statistics dialog
        dialog = DatasetStatisticsDialog(stats, self.project_path, self)
        dialog.exec()

    def on_start_training(self, config=None):
        """Start training"""
        if not self.model_trainer or not self.dataset_manager:
            QMessageBox.warning(self, "Warning", "Please create or open a project first")
            return

        # Check if classes are defined
        if not self.classes:
            QMessageBox.warning(
                self,
                "No Classes",
                "Please add classes before training.\n\n"
                "Use the Classes panel to add object classes."
            )
            return

        # Create data.yaml
        try:
            data_yaml = self.dataset_manager.create_data_yaml(self.classes)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create data.yaml:\n{str(e)}")
            return

        # Register callbacks for training updates
        self.model_trainer.register_callback('on_train_start', self._on_train_start_callback)
        self.model_trainer.register_callback('on_epoch_end', self._on_epoch_end_callback)
        self.model_trainer.register_callback('on_train_end', self._on_train_end_callback)
        self.model_trainer.register_callback('on_train_error', self._on_train_error_callback)

        # Start training
        try:
            # Extract model from config if provided, otherwise use default
            model_name = (config or {}).get('model', 'yolov8s.pt')

            # Remove 'model' from config as it's passed separately
            train_config = {k: v for k, v in (config or {}).items() if k != 'model'}

            self.model_trainer.start_training(
                train_config,
                data_yaml,
                model_name=model_name
            )
            self.status_bar.showMessage(f"Training started with {model_name}...")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Failed to start training:\n{str(e)}")

    def _on_train_start_callback(self, session):
        """Callback when training starts (runs in background thread)"""
        self.training_started.emit()

    def _on_epoch_end_callback(self, session, metrics):
        """Callback when epoch ends (runs in background thread)"""
        # Get current metrics from session
        current_metrics = self.model_trainer.get_current_metrics()
        self.training_epoch_update.emit(current_metrics)

    def _on_train_end_callback(self, session, results):
        """Callback when training ends (runs in background thread)"""
        self.training_finished.emit()

    def _on_train_error_callback(self, error_msg):
        """Callback when training error occurs (runs in background thread)"""
        self.training_finished.emit()

    def on_training_started(self):
        """Handle training started (runs in main thread)"""
        self.training_widget.set_status("Training...")
        self.status_bar.showMessage("Training in progress...")

    def on_training_finished(self):
        """Handle training finished (runs in main thread)"""
        self.training_widget.training_finished()

        if self.model_trainer and self.model_trainer.current_session:
            status = self.model_trainer.current_session.status

            if status == 'completed':
                self.status_bar.showMessage("Training completed!")
                QMessageBox.information(
                    self,
                    "Training Complete",
                    "Model training completed successfully!\n\n"
                    f"Best weights saved to:\n{self.model_trainer.get_best_weights_path()}"
                )
            elif status == 'failed':
                self.status_bar.showMessage("Training failed")
                QMessageBox.warning(
                    self,
                    "Training Failed",
                    "Training failed. Check the console for error details."
                )
            else:
                self.status_bar.showMessage(f"Training {status}")

    def update_training_metrics(self, metrics):
        """Update training metrics display (runs in main thread)"""
        self.metrics_widget.update_metrics(metrics)

        # Update status bar with current epoch
        if 'epoch' in metrics and 'total_epochs' in metrics:
            self.status_bar.showMessage(
                f"Training - Epoch {metrics['epoch']}/{metrics['total_epochs']}"
            )

    def on_stop_training(self):
        """Stop training"""
        if self.model_trainer:
            self.model_trainer.stop_training()

    def view_training_results(self):
        """View training results"""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first.\n\n"
                "Use File → New Project or File → Open Project"
            )
            return

        # Look for training results
        runs_dir = self.project_path / 'runs' / 'train'

        if not runs_dir.exists():
            QMessageBox.warning(
                self,
                "No Training Results",
                "No training results found.\n\n"
                "Please complete model training first using Training → Start Training"
            )
            return

        # Find all training runs
        training_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                              key=lambda x: x.stat().st_mtime,
                              reverse=True)

        if not training_runs:
            QMessageBox.warning(
                self,
                "No Training Results",
                "No training results found in runs/train folder."
            )
            return

        # Use the most recent training run
        latest_run = training_runs[0]

        # Show results dialog
        try:
            dialog = TrainingResultsDialog(latest_run, self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load training results:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def test_model(self):
        """Test trained model on images/videos"""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first to test models."
            )
            return

        # Open model testing dialog
        dialog = ModelTestingDialog(self.project_path, self)
        dialog.exec()

    def export_model(self):
        """Export trained model"""
        # Check if project is open
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first.\n\n"
                "Use File → New Project or File → Open Project"
            )
            return

        # Check if model trainer exists
        if not self.model_trainer:
            QMessageBox.warning(
                self,
                "No Training Session",
                "No training session found.\n\n"
                "Please train a model first using Training → Start Training"
            )
            return

        # Check if trained model exists
        best_weights = self.model_trainer.get_best_weights_path()
        if not best_weights or not best_weights.exists():
            QMessageBox.warning(
                self,
                "No Trained Model",
                "No trained model found.\n\n"
                "Please complete model training first.\n"
                "The best model weights will be available after training completes."
            )
            return

        # Show export dialog
        dialog = ExportDialog(self)
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            formats = dialog.get_selected_formats()

            if not formats:
                QMessageBox.warning(
                    self,
                    "No Format Selected",
                    "Please select at least one export format."
                )
                return

            try:
                import shutil
                import time

                # Show progress message
                self.status_bar.showMessage("Exporting model...")

                results = {}

                # Handle .pt format separately (just copy)
                if 'pt' in formats:
                    try:
                        # Copy to models directory
                        models_dir = self.project_path / 'models'
                        models_dir.mkdir(parents=True, exist_ok=True)

                        dest_path = models_dir / f"best_{int(time.time())}.pt"
                        shutil.copy2(best_weights, dest_path)

                        results['pt'] = {
                            'success': True,
                            'path': str(dest_path)
                        }
                    except Exception as e:
                        results['pt'] = {
                            'success': False,
                            'error': str(e)
                        }

                # Handle other formats using ExportManager
                other_formats = [f for f in formats if f != 'pt']
                if other_formats:
                    export_mgr = ExportManager(best_weights)
                    export_results = export_mgr.export_multiple(other_formats)
                    results.update(export_results)

                # Show success message
                success_count = sum(1 for r in results.values() if r.get('success'))
                failed_count = len(results) - success_count

                if failed_count == 0:
                    msg = f"Successfully exported model to {success_count} format(s):\n\n"
                    for fmt, result in results.items():
                        if result.get('success'):
                            path = result.get('path', 'N/A')
                            msg += f"✓ {fmt.upper()}: {path}\n"

                    QMessageBox.information(self, "Export Complete", msg)
                    self.status_bar.showMessage("Model export completed")
                else:
                    msg = f"Export completed with {failed_count} error(s):\n\n"
                    for fmt, result in results.items():
                        if result.get('success'):
                            msg += f"✓ {fmt.upper()}: {result.get('path', 'N/A')}\n"
                        else:
                            error = result.get('error', 'Unknown error')
                            msg += f"✗ {fmt.upper()}: {error}\n"

                    QMessageBox.warning(self, "Export Completed with Errors", msg)
                    self.status_bar.showMessage("Model export completed with errors")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export model:\n{str(e)}"
                )
                self.status_bar.showMessage("Model export failed")
                import traceback
                traceback.print_exc()

    def refresh_dataset(self):
        """Refresh dataset view"""
        if self.dataset_manager:
            self.dataset_manager.refresh_images_list()
            self.dataset_widget.set_images(self.dataset_manager.images_list)

    def load_project_data(self):
        """Load project data"""
        # Load classes from config.yaml if it exists
        self.load_classes_from_config()

        # Refresh dataset
        self.refresh_dataset()
        self.status_bar.showMessage(f"Loaded project: {self.project_path.name}")

    def load_classes_from_config(self):
        """Load classes from project config.yaml"""
        import yaml

        if not self.project_path:
            return

        config_path = self.project_path / 'config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if config and 'names' in config:
                    # Load classes from config
                    self.classes = config['names']

                    # Update UI
                    self.class_manager.set_classes(self.classes)
                    if self.label_manager:
                        self.label_manager.set_classes(self.classes)
                        self.image_viewer.set_classes(self.classes, self.label_manager.class_colors)

                    print(f"Loaded {len(self.classes)} classes from config.yaml")
            except Exception as e:
                print(f"Error loading classes from config: {e}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            f"{Settings.APP_NAME} v{Settings.APP_VERSION}\n"
            "YOLOv8 Training Tool for object detection"
        )
