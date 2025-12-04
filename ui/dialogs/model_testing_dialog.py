"""
Model Testing Dialog - test trained models on images and videos
"""
from pathlib import Path
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QTextEdit, QTabWidget, QWidget,
                             QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                             QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QImage
import cv2
import numpy as np


class InferenceWorker(QThread):
    """Worker thread for running inference"""
    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_evaluator, file_paths: List[Path], conf_threshold: float, is_video: bool = False):
        super().__init__()
        self.model_evaluator = model_evaluator
        self.file_paths = file_paths
        self.conf_threshold = conf_threshold
        self.is_video = is_video

    def run(self):
        """Run inference"""
        try:
            if self.is_video:
                # Video prediction
                for i, video_path in enumerate(self.file_paths):
                    self.progress.emit(i + 1, len(self.file_paths))
                    result = self.model_evaluator.predict_video(
                        video_path,
                        conf_threshold=self.conf_threshold
                    )
                    result['file_path'] = str(video_path)
                    result['type'] = 'video'
                    self.result_ready.emit(result)
            else:
                # Image prediction
                for i, image_path in enumerate(self.file_paths):
                    self.progress.emit(i + 1, len(self.file_paths))
                    result = self.model_evaluator.predict_image(
                        image_path,
                        conf_threshold=self.conf_threshold
                    )
                    result['type'] = 'image'
                    self.result_ready.emit(result)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class ModelTestingDialog(QDialog):
    """Dialog for testing trained models"""

    def __init__(self, project_path: Path, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.model_evaluator = None
        self.current_model_path = None
        self.test_images = []
        self.test_videos = []
        self.results = []

        self.setWindowTitle("🔍 Model Testing")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("🔍 Test Trained Model")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Model selection
        model_group = self.create_model_selection()
        layout.addWidget(model_group)

        # Tabs for different testing modes
        tabs = QTabWidget()
        tabs.addTab(self.create_image_testing_tab(), "📷 Test Images")
        tabs.addTab(self.create_video_testing_tab(), "🎥 Test Videos")
        tabs.addTab(self.create_batch_testing_tab(), "📊 Batch Testing")
        layout.addWidget(tabs)

        # Results area
        results_group = self.create_results_area()
        layout.addWidget(results_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_clear = QPushButton("Clear Results")
        btn_clear.clicked.connect(self.clear_results)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Load available models
        self.load_available_models()

    def create_model_selection(self) -> QGroupBox:
        """Create model selection group"""
        group = QGroupBox("Model Selection")
        layout = QFormLayout()

        # Model dropdown
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        layout.addRow("Trained Model:", self.model_combo)

        # Model info
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addRow("", self.model_info_label)

        # Confidence threshold
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip("Confidence threshold for detections")
        layout.addRow("Confidence:", self.conf_spin)

        # IoU threshold
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setToolTip("IoU threshold for NMS")
        layout.addRow("IoU (NMS):", self.iou_spin)

        group.setLayout(layout)
        return group

    def create_image_testing_tab(self) -> QWidget:
        """Create image testing tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Instructions
        info = QLabel(
            "📷 <b>Test Single or Multiple Images</b><br>"
            "Upload images to test your trained model's detection accuracy."
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("background-color: #E3F2FD; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)

        # Upload button
        btn_upload = QPushButton("📁 Upload Images")
        btn_upload.setMinimumHeight(40)
        btn_upload.clicked.connect(self.upload_images)
        layout.addWidget(btn_upload)

        # Image list
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        layout.addWidget(QLabel("Selected Images:"))
        layout.addWidget(self.image_list)

        # Test button
        self.btn_test_images = QPushButton("🚀 Run Detection")
        self.btn_test_images.setMinimumHeight(40)
        self.btn_test_images.setEnabled(False)
        self.btn_test_images.clicked.connect(self.test_images_clicked)
        self.btn_test_images.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.btn_test_images)

        # Progress bar
        self.progress_bar_images = QProgressBar()
        self.progress_bar_images.setVisible(False)
        layout.addWidget(self.progress_bar_images)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_video_testing_tab(self) -> QWidget:
        """Create video testing tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Instructions
        info = QLabel(
            "🎥 <b>Test Video Files</b><br>"
            "Upload videos to test real-time detection performance."
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("background-color: #FFF3E0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)

        # Upload button
        btn_upload = QPushButton("📁 Upload Videos")
        btn_upload.setMinimumHeight(40)
        btn_upload.clicked.connect(self.upload_videos)
        layout.addWidget(btn_upload)

        # Video list
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(150)
        layout.addWidget(QLabel("Selected Videos:"))
        layout.addWidget(self.video_list)

        # Test button
        self.btn_test_videos = QPushButton("🚀 Run Detection on Video")
        self.btn_test_videos.setMinimumHeight(40)
        self.btn_test_videos.setEnabled(False)
        self.btn_test_videos.clicked.connect(self.test_videos_clicked)
        self.btn_test_videos.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.btn_test_videos)

        # Progress bar
        self.progress_bar_videos = QProgressBar()
        self.progress_bar_videos.setVisible(False)
        layout.addWidget(self.progress_bar_videos)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_batch_testing_tab(self) -> QWidget:
        """Create batch testing tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Instructions
        info = QLabel(
            "📊 <b>Batch Testing</b><br>"
            "Test multiple images and generate statistics."
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("background-color: #F3E5F5; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)

        # Batch size
        batch_layout = QFormLayout()
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(16)
        batch_layout.addRow("Batch Size:", self.batch_size_spin)

        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        layout.addWidget(batch_widget)

        # Upload folder button
        btn_upload_folder = QPushButton("📁 Select Image Folder")
        btn_upload_folder.setMinimumHeight(40)
        btn_upload_folder.clicked.connect(self.upload_folder)
        layout.addWidget(btn_upload_folder)

        # Folder info
        self.folder_info_label = QLabel("No folder selected")
        self.folder_info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.folder_info_label)

        # Test button
        self.btn_test_batch = QPushButton("🚀 Run Batch Detection")
        self.btn_test_batch.setMinimumHeight(40)
        self.btn_test_batch.setEnabled(False)
        self.btn_test_batch.clicked.connect(self.test_batch_clicked)
        self.btn_test_batch.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8E24AA;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.btn_test_batch)

        # Progress bar
        self.progress_bar_batch = QProgressBar()
        self.progress_bar_batch.setVisible(False)
        layout.addWidget(self.progress_bar_batch)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_results_area(self) -> QGroupBox:
        """Create results display area"""
        group = QGroupBox("Detection Results")
        layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText(
            "Detection results will appear here...\n\n"
            "Results will show:\n"
            "• Number of detections per image/frame\n"
            "• Detected classes and confidence scores\n"
            "• Overall statistics"
        )
        layout.addWidget(self.results_text)

        group.setLayout(layout)
        return group

    def load_available_models(self):
        """Load available trained models"""
        self.model_combo.clear()

        # Check for trained models in runs/train
        runs_dir = self.project_path / 'runs' / 'train'
        if not runs_dir.exists():
            self.model_combo.addItem("No trained models found", None)
            return

        # Find all best.pt files
        models = []
        for run_dir in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if run_dir.is_dir():
                best_weights = run_dir / 'weights' / 'best.pt'
                if best_weights.exists():
                    models.append((run_dir.name, best_weights))

        if not models:
            self.model_combo.addItem("No trained models found", None)
            return

        # Add models to dropdown
        for model_name, model_path in models:
            self.model_combo.addItem(f"{model_name} (best.pt)", model_path)

    def on_model_selected(self, index):
        """Handle model selection"""
        model_path = self.model_combo.currentData()

        if model_path is None:
            self.model_info_label.setText("No model available")
            return

        try:
            from core.model_evaluator import ModelEvaluator

            self.current_model_path = model_path
            self.model_evaluator = ModelEvaluator(model_path)

            # Show model info
            size_mb = model_path.stat().st_size / (1024 * 1024)
            self.model_info_label.setText(f"✓ Model loaded ({size_mb:.1f} MB)")
            self.model_info_label.setStyleSheet("color: green; font-size: 11px;")

        except Exception as e:
            self.model_info_label.setText(f"✗ Error loading model: {str(e)}")
            self.model_info_label.setStyleSheet("color: red; font-size: 11px;")

    def upload_images(self):
        """Upload images for testing"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*.*)"
        )

        if files:
            self.test_images = [Path(f) for f in files]
            self.image_list.clear()
            for img in self.test_images:
                self.image_list.addItem(img.name)

            self.btn_test_images.setEnabled(True and self.model_evaluator is not None)

    def upload_videos(self):
        """Upload videos for testing"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )

        if files:
            self.test_videos = [Path(f) for f in files]
            self.video_list.clear()
            for vid in self.test_videos:
                self.video_list.addItem(vid.name)

            self.btn_test_videos.setEnabled(True and self.model_evaluator is not None)

    def upload_folder(self):
        """Upload folder for batch testing"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            str(Path.home())
        )

        if folder:
            folder_path = Path(folder)
            # Find all images in folder
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            self.test_images = [
                f for f in folder_path.iterdir()
                if f.suffix.lower() in image_exts
            ]

            self.folder_info_label.setText(f"✓ {len(self.test_images)} images found in {folder_path.name}")
            self.folder_info_label.setStyleSheet("color: green;")

            self.btn_test_batch.setEnabled(True and self.model_evaluator is not None and len(self.test_images) > 0)

    def test_images_clicked(self):
        """Test images"""
        if not self.model_evaluator:
            QMessageBox.warning(self, "No Model", "Please select a trained model first.")
            return

        if not self.test_images:
            QMessageBox.warning(self, "No Images", "Please upload images first.")
            return

        self.run_inference(self.test_images, is_video=False)

    def test_videos_clicked(self):
        """Test videos"""
        if not self.model_evaluator:
            QMessageBox.warning(self, "No Model", "Please select a trained model first.")
            return

        if not self.test_videos:
            QMessageBox.warning(self, "No Videos", "Please upload videos first.")
            return

        self.run_inference(self.test_videos, is_video=True)

    def test_batch_clicked(self):
        """Test batch"""
        if not self.model_evaluator:
            QMessageBox.warning(self, "No Model", "Please select a trained model first.")
            return

        if not self.test_images:
            QMessageBox.warning(self, "No Images", "Please select an image folder first.")
            return

        self.run_inference(self.test_images, is_video=False)

    def run_inference(self, file_paths: List[Path], is_video: bool = False):
        """Run inference on files"""
        # Clear previous results
        self.results_text.clear()
        self.results = []

        # Show progress bar
        if is_video:
            self.progress_bar_videos.setVisible(True)
            self.progress_bar_videos.setMaximum(len(file_paths))
            self.progress_bar_videos.setValue(0)
            self.btn_test_videos.setEnabled(False)
        else:
            self.progress_bar_images.setVisible(True)
            self.progress_bar_images.setMaximum(len(file_paths))
            self.progress_bar_images.setValue(0)
            self.btn_test_images.setEnabled(False)
            self.btn_test_batch.setEnabled(False)

        # Create worker thread
        self.worker = InferenceWorker(
            self.model_evaluator,
            file_paths,
            self.conf_spin.value(),
            is_video
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.finished.connect(lambda: self.on_inference_finished(is_video))
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_progress(self, current: int, total: int):
        """Update progress bar"""
        if self.progress_bar_images.isVisible():
            self.progress_bar_images.setValue(current)
        if self.progress_bar_videos.isVisible():
            self.progress_bar_videos.setValue(current)
        if self.progress_bar_batch.isVisible():
            self.progress_bar_batch.setValue(current)

    def on_result_ready(self, result: dict):
        """Handle inference result"""
        self.results.append(result)

        # Display result
        if result['type'] == 'image':
            file_name = Path(result['image_path']).name
            num_detections = len(result['boxes'])

            self.results_text.append(f"\n{'='*60}")
            self.results_text.append(f"📷 {file_name}")
            self.results_text.append(f"   Detections: {num_detections}")

            if num_detections > 0:
                for i, (box, score, cls) in enumerate(zip(result['boxes'], result['scores'], result['classes']), 1):
                    self.results_text.append(f"   [{i}] Class {int(cls)} - Confidence: {score:.2%}")

        elif result['type'] == 'video':
            file_name = Path(result['file_path']).name
            self.results_text.append(f"\n{'='*60}")
            self.results_text.append(f"🎥 {file_name}")
            self.results_text.append(f"   Total Frames: {result['total_frames']}")
            self.results_text.append(f"   Avg Detections/Frame: {result['avg_detections']:.2f}")
            self.results_text.append(f"   Classes Detected: {result['classes_detected']}")

    def on_inference_finished(self, is_video: bool):
        """Handle inference completion"""
        # Hide progress bars
        self.progress_bar_images.setVisible(False)
        self.progress_bar_videos.setVisible(False)
        self.progress_bar_batch.setVisible(False)

        # Re-enable buttons
        self.btn_test_images.setEnabled(True)
        self.btn_test_videos.setEnabled(True)
        self.btn_test_batch.setEnabled(True)

        # Show summary
        self.results_text.append(f"\n{'='*60}")
        self.results_text.append(f"✅ <b>Testing Complete!</b>")
        self.results_text.append(f"   Total Files Processed: {len(self.results)}")

        if not is_video:
            total_detections = sum(len(r['boxes']) for r in self.results)
            avg_detections = total_detections / len(self.results) if self.results else 0
            self.results_text.append(f"   Total Detections: {total_detections}")
            self.results_text.append(f"   Avg Detections/Image: {avg_detections:.2f}")

        QMessageBox.information(
            self,
            "Testing Complete",
            f"Successfully tested {len(self.results)} file(s)!"
        )

    def on_inference_error(self, error_msg: str):
        """Handle inference error"""
        self.progress_bar_images.setVisible(False)
        self.progress_bar_videos.setVisible(False)
        self.progress_bar_batch.setVisible(False)

        self.btn_test_images.setEnabled(True)
        self.btn_test_videos.setEnabled(True)
        self.btn_test_batch.setEnabled(True)

        QMessageBox.critical(self, "Inference Error", f"Error during inference:\n{error_msg}")

    def clear_results(self):
        """Clear results"""
        self.results_text.clear()
        self.results = []
