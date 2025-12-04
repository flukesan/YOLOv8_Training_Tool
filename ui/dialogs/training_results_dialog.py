"""
Training Results Dialog - Display training results and graphs
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTabWidget, QWidget, QScrollArea,
                             QGridLayout, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
from pathlib import Path
import csv


class TrainingResultsDialog(QDialog):
    """Dialog for displaying training results"""

    def __init__(self, results_dir: Path, parent=None):
        super().__init__(parent)
        self.results_dir = Path(results_dir)
        self.setWindowTitle("Training Results")
        self.setMinimumSize(1000, 700)

        self.init_ui()
        self.load_results()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Training Results")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()

        # Tab 1: Summary
        self.summary_tab = QWidget()
        self.init_summary_tab()
        self.tabs.addTab(self.summary_tab, "📊 Summary")

        # Tab 2: Graphs
        self.graphs_tab = QWidget()
        self.init_graphs_tab()
        self.tabs.addTab(self.graphs_tab, "📈 Graphs")

        # Tab 3: Detailed Metrics
        self.metrics_tab = QWidget()
        self.init_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "📋 Detailed Metrics")

        layout.addWidget(self.tabs)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_open_folder = QPushButton("Open Results Folder")
        self.btn_open_folder.clicked.connect(self.open_results_folder)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_open_folder)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def init_summary_tab(self):
        """Initialize summary tab"""
        layout = QVBoxLayout()

        # Metrics Grid
        self.metrics_grid = QGridLayout()

        # Create metric cards
        self.metric_labels = {}
        metrics_info = [
            ('mAP50', 'mAP@0.5', 0, 0),
            ('mAP50-95', 'mAP@0.5:0.95', 0, 1),
            ('Precision', 'Precision', 1, 0),
            ('Recall', 'Recall', 1, 1),
            ('Loss', 'Total Loss', 2, 0),
            ('Epochs', 'Epochs Trained', 2, 1),
        ]

        for key, title, row, col in metrics_info:
            card = self.create_metric_card(title)
            self.metric_labels[key] = card['value_label']
            self.metrics_grid.addWidget(card['widget'], row, col)

        layout.addLayout(self.metrics_grid)

        # Model Info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()

        self.model_info_label = QLabel("Loading...")
        self.model_info_label.setWordWrap(True)
        info_layout.addWidget(self.model_info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.summary_tab.setLayout(layout)

    def create_metric_card(self, title):
        """Create a metric display card"""
        widget = QGroupBox(title)
        widget.setStyleSheet("""
            QGroupBox {
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout()

        value_label = QLabel("--")
        value_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("color: #2196F3;")

        layout.addWidget(value_label)
        widget.setLayout(layout)

        return {'widget': widget, 'value_label': value_label}

    def init_graphs_tab(self):
        """Initialize graphs tab"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout()

        # Graph images
        self.graph_labels = {}

        graphs = [
            ('results', 'Training Results Overview'),
            ('confusion_matrix', 'Confusion Matrix'),
            ('F1_curve', 'F1-Confidence Curve'),
            ('P_curve', 'Precision-Confidence Curve'),
            ('R_curve', 'Recall-Confidence Curve'),
            ('PR_curve', 'Precision-Recall Curve'),
        ]

        for key, title in graphs:
            group = QGroupBox(title)
            group_layout = QVBoxLayout()

            label = QLabel("Graph not found")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(300)
            label.setStyleSheet("border: 1px solid #ccc; background-color: white;")

            self.graph_labels[key] = label
            group_layout.addWidget(label)
            group.setLayout(group_layout)

            layout.addWidget(group)

        content.setLayout(layout)
        scroll.setWidget(content)

        tab_layout = QVBoxLayout()
        tab_layout.addWidget(scroll)
        self.graphs_tab.setLayout(tab_layout)

    def init_metrics_tab(self):
        """Initialize metrics tab"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout()

        info = QLabel("Detailed metrics per epoch:")
        info.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(info)

        self.metrics_text = QLabel("Loading...")
        self.metrics_text.setWordWrap(True)
        self.metrics_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.metrics_text)

        content.setLayout(layout)
        scroll.setWidget(content)

        tab_layout = QVBoxLayout()
        tab_layout.addWidget(scroll)
        self.metrics_tab.setLayout(tab_layout)

    def load_results(self):
        """Load training results"""
        try:
            # Load final metrics from results.csv
            results_csv = self.results_dir / 'results.csv'
            if results_csv.exists():
                self.load_metrics_from_csv(results_csv)

            # Load graphs
            self.load_graphs()

            # Load model info
            self.load_model_info()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Loading Results",
                f"Failed to load training results:\n{str(e)}"
            )

    def load_metrics_from_csv(self, csv_path):
        """Load metrics from results.csv"""
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    return

                # Get last row (final metrics)
                last_row = rows[-1]

                # Update summary cards
                metrics_map = {
                    'mAP50': 'metrics/mAP50(B)',
                    'mAP50-95': 'metrics/mAP50-95(B)',
                    'Precision': 'metrics/precision(B)',
                    'Recall': 'metrics/recall(B)',
                    'Loss': 'train/box_loss',
                    'Epochs': 'epoch',
                }

                for key, csv_key in metrics_map.items():
                    if csv_key in last_row:
                        value = last_row[csv_key].strip()
                        try:
                            if key == 'Epochs':
                                display_value = f"{int(float(value)) + 1}"
                            else:
                                display_value = f"{float(value):.4f}"
                            self.metric_labels[key].setText(display_value)
                        except:
                            self.metric_labels[key].setText(value)

                # Load detailed metrics
                metrics_text = "Epoch\tmAP50\tmAP50-95\tPrecision\tRecall\tLoss\n"
                metrics_text += "─" * 70 + "\n"

                for i, row in enumerate(rows[-10:]):  # Last 10 epochs
                    epoch = int(float(row.get('epoch', 0))) + 1
                    map50 = float(row.get('metrics/mAP50(B)', 0))
                    map5095 = float(row.get('metrics/mAP50-95(B)', 0))
                    precision = float(row.get('metrics/precision(B)', 0))
                    recall = float(row.get('metrics/recall(B)', 0))
                    loss = float(row.get('train/box_loss', 0))

                    metrics_text += f"{epoch}\t{map50:.4f}\t{map5095:.4f}\t{precision:.4f}\t{recall:.4f}\t{loss:.4f}\n"

                self.metrics_text.setText(metrics_text)

        except Exception as e:
            print(f"Error loading metrics: {e}")

    def load_graphs(self):
        """Load graph images"""
        for key, label in self.graph_labels.items():
            img_path = self.results_dir / f'{key}.png'

            if img_path.exists():
                pixmap = QPixmap(str(img_path))
                if not pixmap.isNull():
                    # Scale to fit
                    scaled_pixmap = pixmap.scaled(
                        800, 600,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    label.setPixmap(scaled_pixmap)
                else:
                    label.setText(f"Failed to load {key}.png")
            else:
                label.setText(f"{key}.png not found")

    def load_model_info(self):
        """Load model information"""
        info_text = f"<b>Results Directory:</b><br>{self.results_dir}<br><br>"

        # Check for weights
        best_weights = self.results_dir / 'weights' / 'best.pt'
        last_weights = self.results_dir / 'weights' / 'last.pt'

        if best_weights.exists():
            size_mb = best_weights.stat().st_size / (1024 * 1024)
            info_text += f"<b>Best Weights:</b> {best_weights}<br>"
            info_text += f"<b>Size:</b> {size_mb:.2f} MB<br><br>"

        if last_weights.exists():
            size_mb = last_weights.stat().st_size / (1024 * 1024)
            info_text += f"<b>Last Weights:</b> {last_weights}<br>"
            info_text += f"<b>Size:</b> {size_mb:.2f} MB<br>"

        self.model_info_label.setText(info_text)

    def open_results_folder(self):
        """Open results folder in file explorer"""
        import subprocess
        import sys

        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', str(self.results_dir)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(self.results_dir)])
            else:
                subprocess.Popen(['xdg-open', str(self.results_dir)])
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open folder:\n{str(e)}"
            )
