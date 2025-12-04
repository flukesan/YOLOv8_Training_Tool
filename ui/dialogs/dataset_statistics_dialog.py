"""
Dataset Statistics Dialog - displays comprehensive dataset information
"""
from pathlib import Path
from typing import Dict
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QWidget, QGroupBox, QGridLayout,
                             QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class DatasetStatisticsDialog(QDialog):
    """Dialog for displaying dataset statistics"""

    def __init__(self, stats: Dict, project_path: Path, parent=None):
        super().__init__(parent)
        self.stats = stats
        self.project_path = project_path
        self.setWindowTitle("Dataset Statistics")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("📊 Dataset Statistics")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Project path
        path_label = QLabel(f"Project: {self.project_path.name}")
        path_label.setStyleSheet("color: gray; font-size: 11px;")
        path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(path_label)

        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_overview_tab(), "📈 Overview")
        tabs.addTab(self.create_distribution_tab(), "📊 Distribution")
        tabs.addTab(self.create_details_tab(), "📋 Details")

        layout.addWidget(tabs)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumHeight(35)
        layout.addWidget(btn_close)

        self.setLayout(layout)

        # Apply styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

    def create_overview_tab(self) -> QWidget:
        """Create overview tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Image statistics
        img_group = QGroupBox("Image Statistics")
        img_layout = QGridLayout()

        # Total images
        total_card = self.create_stat_card(
            "Total Images",
            str(self.stats.get('total_images', 0)),
            "#2196F3"
        )
        img_layout.addWidget(total_card, 0, 0)

        # Train images
        train_card = self.create_stat_card(
            "Training Set",
            str(self.stats.get('train_images', 0)),
            "#4CAF50"
        )
        img_layout.addWidget(train_card, 0, 1)

        # Val images
        val_card = self.create_stat_card(
            "Validation Set",
            str(self.stats.get('val_images', 0)),
            "#FF9800"
        )
        img_layout.addWidget(val_card, 1, 0)

        # Test images
        test_card = self.create_stat_card(
            "Test Set",
            str(self.stats.get('test_images', 0)),
            "#9C27B0"
        )
        img_layout.addWidget(test_card, 1, 1)

        img_group.setLayout(img_layout)
        layout.addWidget(img_group)

        # Annotation statistics
        ann_group = QGroupBox("Annotation Statistics")
        ann_layout = QGridLayout()

        # Total annotations
        total_ann_card = self.create_stat_card(
            "Total Annotations",
            str(self.stats.get('total_annotations', 0)),
            "#E91E63"
        )
        ann_layout.addWidget(total_ann_card, 0, 0)

        # Classes count
        classes = self.stats.get('classes', {})
        classes_card = self.create_stat_card(
            "Number of Classes",
            str(len(classes)),
            "#00BCD4"
        )
        ann_layout.addWidget(classes_card, 0, 1)

        # Average annotations per image
        total_imgs = self.stats.get('total_images', 0)
        total_anns = self.stats.get('total_annotations', 0)
        avg_anns = f"{total_anns / total_imgs:.2f}" if total_imgs > 0 else "0"
        avg_card = self.create_stat_card(
            "Avg Annotations/Image",
            avg_anns,
            "#795548"
        )
        ann_layout.addWidget(avg_card, 1, 0, 1, 2)

        ann_group.setLayout(ann_layout)
        layout.addWidget(ann_group)

        # Split ratios
        if total_imgs > 0:
            ratio_group = QGroupBox("Dataset Split Ratios")
            ratio_layout = QVBoxLayout()

            train_pct = (self.stats.get('train_images', 0) / total_imgs) * 100
            val_pct = (self.stats.get('val_images', 0) / total_imgs) * 100
            test_pct = (self.stats.get('test_images', 0) / total_imgs) * 100

            ratio_text = QLabel(
                f"<div style='font-size: 13px; line-height: 1.8;'>"
                f"<b>Train:</b> {train_pct:.1f}% "
                f"({self.stats.get('train_images', 0)} images)<br>"
                f"<b>Validation:</b> {val_pct:.1f}% "
                f"({self.stats.get('val_images', 0)} images)<br>"
                f"<b>Test:</b> {test_pct:.1f}% "
                f"({self.stats.get('test_images', 0)} images)"
                f"</div>"
            )
            ratio_text.setTextFormat(Qt.TextFormat.RichText)
            ratio_layout.addWidget(ratio_text)

            # Progress bars (text-based)
            bars_widget = QWidget()
            bars_layout = QVBoxLayout()
            bars_layout.setSpacing(10)

            bars_layout.addWidget(self.create_progress_bar("Train", train_pct, "#4CAF50"))
            bars_layout.addWidget(self.create_progress_bar("Val", val_pct, "#FF9800"))
            bars_layout.addWidget(self.create_progress_bar("Test", test_pct, "#9C27B0"))

            bars_widget.setLayout(bars_layout)
            ratio_layout.addWidget(bars_widget)

            ratio_group.setLayout(ratio_layout)
            layout.addWidget(ratio_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_distribution_tab(self) -> QWidget:
        """Create class distribution tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        classes = self.stats.get('classes', {})

        if not classes:
            no_data = QLabel("No class data available.\nPlease annotate images first.")
            no_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data.setStyleSheet("color: gray; font-size: 14px; padding: 50px;")
            layout.addWidget(no_data)
        else:
            # Class distribution group
            class_group = QGroupBox(f"Class Distribution ({len(classes)} classes)")
            class_layout = QVBoxLayout()

            # Create table
            table = QTableWidget()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Class ID", "Class Name", "Count"])
            table.setRowCount(len(classes))

            # Sort by count (descending)
            sorted_classes = sorted(classes.items(), key=lambda x: x[1]['count'], reverse=True)

            total_count = sum(c['count'] for c in classes.values())

            for i, (class_id, class_info) in enumerate(sorted_classes):
                # Class ID
                id_item = QTableWidgetItem(str(class_id))
                id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, 0, id_item)

                # Class name
                name_item = QTableWidgetItem(class_info.get('name', f'Class {class_id}'))
                table.setItem(i, 1, name_item)

                # Count with percentage
                count = class_info['count']
                percentage = (count / total_count * 100) if total_count > 0 else 0
                count_item = QTableWidgetItem(f"{count} ({percentage:.1f}%)")
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, 2, count_item)

            # Resize columns
            table.resizeColumnsToContents()
            table.horizontalHeader().setStretchLastSection(True)

            class_layout.addWidget(table)

            # Class balance warning
            if len(classes) >= 2:
                counts = [c['count'] for c in classes.values()]
                max_count = max(counts)
                min_count = min(counts)
                ratio = max_count / min_count if min_count > 0 else float('inf')

                if ratio > 3:
                    warning = QLabel(
                        "⚠️ <b>Class Imbalance Detected:</b><br>"
                        f"Ratio between most and least common class: {ratio:.1f}x<br>"
                        "<i>Consider re-balancing your dataset for better training results.</i>"
                    )
                    warning.setTextFormat(Qt.TextFormat.RichText)
                    warning.setStyleSheet(
                        "background-color: #FFF3CD; "
                        "border: 2px solid #FFB900; "
                        "border-radius: 5px; "
                        "padding: 10px; "
                        "color: #856404;"
                    )
                    warning.setWordWrap(True)
                    class_layout.addWidget(warning)

            class_group.setLayout(class_layout)
            layout.addWidget(class_group)

        widget.setLayout(layout)
        return widget

    def create_details_tab(self) -> QWidget:
        """Create details tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        details_group = QGroupBox("Detailed Information")
        details_layout = QVBoxLayout()

        # Create detailed text
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setMinimumHeight(400)

        text = self.format_detailed_stats()
        details_text.setPlainText(text)

        details_layout.addWidget(details_text)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        widget.setLayout(layout)
        return widget

    def create_stat_card(self, title: str, value: str, color: str) -> QWidget:
        """Create a statistics card"""
        card = QGroupBox()
        card.setMinimumHeight(100)
        card.setStyleSheet(f"""
            QGroupBox {{
                background-color: {color}15;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 15px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Value
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(value_label, 1)

        card.setLayout(layout)
        return card

    def create_progress_bar(self, label: str, percentage: float, color: str) -> QWidget:
        """Create a text-based progress bar"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label with percentage
        label_widget = QLabel(f"{label}: {percentage:.1f}%")
        label_widget.setFont(QFont("Arial", 10))
        layout.addWidget(label_widget)

        # Progress bar (using styled widget)
        bar = QWidget()
        bar.setMinimumHeight(25)
        bar.setStyleSheet(f"""
            QWidget {{
                background-color: #f0f0f0;
                border-radius: 5px;
            }}
        """)

        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(0)

        # Filled portion
        filled = QWidget()
        filled.setMaximumWidth(int(700 * percentage / 100))
        filled.setStyleSheet(f"""
            QWidget {{
                background-color: {color};
                border-radius: 5px;
            }}
        """)
        bar_layout.addWidget(filled)
        bar_layout.addStretch()

        bar.setLayout(bar_layout)
        layout.addWidget(bar)

        widget.setLayout(layout)
        return widget

    def format_detailed_stats(self) -> str:
        """Format detailed statistics as text"""
        lines = []
        lines.append("=" * 60)
        lines.append("DATASET STATISTICS - DETAILED REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Project: {self.project_path.name}")
        lines.append(f"Location: {self.project_path}")
        lines.append("")

        lines.append("IMAGE STATISTICS:")
        lines.append("-" * 60)
        lines.append(f"  Total Images:      {self.stats.get('total_images', 0)}")
        lines.append(f"  Training Images:   {self.stats.get('train_images', 0)}")
        lines.append(f"  Validation Images: {self.stats.get('val_images', 0)}")
        lines.append(f"  Test Images:       {self.stats.get('test_images', 0)}")
        lines.append("")

        total = self.stats.get('total_images', 0)
        if total > 0:
            train_pct = (self.stats.get('train_images', 0) / total) * 100
            val_pct = (self.stats.get('val_images', 0) / total) * 100
            test_pct = (self.stats.get('test_images', 0) / total) * 100
            lines.append("SPLIT RATIOS:")
            lines.append("-" * 60)
            lines.append(f"  Training:   {train_pct:.1f}%")
            lines.append(f"  Validation: {val_pct:.1f}%")
            lines.append(f"  Test:       {test_pct:.1f}%")
            lines.append("")

        lines.append("ANNOTATION STATISTICS:")
        lines.append("-" * 60)
        lines.append(f"  Total Annotations: {self.stats.get('total_annotations', 0)}")

        total_imgs = self.stats.get('total_images', 0)
        total_anns = self.stats.get('total_annotations', 0)
        if total_imgs > 0:
            avg = total_anns / total_imgs
            lines.append(f"  Avg per Image:     {avg:.2f}")
        lines.append("")

        classes = self.stats.get('classes', {})
        if classes:
            lines.append("CLASS DISTRIBUTION:")
            lines.append("-" * 60)

            sorted_classes = sorted(classes.items(), key=lambda x: x[1]['count'], reverse=True)
            total_count = sum(c['count'] for c in classes.values())

            for class_id, class_info in sorted_classes:
                name = class_info.get('name', f'Class {class_id}')
                count = class_info['count']
                pct = (count / total_count * 100) if total_count > 0 else 0
                lines.append(f"  [{class_id}] {name:20s} : {count:5d} ({pct:5.1f}%)")

            lines.append("")
            lines.append("CLASS BALANCE ANALYSIS:")
            lines.append("-" * 60)

            counts = [c['count'] for c in classes.values()]
            max_count = max(counts)
            min_count = min(counts)
            ratio = max_count / min_count if min_count > 0 else float('inf')

            lines.append(f"  Most common class:  {max_count} annotations")
            lines.append(f"  Least common class: {min_count} annotations")
            lines.append(f"  Imbalance ratio:    {ratio:.2f}x")
            lines.append("")

            if ratio > 3:
                lines.append("  ⚠️  WARNING: Significant class imbalance detected!")
                lines.append("     Consider re-balancing your dataset.")
            elif ratio > 1.5:
                lines.append("  ⚡ NOTICE: Moderate class imbalance present.")
            else:
                lines.append("  ✓  Classes are well balanced.")

        lines.append("")
        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
