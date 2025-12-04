"""
Image Viewer Widget - displays images with bounding boxes
"""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from pathlib import Path
from typing import List, Optional
import numpy as np


class ImageViewer(QWidget):
    """Widget for displaying and annotating images"""

    box_added = pyqtSignal(int, int, int, int, int)  # x1, y1, x2, y2, class_id
    box_selected = pyqtSignal(int)  # box index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap = None
        self.annotations = []
        self.current_class = 0
        self.class_colors = {}
        self.class_names = []

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.selected_box = -1

        # Zoom
        self.zoom_factor = 1.0

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)

        layout.addWidget(self.scroll)
        self.setLayout(layout)

    def load_image(self, image_path: Path):
        """Load and display image"""
        self.image_path = image_path
        self.pixmap = QPixmap(str(image_path))
        self.update_display()

    def set_annotations(self, annotations: List):
        """Set bounding box annotations"""
        self.annotations = annotations
        self.update_display()

    def set_classes(self, class_names: List[str], class_colors: dict):
        """Set class names and colors"""
        self.class_names = class_names
        self.class_colors = class_colors

    def set_current_class(self, class_id: int):
        """Set current class for drawing"""
        self.current_class = class_id

    def update_display(self):
        """Update image display with annotations"""
        if self.pixmap is None:
            return

        # Create a copy of pixmap to draw on
        display_pixmap = self.pixmap.copy()
        painter = QPainter(display_pixmap)

        # Draw existing annotations
        for i, bbox in enumerate(self.annotations):
            x1, y1, x2, y2 = bbox.to_absolute(
                self.pixmap.width(), self.pixmap.height()
            )

            color = self.class_colors.get(bbox.class_id, (255, 255, 255))
            pen = QPen(QColor(*color))
            pen.setWidth(3 if i == self.selected_box else 2)
            painter.setPen(pen)

            painter.drawRect(x1, y1, x2-x1, y2-y1)

            # Draw label
            if bbox.class_id < len(self.class_names):
                label = self.class_names[bbox.class_id]
                painter.drawText(x1, y1-5, label)

        # Draw current rectangle being drawn
        if self.current_rect:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        painter.end()

        # Apply zoom
        if self.zoom_factor != 1.0:
            display_pixmap = display_pixmap.scaled(
                int(display_pixmap.width() * self.zoom_factor),
                int(display_pixmap.height() * self.zoom_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        self.image_label.setPixmap(display_pixmap)

    def mousePressEvent(self, event):
        """Handle mouse press for drawing"""
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap:
            pos = self.image_label.mapFrom(self, event.pos())
            self.start_point = pos
            self.drawing = True

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing"""
        if self.drawing and self.start_point:
            pos = self.image_label.mapFrom(self, event.pos())
            self.current_rect = QRect(self.start_point, pos).normalized()
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish drawing"""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            pos = self.image_label.mapFrom(self, event.pos())
            rect = QRect(self.start_point, pos).normalized()

            # Convert to image coordinates
            x1 = int(rect.x() / self.zoom_factor)
            y1 = int(rect.y() / self.zoom_factor)
            x2 = int(rect.right() / self.zoom_factor)
            y2 = int(rect.bottom() / self.zoom_factor)

            # Emit signal
            self.box_added.emit(x1, y1, x2, y2, self.current_class)

            self.drawing = False
            self.start_point = None
            self.current_rect = None

    def zoom_in(self):
        """Zoom in"""
        self.zoom_factor *= 1.2
        self.update_display()

    def zoom_out(self):
        """Zoom out"""
        self.zoom_factor /= 1.2
        self.update_display()

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.update_display()
