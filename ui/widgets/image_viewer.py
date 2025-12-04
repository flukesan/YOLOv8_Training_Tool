"""
Image Viewer Widget - displays images with bounding boxes and polygons
"""
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QPushButton, QSlider, QButtonGroup, QRadioButton, QGroupBox)
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF, QWheelEvent
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class ImageViewer(QWidget):
    """Widget for displaying and annotating images"""

    box_added = pyqtSignal(int, int, int, int, int)  # x1, y1, x2, y2, class_id
    polygon_added = pyqtSignal(list, int)  # points, class_id
    annotation_selected = pyqtSignal(int)  # annotation index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap = None
        self.annotations = []
        self.current_class = 0
        self.class_colors = {}
        self.class_names = []

        # Drawing state
        self.annotation_mode = "box"  # "box" or "polygon"
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.current_polygon_points = []
        self.selected_annotation = -1

        # Zoom
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout()

        # Top toolbar
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)

        main_layout.addWidget(self.scroll)

        self.setLayout(main_layout)

    def create_toolbar(self):
        """Create toolbar with annotation mode and zoom controls"""
        toolbar = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Annotation mode selector
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QHBoxLayout()

        self.btn_box_mode = QRadioButton("Box")
        self.btn_box_mode.setChecked(True)
        self.btn_box_mode.toggled.connect(lambda: self.set_annotation_mode("box"))

        self.btn_polygon_mode = QRadioButton("Polygon")
        self.btn_polygon_mode.toggled.connect(lambda: self.set_annotation_mode("polygon"))

        mode_layout.addWidget(self.btn_box_mode)
        mode_layout.addWidget(self.btn_polygon_mode)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        layout.addStretch()

        # Zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10% = 0.1x
        self.zoom_slider.setMaximum(500)  # 500% = 5.0x
        self.zoom_slider.setValue(100)  # 100% = 1.0x
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_reset = QPushButton("Reset")
        self.btn_zoom_reset.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(self.btn_zoom_reset)

        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)

        toolbar.setLayout(layout)
        return toolbar

    def set_annotation_mode(self, mode: str):
        """Set annotation mode (box or polygon)"""
        self.annotation_mode = mode
        # Clear current drawing
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.current_polygon_points = []
        self.update_display()

    def load_image(self, image_path: Path):
        """Load and display image"""
        self.image_path = image_path
        self.pixmap = QPixmap(str(image_path))
        self.update_display()

    def set_annotations(self, annotations: List):
        """Set annotations"""
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
        for i, ann in enumerate(self.annotations):
            color = self.class_colors.get(ann.class_id, (255, 255, 255))
            pen = QPen(QColor(*color))
            pen.setWidth(3 if i == self.selected_annotation else 2)
            painter.setPen(pen)

            # Draw based on annotation type
            if hasattr(ann, 'annotation_type'):
                if ann.annotation_type == 'box':
                    x1, y1, x2, y2 = ann.to_absolute(
                        self.pixmap.width(), self.pixmap.height()
                    )
                    painter.drawRect(x1, y1, x2-x1, y2-y1)

                    # Draw label
                    if ann.class_id < len(self.class_names):
                        label = self.class_names[ann.class_id]
                        painter.drawText(x1, y1-5, label)

                elif ann.annotation_type == 'polygon':
                    points = ann.to_absolute(
                        self.pixmap.width(), self.pixmap.height()
                    )
                    qpoints = [QPointF(x, y) for x, y in points]
                    polygon = QPolygonF(qpoints)
                    painter.drawPolygon(polygon)

                    # Draw label
                    if ann.class_id < len(self.class_names) and points:
                        label = self.class_names[ann.class_id]
                        painter.drawText(int(points[0][0]), int(points[0][1])-5, label)

        # Draw current drawing
        if self.annotation_mode == "box" and self.current_rect:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        elif self.annotation_mode == "polygon" and len(self.current_polygon_points) > 0:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            # Draw lines between points
            for i in range(len(self.current_polygon_points) - 1):
                p1 = self.current_polygon_points[i]
                p2 = self.current_polygon_points[i + 1]
                painter.drawLine(p1, p2)

            # Draw points
            for point in self.current_polygon_points:
                painter.drawEllipse(point, 3, 3)

            # Draw line from last point to cursor (if drawing)
            if self.drawing and len(self.current_polygon_points) > 0:
                # This will be drawn during mouse move

                pass

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
        if not self.pixmap:
            return

        pos = self.image_label.mapFrom(self, event.pos())

        if event.button() == Qt.MouseButton.LeftButton:
            if self.annotation_mode == "box":
                self.start_point = pos
                self.drawing = True

            elif self.annotation_mode == "polygon":
                # Add point to polygon
                self.current_polygon_points.append(pos)
                self.drawing = True
                self.update_display()

        elif event.button() == Qt.MouseButton.RightButton:
            # Finish polygon on right click
            if self.annotation_mode == "polygon" and len(self.current_polygon_points) >= 3:
                self.finish_polygon()

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing"""
        if not self.pixmap:
            return

        pos = self.image_label.mapFrom(self, event.pos())

        if self.annotation_mode == "box" and self.drawing and self.start_point:
            self.current_rect = QRect(self.start_point, pos).normalized()
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish drawing"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.annotation_mode == "box" and self.drawing:
                self.finish_box()

    def mouseDoubleClickEvent(self, event):
        """Handle double click to finish polygon"""
        if self.annotation_mode == "polygon" and len(self.current_polygon_points) >= 3:
            self.finish_polygon()

    def finish_box(self):
        """Finish drawing bounding box"""
        if not self.start_point or not self.current_rect:
            return

        # Convert to image coordinates
        x1 = int(self.current_rect.x() / self.zoom_factor)
        y1 = int(self.current_rect.y() / self.zoom_factor)
        x2 = int(self.current_rect.right() / self.zoom_factor)
        y2 = int(self.current_rect.bottom() / self.zoom_factor)

        # Emit signal
        self.box_added.emit(x1, y1, x2, y2, self.current_class)

        # Reset
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.update_display()

    def finish_polygon(self):
        """Finish drawing polygon"""
        if len(self.current_polygon_points) < 3:
            return

        # Convert to image coordinates
        points = []
        for point in self.current_polygon_points:
            x = int(point.x() / self.zoom_factor)
            y = int(point.y() / self.zoom_factor)
            points.append((x, y))

        # Emit signal
        self.polygon_added.emit(points, self.current_class)

        # Reset
        self.drawing = False
        self.current_polygon_points = []
        self.update_display()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_zoom_slider_changed(self, value):
        """Handle zoom slider change"""
        self.zoom_factor = value / 100.0
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
        self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
        self.update_display()

    def zoom_in(self):
        """Zoom in"""
        self.zoom_factor *= 1.2
        self.zoom_factor = min(self.zoom_factor, self.max_zoom)
        self.update_zoom_ui()
        self.update_display()

    def zoom_out(self):
        """Zoom out"""
        self.zoom_factor /= 1.2
        self.zoom_factor = max(self.zoom_factor, self.min_zoom)
        self.update_zoom_ui()
        self.update_display()

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.update_zoom_ui()
        self.update_display()

    def update_zoom_ui(self):
        """Update zoom UI elements"""
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
