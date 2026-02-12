"""
Image Viewer Widget - displays images with bounding boxes and polygons
Supports selecting, moving, and resizing existing annotations.
"""
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QPushButton, QSlider, QButtonGroup, QRadioButton, QGroupBox)
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QPolygonF, QWheelEvent
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


# Handle size in image pixels for resize grips
HANDLE_SIZE = 6

# Hit-test tolerance in image pixels
HIT_TOLERANCE = 6


class ImageViewer(QWidget):
    """Widget for displaying and annotating images"""

    box_added = pyqtSignal(int, int, int, int, int)  # x1, y1, x2, y2, class_id
    polygon_added = pyqtSignal(list, int)  # points, class_id
    annotation_selected = pyqtSignal(int)  # annotation index (-1 = deselect)
    annotation_updated = pyqtSignal(int)   # annotation index that was moved/resized

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

        # Edit state
        self._edit_action = None   # None, "move", "resize"
        self._resize_handle = None  # which handle is being dragged
        self._drag_origin = None   # mouse position at drag start (image coords)
        self._orig_box = None      # original box coords at drag start (x1,y1,x2,y2)

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
        self.selected_annotation = -1
        self._edit_action = None
        self.update_display()

    def clear_image(self):
        """Clear the current image"""
        self.image_path = None
        self.pixmap = None
        self.annotations = []
        self.selected_annotation = -1
        self._edit_action = None
        self.image_label.clear()

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

    def select_annotation(self, index: int):
        """Programmatically select an annotation (called from LabelWidget)"""
        self.selected_annotation = index
        self._edit_action = None
        self.update_display()

    # ─── Hit Testing ─────────────────────────────────────────────

    def _get_box_abs(self, ann) -> Optional[Tuple[int, int, int, int]]:
        """Get absolute (x1,y1,x2,y2) for a box annotation"""
        if not self.pixmap or not hasattr(ann, 'annotation_type'):
            return None
        if ann.annotation_type == 'box':
            return ann.to_absolute(self.pixmap.width(), self.pixmap.height())
        return None

    def _handle_rects(self, x1, y1, x2, y2):
        """Return dict of handle-name → (cx, cy) center positions for a box"""
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        return {
            'tl': (x1, y1), 'tm': (mx, y1), 'tr': (x2, y1),
            'ml': (x1, my),                  'mr': (x2, my),
            'bl': (x1, y2), 'bm': (mx, y2), 'br': (x2, y2),
        }

    def _hit_test_handles(self, x, y, x1, y1, x2, y2):
        """Check if (x,y) hits any resize handle. Returns handle name or None."""
        hs = HANDLE_SIZE + 2  # slightly larger for easier grab
        for name, (cx, cy) in self._handle_rects(x1, y1, x2, y2).items():
            if abs(x - cx) <= hs and abs(y - cy) <= hs:
                return name
        return None

    def _hit_test_annotations(self, x, y):
        """Find which annotation is under (x,y). Returns index or -1."""
        if not self.pixmap:
            return -1
        w, h = self.pixmap.width(), self.pixmap.height()
        # Iterate in reverse so topmost (last drawn) is hit first
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            if not hasattr(ann, 'annotation_type'):
                continue
            if ann.annotation_type == 'box':
                bx1, by1, bx2, by2 = ann.to_absolute(w, h)
                if bx1 - HIT_TOLERANCE <= x <= bx2 + HIT_TOLERANCE and \
                   by1 - HIT_TOLERANCE <= y <= by2 + HIT_TOLERANCE:
                    return i
            elif ann.annotation_type == 'polygon':
                pts = ann.to_absolute(w, h)
                if self._point_in_polygon(x, y, pts):
                    return i
        return -1

    @staticmethod
    def _point_in_polygon(px, py, polygon_pts):
        """Ray-casting point-in-polygon test"""
        n = len(polygon_pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon_pts[i]
            xj, yj = polygon_pts[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    # ─── Display ─────────────────────────────────────────────────

    def update_display(self):
        """Update image display with annotations"""
        if self.pixmap is None:
            return

        # Create a copy of pixmap to draw on
        display_pixmap = self.pixmap.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        img_w, img_h = self.pixmap.width(), self.pixmap.height()

        # Draw existing annotations
        for i, ann in enumerate(self.annotations):
            is_selected = (i == self.selected_annotation)
            color = self.class_colors.get(ann.class_id, (255, 255, 255))
            qcolor = QColor(*color)

            # Pen: thicker + dashed for selected
            pen = QPen(qcolor)
            if is_selected:
                pen.setWidth(3)
            else:
                pen.setWidth(2)
            painter.setPen(pen)

            # Semi-transparent fill for selected
            if is_selected:
                fill = QColor(*color, 40)
                painter.setBrush(QBrush(fill))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)

            # Draw based on annotation type
            if hasattr(ann, 'annotation_type'):
                if ann.annotation_type == 'box':
                    x1, y1, x2, y2 = ann.to_absolute(img_w, img_h)
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                    # Draw label background + text
                    if ann.class_id < len(self.class_names):
                        label = self.class_names[ann.class_id]
                        self._draw_label_tag(painter, label, x1, y1, qcolor)

                    # Draw resize handles for selected box
                    if is_selected:
                        self._draw_handles(painter, x1, y1, x2, y2, qcolor)

                elif ann.annotation_type == 'polygon':
                    points = ann.to_absolute(img_w, img_h)
                    qpoints = [QPointF(x, y) for x, y in points]
                    polygon = QPolygonF(qpoints)
                    painter.drawPolygon(polygon)

                    # Draw label
                    if ann.class_id < len(self.class_names) and points:
                        label = self.class_names[ann.class_id]
                        self._draw_label_tag(painter, label, int(points[0][0]), int(points[0][1]), qcolor)

            painter.setBrush(Qt.BrushStyle.NoBrush)

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
            for idx in range(len(self.current_polygon_points) - 1):
                p1 = self.current_polygon_points[idx]
                p2 = self.current_polygon_points[idx + 1]
                painter.drawLine(p1, p2)

            # Draw points
            for point in self.current_polygon_points:
                painter.drawEllipse(point, 3, 3)

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

    def _draw_label_tag(self, painter: QPainter, label: str, x: int, y: int, color: QColor):
        """Draw a label tag with background above annotation"""
        font = painter.font()
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label) + 8
        text_h = metrics.height() + 4
        tag_y = y - text_h
        if tag_y < 0:
            tag_y = y

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(x, tag_y, text_w, text_h)

        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(x + 4, tag_y + metrics.ascent() + 2, label)

    def _draw_handles(self, painter: QPainter, x1, y1, x2, y2, color: QColor):
        """Draw resize handles on selected box annotation"""
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QBrush(color))
        hs = HANDLE_SIZE
        for _, (cx, cy) in self._handle_rects(x1, y1, x2, y2).items():
            painter.drawRect(cx - hs, cy - hs, hs * 2, hs * 2)

    # ─── Coordinate Conversion ───────────────────────────────────

    def get_image_coordinates(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if not self.pixmap:
            return None

        # Get the displayed pixmap size (with zoom)
        displayed_width = int(self.pixmap.width() * self.zoom_factor)
        displayed_height = int(self.pixmap.height() * self.zoom_factor)

        # Get label size
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Calculate offset (image is centered in label)
        offset_x = max(0, (label_width - displayed_width) // 2)
        offset_y = max(0, (label_height - displayed_height) // 2)

        # Adjust for offset
        img_x = widget_pos.x() - offset_x
        img_y = widget_pos.y() - offset_y

        # Check if click is within image bounds
        if img_x < 0 or img_y < 0 or img_x >= displayed_width or img_y >= displayed_height:
            return None

        # Convert from zoomed coordinates to original image coordinates
        original_x = int(img_x / self.zoom_factor)
        original_y = int(img_y / self.zoom_factor)

        return QPoint(original_x, original_y)

    # ─── Mouse Events ────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Handle mouse press for drawing and editing"""
        if not self.pixmap:
            return

        widget_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        img_pos = self.get_image_coordinates(widget_pos)

        if img_pos is None:
            return

        ix, iy = img_pos.x(), img_pos.y()

        if event.button() == Qt.MouseButton.LeftButton:
            # --- Check if clicking on a resize handle of the selected box ---
            if self.selected_annotation >= 0:
                ann = self.annotations[self.selected_annotation]
                box = self._get_box_abs(ann)
                if box:
                    handle = self._hit_test_handles(ix, iy, *box)
                    if handle:
                        self._edit_action = "resize"
                        self._resize_handle = handle
                        self._drag_origin = (ix, iy)
                        self._orig_box = box
                        return

            # --- Check if clicking on any annotation body ---
            hit_idx = self._hit_test_annotations(ix, iy)
            if hit_idx >= 0:
                # Select the annotation and prepare to move
                self.selected_annotation = hit_idx
                self._edit_action = "move"
                self._drag_origin = (ix, iy)
                ann = self.annotations[hit_idx]
                box = self._get_box_abs(ann)
                if box:
                    self._orig_box = box
                else:
                    self._orig_box = None
                self.annotation_selected.emit(hit_idx)
                self.update_display()
                return

            # --- Click on empty area: deselect and start drawing ---
            if self.selected_annotation >= 0:
                self.selected_annotation = -1
                self._edit_action = None
                self.annotation_selected.emit(-1)
                self.update_display()

            # Start drawing
            if self.annotation_mode == "box":
                self.start_point = img_pos
                self.drawing = True

            elif self.annotation_mode == "polygon":
                self.current_polygon_points.append(img_pos)
                self.drawing = True
                self.update_display()

        elif event.button() == Qt.MouseButton.RightButton:
            # Finish polygon on right click
            if self.annotation_mode == "polygon" and len(self.current_polygon_points) >= 3:
                self.finish_polygon()

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing and editing"""
        if not self.pixmap:
            return

        widget_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
        img_pos = self.get_image_coordinates(widget_pos)

        if img_pos is None:
            return

        ix, iy = img_pos.x(), img_pos.y()

        # --- Moving a selected annotation ---
        if self._edit_action == "move" and self._drag_origin and self._orig_box:
            dx = ix - self._drag_origin[0]
            dy = iy - self._drag_origin[1]
            ox1, oy1, ox2, oy2 = self._orig_box
            img_w, img_h = self.pixmap.width(), self.pixmap.height()

            # Clamp to image bounds
            nx1 = max(0, min(ox1 + dx, img_w - (ox2 - ox1)))
            ny1 = max(0, min(oy1 + dy, img_h - (oy2 - oy1)))
            nx2 = nx1 + (ox2 - ox1)
            ny2 = ny1 + (oy2 - oy1)

            self._apply_box_coords(self.selected_annotation, nx1, ny1, nx2, ny2)
            self.update_display()
            return

        # --- Resizing a selected annotation ---
        if self._edit_action == "resize" and self._drag_origin and self._orig_box:
            ox1, oy1, ox2, oy2 = self._orig_box
            dx = ix - self._drag_origin[0]
            dy = iy - self._drag_origin[1]
            nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
            h = self._resize_handle
            img_w, img_h = self.pixmap.width(), self.pixmap.height()

            # Adjust edges based on which handle
            if h in ('tl', 'ml', 'bl'):
                nx1 = max(0, min(ox1 + dx, ox2 - 4))
            if h in ('tr', 'mr', 'br'):
                nx2 = max(ox1 + 4, min(ox2 + dx, img_w))
            if h in ('tl', 'tm', 'tr'):
                ny1 = max(0, min(oy1 + dy, oy2 - 4))
            if h in ('bl', 'bm', 'br'):
                ny2 = max(oy1 + 4, min(oy2 + dy, img_h))

            self._apply_box_coords(self.selected_annotation, nx1, ny1, nx2, ny2)
            self.update_display()
            return

        # --- Drawing new box ---
        if self.annotation_mode == "box" and self.drawing and self.start_point:
            self.current_rect = QRect(self.start_point, img_pos).normalized()
            self.update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish drawing or editing"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._edit_action in ("move", "resize"):
                # Emit update signal so MainWindow can save
                if self.selected_annotation >= 0:
                    self.annotation_updated.emit(self.selected_annotation)
                self._edit_action = None
                self._drag_origin = None
                self._orig_box = None
                self._resize_handle = None
                return

            if self.annotation_mode == "box" and self.drawing:
                self.finish_box()

    def mouseDoubleClickEvent(self, event):
        """Handle double click to finish polygon"""
        if self.annotation_mode == "polygon" and len(self.current_polygon_points) >= 3:
            self.finish_polygon()

    def keyPressEvent(self, event):
        """Handle Escape to deselect"""
        if event.key() == Qt.Key.Key_Escape:
            if self.selected_annotation >= 0:
                self.selected_annotation = -1
                self._edit_action = None
                self.annotation_selected.emit(-1)
                self.update_display()
        super().keyPressEvent(event)

    # ─── Annotation Editing Helpers ──────────────────────────────

    def _apply_box_coords(self, idx, x1, y1, x2, y2):
        """Apply new absolute coords to a box annotation in-place"""
        ann = self.annotations[idx]
        if ann.annotation_type != 'box':
            return
        img_w, img_h = self.pixmap.width(), self.pixmap.height()
        ann.x_center = ((x1 + x2) / 2) / img_w
        ann.y_center = ((y1 + y2) / 2) / img_h
        ann.width = (x2 - x1) / img_w
        ann.height = (y2 - y1) / img_h

    # ─── Drawing Finishers ───────────────────────────────────────

    def finish_box(self):
        """Finish drawing bounding box"""
        if not self.start_point or not self.current_rect:
            return

        x1 = int(self.current_rect.x())
        y1 = int(self.current_rect.y())
        x2 = int(self.current_rect.right())
        y2 = int(self.current_rect.bottom())

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

        points = []
        for point in self.current_polygon_points:
            x = int(point.x())
            y = int(point.y())
            points.append((x, y))

        # Emit signal
        self.polygon_added.emit(points, self.current_class)

        # Reset
        self.drawing = False
        self.current_polygon_points = []
        self.update_display()

    # ─── Zoom ────────────────────────────────────────────────────

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
