"""
Dataset Widget - browse dataset images with modern UI
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeySequence, QShortcut


class DatasetWidget(QWidget):
    """Widget for browsing dataset"""

    image_selected = pyqtSignal(str)
    images_deleted = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = []
        self.class_names = []
        self.class_colors = {}
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Title row with count
        title_row = QHBoxLayout()
        title = QLabel("Images")
        title.setStyleSheet("font-weight: bold; font-size: 15px;")
        title_row.addWidget(title)

        self.count_label = QLabel("0 images")
        self.count_label.setStyleSheet("color: #8891a0; font-size: 12px;")
        title_row.addStretch()
        title_row.addWidget(self.count_label)
        layout.addLayout(title_row)

        # Image list
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.itemClicked.connect(self._on_image_selected)
        layout.addWidget(self.image_list)

        # Annotation summary for current image
        self.annotation_summary = QLabel("")
        self.annotation_summary.setWordWrap(True)
        self.annotation_summary.setStyleSheet(
            "background-color: #1e2229; border-radius: 6px; padding: 8px 10px; "
            "font-size: 12px; color: #c8cdd3;"
        )
        self.annotation_summary.setVisible(False)
        layout.addWidget(self.annotation_summary)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)

        self.btn_prev = QPushButton("<  Previous")
        self.btn_prev.setStyleSheet(
            "QPushButton { padding: 8px 14px; border-radius: 8px; }"
        )
        self.btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self.btn_prev)

        # Current position label
        self.pos_label = QLabel("- / -")
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pos_label.setStyleSheet("color: #8891a0; font-size: 12px;")
        self.pos_label.setFixedWidth(60)
        nav_layout.addWidget(self.pos_label)

        self.btn_next = QPushButton("Next  >")
        self.btn_next.setStyleSheet(
            "QPushButton { padding: 8px 14px; border-radius: 8px; }"
        )
        self.btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self.btn_next)

        layout.addLayout(nav_layout)

        # Delete button
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: #ffffff; "
            "border: none; border-radius: 8px; padding: 8px 16px; "
            "font-weight: 600; }"
            "QPushButton:hover { background-color: #d94435; }"
        )
        self.btn_delete.clicked.connect(self._on_delete)
        layout.addWidget(self.btn_delete)

        # Keyboard shortcut
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self.delete_shortcut.activated.connect(self._on_delete)

        self.setLayout(layout)

    def set_classes(self, class_names, class_colors):
        """Set class info for annotation summary display"""
        self.class_names = class_names
        self.class_colors = class_colors

    def update_annotation_summary(self, annotations):
        """Update annotation summary showing counts per class for current image"""
        if not annotations:
            self.annotation_summary.setText(
                '<span style="color:#8891a0;">No annotations</span>'
            )
            self.annotation_summary.setVisible(True)
            return

        # Count annotations per class
        class_counts = {}
        for ann in annotations:
            class_id = ann.class_id if hasattr(ann, 'class_id') else 0
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # Build HTML summary
        total_classes = len(class_counts)
        header = (f'<b style="color:#ffffff;">{total_classes} class{"es" if total_classes != 1 else ""}</b>'
                  f' &nbsp;|&nbsp; '
                  f'<b style="color:#ffffff;">{len(annotations)} annotation{"s" if len(annotations) != 1 else ""}</b>')

        lines = []
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            color = self.class_colors.get(class_id, (255, 255, 255))
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            if class_id < len(self.class_names):
                name = self.class_names[class_id]
            else:
                name = f"Class {class_id}"
            lines.append(
                f'<span style="color:{hex_color};">&#9632;</span> '
                f'{name}: <b>{count}</b>'
            )

        html = header + '<br>' + '<br>'.join(lines)
        self.annotation_summary.setText(html)
        self.annotation_summary.setVisible(True)

    def set_images(self, images):
        """Set image list"""
        self.images = images
        self.image_list.clear()
        for img in images:
            self.image_list.addItem(img.name)

        self.count_label.setText(f"{len(images)} images")
        self._update_position()

    def _update_position(self):
        """Update position label"""
        current = self.image_list.currentRow()
        total = self.image_list.count()
        if current >= 0 and total > 0:
            self.pos_label.setText(f"{current + 1} / {total}")
        else:
            self.pos_label.setText(f"- / {total}")

    def _on_image_selected(self, item):
        """Handle image selection"""
        index = self.image_list.row(item)
        if 0 <= index < len(self.images):
            self.image_selected.emit(str(self.images[index]))
            self._update_position()

    def _on_prev(self):
        """Navigate to previous image"""
        current = self.image_list.currentRow()
        if current > 0:
            self.image_list.setCurrentRow(current - 1)
            self._on_image_selected(self.image_list.currentItem())

    def _on_next(self):
        """Navigate to next image"""
        current = self.image_list.currentRow()
        if current < self.image_list.count() - 1:
            self.image_list.setCurrentRow(current + 1)
            self._on_image_selected(self.image_list.currentItem())

    def _on_delete(self):
        """Handle delete images"""
        selected_items = self.image_list.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select images to delete")
            return

        count = len(selected_items)
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {count} image(s)?\n\n"
            "This will also delete corresponding label files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            selected_indices = [self.image_list.row(item) for item in selected_items]
            selected_images = [self.images[i] for i in selected_indices if i < len(self.images)]
            self.images_deleted.emit(selected_images)
