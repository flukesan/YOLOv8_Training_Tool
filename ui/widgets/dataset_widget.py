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
