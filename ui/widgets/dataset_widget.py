"""
Dataset Widget - browse dataset images
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeySequence, QShortcut


class DatasetWidget(QWidget):
    """Widget for browsing dataset"""

    image_selected = pyqtSignal(str)
    images_deleted = pyqtSignal(list)  # Signal for deleted images

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Dataset Images")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Image list with multi-selection enabled
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.itemClicked.connect(self._on_image_selected)
        layout.addWidget(self.image_list)

        # Navigation buttons
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._on_next)

        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)

        layout.addLayout(btn_layout)

        # Delete button
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_delete.setStyleSheet("background-color: #d32f2f; color: white;")
        layout.addWidget(self.btn_delete)

        # Keyboard shortcut for Delete key
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self.delete_shortcut.activated.connect(self._on_delete)

        self.setLayout(layout)

    def set_images(self, images):
        """Set image list"""
        self.images = images
        self.image_list.clear()
        for img in images:
            self.image_list.addItem(img.name)

    def _on_image_selected(self, item):
        """Handle image selection"""
        index = self.image_list.row(item)
        if 0 <= index < len(self.images):
            self.image_selected.emit(str(self.images[index]))

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

        # Confirm deletion
        count = len(selected_items)
        msg = f"Are you sure you want to delete {count} image(s)?\nThis will also delete corresponding label files."
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Get selected image paths
            selected_indices = [self.image_list.row(item) for item in selected_items]
            selected_images = [self.images[i] for i in selected_indices if i < len(self.images)]

            # Emit signal with list of images to delete
            self.images_deleted.emit(selected_images)
