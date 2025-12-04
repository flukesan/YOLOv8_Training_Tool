"""
Dataset Widget - browse dataset images
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QPushButton,
                             QLabel, QHBoxLayout)
from PyQt6.QtCore import pyqtSignal


class DatasetWidget(QWidget):
    """Widget for browsing dataset"""

    image_selected = pyqtSignal(str)

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

        # Image list
        self.image_list = QListWidget()
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
