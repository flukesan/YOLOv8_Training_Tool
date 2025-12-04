#!/usr/bin/env python3
"""
YOLOv8 Training Tool
Main entry point
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window import MainWindow
from config.settings import Settings


def main():
    """Main function"""
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(Settings.APP_NAME)
    app.setApplicationVersion(Settings.APP_VERSION)

    # Load stylesheet if exists
    style_path = Settings.STYLES_DIR / 'style.qss'
    if style_path.exists():
        with open(style_path, 'r') as f:
            app.setStyleSheet(f.read())

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
