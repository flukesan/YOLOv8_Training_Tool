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
    # Create application directories with error handling
    try:
        Settings.create_default_directories()
    except PermissionError as e:
        print(f"Error: Cannot create application directories: {e}")
        print("Please check file permissions or run with appropriate privileges.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(Settings.APP_NAME)
    app.setApplicationVersion(Settings.APP_VERSION)

    # Load stylesheet if exists (with error handling)
    style_path = Settings.STYLES_DIR / 'style.qss'
    if style_path.exists():
        try:
            with open(style_path, 'r', encoding='utf-8') as f:
                app.setStyleSheet(f.read())
        except Exception as e:
            print(f"Warning: Could not load stylesheet: {e}")
            # Continue without stylesheet

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
