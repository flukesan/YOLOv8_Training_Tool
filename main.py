#!/usr/bin/env python3
"""
YOLOv8 Training Tool
Main entry point
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window import MainWindow
from config.settings import Settings
from core.logger import AppLogger, get_logger
from core.crash_handler import CrashHandler


def show_crash_dialog(report_path, exc_type, exc_value):
    """Show crash dialog to user"""
    try:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Application Error")
        msg.setText(f"The application encountered an unexpected error: {exc_type.__name__}")
        msg.setInformativeText(
            f"Details: {exc_value}\n\n"
            f"A crash report has been saved to:\n{report_path}\n\n"
            f"Please send this report when reporting the issue."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    except Exception:
        pass  # If even the dialog fails, just continue


def main():
    """Main function"""
    # Initialize logging FIRST - everything else depends on it
    try:
        AppLogger.initialize(level='INFO')
        logger = get_logger(__name__)
        logger.info(f"Starting {Settings.APP_NAME} v{Settings.APP_VERSION}")
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}", file=sys.stderr)
        logger = None

    # Initialize crash handler
    try:
        CrashHandler.initialize(on_crash=show_crash_dialog)
        if logger:
            logger.info("Crash handler installed")

        # Clean up old crashes
        CrashHandler.cleanup_old_crashes(keep_days=30)
    except Exception as e:
        if logger:
            logger.warning(f"Could not initialize crash handler: {e}")
        else:
            print(f"Warning: Could not initialize crash handler: {e}", file=sys.stderr)

    # Create application directories with error handling
    try:
        Settings.create_default_directories()
        if logger:
            logger.info("Application directories ready")
    except PermissionError as e:
        msg = f"Cannot create application directories: {e}"
        if logger:
            logger.critical(msg)
        print(f"Error: {msg}", file=sys.stderr)
        print("Please check file permissions or run with appropriate privileges.")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.critical(f"Error during initialization: {e}", exc_info=True)
        print(f"Error during initialization: {e}", file=sys.stderr)
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
            if logger:
                logger.debug("Stylesheet loaded")
        except Exception as e:
            if logger:
                logger.warning(f"Could not load stylesheet: {e}")

    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        if logger:
            logger.info("Application started successfully")
    except Exception as e:
        if logger:
            logger.critical(f"Failed to create main window: {e}", exc_info=True)
        raise

    # Run application
    exit_code = app.exec()
    if logger:
        logger.info(f"Application exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
