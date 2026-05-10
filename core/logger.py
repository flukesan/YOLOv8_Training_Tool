"""
Centralized logging framework for YOLOv8 Training Tool

Provides a consistent logging interface across the application with:
- File and console output
- Rotating log files
- Per-module loggers
- Configurable log levels
- Thread-safe logging
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class AppLogger:
    """Application logger with rotating file handler and console output"""

    _initialized = False
    _log_dir: Optional[Path] = None
    _log_level = logging.INFO

    @classmethod
    def initialize(cls, log_dir: Optional[Path] = None,
                  level: str = 'INFO',
                  max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                  backup_count: int = 5):
        """Initialize the logging framework
        Args:
            log_dir: Directory for log files (default: ~/.yolo_training_tool/logs)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximum bytes per log file before rotation
            backup_count: Number of backup files to keep
        """
        if cls._initialized:
            return

        # Determine log directory
        if log_dir is None:
            log_dir = Path.home() / ".yolo_training_tool" / "logs"
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        # Set log level
        cls._log_level = getattr(logging, level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger('yolo_tool')
        root_logger.setLevel(cls._log_level)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Format
        log_format = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler with rotation
        log_file = cls._log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(cls._log_level)
        root_logger.addHandler(file_handler)

        # Console handler (only WARNING and above by default)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(logging.WARNING)
        root_logger.addHandler(console_handler)

        # Error file handler (only ERROR and above to a separate file)
        error_file = cls._log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(log_format)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)

        cls._initialized = True
        root_logger.info(f"Logging initialized. Log directory: {cls._log_dir}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific module
        Args:
            name: Module name (typically __name__)
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.initialize()

        # Use hierarchical naming under 'yolo_tool'
        if not name.startswith('yolo_tool'):
            name = f"yolo_tool.{name}"

        return logging.getLogger(name)

    @classmethod
    def get_log_dir(cls) -> Optional[Path]:
        """Get the log directory path"""
        return cls._log_dir

    @classmethod
    def set_console_level(cls, level: str):
        """Set the console output log level
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.WARNING)
        root_logger = logging.getLogger('yolo_tool')

        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and \
               not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger for a module"""
    return AppLogger.get_logger(name)
