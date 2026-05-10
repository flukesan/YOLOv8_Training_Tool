"""
Unit tests for core.logger
"""
import pytest
import logging
import tempfile
from pathlib import Path

from core.logger import AppLogger, get_logger


@pytest.fixture(autouse=True)
def enable_logging_for_tests():
    """Override conftest's disable_logging for these tests"""
    logging.disable(logging.NOTSET)
    yield
    logging.disable(logging.NOTSET)


class TestAppLogger:
    """Tests for AppLogger"""

    def setup_method(self):
        """Reset logger state before each test"""
        AppLogger._initialized = False
        AppLogger._log_dir = None
        # Re-enable logging in case previous test disabled it
        logging.disable(logging.NOTSET)

    def test_initialize_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        AppLogger.initialize(log_dir=log_dir, level='DEBUG')

        assert log_dir.exists()
        assert AppLogger._initialized is True
        assert AppLogger.get_log_dir() == log_dir

    def test_get_logger_returns_logger(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path, level='INFO')
        logger = get_logger('test_module')

        assert isinstance(logger, logging.Logger)
        assert 'test_module' in logger.name

    def test_logger_writes_to_file(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path, level='DEBUG')
        logger = get_logger('test_write')
        logger.info("Test message")

        # Force flush
        for handler in logging.getLogger('yolo_tool').handlers:
            handler.flush()

        log_file = tmp_path / "app.log"
        assert log_file.exists()
        content = log_file.read_text(encoding='utf-8')
        assert "Test message" in content

    def test_error_logger_writes_to_separate_file(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path, level='DEBUG')
        logger = get_logger('test_error')
        logger.error("Error happened")

        for handler in logging.getLogger('yolo_tool').handlers:
            handler.flush()

        error_file = tmp_path / "errors.log"
        assert error_file.exists()
        content = error_file.read_text(encoding='utf-8')
        assert "Error happened" in content

    def test_only_initialize_once(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path)
        first_dir = AppLogger.get_log_dir()

        # Try to re-initialize with different dir - should be ignored
        AppLogger.initialize(log_dir=tmp_path / "other")
        assert AppLogger.get_log_dir() == first_dir

    def test_set_console_level(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path)
        # Should not raise
        AppLogger.set_console_level('DEBUG')
        AppLogger.set_console_level('ERROR')

    def test_get_logger_hierarchical_naming(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path)
        logger = get_logger('mymodule')
        assert logger.name == 'yolo_tool.mymodule'

    def test_get_logger_already_prefixed(self, tmp_path):
        AppLogger.initialize(log_dir=tmp_path)
        logger = get_logger('yolo_tool.somemodule')
        assert logger.name == 'yolo_tool.somemodule'
