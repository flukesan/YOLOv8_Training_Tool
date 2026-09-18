"""
Unit tests for core.crash_handler
"""
import sys
import json
import pytest
from pathlib import Path

from core.crash_handler import CrashHandler


class TestCrashHandler:
    """Tests for CrashHandler"""

    def setup_method(self):
        """Reset handler state"""
        CrashHandler._crash_dir = None
        CrashHandler._on_crash_callback = None

    def teardown_method(self):
        """Restore original excepthook"""
        if CrashHandler._original_excepthook:
            sys.excepthook = CrashHandler._original_excepthook

    def test_initialize_creates_dir(self, tmp_path):
        crash_dir = tmp_path / "crashes"
        CrashHandler.initialize(crash_dir=crash_dir)
        assert crash_dir.exists()

    def test_generate_crash_report_creates_files(self, tmp_path):
        CrashHandler.initialize(crash_dir=tmp_path)

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()

        report_path = CrashHandler.generate_crash_report(
            exc_type, exc_value, exc_tb
        )

        assert report_path.exists()
        assert report_path.suffix == '.json'

        # Verify text version
        text_path = report_path.with_suffix('.txt')
        assert text_path.exists()

    def test_crash_report_contains_exception_info(self, tmp_path):
        CrashHandler.initialize(crash_dir=tmp_path)

        try:
            raise RuntimeError("Specific error message")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()

        report_path = CrashHandler.generate_crash_report(
            exc_type, exc_value, exc_tb
        )

        with open(report_path) as f:
            data = json.load(f)

        assert data['exception']['type'] == 'RuntimeError'
        assert 'Specific error message' in data['exception']['message']

    def test_crash_report_contains_system_info(self, tmp_path):
        CrashHandler.initialize(crash_dir=tmp_path)

        try:
            raise ValueError("test")
        except ValueError:
            exc_info = sys.exc_info()

        report_path = CrashHandler.generate_crash_report(*exc_info)

        with open(report_path) as f:
            data = json.load(f)

        assert 'system' in data
        assert 'platform' in data['system']
        assert 'python_version' in data['system']

    def test_get_recent_crashes_empty(self, tmp_path):
        CrashHandler.initialize(crash_dir=tmp_path)
        crashes = CrashHandler.get_recent_crashes()
        assert crashes == []

    def test_get_recent_crashes_with_data(self, tmp_path):
        import time
        CrashHandler.initialize(crash_dir=tmp_path)

        # Create multiple crash reports with delays to get unique timestamps
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except ValueError:
                CrashHandler.generate_crash_report(*sys.exc_info())
            time.sleep(1.1)  # Ensure different timestamps (second granularity)

        crashes = CrashHandler.get_recent_crashes(limit=10)
        assert len(crashes) == 3

    def test_cleanup_old_crashes(self, tmp_path):
        import time
        CrashHandler.initialize(crash_dir=tmp_path)

        # Create old fake crash file
        old_file = tmp_path / "crash_old.json"
        old_file.write_text("{}")
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        import os
        os.utime(old_file, (old_time, old_time))

        # Create recent file
        new_file = tmp_path / "crash_new.json"
        new_file.write_text("{}")

        CrashHandler.cleanup_old_crashes(keep_days=30)

        assert not old_file.exists()
        assert new_file.exists()

    def test_keyboard_interrupt_not_caught(self, tmp_path):
        """KeyboardInterrupt should be passed through, not caught"""
        CrashHandler.initialize(crash_dir=tmp_path)

        # Simulate KeyboardInterrupt handling
        # Should not generate crash report
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            exc_info = sys.exc_info()

        # The handler should pass through, not generate report
        # We just verify it doesn't crash when called
        try:
            CrashHandler._handle_exception(*exc_info)
        except Exception:
            pass

        # Should be no crash report
        crashes = CrashHandler.get_recent_crashes()
        assert len(crashes) == 0
