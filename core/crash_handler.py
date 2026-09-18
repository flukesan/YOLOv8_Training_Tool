"""
Crash reporting and exception handling

Captures unhandled exceptions, generates crash reports with system info,
and provides a recovery mechanism for the application.
"""
import sys
import traceback
import platform
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from core.logger import get_logger

logger = get_logger(__name__)


class CrashHandler:
    """Global crash handler for unhandled exceptions"""

    _crash_dir: Optional[Path] = None
    _on_crash_callback: Optional[Callable] = None
    _original_excepthook = None

    @classmethod
    def initialize(cls, crash_dir: Optional[Path] = None,
                  on_crash: Optional[Callable] = None):
        """Install global exception handler
        Args:
            crash_dir: Directory for crash reports (default: ~/.yolo_training_tool/crashes)
            on_crash: Optional callback to invoke on crash (e.g., show dialog)
        """
        if crash_dir is None:
            crash_dir = Path.home() / ".yolo_training_tool" / "crashes"
        cls._crash_dir = Path(crash_dir)
        cls._crash_dir.mkdir(parents=True, exist_ok=True)

        cls._on_crash_callback = on_crash

        # Save original exception hook
        cls._original_excepthook = sys.excepthook

        # Install our hook
        sys.excepthook = cls._handle_exception

        logger.info(f"Crash handler initialized. Reports: {cls._crash_dir}")

    @classmethod
    def _handle_exception(cls, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        # Don't catch keyboard interrupts
        if issubclass(exc_type, KeyboardInterrupt):
            if cls._original_excepthook:
                cls._original_excepthook(exc_type, exc_value, exc_traceback)
            return

        # Generate crash report
        try:
            report_path = cls.generate_crash_report(
                exc_type, exc_value, exc_traceback
            )
            logger.critical(f"Application crashed. Report saved to: {report_path}")

            # Invoke callback if provided
            if cls._on_crash_callback:
                try:
                    cls._on_crash_callback(report_path, exc_type, exc_value)
                except Exception as e:
                    logger.error(f"Error in crash callback: {e}")

        except Exception as e:
            # Last resort - log to stderr if everything else fails
            print(f"FATAL: Crash handler failed: {e}", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                     file=sys.stderr)

        # Call original handler
        if cls._original_excepthook:
            cls._original_excepthook(exc_type, exc_value, exc_traceback)

    @classmethod
    def generate_crash_report(cls, exc_type, exc_value, exc_traceback) -> Path:
        """Generate a detailed crash report
        Returns:
            Path to crash report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = cls._crash_dir / f"crash_{timestamp}.json"

        # Format traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_str = ''.join(tb_lines)

        # Collect system info
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'exception': {
                'type': exc_type.__name__ if exc_type else 'Unknown',
                'message': str(exc_value) if exc_value else '',
                'traceback': tb_str,
            },
            'system': cls._get_system_info(),
            'environment': cls._get_environment_info(),
        }

        # Write JSON report
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not write JSON crash report: {e}")

        # Also write a human-readable text version
        text_path = cls._crash_dir / f"crash_{timestamp}.txt"
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(cls._format_text_report(report_data))
        except Exception as e:
            logger.error(f"Could not write text crash report: {e}")

        return report_path

    @staticmethod
    def _get_system_info() -> dict:
        """Collect system information"""
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }

        # Try to get GPU info
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_version'] = torch.version.cuda
        except ImportError:
            info['torch'] = 'not installed'
        except Exception as e:
            info['torch_error'] = str(e)

        return info

    @staticmethod
    def _get_environment_info() -> dict:
        """Collect environment information"""
        env_info = {}

        # Try to get installed package versions
        try:
            import pkg_resources
            packages_of_interest = ['ultralytics', 'torch', 'PyQt6',
                                    'opencv-python', 'numpy', 'pyyaml']
            installed = {}
            for pkg in packages_of_interest:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    installed[pkg] = version
                except pkg_resources.DistributionNotFound:
                    installed[pkg] = 'not installed'
            env_info['packages'] = installed
        except Exception as e:
            env_info['package_error'] = str(e)

        return env_info

    @staticmethod
    def _format_text_report(report_data: dict) -> str:
        """Format crash report as readable text"""
        lines = [
            "=" * 70,
            "YOLOv8 Training Tool - Crash Report",
            "=" * 70,
            f"Timestamp: {report_data['timestamp']}",
            "",
            "EXCEPTION:",
            "-" * 70,
            f"Type: {report_data['exception']['type']}",
            f"Message: {report_data['exception']['message']}",
            "",
            "Traceback:",
            report_data['exception']['traceback'],
            "",
            "SYSTEM INFO:",
            "-" * 70,
        ]

        for key, value in report_data['system'].items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "PACKAGES:",
            "-" * 70,
        ])

        packages = report_data.get('environment', {}).get('packages', {})
        for pkg, ver in packages.items():
            lines.append(f"  {pkg}: {ver}")

        lines.extend([
            "",
            "=" * 70,
            "Please send this report to support@example.com when reporting issues.",
            "=" * 70,
        ])

        return '\n'.join(lines)

    @classmethod
    def get_recent_crashes(cls, limit: int = 10) -> list:
        """Get list of recent crash reports
        Args:
            limit: Maximum number of reports to return
        Returns:
            List of crash report paths sorted by recency
        """
        if cls._crash_dir is None or not cls._crash_dir.exists():
            return []

        reports = sorted(
            cls._crash_dir.glob("crash_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return reports[:limit]

    @classmethod
    def cleanup_old_crashes(cls, keep_days: int = 30):
        """Clean up old crash reports
        Args:
            keep_days: Keep reports newer than this many days
        """
        if cls._crash_dir is None or not cls._crash_dir.exists():
            return

        import time
        cutoff = time.time() - (keep_days * 24 * 3600)

        for report in cls._crash_dir.iterdir():
            if report.is_file() and report.stat().st_mtime < cutoff:
                try:
                    report.unlink()
                    logger.debug(f"Deleted old crash report: {report.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {report}: {e}")
