"""
Auto-save and recovery functionality

Periodically saves application state and provides recovery from crashes.
Saves: project state, training progress, annotation work, UI preferences.
"""
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from core.logger import get_logger

logger = get_logger(__name__)


class AutoSaveManager:
    """Manages automatic state saving and crash recovery"""

    def __init__(self, save_dir: Optional[Path] = None,
                interval: int = 300):  # 5 minutes default
        """
        Args:
            save_dir: Directory to save state (default: ~/.yolo_training_tool/autosave)
            interval: Auto-save interval in seconds
        """
        if save_dir is None:
            save_dir = Path.home() / ".yolo_training_tool" / "autosave"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.interval = interval
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._state_providers: Dict[str, Callable] = {}
        self._enabled = False

        logger.info(f"Auto-save initialized. Directory: {self.save_dir}, "
                   f"Interval: {interval}s")

    def register_state_provider(self, key: str, provider: Callable):
        """Register a function that returns state to be saved
        Args:
            key: Unique key for this state component
            provider: Function returning a JSON-serializable dict
        """
        with self._lock:
            self._state_providers[key] = provider
        logger.debug(f"Registered state provider: {key}")

    def unregister_state_provider(self, key: str):
        """Remove a state provider"""
        with self._lock:
            self._state_providers.pop(key, None)

    def start(self):
        """Start auto-save timer"""
        if self._enabled:
            return

        self._enabled = True
        self._schedule_save()
        logger.info("Auto-save started")

    def stop(self):
        """Stop auto-save timer"""
        self._enabled = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("Auto-save stopped")

    def _schedule_save(self):
        """Schedule next auto-save"""
        if not self._enabled:
            return

        self._timer = threading.Timer(self.interval, self._auto_save)
        self._timer.daemon = True
        self._timer.start()

    def _auto_save(self):
        """Perform auto-save and reschedule"""
        try:
            self.save_now()
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
        finally:
            self._schedule_save()

    def save_now(self) -> Optional[Path]:
        """Force an immediate save
        Returns:
            Path to saved state file, or None if no providers
        """
        with self._lock:
            if not self._state_providers:
                return None

            # Collect state from all providers
            state = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'data': {}
            }

            for key, provider in self._state_providers.items():
                try:
                    state['data'][key] = provider()
                except Exception as e:
                    logger.warning(f"State provider {key} failed: {e}")
                    state['data'][key] = {'_error': str(e)}

            # Save with timestamp
            save_path = self.save_dir / "current.json"
            backup_path = self.save_dir / "previous.json"

            # Rotate: current -> previous
            try:
                if save_path.exists():
                    if backup_path.exists():
                        backup_path.unlink()
                    save_path.rename(backup_path)
            except Exception as e:
                logger.warning(f"Could not rotate save files: {e}")

            # Write new state atomically (write to temp, then rename)
            temp_path = self.save_dir / "current.tmp"
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, default=str)
                temp_path.rename(save_path)

                logger.debug(f"Auto-saved {len(state['data'])} state components")
                return save_path

            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                # Clean up temp file
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return None

    def has_recovery_data(self) -> bool:
        """Check if there is recovery data available"""
        save_path = self.save_dir / "current.json"
        return save_path.exists()

    def load_recovery_data(self) -> Optional[Dict[str, Any]]:
        """Load saved state for recovery
        Returns:
            Saved state dict, or None if not available
        """
        save_path = self.save_dir / "current.json"
        if not save_path.exists():
            return None

        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"Loaded recovery data from {state.get('timestamp', 'unknown time')}")
            return state
        except Exception as e:
            logger.error(f"Could not load recovery data: {e}")

            # Try backup
            backup_path = self.save_dir / "previous.json"
            if backup_path.exists():
                try:
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    logger.info("Loaded backup recovery data")
                    return state
                except Exception as e2:
                    logger.error(f"Could not load backup: {e2}")

            return None

    def clear_recovery_data(self):
        """Clear saved recovery data (after successful recovery or fresh start)"""
        for filename in ['current.json', 'previous.json', 'current.tmp']:
            path = self.save_dir / filename
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {path}: {e}")
        logger.info("Recovery data cleared")

    def get_save_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current save without loading full data"""
        save_path = self.save_dir / "current.json"
        if not save_path.exists():
            return None

        try:
            stat = save_path.stat()
            return {
                'path': str(save_path),
                'size_bytes': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'age_seconds': time.time() - stat.st_mtime,
            }
        except Exception as e:
            logger.error(f"Could not get save info: {e}")
            return None
