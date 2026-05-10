"""
Unit tests for core.auto_save
"""
import json
import time
import pytest
from pathlib import Path

from core.auto_save import AutoSaveManager


class TestAutoSaveManager:
    """Tests for AutoSaveManager"""

    def test_initialization(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path, interval=60)
        assert manager.save_dir == tmp_path
        assert manager.interval == 60
        assert tmp_path.exists()

    def test_register_state_provider(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)

        manager.register_state_provider('test', lambda: {'data': 'value'})
        assert 'test' in manager._state_providers

    def test_unregister_state_provider(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {})
        manager.unregister_state_provider('test')
        assert 'test' not in manager._state_providers

    def test_save_now_no_providers(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        result = manager.save_now()
        assert result is None

    def test_save_now_with_providers(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('app', lambda: {'window_size': [800, 600]})
        manager.register_state_provider('project', lambda: {'name': 'test'})

        save_path = manager.save_now()
        assert save_path is not None
        assert save_path.exists()

        # Verify content
        with open(save_path) as f:
            state = json.load(f)

        assert 'data' in state
        assert 'app' in state['data']
        assert 'project' in state['data']
        assert state['data']['project']['name'] == 'test'

    def test_save_now_handles_provider_errors(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)

        def good_provider():
            return {'value': 1}

        def bad_provider():
            raise RuntimeError("Provider failed")

        manager.register_state_provider('good', good_provider)
        manager.register_state_provider('bad', bad_provider)

        save_path = manager.save_now()
        assert save_path is not None

        with open(save_path) as f:
            state = json.load(f)

        assert state['data']['good'] == {'value': 1}
        assert '_error' in state['data']['bad']

    def test_has_recovery_data_initially_false(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        assert manager.has_recovery_data() is False

    def test_has_recovery_data_after_save(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {'x': 1})
        manager.save_now()
        assert manager.has_recovery_data() is True

    def test_load_recovery_data(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {'value': 42})
        manager.save_now()

        # New manager should be able to load
        manager2 = AutoSaveManager(save_dir=tmp_path)
        recovery = manager2.load_recovery_data()

        assert recovery is not None
        assert recovery['data']['test']['value'] == 42

    def test_load_recovery_data_none(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        result = manager.load_recovery_data()
        assert result is None

    def test_clear_recovery_data(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {'x': 1})
        manager.save_now()

        assert manager.has_recovery_data() is True

        manager.clear_recovery_data()
        assert manager.has_recovery_data() is False

    def test_save_creates_backup(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {'value': 'first'})
        manager.save_now()

        manager.unregister_state_provider('test')
        manager.register_state_provider('test', lambda: {'value': 'second'})
        manager.save_now()

        backup_path = tmp_path / "previous.json"
        current_path = tmp_path / "current.json"

        assert current_path.exists()
        assert backup_path.exists()

        # Backup should have the old value
        with open(backup_path) as f:
            backup = json.load(f)
        assert backup['data']['test']['value'] == 'first'

    def test_get_save_info(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        manager.register_state_provider('test', lambda: {'x': 1})
        manager.save_now()

        info = manager.get_save_info()
        assert info is not None
        assert 'size_bytes' in info
        assert 'modified' in info
        assert info['size_bytes'] > 0

    def test_get_save_info_no_save(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path)
        info = manager.get_save_info()
        assert info is None

    def test_start_stop(self, tmp_path):
        manager = AutoSaveManager(save_dir=tmp_path, interval=60)

        manager.start()
        assert manager._enabled is True

        manager.stop()
        assert manager._enabled is False
