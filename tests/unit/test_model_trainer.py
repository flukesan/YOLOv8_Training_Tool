"""
Unit tests for core.model_trainer (validation logic without actually training)
"""
import pytest
from pathlib import Path

from core.model_trainer import ModelTrainer, TrainingSession


class TestTrainingSession:
    """Tests for TrainingSession class"""

    def test_initialization(self):
        config = {'epochs': 100, 'batch': 16}
        session = TrainingSession('test_session', config)

        assert session.session_id == 'test_session'
        assert session.total_epochs == 100
        assert session.status == 'initialized'
        assert session.current_epoch == 0

    def test_get_progress_zero(self):
        session = TrainingSession('test', {'epochs': 100})
        assert session.get_progress() == 0.0

    def test_get_progress_50_percent(self):
        session = TrainingSession('test', {'epochs': 100})
        session.current_epoch = 50
        assert session.get_progress() == 50.0

    def test_get_progress_zero_epochs(self):
        session = TrainingSession('test', {'epochs': 0})
        # Should not divide by zero
        assert session.get_progress() == 0.0

    def test_update_metrics(self):
        session = TrainingSession('test', {'epochs': 100})
        metrics = {
            'train_loss': 0.5,
            'precision': 0.85,
            'recall': 0.80,
            'mAP50': 0.90,
        }
        session.update_metrics(metrics)

        assert 0.5 in session.metrics['train_loss']
        assert 0.85 in session.metrics['precision']

    def test_update_metrics_unknown_key_ignored(self):
        session = TrainingSession('test', {'epochs': 100})
        session.update_metrics({'unknown_metric': 999})
        assert 'unknown_metric' not in session.metrics


class TestModelTrainer:
    """Tests for ModelTrainer class"""

    def test_initialization(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)
        assert trainer.project_path == temp_project_dir
        assert trainer.current_session is None
        assert not trainer.is_training()

    def test_callbacks_registration(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)

        called = []
        def my_callback():
            called.append(True)

        trainer.register_callback('on_epoch_end', my_callback)
        trainer._trigger_callbacks('on_epoch_end')

        assert len(called) == 1

    def test_callbacks_includes_error_event(self, temp_project_dir):
        """Bug #7 - on_train_error must be registered"""
        trainer = ModelTrainer(temp_project_dir)
        assert 'on_train_error' in trainer.callbacks

    def test_callback_handles_exception(self, temp_project_dir):
        """Callback errors shouldn't crash the trainer"""
        trainer = ModelTrainer(temp_project_dir)

        def bad_callback():
            raise RuntimeError("Callback failed")

        trainer.register_callback('on_epoch_end', bad_callback)
        # Should not raise
        trainer._trigger_callbacks('on_epoch_end')

    # === Configuration Validation Tests ===

    def test_validate_config_valid(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\nnames: [a, b]\n")

        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'data': str(data_yaml),
            'patience': 50,
        }
        valid, error = trainer._validate_config(config)
        assert valid is True
        assert error is None

    def test_validate_config_zero_epochs(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        config = {
            'epochs': 0,
            'batch': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'data': str(data_yaml),
        }
        valid, error = trainer._validate_config(config)
        assert valid is False
        assert "epochs" in error.lower()

    def test_validate_config_invalid_imgsz(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        # 100 is not divisible by 32
        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 100,
            'lr0': 0.01,
            'data': str(data_yaml),
        }
        valid, error = trainer._validate_config(config)
        assert valid is False
        assert "32" in error

    def test_validate_config_invalid_imgsz_too_small(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 16,  # Too small
            'lr0': 0.01,
            'data': str(data_yaml),
        }
        valid, error = trainer._validate_config(config)
        assert valid is False

    def test_validate_config_negative_batch(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        config = {
            'epochs': 100,
            'batch': -5,
            'imgsz': 640,
            'lr0': 0.01,
            'data': str(data_yaml),
        }
        valid, error = trainer._validate_config(config)
        assert valid is False
        assert "batch" in error.lower()

    def test_validate_config_invalid_lr(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'lr0': 1.5,  # > 1
            'data': str(data_yaml),
        }
        valid, error = trainer._validate_config(config)
        assert valid is False
        assert "learning rate" in error.lower()

    def test_validate_config_missing_data(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)
        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'data': '/nonexistent/file.yaml',
        }
        valid, error = trainer._validate_config(config)
        assert valid is False
        assert "data" in error.lower() or "not found" in error.lower()

    def test_validate_config_negative_patience(self, temp_project_dir, tmp_path):
        trainer = ModelTrainer(temp_project_dir)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("test")

        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'data': str(data_yaml),
            'patience': -1,
        }
        valid, error = trainer._validate_config(config)
        assert valid is False

    # === Best Weights Path Tests ===

    def test_get_best_weights_path_no_runs(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)
        result = trainer.get_best_weights_path()
        assert result is None

    def test_get_best_weights_path_with_runs(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)

        # Create fake training run
        run_dir = temp_project_dir / 'runs' / 'train' / 'exp1' / 'weights'
        run_dir.mkdir(parents=True)
        best_weights = run_dir / 'best.pt'
        best_weights.write_bytes(b'\x00' * 2000)

        result = trainer.get_best_weights_path()
        assert result is not None
        assert result == best_weights

    def test_get_best_weights_picks_most_recent(self, temp_project_dir):
        import time
        trainer = ModelTrainer(temp_project_dir)

        # Create two training runs with different mtimes
        run1 = temp_project_dir / 'runs' / 'train' / 'old' / 'weights'
        run1.mkdir(parents=True)
        old_weights = run1 / 'best.pt'
        old_weights.write_bytes(b'\x00' * 2000)

        time.sleep(0.1)  # Ensure different mtime

        run2 = temp_project_dir / 'runs' / 'train' / 'new' / 'weights'
        run2.mkdir(parents=True)
        new_weights = run2 / 'best.pt'
        new_weights.write_bytes(b'\x00' * 2000)

        result = trainer.get_best_weights_path()
        assert result == new_weights

    def test_get_current_metrics_no_session(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)
        metrics = trainer.get_current_metrics()
        assert metrics == {}

    # === Pre-flight checks ===

    def test_preflight_disk_space_check(self, temp_project_dir):
        trainer = ModelTrainer(temp_project_dir)

        config = {
            'project': str(temp_project_dir),
            'device': 'cpu',
            'batch': 16,
            'imgsz': 640,
        }
        ok, error = trainer._preflight_checks(config)
        # Should pass on systems with disk space
        assert ok is True or "disk" in error.lower()
