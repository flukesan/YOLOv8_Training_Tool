"""
Integration tests for the full dataset preparation workflow
"""
import pytest
import yaml
from pathlib import Path

from core.dataset_manager import DatasetManager
from core.label_manager import LabelManager, BoundingBox


@pytest.mark.integration
class TestDatasetWorkflow:
    """End-to-end tests for dataset preparation workflow"""

    def test_full_workflow_import_split_validate(self, tmp_path):
        """Test: Import images -> Create labels -> Split -> Validate"""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Setup project structure
        from config.settings import Settings
        Settings.create_project_structure(project_dir)

        # 1. Create source images
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        for i in range(20):
            (source_dir / f"img_{i:03d}.jpg").write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)

        # 2. Import images
        ds_manager = DatasetManager(project_dir)
        import_stats = ds_manager.import_images([str(source_dir)])
        assert import_stats['imported'] == 20

        # 3. Create labels for some images
        label_manager = LabelManager(project_dir)
        label_manager.set_classes(['Good', 'Defect'])

        labels_dir = project_dir / 'labels'
        for i in range(15):  # Label first 15
            label_file = labels_dir / f"img_{i:03d}.txt"
            bbox = BoundingBox(i % 2, 0.5, 0.5, 0.2, 0.3)
            label_manager.save_annotations(label_file, [bbox])

        # 4. Split dataset
        split_stats = ds_manager.split_dataset(
            ratios={'train': 0.7, 'val': 0.2, 'test': 0.1},
            random_seed=42
        )
        assert split_stats['train'] == 14
        assert split_stats['val'] == 4
        assert split_stats['test'] == 2

        # 5. Create data.yaml
        yaml_path = ds_manager.create_data_yaml(['Good', 'Defect'])
        assert yaml_path.exists()

        # Verify yaml content
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data['nc'] == 2
        assert data['names'] == ['Good', 'Defect']

        # 6. Validate dataset
        is_valid, issues = ds_manager.validate_dataset()
        # Should be valid since we have train and val
        assert is_valid is True or len(issues) == 0

        # 7. Get statistics
        stats = ds_manager.get_dataset_statistics(['Good', 'Defect'])
        assert stats['total_images'] == 20
        assert stats['total_annotations'] > 0

    def test_workflow_handles_missing_labels(self, tmp_path):
        """Workflow should handle images without labels gracefully"""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        from config.settings import Settings
        Settings.create_project_structure(project_dir)

        # Create images but no labels
        for i in range(10):
            img = project_dir / 'images' / f"img_{i}.jpg"
            img.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)

        ds_manager = DatasetManager(project_dir)
        # Should work even without labels
        ds_manager.split_dataset(random_seed=42)

        # Check unlabeled images
        unlabeled = ds_manager.get_images_without_annotations('train')
        assert len(unlabeled) > 0

    def test_workflow_reproducibility(self, tmp_path):
        """Same inputs + seed = same outputs"""
        # Setup two identical projects
        for run in [1, 2]:
            project_dir = tmp_path / f"project_{run}"
            project_dir.mkdir()

            from config.settings import Settings
            Settings.create_project_structure(project_dir)

            # Create same images
            for i in range(15):
                (project_dir / 'images' / f"img_{i:03d}.jpg").write_bytes(
                    b'\xff\xd8\xff\xe0' + b'\x00' * 100
                )

        # Split both with same seed
        ds_manager_1 = DatasetManager(tmp_path / "project_1")
        ds_manager_2 = DatasetManager(tmp_path / "project_2")

        stats_1 = ds_manager_1.split_dataset(random_seed=42)
        stats_2 = ds_manager_2.split_dataset(random_seed=42)

        # Same seed should produce same split sizes
        assert stats_1 == stats_2

        # And same images should be in same splits
        train_1 = sorted([f.name for f in (tmp_path / "project_1" / 'train' / 'images').iterdir()])
        train_2 = sorted([f.name for f in (tmp_path / "project_2" / 'train' / 'images').iterdir()])
        assert train_1 == train_2

    def test_workflow_with_corrupted_labels(self, tmp_path):
        """Workflow should be resilient to corrupted label files"""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        from config.settings import Settings
        Settings.create_project_structure(project_dir)

        # Create images
        for i in range(5):
            (project_dir / 'images' / f"img_{i}.jpg").write_bytes(b'\xff\xd8\xff\xe0')

        # Create some valid and some corrupted labels
        label_manager = LabelManager(project_dir)
        label_manager.set_classes(['A', 'B'])

        # Valid label
        valid_file = project_dir / 'labels' / 'img_0.txt'
        valid_file.write_text("0 0.5 0.5 0.2 0.3\n")

        # Corrupted labels
        corrupt_file = project_dir / 'labels' / 'img_1.txt'
        corrupt_file.write_text("invalid garbage data\nthis is not yolo format\n")

        # Mix of valid and invalid
        mixed_file = project_dir / 'labels' / 'img_2.txt'
        mixed_file.write_text(
            "0 0.5 0.5 0.2 0.3\n"  # Valid
            "abc def ghi jkl mno\n"  # Invalid
            "1 0.3 0.7 0.1 0.15\n"  # Valid
        )

        # Should not crash on any of these
        valid_anns = label_manager.load_annotations(valid_file)
        corrupt_anns = label_manager.load_annotations(corrupt_file)
        mixed_anns = label_manager.load_annotations(mixed_file)

        assert len(valid_anns) == 1
        assert len(corrupt_anns) == 0  # All invalid - none loaded
        assert len(mixed_anns) == 2  # Only valid lines


@pytest.mark.integration
class TestLoggingAndCrashHandling:
    """Integration test for logging + crash handling"""

    def test_logger_and_crash_handler_work_together(self, tmp_path):
        from core.logger import AppLogger, get_logger
        from core.crash_handler import CrashHandler
        import sys

        # Initialize both
        AppLogger._initialized = False  # Force re-init
        AppLogger.initialize(log_dir=tmp_path / "logs")
        CrashHandler.initialize(crash_dir=tmp_path / "crashes")

        logger = get_logger('integration_test')
        logger.info("Starting integration test")
        logger.error("Test error message")

        # Generate a crash report
        try:
            raise RuntimeError("Integration test exception")
        except RuntimeError:
            CrashHandler.generate_crash_report(*sys.exc_info())

        # Verify both logs and crash reports exist
        log_file = tmp_path / "logs" / "app.log"
        assert log_file.exists()

        crashes = CrashHandler.get_recent_crashes()
        assert len(crashes) >= 1


@pytest.mark.integration
class TestAutoSaveIntegration:
    """Integration tests for auto-save with real state"""

    def test_save_and_recover_app_state(self, tmp_path):
        from core.auto_save import AutoSaveManager

        # Simulate app state
        app_state = {
            'window_geometry': [100, 100, 1600, 900],
            'last_project': '/path/to/project',
            'recent_files': ['file1.yaml', 'file2.yaml'],
        }

        training_state = {
            'in_progress': True,
            'current_epoch': 42,
            'best_metric': 0.95,
        }

        # First session - save state
        manager = AutoSaveManager(save_dir=tmp_path, interval=60)
        manager.register_state_provider('app', lambda: app_state)
        manager.register_state_provider('training', lambda: training_state)
        manager.save_now()

        # Simulate restart - new manager
        manager2 = AutoSaveManager(save_dir=tmp_path)

        assert manager2.has_recovery_data() is True

        recovery = manager2.load_recovery_data()
        assert recovery['data']['app']['window_geometry'] == [100, 100, 1600, 900]
        assert recovery['data']['training']['current_epoch'] == 42

        # After successful recovery, clear data
        manager2.clear_recovery_data()
        assert manager2.has_recovery_data() is False
