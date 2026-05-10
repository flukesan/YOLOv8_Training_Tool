"""
Unit tests for core.dataset_manager
"""
import pytest
from pathlib import Path

from core.dataset_manager import DatasetManager


class TestDatasetManager:
    """Tests for DatasetManager"""

    def test_initialization(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        assert manager.project_path == temp_dataset_dir
        assert manager.classes == []
        assert manager.images_list == []

    def test_is_image_file_jpg(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        assert manager._is_image_file(Path("test.jpg")) is True
        assert manager._is_image_file(Path("test.JPG")) is True
        assert manager._is_image_file(Path("test.png")) is True
        assert manager._is_image_file(Path("test.txt")) is False

    def test_refresh_images_list(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        manager.refresh_images_list()
        assert len(manager.images_list) == 10
        # Verify sorted
        names = [p.name for p in manager.images_list]
        assert names == sorted(names)

    def test_split_dataset_valid_ratios(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        stats = manager.split_dataset(ratios=ratios, random_seed=42)

        assert stats['train'] == 7
        assert stats['val'] == 2
        assert stats['test'] == 1

    def test_split_dataset_invalid_ratios(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        ratios = {'train': 0.5, 'val': 0.3, 'test': 0.3}  # Sums to 1.1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            manager.split_dataset(ratios=ratios)

    def test_split_dataset_negative_ratio(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        ratios = {'train': 1.2, 'val': -0.1, 'test': -0.1}  # Sums to 1.0 but invalid
        with pytest.raises(ValueError):
            manager.split_dataset(ratios=ratios)

    def test_split_empty_dataset(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        with pytest.raises(ValueError, match="No images found"):
            manager.split_dataset()

    def test_split_too_few_images(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        # Add only 2 images (need at least 3)
        for i in range(2):
            (temp_dataset_dir / 'images' / f'img_{i}.jpg').write_bytes(b'\xff\xd8\xff\xe0')

        with pytest.raises(ValueError, match="at least 3 images"):
            manager.split_dataset()

    def test_split_reproducibility(self, temp_dataset_dir, sample_image_paths):
        """Same seed should produce same split"""
        manager1 = DatasetManager(temp_dataset_dir)
        stats1 = manager1.split_dataset(random_seed=42)

        # Get train images for first split
        train_dir = temp_dataset_dir / 'train' / 'images'
        train_images_1 = sorted([f.name for f in train_dir.iterdir()])

        # Reset by clearing splits
        import shutil
        for split in ['train', 'val', 'test']:
            split_dir = temp_dataset_dir / split / 'images'
            if split_dir.exists():
                for f in split_dir.iterdir():
                    f.unlink()

        manager2 = DatasetManager(temp_dataset_dir)
        stats2 = manager2.split_dataset(random_seed=42)
        train_images_2 = sorted([f.name for f in train_dir.iterdir()])

        # With same seed, splits should be identical
        assert train_images_1 == train_images_2
        assert stats1 == stats2

    def test_create_data_yaml(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        classes = ['Good', 'Defect']
        yaml_path = manager.create_data_yaml(classes)

        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert 'nc: 2' in content
        assert 'Good' in content
        assert 'Defect' in content

    def test_validate_empty_dataset(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        is_valid, issues = manager.validate_dataset()
        assert is_valid is False
        assert len(issues) > 0

    def test_get_statistics_empty(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)
        stats = manager.get_dataset_statistics()
        assert stats['total_images'] == 0
        assert stats['total_annotations'] == 0

    def test_get_statistics_with_data(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        manager.split_dataset(random_seed=42)

        # Add some labels
        for split in ['train', 'val']:
            label_dir = temp_dataset_dir / split / 'labels'
            for img_file in (temp_dataset_dir / split / 'images').iterdir():
                label_file = label_dir / f"{img_file.stem}.txt"
                label_file.write_text("0 0.5 0.5 0.2 0.3\n")

        stats = manager.get_dataset_statistics(class_names=['Good', 'Defect'])
        assert stats['total_images'] > 0
        assert stats['total_annotations'] > 0

    def test_get_images_without_annotations(self, temp_dataset_dir, sample_image_paths):
        manager = DatasetManager(temp_dataset_dir)
        manager.split_dataset(random_seed=42)

        # No labels exist yet
        unlabeled = manager.get_images_without_annotations('train')
        assert len(unlabeled) > 0

    def test_import_images_from_directory(self, temp_dataset_dir, tmp_path):
        # Create source directory with images
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(5):
            (source_dir / f"src_img_{i}.jpg").write_bytes(b'\xff\xd8\xff\xe0')

        # Also add a non-image file (should be ignored)
        (source_dir / "readme.txt").write_text("not an image")

        manager = DatasetManager(temp_dataset_dir)
        stats = manager.import_images([str(source_dir)], copy=True)

        assert stats['imported'] == 5
        # Source files should still exist (copy mode)
        assert (source_dir / "src_img_0.jpg").exists()

    def test_import_handles_duplicate_names(self, temp_dataset_dir):
        manager = DatasetManager(temp_dataset_dir)

        # Create same-named file twice
        source_dir = temp_dataset_dir.parent / "source_dup"
        source_dir.mkdir(exist_ok=True)
        (source_dir / "duplicate.jpg").write_bytes(b'\xff\xd8\xff\xe0')

        stats1 = manager.import_images([str(source_dir / "duplicate.jpg")])
        stats2 = manager.import_images([str(source_dir / "duplicate.jpg")])

        assert stats1['imported'] == 1
        assert stats2['imported'] == 1

        # Both files should exist with different names
        images = list((temp_dataset_dir / 'images').iterdir())
        assert len(images) == 2

        # Cleanup
        import shutil
        shutil.rmtree(source_dir, ignore_errors=True)
