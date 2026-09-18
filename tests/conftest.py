"""
Shared pytest fixtures for all tests
"""
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add project root to path so 'core' and 'config' are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for tests"""
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dataset_dir(temp_project_dir):
    """Create a temporary dataset structure"""
    structure = {
        'images': temp_project_dir / 'images',
        'labels': temp_project_dir / 'labels',
        'train': temp_project_dir / 'train',
        'val': temp_project_dir / 'val',
        'test': temp_project_dir / 'test',
    }

    for split_dir in [structure['train'], structure['val'], structure['test']]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

    structure['images'].mkdir(parents=True, exist_ok=True)
    structure['labels'].mkdir(parents=True, exist_ok=True)

    return temp_project_dir


@pytest.fixture
def sample_image_paths(temp_dataset_dir):
    """Create sample image files for testing"""
    images_dir = temp_dataset_dir / 'images'
    image_paths = []

    for i in range(10):
        img_path = images_dir / f'img_{i:03d}.jpg'
        # Create empty file with .jpg extension
        img_path.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)  # Minimal JPEG header
        image_paths.append(img_path)

    return image_paths


@pytest.fixture
def sample_label_content():
    """Sample valid YOLO label content"""
    return "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.15\n"


@pytest.fixture
def sample_label_file(tmp_path, sample_label_content):
    """Create a sample label file"""
    label_file = tmp_path / "test.txt"
    label_file.write_text(sample_label_content)
    return label_file


@pytest.fixture
def invalid_label_content():
    """Invalid YOLO label content for testing error handling"""
    return "0 abc 0.5 0.2 0.3\n-1 0.3 0.7 0.1 0.15\n0 0.5 0.5 1.5 0.3\n"


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging output during tests to keep output clean"""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
