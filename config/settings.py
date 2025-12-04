"""
Application settings and paths configuration
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Settings:
    """Application settings manager"""

    # Application info
    APP_NAME = "YOLOv8 Training Tool"
    APP_VERSION = "1.0.0"

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    RESOURCES_DIR = BASE_DIR / "resources"
    ICONS_DIR = RESOURCES_DIR / "icons"
    STYLES_DIR = RESOURCES_DIR / "styles"

    # Default directories
    DEFAULT_PROJECTS_DIR = Path.home() / "YOLOv8_Projects"
    DEFAULT_MODELS_DIR = Path.home() / "YOLOv8_Models"

    # Training defaults (Updated for Ultralytics 8.3.x API)
    DEFAULT_TRAIN_PARAMS = {
        'epochs': 100,
        'batch': 16,              # Changed from 'batch_size'
        'imgsz': 640,             # Changed from 'img_size'
        'patience': 50,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'optimizer': 'SGD',
        'cos_lr': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'single_cls': False,
        'rect': False,
        'resume': False,
        'cache': None,
        'device': '',
        'workers': 8,
        'project': None,
        'name': None,
        'exist_ok': False,
        'save_period': -1,
        'seed': 0,
        # Removed deprecated parameters: nosave, noval, noautoanchor, noplots,
        # evolve, bucket, image_weights, quad, linear_lr, local_rank, label_smoothing
    }

    # Dataset split ratios
    DATASET_SPLIT = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1
    }

    # Supported image formats
    IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']

    # Supported annotation formats
    ANNOTATION_FORMATS = ['yolo', 'coco', 'voc', 'labelme']

    # Export formats
    EXPORT_FORMATS = ['pt', 'onnx', 'tflite', 'torchscript', 'coreml', 'tfjs']

    # UI settings
    UI_SETTINGS = {
        'window_width': 1600,
        'window_height': 900,
        'theme': 'dark',
        'auto_save': True,
        'auto_save_interval': 300,  # seconds
        'show_confidence': True,
        'default_confidence': 0.25,
        'show_labels': True,
        'box_thickness': 2,
        'font_scale': 0.5,
    }

    # Colors for different classes (will be generated dynamically)
    CLASS_COLORS = {}

    # Keyboard shortcuts
    SHORTCUTS = {
        'save': 'Ctrl+S',
        'open': 'Ctrl+O',
        'new_project': 'Ctrl+N',
        'delete_annotation': 'Delete',
        'next_image': 'D',
        'prev_image': 'A',
        'zoom_in': 'Ctrl++',
        'zoom_out': 'Ctrl+-',
        'reset_zoom': 'Ctrl+0',
        'copy_annotation': 'Ctrl+C',
        'paste_annotation': 'Ctrl+V',
        'undo': 'Ctrl+Z',
        'redo': 'Ctrl+Y',
        'start_training': 'F5',
        'stop_training': 'Shift+F5',
    }

    # YOLO model variants
    YOLO_MODELS = {
        'YOLOv8n': 'yolov8n.pt',
        'YOLOv8s': 'yolov8s.pt',
        'YOLOv8m': 'yolov8m.pt',
        'YOLOv8l': 'yolov8l.pt',
        'YOLOv8x': 'yolov8x.pt',
    }

    @classmethod
    def load_settings(cls, config_path: str = None) -> Dict[str, Any]:
        """Load settings from config file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    @classmethod
    def save_settings(cls, settings: Dict[str, Any], config_path: str):
        """Save settings to config file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(settings, f, default_flow_style=False)

    @classmethod
    def create_default_directories(cls):
        """Create default application directories"""
        cls.DEFAULT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        cls.ICONS_DIR.mkdir(parents=True, exist_ok=True)
        cls.STYLES_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_project_structure(cls, project_path: Path) -> Dict[str, Path]:
        """Get project directory structure"""
        return {
            'root': project_path,
            'images': project_path / 'images',
            'labels': project_path / 'labels',
            'train': project_path / 'train',
            'val': project_path / 'val',
            'test': project_path / 'test',
            'models': project_path / 'models',
            'runs': project_path / 'runs',
            'config': project_path / 'config.yaml',
        }

    @classmethod
    def create_project_structure(cls, project_path: Path):
        """Create project directory structure"""
        structure = cls.get_project_structure(project_path)

        # Create directories
        for key, path in structure.items():
            if key != 'config':
                path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for train/val/test
        for split in ['train', 'val', 'test']:
            (structure[split] / 'images').mkdir(parents=True, exist_ok=True)
            (structure[split] / 'labels').mkdir(parents=True, exist_ok=True)


# Initialize default directories on import
Settings.create_default_directories()
