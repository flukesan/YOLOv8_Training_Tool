"""
Dataset Manager - handles dataset operations
"""
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from config.settings import Settings


class DatasetManager:
    """Manages dataset operations including import, split, and organization"""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.structure = Settings.get_project_structure(self.project_path)
        self.classes = []
        self.images_list = []

    def import_images(self, source_paths: List[str], copy: bool = True) -> Dict[str, int]:
        """
        Import images from source paths
        Args:
            source_paths: List of file or directory paths
            copy: Whether to copy files (True) or move them (False)
        Returns:
            Dictionary with import statistics
        """
        stats = {'imported': 0, 'skipped': 0, 'errors': 0}
        images_dir = self.structure['images']
        images_dir.mkdir(parents=True, exist_ok=True)

        for source in source_paths:
            source_path = Path(source)

            if source_path.is_file():
                if self._is_image_file(source_path):
                    if self._import_single_image(source_path, images_dir, copy):
                        stats['imported'] += 1
                    else:
                        stats['errors'] += 1
                else:
                    stats['skipped'] += 1

            elif source_path.is_dir():
                for img_file in source_path.rglob('*'):
                    if img_file.is_file() and self._is_image_file(img_file):
                        if self._import_single_image(img_file, images_dir, copy):
                            stats['imported'] += 1
                        else:
                            stats['errors'] += 1

        self.refresh_images_list()
        return stats

    def _import_single_image(self, source: Path, dest_dir: Path, copy: bool) -> bool:
        """Import a single image file"""
        try:
            dest_path = dest_dir / source.name

            # Handle duplicate names
            counter = 1
            while dest_path.exists():
                stem = source.stem
                suffix = source.suffix
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            if copy:
                shutil.copy2(source, dest_path)
            else:
                shutil.move(str(source), dest_path)

            return True
        except Exception as e:
            print(f"Error importing {source}: {e}")
            return False

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image format"""
        return file_path.suffix.lower() in Settings.IMAGE_FORMATS

    def split_dataset(self, ratios: Dict[str, float] = None,
                     random_seed: int = 42) -> Dict[str, int]:
        """
        Split dataset into train/val/test sets
        Args:
            ratios: Dictionary with 'train', 'val', 'test' ratios
            random_seed: Random seed for reproducibility
        Returns:
            Dictionary with split counts
        """
        if ratios is None:
            ratios = Settings.DATASET_SPLIT

        random.seed(random_seed)

        images_dir = self.structure['images']
        labels_dir = self.structure['labels']

        # Get all images
        images = [f for f in images_dir.iterdir() if self._is_image_file(f)]

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        total = len(images)
        train_end = int(total * ratios['train'])
        val_end = train_end + int(total * ratios['val'])

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        stats = {}

        # Move files to respective directories
        for split_name, split_images in splits.items():
            split_img_dir = self.structure[split_name] / 'images'
            split_lbl_dir = self.structure[split_name] / 'labels'

            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in split_images:
                # Move image
                dest_img = split_img_dir / img_file.name
                shutil.copy2(img_file, dest_img)

                # Move corresponding label if exists
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_lbl = split_lbl_dir / label_file.name
                    shutil.copy2(label_file, dest_lbl)

            stats[split_name] = len(split_images)

        return stats

    def refresh_images_list(self):
        """Refresh the list of images in the dataset"""
        images_dir = self.structure['images']
        if images_dir.exists():
            self.images_list = sorted([
                f for f in images_dir.iterdir()
                if self._is_image_file(f)
            ])
        else:
            self.images_list = []

    def get_dataset_statistics(self, class_names: List[str] = None) -> Dict:
        """Get statistics about the dataset

        Args:
            class_names: List of class names (optional)
        """
        stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'total_annotations': 0,
            'classes': {},
            'images_per_class': {}
        }

        # Count images in each split
        for split in ['train', 'val', 'test']:
            split_dir = self.structure[split] / 'images'
            if split_dir.exists():
                count = len([f for f in split_dir.iterdir() if self._is_image_file(f)])
                stats[f'{split}_images'] = count
                stats['total_images'] += count

        # Count annotations and class distribution
        self._count_annotations(stats, class_names)

        return stats

    def _count_annotations(self, stats: Dict, class_names: List[str] = None):
        """Count annotations and class distribution

        Args:
            stats: Statistics dictionary to update
            class_names: List of class names (optional)
        """
        for split in ['train', 'val', 'test']:
            labels_dir = self.structure[split] / 'labels'
            if not labels_dir.exists():
                continue

            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    stats['total_annotations'] += len(lines)

                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])

                            # Initialize class entry if not exists
                            if class_id not in stats['classes']:
                                stats['classes'][class_id] = {
                                    'count': 0,
                                    'name': class_names[class_id] if class_names and class_id < len(class_names) else f'Class {class_id}'
                                }

                            stats['classes'][class_id]['count'] += 1

    def create_data_yaml(self, classes: List[str], save_path: Path = None) -> Path:
        """
        Create data.yaml configuration file for YOLO training
        Args:
            classes: List of class names
            save_path: Where to save the yaml file
        Returns:
            Path to the created yaml file
        """
        if save_path is None:
            save_path = self.structure['config']

        data_config = {
            'path': str(self.project_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }

        with open(save_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        return save_path

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset structure and files
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if splits exist
        for split in ['train', 'val']:
            split_img_dir = self.structure[split] / 'images'
            if not split_img_dir.exists() or not any(split_img_dir.iterdir()):
                issues.append(f"No images found in {split} split")

        # Check if config.yaml exists
        if not self.structure['config'].exists():
            issues.append("data.yaml configuration file not found")

        # Check for orphaned labels (labels without images)
        for split in ['train', 'val', 'test']:
            self._check_orphaned_files(split, issues)

        return len(issues) == 0, issues

    def _check_orphaned_files(self, split: str, issues: List[str]):
        """Check for orphaned labels or images"""
        img_dir = self.structure[split] / 'images'
        lbl_dir = self.structure[split] / 'labels'

        if not img_dir.exists() or not lbl_dir.exists():
            return

        # Get image and label stems
        img_stems = {f.stem for f in img_dir.iterdir() if self._is_image_file(f)}
        lbl_stems = {f.stem for f in lbl_dir.iterdir() if f.suffix == '.txt'}

        # Find orphaned labels
        orphaned_labels = lbl_stems - img_stems
        if orphaned_labels:
            issues.append(f"{split}: {len(orphaned_labels)} labels without corresponding images")

    def get_images_without_annotations(self, split: str = 'train') -> List[Path]:
        """Get list of images that don't have annotations"""
        img_dir = self.structure[split] / 'images'
        lbl_dir = self.structure[split] / 'labels'

        if not img_dir.exists():
            return []

        images_without_labels = []
        for img_file in img_dir.iterdir():
            if self._is_image_file(img_file):
                label_file = lbl_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    images_without_labels.append(img_file)

        return images_without_labels

    def auto_organize(self, source_dir: Path) -> Dict[str, int]:
        """
        Auto-organize dataset from various formats
        Args:
            source_dir: Source directory containing images and possibly labels
        Returns:
            Statistics dictionary
        """
        source_dir = Path(source_dir)
        stats = {'images': 0, 'labels': 0}

        # Import images
        import_stats = self.import_images([str(source_dir)])
        stats['images'] = import_stats['imported']

        # Try to import labels if they exist
        labels_src = source_dir / 'labels'
        if labels_src.exists():
            labels_dest = self.structure['labels']
            labels_dest.mkdir(parents=True, exist_ok=True)

            for label_file in labels_src.glob('*.txt'):
                shutil.copy2(label_file, labels_dest / label_file.name)
                stats['labels'] += 1

        return stats
