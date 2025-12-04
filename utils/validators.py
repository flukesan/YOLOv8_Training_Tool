"""
Data validators
"""
from pathlib import Path
from typing import List, Tuple


def validate_image_path(path: Path) -> Tuple[bool, str]:
    """Validate image file path"""
    if not path.exists():
        return False, "File does not exist"
    if not path.is_file():
        return False, "Path is not a file"
    if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return False, "Unsupported image format"
    return True, ""


def validate_label_file(label_path: Path) -> Tuple[bool, List[str]]:
    """Validate YOLO label file"""
    errors = []
    if not label_path.exists():
        return False, ["File does not exist"]

    with open(label_path, 'r') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                errors.append(f"Line {i}: Invalid format")
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])

                if not (0 <= x <= 1 and 0 <= y <= 1):
                    errors.append(f"Line {i}: Coordinates out of range")
                if not (0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"Line {i}: Size out of range")
            except ValueError:
                errors.append(f"Line {i}: Invalid number format")

    return len(errors) == 0, errors


def validate_dataset_split(train_ratio: float, val_ratio: float,
                          test_ratio: float) -> Tuple[bool, str]:
    """Validate dataset split ratios"""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        return False, f"Ratios must sum to 1.0 (got {total})"
    if train_ratio <= 0:
        return False, "Train ratio must be positive"
    return True, ""
