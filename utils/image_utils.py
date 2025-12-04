"""
Image utilities for processing and manipulation
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load image using OpenCV"""
    img = cv2.imread(str(image_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, output_path: Path):
    """Save image using OpenCV"""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)


def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                keep_aspect: bool = True) -> np.ndarray:
    """Resize image"""
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def draw_bounding_box(image: np.ndarray, box: Tuple[int, int, int, int],
                     label: str, color: Tuple[int, int, int],
                     thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    x1, y1, x2, y2 = box
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, thickness)
    return img


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image width and height"""
    img = Image.open(image_path)
    return img.size


def apply_augmentation(image: np.ndarray, augmentation_type: str) -> np.ndarray:
    """Apply data augmentation"""
    if augmentation_type == 'flip_horizontal':
        return cv2.flip(image, 1)
    elif augmentation_type == 'flip_vertical':
        return cv2.flip(image, 0)
    elif augmentation_type == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif augmentation_type == 'brightness':
        return cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    return image
