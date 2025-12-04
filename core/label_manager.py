"""
Label Manager - handles annotation and labeling operations
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class BoundingBox:
    """Represents a bounding box annotation"""

    def __init__(self, class_id: int, x_center: float, y_center: float,
                 width: float, height: float, confidence: float = 1.0):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence

    def to_yolo_format(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_absolute(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized coordinates to absolute pixel coordinates
        Returns: (x1, y1, x2, y2)
        """
        x_center_abs = self.x_center * img_width
        y_center_abs = self.y_center * img_height
        width_abs = self.width * img_width
        height_abs = self.height * img_height

        x1 = int(x_center_abs - width_abs / 2)
        y1 = int(y_center_abs - height_abs / 2)
        x2 = int(x_center_abs + width_abs / 2)
        y2 = int(y_center_abs + height_abs / 2)

        return x1, y1, x2, y2

    @classmethod
    def from_absolute(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                     img_width: int, img_height: int):
        """Create BoundingBox from absolute pixel coordinates"""
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        return cls(class_id, x_center, y_center, width, height)

    @classmethod
    def from_yolo_format(cls, line: str):
        """Create BoundingBox from YOLO format line"""
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0

        return cls(class_id, x_center, y_center, width, height, confidence)

    def __repr__(self):
        return f"BoundingBox(class={self.class_id}, x={self.x_center:.3f}, y={self.y_center:.3f}, w={self.width:.3f}, h={self.height:.3f})"


class LabelManager:
    """Manages labels and annotations for images"""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.classes = []
        self.class_colors = {}

    def set_classes(self, classes: List[str]):
        """Set the list of class names"""
        self.classes = classes
        self._generate_class_colors()

    def _generate_class_colors(self):
        """Generate unique colors for each class"""
        random.seed(42)
        for i, class_name in enumerate(self.classes):
            # Generate distinct colors
            hue = (i * 137.5) % 360  # Golden angle for better distribution
            self.class_colors[i] = self._hsv_to_rgb(hue, 0.8, 0.9)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB color"""
        h = h / 60.0
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c

        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

    def load_annotations(self, label_path: Path) -> List[BoundingBox]:
        """Load annotations from a YOLO format label file"""
        if not label_path.exists():
            return []

        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                bbox = BoundingBox.from_yolo_format(line)
                if bbox:
                    annotations.append(bbox)

        return annotations

    def save_annotations(self, label_path: Path, annotations: List[BoundingBox]):
        """Save annotations to a YOLO format label file"""
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, 'w') as f:
            for bbox in annotations:
                f.write(bbox.to_yolo_format() + '\n')

    def add_annotation(self, label_path: Path, bbox: BoundingBox):
        """Add a new annotation to a label file"""
        annotations = self.load_annotations(label_path)
        annotations.append(bbox)
        self.save_annotations(label_path, annotations)

    def delete_annotation(self, label_path: Path, index: int) -> bool:
        """Delete an annotation by index"""
        annotations = self.load_annotations(label_path)
        if 0 <= index < len(annotations):
            annotations.pop(index)
            self.save_annotations(label_path, annotations)
            return True
        return False

    def update_annotation(self, label_path: Path, index: int, bbox: BoundingBox) -> bool:
        """Update an existing annotation"""
        annotations = self.load_annotations(label_path)
        if 0 <= index < len(annotations):
            annotations[index] = bbox
            self.save_annotations(label_path, annotations)
            return True
        return False

    def get_annotation_stats(self, label_path: Path) -> Dict:
        """Get statistics about annotations in a label file"""
        annotations = self.load_annotations(label_path)

        stats = {
            'total': len(annotations),
            'by_class': {}
        }

        for bbox in annotations:
            class_id = bbox.class_id
            stats['by_class'][class_id] = stats['by_class'].get(class_id, 0) + 1

        return stats

    def validate_annotation(self, bbox: BoundingBox) -> Tuple[bool, Optional[str]]:
        """
        Validate a bounding box annotation
        Returns: (is_valid, error_message)
        """
        # Check if coordinates are in valid range [0, 1]
        if not (0 <= bbox.x_center <= 1 and 0 <= bbox.y_center <= 1):
            return False, "Center coordinates must be between 0 and 1"

        if not (0 < bbox.width <= 1 and 0 < bbox.height <= 1):
            return False, "Width and height must be between 0 and 1"

        # Check if class_id is valid
        if bbox.class_id < 0 or bbox.class_id >= len(self.classes):
            return False, f"Invalid class ID: {bbox.class_id}"

        # Check if box is too small (might be noise)
        if bbox.width < 0.001 or bbox.height < 0.001:
            return False, "Bounding box is too small"

        return True, None

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get RGB color for a class"""
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        return (255, 255, 255)  # Default white

    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"Unknown ({class_id})"

    def auto_annotate(self, image_path: Path, model_path: Path,
                     confidence_threshold: float = 0.25) -> List[BoundingBox]:
        """
        Auto-annotate an image using a pre-trained model
        Args:
            image_path: Path to image
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        Returns:
            List of detected bounding boxes
        """
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            results = model(str(image_path), conf=confidence_threshold, verbose=False)

            annotations = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates (normalized)
                        x_center = float(box.xywhn[0][0])
                        y_center = float(box.xywhn[0][1])
                        width = float(box.xywhn[0][2])
                        height = float(box.xywhn[0][3])
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        bbox = BoundingBox(class_id, x_center, y_center,
                                          width, height, confidence)
                        annotations.append(bbox)

            return annotations
        except Exception as e:
            print(f"Error in auto-annotation: {e}")
            return []

    def copy_annotations(self, from_image: Path, to_images: List[Path]):
        """Copy annotations from one image to multiple images"""
        # Determine label path
        from_label = from_image.parent.parent / 'labels' / f"{from_image.stem}.txt"

        if not from_label.exists():
            return

        annotations = self.load_annotations(from_label)

        for to_image in to_images:
            to_label = to_image.parent.parent / 'labels' / f"{to_image.stem}.txt"
            self.save_annotations(to_label, annotations)

    def merge_annotations(self, label_paths: List[Path], output_path: Path):
        """Merge annotations from multiple files into one"""
        all_annotations = []

        for label_path in label_paths:
            annotations = self.load_annotations(label_path)
            all_annotations.extend(annotations)

        self.save_annotations(output_path, all_annotations)

    def filter_annotations_by_class(self, label_path: Path,
                                   allowed_classes: List[int]) -> List[BoundingBox]:
        """Filter annotations to only include specific classes"""
        annotations = self.load_annotations(label_path)
        filtered = [bbox for bbox in annotations if bbox.class_id in allowed_classes]
        return filtered

    def remap_classes(self, label_path: Path, class_mapping: Dict[int, int]):
        """
        Remap class IDs in annotations
        Args:
            label_path: Path to label file
            class_mapping: Dictionary mapping old class IDs to new ones
        """
        annotations = self.load_annotations(label_path)

        for bbox in annotations:
            if bbox.class_id in class_mapping:
                bbox.class_id = class_mapping[bbox.class_id]

        self.save_annotations(label_path, annotations)
