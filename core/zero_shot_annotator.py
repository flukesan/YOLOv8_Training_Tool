"""
Zero-Shot Annotator - auto-generate YOLO annotations from text prompts using
Grounding DINO via the Autodistill framework.

This is an OPTIONAL feature. autodistill / autodistill-grounding-dino /
supervision are heavy dependencies (they pull in PyTorch + GroundingDINO
weights) and are intentionally NOT part of the default requirements.txt.
Install requirements-autoannotate.txt to enable this module. The import is
guarded so the rest of the application is unaffected when it is absent -
`is_available()` reports the status and callers should check it before
constructing `ZeroShotAnnotator`.
"""
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from core.logger import get_logger

logger = get_logger(__name__)

try:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO
    _AUTODISTILL_AVAILABLE = True
except ImportError:
    CaptionOntology = None
    GroundingDINO = None
    _AUTODISTILL_AVAILABLE = False


DEFAULT_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_available() -> bool:
    """Whether autodistill + Grounding DINO are installed."""
    return _AUTODISTILL_AVAILABLE


def get_availability_message() -> str:
    """Status / setup instructions, for display in the UI."""
    if _AUTODISTILL_AVAILABLE:
        return "Grounding DINO: Ready"
    return (
        "Grounding DINO is not installed.\n\n"
        "To enable zero-shot auto-annotation:\n"
        "  pip install -r requirements-autoannotate.txt\n\n"
        "This pulls in autodistill, autodistill-grounding-dino, and "
        "supervision (heavy: includes PyTorch + GroundingDINO weights, "
        "downloaded on first use). A CUDA GPU is strongly recommended."
    )


class ZeroShotAnnotator:
    """Auto-annotates a folder of images with Grounding DINO (zero-shot),
    producing a YOLO-format dataset: images/, labels/, data.yaml."""

    DEFAULT_BOX_THRESHOLD = 0.35
    DEFAULT_TEXT_THRESHOLD = 0.25
    DEFAULT_NMS_THRESHOLD = 0.5

    def __init__(self, ontology: Dict[str, str],
                box_threshold: float = DEFAULT_BOX_THRESHOLD,
                text_threshold: float = DEFAULT_TEXT_THRESHOLD):
        """
        Args:
            ontology: {caption prompt: YOLO class name}, e.g.
                {"iced coffee cup": "cup", "stainless steel straw": "straw"}
            box_threshold: confidence cutoff for keeping a detected box
            text_threshold: confidence that the caption matches what's seen
        """
        if not is_available():
            raise RuntimeError(get_availability_message())
        if not ontology:
            raise ValueError(
                "Ontology must have at least one caption -> class mapping")

        # Dict order == class index order used by CaptionOntology, and is
        # what we write into data.yaml, so detections.class_id lines up.
        self.class_names: List[str] = list(ontology.values())
        self._model = GroundingDINO(
            ontology=CaptionOntology(ontology),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    def run(self, source_dir: Path, output_dir: Path,
           use_nms: bool = True,
           nms_threshold: float = DEFAULT_NMS_THRESHOLD,
           class_agnostic_nms: bool = True,
           extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS,
           progress_callback: Optional[Callable[[int, int, str], None]] = None,
           should_stop: Optional[Callable[[], bool]] = None) -> Dict:
        """
        Annotate every image in source_dir and write a YOLO dataset.

        Args:
            source_dir: folder of input images (.jpg/.jpeg/.png by default)
            output_dir: destination - creates images/, labels/, data.yaml
            use_nms: remove overlapping duplicate boxes on the same object
            nms_threshold: IoU threshold above which boxes are considered
                duplicates
            class_agnostic_nms: suppress overlaps across different classes
                too (useful when prompts can double-fire on one object)
            extensions: file extensions to scan for in source_dir
            progress_callback: called as (current, total, message) after
                each image is processed
            should_stop: polled before each image; return True to abort

        Returns:
            stats dict: images_found, processed, annotated,
            skipped_no_detection, skipped_error, total_objects, errors,
            output_dir, classes
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        images_out = output_dir / 'images'
        labels_out = output_dir / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        image_paths = self._collect_images(source_dir, extensions)
        total = len(image_paths)

        stats = {
            'images_found': total,
            'processed': 0,
            'annotated': 0,
            'skipped_no_detection': 0,
            'skipped_error': 0,
            'total_objects': 0,
            'errors': [],
            'output_dir': str(output_dir),
            'classes': self.class_names,
        }

        for i, image_path in enumerate(image_paths, 1):
            if should_stop is not None and should_stop():
                logger.info("Auto-annotation stopped by user request")
                break

            message = ""
            try:
                detections = self._model.predict(str(image_path))

                if use_nms and len(detections) > 0:
                    detections = detections.with_nms(
                        threshold=nms_threshold,
                        class_agnostic=class_agnostic_nms,
                    )

                if len(detections) == 0:
                    stats['skipped_no_detection'] += 1
                    message = f"{image_path.name} -> 0 detections"
                else:
                    dest_image = self._copy_image(image_path, images_out)
                    label_lines = self._to_yolo_lines(detections, image_path)
                    if label_lines:
                        label_path = labels_out / f"{dest_image.stem}.txt"
                        with open(label_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(label_lines) + '\n')
                        stats['annotated'] += 1
                        stats['total_objects'] += len(label_lines)
                        message = (f"{image_path.name} -> {len(label_lines)} "
                                  f"object(s){' (NMS applied)' if use_nms else ''}")
                    else:
                        stats['skipped_no_detection'] += 1
                        message = f"{image_path.name} -> 0 detections"

                stats['processed'] += 1

            except Exception as e:
                stats['skipped_error'] += 1
                stats['errors'].append(f"{image_path.name}: {e}")
                logger.warning(f"Auto-annotate failed on {image_path.name}: {e}")
                message = f"skipped {image_path.name} - {e}"

            if progress_callback is not None:
                progress_callback(i, total, message)

        if stats['annotated'] > 0:
            self._write_data_yaml(output_dir)

        logger.info(
            f"Auto-annotation complete: {stats['annotated']}/{stats['images_found']} "
            f"images annotated, {stats['total_objects']} objects, "
            f"{stats['skipped_no_detection']} with no detections, "
            f"{stats['skipped_error']} errors"
        )
        return stats

    @staticmethod
    def _collect_images(source_dir: Path, extensions) -> List[Path]:
        """Flat (non-recursive) scan of source_dir for the given extensions."""
        seen = set()
        images = []
        for ext in extensions:
            for pattern in (f'*{ext}', f'*{ext.upper()}'):
                for path in source_dir.glob(pattern):
                    if path.is_file() and path not in seen:
                        seen.add(path)
                        images.append(path)
        return sorted(images)

    @staticmethod
    def _copy_image(src: Path, images_out: Path) -> Path:
        dest = images_out / src.name
        counter = 1
        while dest.exists():
            dest = images_out / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, dest)
        return dest

    @staticmethod
    def _to_yolo_lines(detections, image_path: Path) -> List[str]:
        """Convert a supervision Detections object (absolute pixel xyxy) to
        normalized 'class_id x_center y_center width height' lines."""
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            return []

        lines = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            if class_id is None:
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            if x2 <= x1 or y2 <= y1:
                continue

            xc = min(max(((x1 + x2) / 2) / width, 0.0), 1.0)
            yc = min(max(((y1 + y2) / 2) / height, 0.0), 1.0)
            w = min(max((x2 - x1) / width, 0.0), 1.0)
            h = min(max((y2 - y1) / height, 0.0), 1.0)
            lines.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return lines

    def _write_data_yaml(self, output_dir: Path) -> Path:
        data = {
            'path': str(output_dir.resolve()),
            'train': 'images',
            'val': 'images',
            'nc': len(self.class_names),
            'names': self.class_names,
        }
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return yaml_path
