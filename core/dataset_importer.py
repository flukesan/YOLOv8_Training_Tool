"""
Dataset Importer - import external datasets (COCO JSON / YOLO folder exports,
e.g. from FiftyOne) into a project.

Copies images into project/images, converts+remaps annotations into YOLO txt
files in project/labels, and merges the source classes into the project's
class list by name (case-insensitive). Existing class indices are preserved;
unknown classes are appended.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from config.settings import Settings
from core.logger import get_logger

logger = get_logger(__name__)


def detect_format(source_dir: Path) -> Optional[str]:
    """Best-effort format detection: 'coco' or 'yolo' (None if unknown)."""
    source_dir = Path(source_dir)
    if _find_coco_json(source_dir):
        return 'coco'
    if _find_yolo_yaml(source_dir) or _find_labels_dir(source_dir):
        return 'yolo'
    return None


def _find_coco_json(source_dir: Path) -> Optional[Path]:
    """Find a COCO annotations json (has images/annotations/categories keys)."""
    candidates = sorted(source_dir.rglob('*.json'))
    for path in candidates[:20]:  # avoid scanning huge trees exhaustively
        try:
            with open(path, 'r', encoding='utf-8') as f:
                head = f.read(4096)
            # Cheap check before full parse
            if '"images"' in head and '"annotations"' in head:
                return path
        except OSError:
            continue
    return None


def _find_yolo_yaml(source_dir: Path) -> Optional[Path]:
    """Find dataset.yaml/data.yaml (or any yaml with a 'names' key)."""
    named = [source_dir / 'dataset.yaml', source_dir / 'data.yaml']
    others = sorted(source_dir.glob('*.yaml')) + sorted(source_dir.glob('*.yml'))
    for path in named + others:
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and 'names' in data:
                return path
        except (OSError, yaml.YAMLError):
            continue
    return None


def _find_labels_dir(source_dir: Path) -> Optional[Path]:
    """Find a labels/ directory containing .txt files."""
    for candidate in sorted(source_dir.rglob('labels')):
        if candidate.is_dir() and any(candidate.rglob('*.txt')):
            return candidate
    return None


def _yaml_names_to_list(names) -> List[str]:
    """YOLO yaml 'names' can be a list or an {index: name} dict."""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in (names or [])]


class DatasetImporter:
    """Imports a COCO or YOLO dataset folder into the project."""

    def __init__(self, project_path: Path, existing_classes: List[str]):
        self.project_path = Path(project_path)
        self.existing_classes = list(existing_classes)
        self.structure = Settings.get_project_structure(self.project_path)

    # ------------------------------------------------------------------ scan
    def scan(self, source_dir: Path, fmt: Optional[str] = None) -> Dict:
        """Inspect the source folder without importing.

        Returns {'format', 'classes', 'image_count', 'annotation_count'}.
        Raises ValueError if the folder is not a recognisable dataset.
        """
        source_dir = Path(source_dir)
        fmt = fmt or detect_format(source_dir)
        if fmt == 'coco':
            return self._scan_coco(source_dir)
        if fmt == 'yolo':
            return self._scan_yolo(source_dir)
        raise ValueError(
            "Could not detect a COCO json or YOLO dataset.yaml/labels folder "
            f"in: {source_dir}"
        )

    def _scan_coco(self, source_dir: Path) -> Dict:
        json_path = _find_coco_json(source_dir)
        if not json_path:
            raise ValueError("No COCO annotations .json found")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        classes = [str(c['name']) for c in
                   sorted(data.get('categories', []), key=lambda c: c['id'])]
        return {
            'format': 'coco',
            'json_path': json_path,
            'classes': classes,
            'image_count': len(data.get('images', [])),
            'annotation_count': len(data.get('annotations', [])),
        }

    def _scan_yolo(self, source_dir: Path) -> Dict:
        yaml_path = _find_yolo_yaml(source_dir)
        classes: List[str] = []
        if yaml_path:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                classes = _yaml_names_to_list(yaml.safe_load(f).get('names'))
        pairs = self._collect_yolo_pairs(source_dir)
        ann_count = sum(1 for _, lbl in pairs if lbl is not None)
        if not classes:
            # Derive class ids from the label files themselves
            max_id = -1
            for _, lbl in pairs:
                if lbl is None:
                    continue
                try:
                    with open(lbl, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                max_id = max(max_id, int(float(parts[0])))
                except (OSError, ValueError):
                    continue
            classes = [f"class_{i}" for i in range(max_id + 1)]
        return {
            'format': 'yolo',
            'classes': classes,
            'image_count': len(pairs),
            'annotation_count': ann_count,
        }

    @staticmethod
    def _collect_yolo_pairs(source_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
        """Find (image, label-or-None) pairs, handling images/-labels/ trees
        (FiftyOne YOLOv5 exports) and flat side-by-side layouts."""
        pairs = []
        seen = set()
        for ext in Settings.IMAGE_FORMATS:
            for pattern in (f'*{ext}', f'*{ext.upper()}'):
                for img in source_dir.rglob(pattern):
                    if not img.is_file() or img in seen:
                        continue
                    seen.add(img)
                    label = None
                    parts = list(img.parts)
                    if 'images' in parts:
                        idx = len(parts) - 1 - parts[::-1].index('images')
                        cand = Path(*parts[:idx], 'labels',
                                    *parts[idx + 1:]).with_suffix('.txt')
                        if cand.exists():
                            label = cand
                    if label is None:
                        cand = img.with_suffix('.txt')
                        if cand.exists():
                            label = cand
                    pairs.append((img, label))
        return pairs

    # ---------------------------------------------------------------- import
    def build_class_mapping(self, source_classes: List[str],
                            selected: Optional[List[str]] = None
                            ) -> Tuple[List[str], Dict[int, int]]:
        """Merge source classes into the project class list by name.

        Returns (merged_class_list, {source_index: merged_index}). Source
        classes not in `selected` (when given) are omitted from the mapping,
        which drops their annotations at import time.
        """
        merged = list(self.existing_classes)
        lower_index = {name.lower(): i for i, name in enumerate(merged)}
        selected_lower = ({s.lower() for s in selected}
                          if selected is not None else None)

        mapping: Dict[int, int] = {}
        for src_idx, name in enumerate(source_classes):
            if selected_lower is not None and name.lower() not in selected_lower:
                continue
            key = name.lower()
            if key not in lower_index:
                lower_index[key] = len(merged)
                merged.append(name)
            mapping[src_idx] = lower_index[key]
        return merged, mapping

    def import_dataset(self, source_dir: Path, fmt: Optional[str] = None,
                       selected_classes: Optional[List[str]] = None,
                       filename_prefix: str = '',
                       include_unlabeled: bool = False) -> Dict:
        """Run the import. Returns stats including the merged class list."""
        source_dir = Path(source_dir)
        info = self.scan(source_dir, fmt)
        merged_classes, mapping = self.build_class_mapping(
            info['classes'], selected_classes)

        images_dir = self.structure['images']
        labels_dir = self.structure['labels']
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stats = {'imported': 0, 'skipped': 0, 'errors': 0,
                 'annotations': 0, 'classes': merged_classes}

        if info['format'] == 'coco':
            items = self._iter_coco_items(source_dir, info['json_path'], mapping)
        else:
            items = self._iter_yolo_items(source_dir, mapping)

        for src_image, label_lines in items:
            if not label_lines and not include_unlabeled:
                stats['skipped'] += 1
                continue
            try:
                dest = self._unique_dest(images_dir, src_image, filename_prefix)
                shutil.copy2(src_image, dest)
                if label_lines:
                    label_path = labels_dir / f"{dest.stem}.txt"
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(label_lines) + '\n')
                    stats['annotations'] += len(label_lines)
                stats['imported'] += 1
            except OSError as e:
                logger.error(f"Failed to import {src_image}: {e}")
                stats['errors'] += 1

        logger.info(
            f"Dataset import from {source_dir}: {stats['imported']} images, "
            f"{stats['annotations']} annotations, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )
        return stats

    @staticmethod
    def _unique_dest(images_dir: Path, src_image: Path, prefix: str) -> Path:
        stem = f"{prefix}{src_image.stem}" if prefix else src_image.stem
        dest = images_dir / f"{stem}{src_image.suffix.lower()}"
        counter = 1
        while dest.exists():
            dest = images_dir / f"{stem}_{counter}{src_image.suffix.lower()}"
            counter += 1
        return dest

    def _iter_coco_items(self, source_dir: Path, json_path: Path,
                         mapping: Dict[int, int]):
        """Yield (image_path, [yolo label lines]) from a COCO export."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # COCO category ids are arbitrary; map them through their position in
        # the sorted category list (same order used by _scan_coco).
        cats = sorted(data.get('categories', []), key=lambda c: c['id'])
        catid_to_srcidx = {c['id']: i for i, c in enumerate(cats)}

        anns_by_image: Dict[int, list] = {}
        for ann in data.get('annotations', []):
            if ann.get('iscrowd'):
                continue
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        for img_info in data.get('images', []):
            file_name = img_info.get('file_name', '')
            img_path = self._resolve_coco_image(source_dir, json_path, file_name)
            if img_path is None:
                logger.warning(f"COCO image not found: {file_name}")
                continue

            width = img_info.get('width') or 0
            height = img_info.get('height') or 0
            if not width or not height:
                logger.warning(f"COCO image missing size, skipped: {file_name}")
                continue

            lines = []
            for ann in anns_by_image.get(img_info['id'], []):
                src_idx = catid_to_srcidx.get(ann.get('category_id'))
                if src_idx is None or src_idx not in mapping:
                    continue
                x, y, w, h = ann.get('bbox', (0, 0, 0, 0))
                if w <= 0 or h <= 0:
                    continue
                xc = min(max((x + w / 2) / width, 0.0), 1.0)
                yc = min(max((y + h / 2) / height, 0.0), 1.0)
                nw = min(w / width, 1.0)
                nh = min(h / height, 1.0)
                lines.append(
                    f"{mapping[src_idx]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            yield img_path, lines

    @staticmethod
    def _resolve_coco_image(source_dir: Path, json_path: Path,
                            file_name: str) -> Optional[Path]:
        if not file_name:
            return None
        for base in (json_path.parent, source_dir,
                     source_dir / 'images', source_dir / 'data'):
            cand = base / file_name
            if cand.exists():
                return cand
        matches = list(source_dir.rglob(Path(file_name).name))
        return matches[0] if matches else None

    def _iter_yolo_items(self, source_dir: Path, mapping: Dict[int, int]):
        """Yield (image_path, [remapped yolo label lines]) from a YOLO tree."""
        for img, label in self._collect_yolo_pairs(source_dir):
            lines = []
            if label is not None:
                try:
                    with open(label, 'r', encoding='utf-8') as f:
                        for raw in f:
                            parts = raw.split()
                            if len(parts) < 5:
                                continue
                            try:
                                src_idx = int(float(parts[0]))
                            except ValueError:
                                continue
                            if src_idx not in mapping:
                                continue
                            # Keep box or polygon coordinates as-is; only the
                            # class index is remapped.
                            lines.append(
                                ' '.join([str(mapping[src_idx])] + parts[1:]))
                except OSError as e:
                    logger.warning(f"Cannot read label {label}: {e}")
            yield img, lines
