"""
Annotation format converter
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
from core.label_manager import BoundingBox


class AnnotationConverter:
    """Convert between different annotation formats"""

    @staticmethod
    def yolo_to_coco(yolo_labels: List[BoundingBox], image_info: Dict) -> Dict:
        """Convert YOLO format to COCO format"""
        annotations = []
        for i, bbox in enumerate(yolo_labels):
            x1, y1, x2, y2 = bbox.to_absolute(image_info['width'], image_info['height'])
            annotations.append({
                'id': i,
                'image_id': image_info['id'],
                'category_id': bbox.class_id,
                'bbox': [x1, y1, x2-x1, y2-y1],
                'area': (x2-x1) * (y2-y1),
                'iscrowd': 0
            })
        return annotations

    @staticmethod
    def coco_to_yolo(coco_annotations: List[Dict], image_width: int,
                    image_height: int) -> List[BoundingBox]:
        """Convert COCO format to YOLO format"""
        yolo_boxes = []
        for ann in coco_annotations:
            x, y, w, h = ann['bbox']
            bbox = BoundingBox.from_absolute(
                ann['category_id'], x, y, x+w, y+h,
                image_width, image_height
            )
            yolo_boxes.append(bbox)
        return yolo_boxes

    @staticmethod
    def voc_to_yolo(voc_xml_path: Path, image_width: int,
                   image_height: int) -> List[BoundingBox]:
        """Convert Pascal VOC XML to YOLO format"""
        tree = ET.parse(voc_xml_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)

            bbox = BoundingBox.from_absolute(0, x1, y1, x2, y2,
                                           image_width, image_height)
            boxes.append(bbox)

        return boxes
