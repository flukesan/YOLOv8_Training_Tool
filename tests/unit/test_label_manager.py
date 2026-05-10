"""
Unit tests for core.label_manager
"""
import pytest
from pathlib import Path

from core.label_manager import BoundingBox, Polygon, LabelManager


class TestBoundingBox:
    """Tests for BoundingBox annotation"""

    def test_create_valid_bbox(self):
        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)
        assert bbox.class_id == 0
        assert bbox.x_center == 0.5
        assert bbox.y_center == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.3

    def test_to_yolo_format(self):
        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)
        result = bbox.to_yolo_format()
        assert result.startswith("0 ")
        # Should have 5 space-separated values
        parts = result.split()
        assert len(parts) == 5

    def test_to_absolute(self):
        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)
        x1, y1, x2, y2 = bbox.to_absolute(640, 480)
        # x_center = 320, width = 128 (0.2 * 640)
        # y_center = 240, height = 144 (0.3 * 480)
        assert x1 == 256  # 320 - 64
        assert y1 == 168  # 240 - 72
        assert x2 == 384  # 320 + 64
        assert y2 == 312  # 240 + 72

    def test_from_absolute(self):
        bbox = BoundingBox.from_absolute(0, 192, 168, 448, 312, 640, 480)
        assert bbox.class_id == 0
        assert abs(bbox.x_center - 0.5) < 0.01
        assert abs(bbox.y_center - 0.5) < 0.01

    def test_from_yolo_format_valid(self):
        line = "0 0.5 0.5 0.2 0.3"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is not None
        assert bbox.class_id == 0
        assert bbox.x_center == 0.5

    def test_from_yolo_format_invalid_few_values(self):
        line = "0 0.5"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is None

    def test_from_yolo_format_invalid_data_type(self):
        line = "0 abc 0.5 0.2 0.3"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is None

    def test_from_yolo_format_negative_class(self):
        line = "-1 0.5 0.5 0.2 0.3"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is None

    def test_from_yolo_format_out_of_range(self):
        line = "0 1.5 0.5 0.2 0.3"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is None

    def test_from_yolo_format_zero_width(self):
        line = "0 0.5 0.5 0.0 0.3"
        bbox = BoundingBox.from_yolo_format(line)
        assert bbox is None


class TestPolygon:
    """Tests for Polygon annotation"""

    def test_create_valid_polygon(self):
        points = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
        poly = Polygon(0, points)
        assert poly.class_id == 0
        assert len(poly.points) == 4

    def test_to_yolo_seg_format(self):
        points = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]
        poly = Polygon(1, points)
        result = poly.to_yolo_seg_format()
        assert result.startswith("1 ")

    def test_get_bounding_box(self):
        points = [(0.1, 0.2), (0.5, 0.2), (0.5, 0.6), (0.1, 0.6)]
        poly = Polygon(0, points)
        x_min, y_min, x_max, y_max = poly.get_bounding_box()
        assert x_min == 0.1
        assert y_min == 0.2
        assert x_max == 0.5
        assert y_max == 0.6

    def test_from_yolo_seg_format_valid(self):
        line = "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5"
        poly = Polygon.from_yolo_seg_format(line)
        assert poly is not None
        assert poly.class_id == 0
        assert len(poly.points) == 4

    def test_from_yolo_seg_format_too_few_points(self):
        line = "0 0.1 0.1 0.5 0.1"  # Only 2 points
        poly = Polygon.from_yolo_seg_format(line)
        assert poly is None

    def test_from_yolo_seg_format_negative_class(self):
        line = "-1 0.1 0.1 0.5 0.1 0.5 0.5"
        poly = Polygon.from_yolo_seg_format(line)
        assert poly is None

    def test_from_yolo_seg_format_out_of_range(self):
        line = "0 0.1 0.1 1.5 0.1 0.5 0.5"  # 1.5 out of range
        poly = Polygon.from_yolo_seg_format(line)
        assert poly is None


class TestLabelManager:
    """Tests for LabelManager"""

    def test_initialization(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        assert manager.project_path == temp_project_dir
        assert manager.classes == []

    def test_set_classes(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        classes = ['Good', 'Defect']
        manager.set_classes(classes)
        assert manager.classes == classes
        assert 0 in manager.class_colors
        assert 1 in manager.class_colors

    def test_load_annotations_valid(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)
        annotations = manager.load_annotations(sample_label_file)
        assert len(annotations) == 2
        assert all(isinstance(a, BoundingBox) for a in annotations)

    def test_load_annotations_nonexistent(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        result = manager.load_annotations(temp_project_dir / "nonexistent.txt")
        assert result == []

    def test_save_and_load_annotations(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)

        label_file = temp_project_dir / "test_save.txt"
        manager.save_annotations(label_file, [bbox])

        loaded = manager.load_annotations(label_file)
        assert len(loaded) == 1
        assert loaded[0].class_id == 0

    def test_validate_bbox_valid(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['ClassA', 'ClassB'])

        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)
        valid, error = manager.validate_annotation(bbox)
        assert valid is True
        assert error is None

    def test_validate_bbox_invalid_class(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['ClassA'])

        bbox = BoundingBox(5, 0.5, 0.5, 0.2, 0.3)  # Invalid class ID
        valid, error = manager.validate_annotation(bbox)
        assert valid is False
        assert "class" in error.lower()

    def test_validate_bbox_too_small(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['ClassA'])

        bbox = BoundingBox(0, 0.5, 0.5, 0.0001, 0.0001)
        valid, error = manager.validate_annotation(bbox)
        assert valid is False

    def test_validate_polygon_valid(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['ClassA'])

        poly = Polygon(0, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)])
        valid, error = manager.validate_annotation(poly)
        assert valid is True

    def test_validate_polygon_too_few_points(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)

        # Manually create polygon with too few points (bypassing validation)
        poly = Polygon.__new__(Polygon)
        poly.class_id = 0
        poly.points = [(0.1, 0.1), (0.5, 0.5)]
        poly.confidence = 1.0
        poly.annotation_type = "polygon"

        valid, error = manager.validate_annotation(poly)
        assert valid is False

    def test_get_class_color(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['A', 'B', 'C'])

        color_a = manager.get_class_color(0)
        color_b = manager.get_class_color(1)

        assert isinstance(color_a, tuple)
        assert len(color_a) == 3
        assert color_a != color_b  # Different colors

    def test_get_class_name(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        manager.set_classes(['Good', 'Defect'])

        assert manager.get_class_name(0) == 'Good'
        assert manager.get_class_name(1) == 'Defect'
        assert 'Unknown' in manager.get_class_name(99)

    def test_add_annotation(self, temp_project_dir):
        manager = LabelManager(temp_project_dir)
        label_file = temp_project_dir / "add_test.txt"

        bbox = BoundingBox(0, 0.5, 0.5, 0.2, 0.3)
        manager.add_annotation(label_file, bbox)

        loaded = manager.load_annotations(label_file)
        assert len(loaded) == 1

    def test_delete_annotation(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)

        # Copy sample file to manage with manager
        result = manager.delete_annotation(sample_label_file, 0)
        assert result is True

        loaded = manager.load_annotations(sample_label_file)
        assert len(loaded) == 1  # Was 2, now 1

    def test_delete_annotation_invalid_index(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)
        result = manager.delete_annotation(sample_label_file, 999)
        assert result is False

    def test_get_annotation_stats(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)
        stats = manager.get_annotation_stats(sample_label_file)
        assert stats['total'] == 2
        assert 0 in stats['by_class']
        assert 1 in stats['by_class']

    def test_filter_annotations_by_class(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)
        filtered = manager.filter_annotations_by_class(sample_label_file, [0])
        assert len(filtered) == 1
        assert filtered[0].class_id == 0

    def test_remap_classes(self, temp_project_dir, sample_label_file):
        manager = LabelManager(temp_project_dir)
        manager.remap_classes(sample_label_file, {0: 5, 1: 10})

        annotations = manager.load_annotations(sample_label_file)
        class_ids = [a.class_id for a in annotations]
        assert 5 in class_ids
        assert 10 in class_ids
