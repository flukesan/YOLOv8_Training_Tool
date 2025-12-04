"""
Model Evaluator - handles model evaluation and testing
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


class ModelEvaluator:
    """Evaluates trained YOLO models"""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict_image(self, image_path: Path, conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Run prediction on a single image
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {}

        try:
            results = self.model(str(image_path),
                               conf=conf_threshold,
                               iou=iou_threshold,
                               verbose=False)

            if not results or len(results) == 0:
                return {
                    'boxes': [],
                    'scores': [],
                    'classes': [],
                    'image_path': str(image_path)
                }

            result = results[0]

            predictions = {
                'boxes': [],
                'scores': [],
                'classes': [],
                'image_path': str(image_path)
            }

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Extract normalized coordinates
                if hasattr(boxes, 'xywhn'):
                    predictions['boxes'] = boxes.xywhn.cpu().numpy().tolist()

                # Extract confidence scores
                if hasattr(boxes, 'conf'):
                    predictions['scores'] = boxes.conf.cpu().numpy().tolist()

                # Extract class IDs
                if hasattr(boxes, 'cls'):
                    predictions['classes'] = boxes.cls.cpu().numpy().tolist()

            return predictions

        except Exception as e:
            print(f"Error in prediction: {e}")
            return {}

    def predict_video(self, video_path: Path, conf_threshold: float = 0.25,
                     output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run prediction on a video
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold
            output_path: Path to save output video
        Returns:
            Dictionary with prediction statistics
        """
        if self.model is None:
            return {}

        try:
            results = self.model(str(video_path),
                               conf=conf_threshold,
                               stream=True,
                               verbose=False)

            stats = {
                'total_frames': 0,
                'detections_per_frame': [],
                'classes_detected': set()
            }

            for result in results:
                stats['total_frames'] += 1

                if result.boxes is not None:
                    num_detections = len(result.boxes)
                    stats['detections_per_frame'].append(num_detections)

                    if hasattr(result.boxes, 'cls'):
                        classes = result.boxes.cls.cpu().numpy()
                        stats['classes_detected'].update(classes.tolist())

            stats['classes_detected'] = list(stats['classes_detected'])
            stats['avg_detections'] = np.mean(stats['detections_per_frame']) if stats['detections_per_frame'] else 0

            return stats

        except Exception as e:
            print(f"Error in video prediction: {e}")
            return {}

    def evaluate_dataset(self, data_yaml_path: Path,
                        split: str = 'val') -> Dict[str, Any]:
        """
        Evaluate model on a dataset split
        Args:
            data_yaml_path: Path to data.yaml
            split: Dataset split to evaluate ('val' or 'test')
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            return {}

        try:
            results = self.model.val(
                data=str(data_yaml_path),
                split=split,
                verbose=False
            )

            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict

            # Add confusion matrix if available
            if hasattr(results, 'confusion_matrix'):
                metrics['confusion_matrix'] = results.confusion_matrix.matrix.tolist()

            return metrics

        except Exception as e:
            print(f"Error in dataset evaluation: {e}")
            return {}

    def get_confusion_matrix(self, data_yaml_path: Path,
                            split: str = 'val') -> Optional[np.ndarray]:
        """Get confusion matrix for dataset"""
        metrics = self.evaluate_dataset(data_yaml_path, split)
        return metrics.get('confusion_matrix')

    def calculate_precision_recall_curve(self, data_yaml_path: Path,
                                        split: str = 'val') -> Dict[int, Dict]:
        """
        Calculate precision-recall curve for each class
        Args:
            data_yaml_path: Path to data.yaml
            split: Dataset split
        Returns:
            Dictionary mapping class IDs to P-R curve data
        """
        if self.model is None:
            return {}

        try:
            results = self.model.val(
                data=str(data_yaml_path),
                split=split,
                verbose=False,
                plots=True
            )

            pr_curves = {}

            if hasattr(results, 'curves'):
                # Extract P-R curves if available
                pass  # Implementation depends on ultralytics version

            return pr_curves

        except Exception as e:
            print(f"Error calculating P-R curves: {e}")
            return {}

    def benchmark_speed(self, image_path: Path, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed
        Args:
            image_path: Path to test image
            iterations: Number of iterations
        Returns:
            Speed metrics
        """
        if self.model is None:
            return {}

        try:
            import time

            times = []

            # Warm-up
            for _ in range(10):
                _ = self.model(str(image_path), verbose=False)

            # Benchmark
            for _ in range(iterations):
                start = time.time()
                _ = self.model(str(image_path), verbose=False)
                end = time.time()
                times.append(end - start)

            times = np.array(times)

            return {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'fps': float(1.0 / np.mean(times)),
                'iterations': iterations
            }

        except Exception as e:
            print(f"Error in speed benchmark: {e}")
            return {}

    def get_class_performance(self, data_yaml_path: Path,
                             split: str = 'val') -> Dict[int, Dict[str, float]]:
        """
        Get per-class performance metrics
        Args:
            data_yaml_path: Path to data.yaml
            split: Dataset split
        Returns:
            Dictionary mapping class IDs to metrics
        """
        if self.model is None:
            return {}

        try:
            results = self.model.val(
                data=str(data_yaml_path),
                split=split,
                verbose=False
            )

            class_metrics = {}

            if hasattr(results, 'results_dict'):
                # Try to extract per-class metrics
                # This depends on the ultralytics version
                pass

            return class_metrics

        except Exception as e:
            print(f"Error getting class performance: {e}")
            return {}

    def detect_objects_batch(self, image_paths: List[Path],
                            conf_threshold: float = 0.25,
                            batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Run batch prediction on multiple images
        Args:
            image_paths: List of image paths
            conf_threshold: Confidence threshold
            batch_size: Batch size for processing
        Returns:
            List of prediction results
        """
        if self.model is None:
            return []

        all_predictions = []

        try:
            # Process in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_paths_str = [str(p) for p in batch_paths]

                results = self.model(batch_paths_str,
                                   conf=conf_threshold,
                                   verbose=False)

                for j, result in enumerate(results):
                    predictions = {
                        'boxes': [],
                        'scores': [],
                        'classes': [],
                        'image_path': str(batch_paths[j])
                    }

                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes

                        if hasattr(boxes, 'xywhn'):
                            predictions['boxes'] = boxes.xywhn.cpu().numpy().tolist()

                        if hasattr(boxes, 'conf'):
                            predictions['scores'] = boxes.conf.cpu().numpy().tolist()

                        if hasattr(boxes, 'cls'):
                            predictions['classes'] = boxes.cls.cpu().numpy().tolist()

                    all_predictions.append(predictions)

            return all_predictions

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return []

    def compare_with_ground_truth(self, predictions: Dict[str, Any],
                                 ground_truth_path: Path,
                                 iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compare predictions with ground truth
        Args:
            predictions: Prediction results
            ground_truth_path: Path to ground truth label file
            iou_threshold: IoU threshold for matching
        Returns:
            Comparison metrics
        """
        from core.label_manager import LabelManager

        label_manager = LabelManager(Path('.'))
        gt_boxes = label_manager.load_annotations(ground_truth_path)

        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)

        matched_gt = set()

        # Compare predictions with ground truth
        for pred_box, pred_class in zip(predictions['boxes'], predictions['classes']):
            matched = False

            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                if int(pred_class) != gt_box.class_id:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(pred_box, [gt_box.x_center, gt_box.y_center,
                                                    gt_box.width, gt_box.height])

                if iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(i)
                    matched = True
                    false_negatives -= 1
                    break

            if not matched:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in xywh format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to xyxy
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
