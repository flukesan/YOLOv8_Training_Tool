# API Reference

Programmatic API for YOLOv8 Training Tool.

## Core Modules

### `core.logger`

Centralized logging framework.

```python
from core.logger import AppLogger, get_logger

# Initialize once at application start
AppLogger.initialize(level='INFO')

# Get logger anywhere in code
logger = get_logger(__name__)
logger.info("Training started")
logger.warning("Batch size may be too large")
logger.error("Training failed", exc_info=True)
```

**Key Methods:**
- `AppLogger.initialize(log_dir, level, max_bytes, backup_count)` - Initialize once
- `AppLogger.get_logger(name)` - Get a named logger
- `AppLogger.set_console_level(level)` - Change console verbosity
- `get_logger(name)` - Convenience function

---

### `core.crash_handler`

Global crash reporting.

```python
from core.crash_handler import CrashHandler

# Initialize at startup
CrashHandler.initialize(on_crash=show_dialog_callback)

# Get recent crashes
crashes = CrashHandler.get_recent_crashes(limit=10)

# Cleanup old reports
CrashHandler.cleanup_old_crashes(keep_days=30)
```

---

### `core.auto_save`

Automatic state saving and recovery.

```python
from core.auto_save import AutoSaveManager

manager = AutoSaveManager(interval=300)  # 5 minutes

# Register what to save
manager.register_state_provider('app_state', lambda: {
    'window_size': [800, 600],
    'last_project': '/path/to/project',
})

# Start auto-save
manager.start()

# Force immediate save
manager.save_now()

# Recovery on next launch
if manager.has_recovery_data():
    state = manager.load_recovery_data()
    # Restore from state...
```

---

### `core.dataset_manager`

Dataset operations.

```python
from core.dataset_manager import DatasetManager
from pathlib import Path

manager = DatasetManager(Path("my_project"))

# Import images
stats = manager.import_images(["/path/to/images"])
print(f"Imported {stats['imported']} images")

# Split dataset
split_stats = manager.split_dataset(
    ratios={'train': 0.7, 'val': 0.2, 'test': 0.1},
    random_seed=42
)

# Create data.yaml
manager.create_data_yaml(['Class1', 'Class2'])

# Get statistics
stats = manager.get_dataset_statistics(class_names=['Class1', 'Class2'])

# Validate dataset
is_valid, issues = manager.validate_dataset()
```

---

### `core.label_manager`

Annotation management.

```python
from core.label_manager import LabelManager, BoundingBox, Polygon
from pathlib import Path

manager = LabelManager(Path("my_project"))
manager.set_classes(['Good', 'Defect'])

# Create bounding box
bbox = BoundingBox(
    class_id=0,
    x_center=0.5, y_center=0.5,
    width=0.2, height=0.3
)

# Save annotations
label_file = Path("my_project/labels/img.txt")
manager.save_annotations(label_file, [bbox])

# Load annotations
annotations = manager.load_annotations(label_file)

# Validate
valid, error = manager.validate_annotation(bbox)

# Auto-annotate using existing model
auto_anns = manager.auto_annotate(
    image_path=Path("img.jpg"),
    model_path=Path("model.pt"),
    confidence_threshold=0.25
)
```

---

### `core.model_trainer`

Training operations.

```python
from core.model_trainer import ModelTrainer
from pathlib import Path

trainer = ModelTrainer(Path("my_project"))

# Register callbacks
def on_epoch_end(session, metrics):
    print(f"Epoch {session.current_epoch}: {metrics}")

trainer.register_callback('on_epoch_end', on_epoch_end)
trainer.register_callback('on_train_end', lambda s, r: print("Done!"))
trainer.register_callback('on_train_error', lambda e: print(f"Error: {e}"))

# Configuration
config = {
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'lr0': 0.01,
    'patience': 50,
    # Augmentation params (optional)
    'mosaic': 1.0,
    'mixup': 0.1,
    # Hardware (optional)
    'device': 'cuda:0',
    'workers': 4,
}

# Start training
session = trainer.start_training(
    config=config,
    data_yaml_path=Path("my_project/config.yaml"),
    model_name='yolov8s.pt',
    async_mode=True  # Non-blocking
)

# Monitor
while trainer.is_training():
    metrics = trainer.get_current_metrics()
    print(f"Progress: {metrics['progress']}%")
    time.sleep(10)

# Get best weights
best_pt = trainer.get_best_weights_path()
```

**Available callbacks:**
- `on_train_start(session)`
- `on_epoch_end(session, metrics)`
- `on_train_end(session, results)`
- `on_train_error(error_msg)`
- `on_val_end(session, metrics)`

---

### `core.export_manager`

Model export to various formats.

```python
from core.export_manager import ExportManager
from pathlib import Path

exporter = ExportManager(Path("model.pt"))

# Export to ONNX
onnx_path = exporter.export_onnx(
    dynamic=False,
    simplify=True,
    opset=12,
    half=False
)

# Export to TFLite
tflite_path = exporter.export_tflite(int8=False, half=True)

# Export to TensorRT (Linux + NVIDIA)
trt_path = exporter.export_tensorrt(workspace=4, half=True)

# Export to multiple formats
results = exporter.export_multiple(
    formats=['onnx', 'tflite', 'torchscript'],
    output_dir=Path("exports/")
)

# Compare model sizes
sizes = exporter.compare_model_sizes(['onnx', 'tflite', 'torchscript'])
for fmt, size_mb in sizes.items():
    print(f"{fmt}: {size_mb:.2f} MB")

# Get model info
info = exporter.get_model_info()
print(f"Classes: {info['classes']}")

# Validate exported model
is_valid = exporter.validate_export(onnx_path, test_image=Path("test.jpg"))
```

---

### `core.model_evaluator`

Model evaluation and inference.

```python
from core.model_evaluator import ModelEvaluator
from pathlib import Path

evaluator = ModelEvaluator(
    model_path=Path("model.pt"),
    project_path=Path("my_project")
)

# Predict on single image
result = evaluator.predict_image(
    image_path=Path("test.jpg"),
    conf_threshold=0.25,
    iou_threshold=0.45
)
print(f"Detected {len(result['boxes'])} objects")

# Predict on video
stats = evaluator.predict_video(
    video_path=Path("video.mp4"),
    conf_threshold=0.25
)
print(f"Total frames: {stats['total_frames']}")

# Evaluate on dataset
metrics = evaluator.evaluate_dataset(
    data_yaml_path=Path("my_project/config.yaml"),
    split='val'
)

# Benchmark inference speed
speed = evaluator.benchmark_speed(
    image_path=Path("test.jpg"),
    iterations=100
)
print(f"Speed: {speed['fps']:.1f} FPS")

# Batch prediction
results = evaluator.detect_objects_batch(
    image_paths=[Path("img1.jpg"), Path("img2.jpg")],
    batch_size=16
)
```

---

## Configuration

### `config.settings.Settings`

Global application settings.

```python
from config.settings import Settings

# Access settings
print(Settings.APP_NAME)
print(Settings.DEFAULT_TRAIN_PARAMS)

# Augmentation presets
preset = Settings.AUGMENTATION_PRESETS['industrial']

# Get project structure
structure = Settings.get_project_structure(project_path)
```

---

## Complete Example: Training Pipeline

```python
from pathlib import Path
from core.logger import AppLogger, get_logger
from core.dataset_manager import DatasetManager
from core.label_manager import LabelManager, BoundingBox
from core.model_trainer import ModelTrainer
from core.export_manager import ExportManager

# 1. Setup
AppLogger.initialize(level='INFO')
logger = get_logger('training_pipeline')

project = Path("/home/user/MyProject")

# 2. Prepare dataset
ds = DatasetManager(project)
ds.import_images(["/path/to/raw_images"])

# 3. Annotate (assuming you have labels ready)
labels = LabelManager(project)
labels.set_classes(['Good', 'Defect'])
# ... annotation work ...

# 4. Split dataset
ds.split_dataset(ratios={'train': 0.7, 'val': 0.2, 'test': 0.1})
yaml_path = ds.create_data_yaml(['Good', 'Defect'])

# 5. Train model
trainer = ModelTrainer(project)
trainer.register_callback('on_train_end',
    lambda s, r: logger.info(f"Training complete: {s.session_id}"))

session = trainer.start_training(
    config={
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 50,
        # Industrial preset for Go/NoGo
        **Settings.AUGMENTATION_PRESETS['industrial'],
    },
    data_yaml_path=yaml_path,
    model_name='yolov8s.pt',
    async_mode=False  # Wait for completion
)

# 6. Get trained model
best_model = trainer.get_best_weights_path()

# 7. Export
exporter = ExportManager(best_model)
results = exporter.export_multiple(
    formats=['onnx', 'tflite', 'torchscript'],
    output_dir=project / 'exports'
)

logger.info(f"Pipeline complete. Exports: {results}")
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip GPU tests
pytest tests/ -v -m "not gpu"

# With coverage
pip install pytest-cov
pytest --cov=core --cov-report=html
```

See generated `htmlcov/index.html` for coverage report.
