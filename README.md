# YOLOv8 Training Tool

โปรแกรมสำหรับ Training โมเดล YOLOv8 พร้อม GUI ที่ใช้งานง่าย

## ฟีเจอร์หลัก

### 1. Project Management
- สร้าง/เปิด/บันทึก project
- Import ภาพจากหลายแหล่ง
- Auto-organize dataset structure

### 2. Dataset Management
- แบ่ง train/val/test อัตโนมัติ
- Data augmentation preview
- Dataset statistics

### 3. Annotation Tools
- วาด bounding box (drag & drop)
- แก้ไข/ลบ annotation
- Copy/Paste annotations
- Keyboard shortcuts
- Auto-save

### 4. Class Management
- เพิ่ม/ลบ/แก้ไข classes
- Color coding แต่ละ class
- Class statistics

### 5. Training
- Real-time training metrics (loss, mAP)
- Training visualization (charts)
- Pause/Resume training
- Hyperparameter tuning
- Multi-GPU support

### 6. Evaluation
- Test โมเดลบนภาพ/วิดีโอ
- Confusion matrix
- Precision/Recall curves
- Export ผลลัพธ์

### 7. Export
- Export เป็น .pt, .onnx, .tflite, .torchscript, .coreml
- Generate data.yaml อัตโนมัติ

## โครงสร้างโปรเจกต์

```
YOLOv8_Training_Tool/
├── main.py                 # Entry point
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py         # App settings & paths
├── core/
│   ├── __init__.py
│   ├── dataset_manager.py  # จัดการ dataset
│   ├── label_manager.py    # จัดการ labels
│   ├── model_trainer.py    # Training engine
│   ├── model_evaluator.py  # ประเมินผล
│   └── export_manager.py   # Export โมเดล
├── ui/
│   ├── __init__.py
│   ├── main_window.py      # Main window
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── image_viewer.py
│   │   ├── label_widget.py
│   │   ├── class_manager.py
│   │   ├── dataset_widget.py
│   │   ├── training_widget.py
│   │   └── metrics_widget.py
│   └── dialogs/
│       ├── __init__.py
│       ├── new_project_dialog.py
│       ├── class_dialog.py
│       └── export_dialog.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── annotation_converter.py
│   └── validators.py
└── resources/
    ├── icons/
    └── styles/
        └── style.qss
```

## การติดตั้ง

1. Clone repository:
```bash
git clone <repository-url>
cd YOLOv8_Training_Tool
```

2. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

3. รันโปรแกรม:
```bash
python main.py
```

## การใช้งาน

### สร้าง Project ใหม่
1. File → New Project
2. ใส่ชื่อ project และเลือกที่เก็บ
3. กด Create

### Import ภาพ
1. File → Import Images
2. เลือกภาพที่ต้องการ
3. ภาพจะถูก import เข้า project

### เพิ่ม Classes
1. ใน Class Manager panel
2. กด Add
3. ใส่ชื่อ class

### Annotate ภาพ
1. เลือกภาพจาก Dataset panel
2. เลือก class ที่ต้องการ
3. Drag & drop บนภาพเพื่อวาด bounding box
4. กด Save (Ctrl+S)

### Split Dataset
1. Dataset → Split Dataset
2. โปรแกรมจะแบ่งเป็น train/val/test อัตโนมัติ

### Train Model
1. ตั้งค่า hyperparameters ใน Training panel
2. กด Start Training
3. ติดตามผลใน Metrics panel

### Export Model
1. Training → Export Model
2. เลือก format ที่ต้องการ
3. กด Export

## Keyboard Shortcuts

- `Ctrl+S` - Save
- `Ctrl+O` - Open
- `Ctrl+N` - New Project
- `Delete` - Delete annotation
- `D` - Next image
- `A` - Previous image
- `Ctrl++` - Zoom in
- `Ctrl+-` - Zoom out
- `Ctrl+0` - Reset zoom
- `F5` - Start training
- `Shift+F5` - Stop training

## ข้อกำหนดของระบบ

- Python 3.8 หรือสูงกว่า
- CUDA-compatible GPU (แนะนำสำหรับ training)
- RAM อย่างน้อย 8GB
- พื้นที่ว่างอย่างน้อย 10GB

## License

MIT License