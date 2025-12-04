# YOLOv8 Training Tool - Installation Guide

## สำหรับ Windows

### Prerequisites
ก่อนติดตั้ง dependencies ต้องติดตั้ง Microsoft C++ Build Tools:

1. ดาวน์โหลด: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. เปิดตัวติดตั้งและเลือก "Desktop development with C++"
3. กด Install

### หรือใช้วิธีง่ายกว่า:
ติดตั้งโดยข้าม stringzilla:

```bash
pip install ultralytics --only-binary :all:
pip install -r requirements_windows.txt
```

## การติดตั้งปกติ (Linux/Mac)

```bash
pip install -r requirements.txt
```

## การรันโปรแกรม

```bash
python main.py
```

## หากยังมีปัญหา

ติดตั้งแบบไม่มี stringzilla:
```bash
pip install ultralytics --no-deps
pip install -r requirements_minimal.txt
```
