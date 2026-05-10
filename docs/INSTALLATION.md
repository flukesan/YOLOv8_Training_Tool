# Installation Guide

Complete installation instructions for YOLOv8 Training Tool.

## System Requirements

### Minimum
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space
- Modern OS: Windows 10+, macOS 10.15+, Ubuntu 20.04+

### Recommended
- Python 3.10 or 3.11
- 16 GB RAM
- 50 GB free disk space (for datasets and models)
- NVIDIA GPU with 8+ GB VRAM

---

## Quick Install

### From Source (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd YOLOv8_Training_Tool

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment:
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

---

## Detailed Installation

### Step 1: Install Python

#### Windows
Download from [python.org](https://python.org)
- Choose Python 3.10 or 3.11
- ✅ Check "Add Python to PATH" during installation

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### Step 2: Verify Python Installation

```bash
python --version  # Should show 3.8+
pip --version
```

### Step 3: Create Virtual Environment

A virtual environment keeps dependencies isolated:

```bash
python -m venv venv
```

Activate it:

| OS | Command |
|----|---------|
| Windows (cmd) | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| Linux/macOS | `source venv/bin/activate` |

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

#### Standard Installation
```bash
pip install -r requirements.txt
```

#### Minimal (CPU only, smaller install)
```bash
pip install -r requirements_minimal.txt
```

#### Windows-specific
```bash
pip install -r requirements_windows.txt
```

### Step 5: Install GPU Support (Optional but Recommended)

#### NVIDIA GPU

1. Verify NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Install CUDA-enabled PyTorch:

   **CUDA 11.8** (older GPUs):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **CUDA 12.1** (newer GPUs):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify CUDA in Python:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Your GPU name
   ```

#### Apple Silicon (M1/M2/M3)

PyTorch supports Metal Performance Shaders (MPS):
```bash
pip install torch torchvision
```

Then use `device='mps'` for training.

### Step 6: First Launch

```bash
python main.py
```

The application will:
1. Create directories in `~/YOLOv8_Projects/`
2. Initialize logging in `~/.yolo_training_tool/logs/`
3. Open the main window

---

## Verify Installation

### Run Tests
```bash
pip install pytest
pytest tests/ -v
```

All tests should pass (~111 tests).

### Test Basic Workflow
1. Launch the app
2. Create a new project (File > New Project)
3. Import a few test images
4. Add a class
5. Annotate one image

If all of these work, installation is complete.

---

## Optional: Pre-trained Models

The tool will download YOLOv8 models on first use. If you want to pre-download:

```python
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Nano (smallest)
YOLO('yolov8s.pt')  # Small (recommended)
YOLO('yolov8m.pt')  # Medium
YOLO('yolov8l.pt')  # Large
YOLO('yolov8x.pt')  # XLarge (best accuracy)
```

These will be cached in `~/YOLOv8_Models/` (or current directory).

---

## Updating

### From Source
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Specific Branch
```bash
git fetch origin
git checkout <branch-name>
pip install -r requirements.txt --upgrade
```

---

## Uninstall

### Remove application
```bash
# Delete cloned directory
rm -rf YOLOv8_Training_Tool/
```

### Remove user data (optional)
```bash
# Linux/macOS
rm -rf ~/.yolo_training_tool/
rm -rf ~/YOLOv8_Projects/
rm -rf ~/YOLOv8_Models/

# Windows
rmdir /s %USERPROFILE%\.yolo_training_tool
rmdir /s %USERPROFILE%\YOLOv8_Projects
rmdir /s %USERPROFILE%\YOLOv8_Models
```

---

## Troubleshooting Installation

### "pip: command not found"
Use `python -m pip` instead of `pip`.

### "ImportError: DLL load failed"
Common on Windows. Install Visual C++ Redistributables:
- Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### Permission errors
- Don't use `sudo pip` (can break system Python)
- Use virtual environment instead
- Or use `pip install --user`

### "Could not find a version that satisfies the requirement"
- Update pip: `pip install --upgrade pip`
- Check Python version: requires 3.8+
- Try without version pinning: edit requirements.txt

### Slow installation
Use a faster mirror:
```bash
pip install -r requirements.txt -i https://pypi.org/simple
```

---

## Docker Installation (Advanced)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t yolo-training-tool .
docker run -it --gpus all yolo-training-tool
```

---

## Next Steps

After installation:
- Read [USER_GUIDE.md](USER_GUIDE.md) for usage instructions
- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) for training tips
