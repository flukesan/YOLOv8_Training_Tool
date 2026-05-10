# Troubleshooting Guide

Solutions to common problems when using YOLOv8 Training Tool.

## Installation Issues

### "No module named 'ultralytics'"

```bash
pip install ultralytics
# Or all dependencies:
pip install -r requirements.txt
```

### "No module named 'PyQt6'"

```bash
pip install PyQt6
```

### CUDA not available
1. Check NVIDIA driver: `nvidia-smi`
2. Install correct PyTorch version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Training Issues

### "No images found in dataset"
**Cause**: Empty images folder or wrong location.

**Fix**:
1. Verify images are in `images/` folder
2. Check supported formats (.jpg, .png, .bmp, .webp, .tiff)
3. Re-import: **File > Import Images**

### "Need at least 3 images to split"
**Cause**: Too few images for train/val/test split.

**Fix**: Import more images (recommended 100+ per class).

### "Invalid training configuration: Image size must be divisible by 32"
**Cause**: Custom image size not multiple of 32.

**Fix**: Use 320, 416, 512, 640, 736, 832, 1024, or 1280.

### "Insufficient disk space"
**Cause**: Less than 2GB free.

**Fix**:
- Free up disk space
- Move project to drive with more space

### "CUDA out of memory"
**Cause**: Batch size or image size too large.

**Fix**:
1. Reduce batch size (try 8, 4, or 2)
2. Reduce image size (try 512 or 416)
3. Use smaller model (YOLOv8n or s)
4. Enable AMP (mixed precision)

### Training stuck at epoch 0
**Cause**: Often data loading issue.

**Fix**:
1. Reduce workers to 0 or 1
2. Set cache='disk' instead of 'ram'
3. Check for very large images (resize first)

### "data.yaml not found"
**Cause**: Dataset not properly initialized.

**Fix**:
1. Run **Dataset > Split Dataset** first
2. Verify config.yaml exists in project root

---

## Annotation Issues

### Bounding box appears in wrong position when zoomed
**Cause**: Coordinate conversion bug (FIXED in v1.0.0).

**Fix**: Update to latest version.

### Cannot delete annotation
**Try**:
1. Click on the annotation first to select it
2. Press Delete key
3. Or right-click → Delete

### Annotations not saving
**Check**:
1. File permissions on labels folder
2. Disk not full
3. Look in `app.log` for errors

---

## Model Loading Issues

### "Model file not found"
**Cause**: Model path is incorrect or file deleted.

**Fix**:
1. Check `runs/train/<exp>/weights/best.pt` exists
2. Re-train if necessary
3. Use **File > Open Model** to browse manually

### "Model file is empty"
**Cause**: Training was interrupted before saving.

**Fix**:
1. Use `last.pt` instead of `best.pt`
2. Or retrain the model

### Cannot load .pt file
**Check**:
- File size > 1KB (typical model is 6-130 MB)
- Same major version of YOLOv8 used to create
- Ultralytics package installed

---

## Export Issues

### "Unsupported export format"
**Fix**: Use one of: pt, onnx, tflite, torchscript, coreml, tfjs, engine, paddle, ncnn

### TensorRT export fails
**Causes**:
- Not on Linux
- TensorRT not installed
- Wrong CUDA version

### CoreML export fails
**Cause**: Only works on macOS.

**Fix**: Use ONNX format on other platforms.

### File move fails (cross-drive)
**Cause**: Earlier versions had this bug (FIXED).

**Fix**: Update to latest version.

---

## UI Issues

### Application won't start
1. Check `~/.yolo_training_tool/logs/app.log`
2. Check `~/.yolo_training_tool/crashes/` for crash reports
3. Try running from terminal to see errors:
   ```bash
   python main.py
   ```

### Buttons don't respond
**Try**:
1. Resize the window
2. Switch tabs and back
3. Restart the application

### Text in Model Testing tabs unreadable
**Cause**: Theme/contrast issue (FIXED in v1.0.0).

**Fix**: Update to latest version.

### High DPI display issues
**Set environment variable** before launch:
```bash
export QT_AUTO_SCREEN_SCALE_FACTOR=1
python main.py
```

---

## Performance Issues

### Slow data loading
**Try**:
1. Use SSD instead of HDD
2. Set cache='ram' (if you have enough RAM)
3. Increase workers (try 4-8)
4. Resize images before training

### Slow training
**Optimize**:
1. Use GPU (10-100x faster than CPU)
2. Enable AMP (`amp=True`)
3. Use cache='ram'
4. Reduce image size if quality permits
5. Use smaller model for prototyping

### High RAM usage
**Reduce**:
1. cache='disk' or 'none' (instead of 'ram')
2. Reduce workers
3. Reduce batch size

---

## Recovery & Crashes

### Application crashed
1. Restart - auto-recovery may offer to restore previous state
2. Check `~/.yolo_training_tool/crashes/` for the crash report
3. Crash report includes:
   - Exception type and message
   - Full traceback
   - System information
   - Installed package versions

### How to report a crash
1. Find the latest `.txt` crash report in `~/.yolo_training_tool/crashes/`
2. Include in your bug report:
   - Steps to reproduce
   - The crash report contents
   - Project size/type if relevant

### Recovery from autosave
- On startup, the tool checks for autosave data
- If found, you'll be prompted to recover
- Auto-save runs every 5 minutes by default
- Saves are in `~/.yolo_training_tool/autosave/`

---

## Logs & Diagnostics

### Where are the logs?

| Type | Location |
|------|----------|
| App logs | `~/.yolo_training_tool/logs/app.log` |
| Errors only | `~/.yolo_training_tool/logs/errors.log` |
| Crash reports | `~/.yolo_training_tool/crashes/crash_*.json` |
| Auto-save | `~/.yolo_training_tool/autosave/current.json` |

### Reading logs
Logs use this format:
```
2026-05-10 14:30:00 [INFO    ] yolo_tool.module - Message here
```

Levels (most to least verbose):
- DEBUG (development details)
- INFO (general info)
- WARNING (potential issues)
- ERROR (errors but recoverable)
- CRITICAL (severe errors)

### Enable DEBUG logging
Set environment variable:
```bash
export YOLO_LOG_LEVEL=DEBUG
python main.py
```

---

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `NameError: name 'X' is not defined` | Code merge conflict | `git reset --hard origin/<branch>` |
| `Permission denied` | File permissions | Run with appropriate privileges |
| `cudnn error` | GPU compatibility | Check CUDA/cuDNN versions |
| `ModuleNotFoundError` | Missing package | `pip install <package>` |
| `OOM (Out of Memory)` | Resource exhausted | Reduce batch/image size |
| `RuntimeError: CUDA error` | GPU issue | Restart, check drivers |

---

## Still Need Help?

1. Check the logs first
2. Search GitHub Issues
3. Create a new issue with:
   - System information
   - Steps to reproduce
   - Error messages from logs
   - Crash report (if applicable)
   - Screenshots if UI-related
