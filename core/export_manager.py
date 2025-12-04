"""
Export Manager - handles model export to various formats
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from config.settings import Settings


class ExportManager:
    """Manages model export operations"""

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

    def export(self, format: str, **kwargs) -> Optional[Path]:
        """
        Export model to specified format
        Args:
            format: Export format ('onnx', 'tflite', 'pt', 'torchscript', etc.)
            **kwargs: Additional export parameters
        Returns:
            Path to exported model
        """
        if self.model is None:
            print("Model not loaded")
            return None

        if format not in Settings.EXPORT_FORMATS:
            print(f"Unsupported export format: {format}")
            return None

        try:
            # Export model
            export_path = self.model.export(format=format, **kwargs)

            if export_path:
                return Path(export_path)

            return None

        except Exception as e:
            print(f"Error exporting to {format}: {e}")
            return None

    def export_onnx(self, dynamic: bool = False, simplify: bool = True,
                   opset: int = 12, half: bool = False) -> Optional[Path]:
        """
        Export model to ONNX format
        Args:
            dynamic: Dynamic input shapes
            simplify: Simplify ONNX model
            opset: ONNX opset version
            half: Export in FP16
        Returns:
            Path to ONNX model
        """
        return self.export(
            format='onnx',
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            half=half
        )

    def export_tflite(self, int8: bool = False, half: bool = False) -> Optional[Path]:
        """
        Export model to TensorFlow Lite format
        Args:
            int8: INT8 quantization
            half: FP16 quantization
        Returns:
            Path to TFLite model
        """
        return self.export(
            format='tflite',
            int8=int8,
            half=half
        )

    def export_torchscript(self, optimize: bool = True) -> Optional[Path]:
        """
        Export model to TorchScript format
        Args:
            optimize: Optimize for mobile
        Returns:
            Path to TorchScript model
        """
        return self.export(
            format='torchscript',
            optimize=optimize
        )

    def export_coreml(self, nms: bool = True, half: bool = False) -> Optional[Path]:
        """
        Export model to CoreML format (macOS/iOS)
        Args:
            nms: Include NMS in model
            half: FP16 precision
        Returns:
            Path to CoreML model
        """
        import platform
        if platform.system() != 'Darwin':
            print("CoreML export only supported on macOS")
            return None

        return self.export(
            format='coreml',
            nms=nms,
            half=half
        )

    def export_tfjs(self) -> Optional[Path]:
        """
        Export model to TensorFlow.js format
        Returns:
            Path to TFJS model directory
        """
        return self.export(format='tfjs')

    def export_tensorrt(self, workspace: int = 4, half: bool = True,
                       int8: bool = False) -> Optional[Path]:
        """
        Export model to TensorRT format
        Args:
            workspace: Workspace size in GB
            half: FP16 precision
            int8: INT8 quantization
        Returns:
            Path to TensorRT model
        """
        import platform
        if platform.system() != 'Linux':
            print("TensorRT export only supported on Linux")
            return None

        return self.export(
            format='engine',
            workspace=workspace,
            half=half,
            int8=int8
        )

    def export_paddle(self) -> Optional[Path]:
        """
        Export model to PaddlePaddle format
        Returns:
            Path to Paddle model
        """
        return self.export(format='paddle')

    def export_ncnn(self, half: bool = False) -> Optional[Path]:
        """
        Export model to NCNN format
        Args:
            half: FP16 precision
        Returns:
            Path to NCNN model
        """
        return self.export(
            format='ncnn',
            half=half
        )

    def export_multiple(self, formats: List[str],
                       output_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
        """
        Export model to multiple formats
        Args:
            formats: List of export formats
            output_dir: Output directory for all exports
        Returns:
            Dictionary mapping format to exported path
        """
        results = {}

        for fmt in formats:
            if fmt in Settings.EXPORT_FORMATS:
                exported_path = self.export(fmt)
                results[fmt] = exported_path

                # Move to output directory if specified
                if output_dir and exported_path:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    new_path = output_dir / exported_path.name
                    exported_path.rename(new_path)
                    results[fmt] = new_path

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}

        info = {
            'model_path': str(self.model_path),
            'model_type': 'YOLOv8',
            'task': getattr(self.model, 'task', 'detect'),
        }

        # Try to get model details
        if hasattr(self.model, 'model'):
            model_obj = self.model.model
            if hasattr(model_obj, 'names'):
                info['classes'] = model_obj.names
                info['num_classes'] = len(model_obj.names)

        # Get file size
        if self.model_path.exists():
            info['file_size_mb'] = self.model_path.stat().st_size / (1024 * 1024)

        return info

    def validate_export(self, exported_path: Path,
                       test_image: Optional[Path] = None) -> bool:
        """
        Validate exported model
        Args:
            exported_path: Path to exported model
            test_image: Optional test image for validation
        Returns:
            True if validation successful
        """
        if not exported_path.exists():
            return False

        try:
            # Try to load and run inference with exported model
            if test_image and test_image.exists():
                from ultralytics import YOLO

                # For ONNX and TorchScript, we can test with YOLO
                if exported_path.suffix in ['.onnx', '.torchscript', '.pt']:
                    test_model = YOLO(str(exported_path))
                    results = test_model(str(test_image), verbose=False)
                    return results is not None and len(results) > 0

            # For other formats, just check if file exists and has size > 0
            return exported_path.stat().st_size > 0

        except Exception as e:
            print(f"Validation failed: {e}")
            return False

    def compare_model_sizes(self, formats: List[str]) -> Dict[str, float]:
        """
        Compare exported model sizes across formats
        Args:
            formats: List of formats to compare
        Returns:
            Dictionary mapping format to size in MB
        """
        sizes = {}

        temp_exports = self.export_multiple(formats)

        for fmt, path in temp_exports.items():
            if path and path.exists():
                sizes[fmt] = path.stat().st_size / (1024 * 1024)

        return sizes

    def optimize_for_inference(self, format: str = 'onnx',
                              quantization: str = 'none') -> Optional[Path]:
        """
        Export and optimize model for inference
        Args:
            format: Export format
            quantization: Quantization type ('none', 'fp16', 'int8')
        Returns:
            Path to optimized model
        """
        kwargs = {}

        if quantization == 'fp16':
            kwargs['half'] = True
        elif quantization == 'int8':
            kwargs['int8'] = True

        if format == 'onnx':
            kwargs['simplify'] = True
            kwargs['dynamic'] = False

        return self.export(format, **kwargs)

    def create_deployment_package(self, output_dir: Path,
                                  formats: List[str],
                                  include_config: bool = True) -> Path:
        """
        Create a complete deployment package
        Args:
            output_dir: Output directory
            formats: Formats to include
            include_config: Include model configuration
        Returns:
            Path to deployment package
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export models
        exported = self.export_multiple(formats, output_dir / 'models')

        # Copy original model info
        if include_config:
            info = self.get_model_info()

            import yaml
            config_path = output_dir / 'model_info.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(info, f)

        # Create README
        readme_path = output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write("# YOLOv8 Model Deployment Package\n\n")
            f.write(f"## Exported Formats\n\n")
            for fmt, path in exported.items():
                if path:
                    f.write(f"- **{fmt}**: {path.name}\n")
            f.write("\n## Model Information\n\n")
            info = self.get_model_info()
            for key, value in info.items():
                f.write(f"- **{key}**: {value}\n")

        return output_dir
