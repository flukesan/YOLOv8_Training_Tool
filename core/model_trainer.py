"""
Model Trainer - handles YOLO model training operations
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import threading
import time
from config.settings import Settings


class TrainingSession:
    """Represents a single training session"""

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'  # initialized, running, paused, completed, failed
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 100)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50-95': []
        }
        self.best_metrics = {}

    def update_metrics(self, epoch_metrics: Dict[str, float]):
        """Update training metrics for current epoch"""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_progress(self) -> float:
        """Get training progress as percentage"""
        if self.total_epochs == 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


class ModelTrainer:
    """Handles YOLO model training"""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.current_session: Optional[TrainingSession] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.session_lock = threading.Lock()  # Thread safety for session access
        self.callbacks = {
            'on_epoch_end': [],
            'on_train_start': [],
            'on_train_end': [],
            'on_val_end': [],
            'on_train_error': []  # Add error callback
        }

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for training events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Error in callback for {event}: {e}")

    def start_training(self, config: Dict[str, Any],
                      data_yaml_path: Path,
                      model_name: str = 'yolov8n.pt',
                      async_mode: bool = True) -> TrainingSession:
        """
        Start a new training session
        Args:
            config: Training configuration
            data_yaml_path: Path to data.yaml file
            model_name: Base model to use
            async_mode: Run training in background thread
        Returns:
            TrainingSession object
        """
        if self.is_training():
            raise RuntimeError("Training is already in progress")

        # Create training session
        session_id = f"session_{int(time.time())}"
        self.current_session = TrainingSession(session_id, config)

        # Reset flags
        self.stop_flag.clear()
        self.pause_flag.clear()

        # Prepare training config
        train_config = self._prepare_training_config(config, data_yaml_path, model_name)

        if async_mode:
            # Run training in background thread
            self.training_thread = threading.Thread(
                target=self._train_worker,
                args=(train_config,),
                daemon=True
            )
            self.training_thread.start()
        else:
            # Run training in foreground
            self._train_worker(train_config)

        return self.current_session

    def _validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate training configuration
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check epochs
        epochs = config.get('epochs', 0)
        if epochs <= 0:
            return False, f"Epochs must be positive, got {epochs}"

        if epochs > 10000:
            return False, f"Epochs too large ({epochs}), maximum is 10000"

        # Check batch size
        batch = config.get('batch', 0)
        if batch <= 0:
            return False, f"Batch size must be positive, got {batch}"

        # Check image size (must be divisible by 32 for YOLO)
        imgsz = config.get('imgsz', 640)
        if imgsz % 32 != 0:
            return False, f"Image size must be divisible by 32, got {imgsz}"

        if imgsz < 32 or imgsz > 2048:
            return False, f"Image size must be between 32 and 2048, got {imgsz}"

        # Check learning rate
        lr0 = config.get('lr0', 0.01)
        if not (0 < lr0 < 1):
            return False, f"Learning rate must be between 0 and 1, got {lr0}"

        # Check data file exists
        data_file = Path(config.get('data', ''))
        if not data_file.exists():
            return False, f"Data file not found: {data_file}"

        # Check patience
        patience = config.get('patience', 50)
        if patience < 0:
            return False, f"Patience must be non-negative, got {patience}"

        return True, None

    def _prepare_training_config(self, config: Dict[str, Any],
                                 data_yaml_path: Path,
                                 model_name: str) -> Dict[str, Any]:
        """Prepare complete training configuration"""
        # Start with default parameters
        train_config = Settings.DEFAULT_TRAIN_PARAMS.copy()

        # Update with user config
        train_config.update(config)

        # Set required paths
        train_config['data'] = str(data_yaml_path)
        train_config['model'] = model_name

        # Set project paths
        if not train_config.get('project'):
            train_config['project'] = str(self.project_path / 'runs' / 'train')

        if not train_config.get('name'):
            train_config['name'] = f"exp_{int(time.time())}"

        # Validate configuration
        valid, error_msg = self._validate_config(train_config)
        if not valid:
            raise ValueError(f"Invalid training configuration: {error_msg}")

        return train_config

    def _preflight_checks(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Pre-flight checks before training to catch issues early
        Returns:
            Tuple of (is_ok, error_message)
        """
        import shutil

        # Check disk space (need at least 2GB free for training)
        project_dir = Path(config.get('project', self.project_path / 'runs' / 'train'))
        check_dir = project_dir.parent if project_dir.parent.exists() else Path.home()
        try:
            stat = shutil.disk_usage(check_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 2:
                return False, f"Insufficient disk space: {free_gb:.1f}GB free, need at least 2GB"
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")

        # Check CUDA availability if GPU device specified
        device = str(config.get('device', '')).lower()
        if device and device not in ('', 'cpu'):
            try:
                import torch
                if not torch.cuda.is_available():
                    return False, "GPU device specified but CUDA is not available. Use 'cpu' or install CUDA."

                # Warn if batch size might be too large
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    batch = config.get('batch', 16)
                    imgsz = config.get('imgsz', 640)

                    # Rough estimate of GPU memory needed (heuristic)
                    estimated_gb = (batch * (imgsz / 640) ** 2) * 0.3
                    if estimated_gb > gpu_mem_gb * 0.85 and batch > 0:
                        print(f"Warning: Batch size {batch} at {imgsz}px may exceed GPU memory "
                              f"({gpu_mem_gb:.1f}GB). Consider reducing batch size.")
            except ImportError:
                print("Warning: torch not available for GPU checks")
            except Exception as e:
                print(f"Warning: Could not check GPU: {e}")

        return True, None

    def _train_worker(self, config: Dict[str, Any]):
        """Worker function for training"""
        try:
            from ultralytics import YOLO

            # Run pre-flight checks
            ok, error_msg = self._preflight_checks(config)
            if not ok:
                raise RuntimeError(f"Pre-flight check failed: {error_msg}")

            self.current_session.status = 'running'
            self.current_session.start_time = time.time()

            self._trigger_callbacks('on_train_start', self.current_session)

            # Separate model path from config (it's not a train() parameter)
            train_kwargs = config.copy()
            model_path = train_kwargs.pop('model', 'yolov8n.pt')

            # Load model
            model = YOLO(model_path)

            # Add custom callbacks to YOLO trainer
            def on_train_epoch_end(trainer):
                """Called at end of each training epoch"""
                if self.stop_flag.is_set():
                    trainer.stop = True
                    return

                # Handle pause
                while self.pause_flag.is_set():
                    time.sleep(0.5)
                    if self.stop_flag.is_set():
                        trainer.stop = True
                        return

                # Extract metrics
                metrics = {}
                if hasattr(trainer, 'loss_items'):
                    loss = trainer.loss_items
                    if loss is not None and len(loss) > 0:
                        metrics['train_loss'] = float(loss[0]) if len(loss) > 0 else 0.0

                if hasattr(trainer, 'metrics') and trainer.metrics:
                    results = trainer.metrics
                    metrics.update({
                        'precision': float(results.get('metrics/precision(B)', 0.0)),
                        'recall': float(results.get('metrics/recall(B)', 0.0)),
                        'mAP50': float(results.get('metrics/mAP50(B)', 0.0)),
                        'mAP50-95': float(results.get('metrics/mAP50-95(B)', 0.0)),
                    })

                # Update session (thread-safe)
                with self.session_lock:
                    self.current_session.current_epoch = trainer.epoch + 1
                    self.current_session.update_metrics(metrics)

                self._trigger_callbacks('on_epoch_end', self.current_session, metrics)

            # Register callback with YOLO model
            model.add_callback('on_train_epoch_end', on_train_epoch_end)

            # Train model (use train_kwargs which has 'model' removed)
            results = model.train(
                **train_kwargs,
                verbose=True,
                plots=True,
            )

            # Training completed successfully
            self.current_session.status = 'completed'
            self.current_session.end_time = time.time()

            # Get best metrics
            if hasattr(results, 'results_dict'):
                self.current_session.best_metrics = results.results_dict

            self._trigger_callbacks('on_train_end', self.current_session, results)

        except Exception as e:
            self.current_session.status = 'failed'
            self.current_session.end_time = time.time()
            error_msg = f"Training failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self._trigger_callbacks('on_train_error', error_msg)

    def pause_training(self):
        """Pause the current training session"""
        if self.is_training():
            self.pause_flag.set()
            self.current_session.status = 'paused'

    def resume_training(self):
        """Resume paused training"""
        if self.current_session and self.current_session.status == 'paused':
            self.pause_flag.clear()
            self.current_session.status = 'running'

    def stop_training(self):
        """Stop the current training session"""
        if self.is_training():
            self.stop_flag.set()
            self.current_session.status = 'stopped'

    def is_training(self) -> bool:
        """Check if training is currently in progress"""
        return (self.current_session is not None and
                self.current_session.status in ['running', 'paused'])

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics (thread-safe)"""
        with self.session_lock:
            if self.current_session is None:
                return {}

            return {
                'epoch': self.current_session.current_epoch,
                'total_epochs': self.current_session.total_epochs,
                'progress': self.current_session.get_progress(),
                'elapsed_time': self.current_session.get_elapsed_time(),
                'status': self.current_session.status,
                'metrics': self.current_session.metrics.copy(),  # Return copy for safety
                'best_metrics': self.current_session.best_metrics.copy() if self.current_session.best_metrics else {}
            }

    def resume_from_checkpoint(self, checkpoint_path: Path,
                              config: Dict[str, Any] = None) -> TrainingSession:
        """Resume training from a checkpoint"""
        if self.is_training():
            raise RuntimeError("Training is already in progress")

        if config is None:
            config = {}

        config['resume'] = str(checkpoint_path)

        # Extract data.yaml path from checkpoint if not provided
        if 'data' not in config:
            # Try to find data.yaml in checkpoint directory
            checkpoint_dir = checkpoint_path.parent.parent
            data_yaml = checkpoint_dir / 'data.yaml'
            if not data_yaml.exists():
                raise FileNotFoundError("data.yaml not found for resume training")
        else:
            data_yaml = Path(config['data'])

        return self.start_training(config, data_yaml, str(checkpoint_path))

    def validate_model(self, model_path: Path, data_yaml_path: Path,
                      config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate a trained model
        Args:
            model_path: Path to model weights
            data_yaml_path: Path to data.yaml
            config: Validation configuration
        Returns:
            Validation metrics
        """
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))

            val_config = config or {}
            val_config['data'] = str(data_yaml_path)

            results = model.val(**val_config)

            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict

            return metrics

        except Exception as e:
            print(f"Validation failed: {e}")
            return {}

    def get_training_history(self) -> Dict[str, list]:
        """Get complete training history"""
        if self.current_session is None:
            return {}

        return self.current_session.metrics

    def get_best_weights_path(self) -> Optional[Path]:
        """Get path to best weights from last training"""
        # First, try to get from current session if available
        if self.current_session is not None:
            project_dir = Path(self.current_session.config.get('project',
                              self.project_path / 'runs' / 'train'))
            name = self.current_session.config.get('name', 'exp')

            best_weights = project_dir / name / 'weights' / 'best.pt'
            if best_weights.exists():
                return best_weights

        # Fallback: Search for most recent training run
        runs_dir = self.project_path / 'runs' / 'train'
        if not runs_dir.exists():
            return None

        # Find all training runs
        training_runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                best_weights = run_dir / 'weights' / 'best.pt'
                if best_weights.exists():
                    # Get modification time for sorting
                    training_runs.append((best_weights, best_weights.stat().st_mtime))

        if not training_runs:
            return None

        # Return the most recent weights file
        training_runs.sort(key=lambda x: x[1], reverse=True)
        return training_runs[0][0]

    def get_last_weights_path(self) -> Optional[Path]:
        """Get path to last weights from last training"""
        # First, try to get from current session if available
        if self.current_session is not None:
            project_dir = Path(self.current_session.config.get('project',
                              self.project_path / 'runs' / 'train'))
            name = self.current_session.config.get('name', 'exp')

            last_weights = project_dir / name / 'weights' / 'last.pt'
            if last_weights.exists():
                return last_weights

        # Fallback: Search for most recent training run
        runs_dir = self.project_path / 'runs' / 'train'
        if not runs_dir.exists():
            return None

        # Find all training runs
        training_runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                last_weights = run_dir / 'weights' / 'last.pt'
                if last_weights.exists():
                    # Get modification time for sorting
                    training_runs.append((last_weights, last_weights.stat().st_mtime))

        if not training_runs:
            return None

        # Return the most recent weights file
        training_runs.sort(key=lambda x: x[1], reverse=True)
        return training_runs[0][0]
