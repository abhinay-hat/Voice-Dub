"""
Model manager for sequential loading/unloading of ML models.
Prevents VRAM exhaustion by ensuring only one model loaded at a time.
"""
import torch
import gc
from typing import Callable, Optional, Any
from pathlib import Path


class ModelManager:
    """
    Manages sequential model loading/unloading for VRAM efficiency.

    Ensures only one model is loaded at a time, properly unloads previous model
    before loading next, and tracks VRAM usage.

    Usage:
        manager = ModelManager()
        model = manager.load_model("whisper", lambda: load_whisper_model())
        # ... use model ...
        model = manager.load_model("seamless", lambda: load_seamless_model())
        # Whisper automatically unloaded before Seamless loads
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize model manager.

        Args:
            verbose: Print VRAM usage during load/unload operations
        """
        self.current_model: Optional[Any] = None
        self.current_model_name: Optional[str] = None
        self.verbose = verbose

    def load_model(self, model_name: str, model_loader_fn: Callable[[], Any]) -> Any:
        """
        Load a model, unloading previous model first.

        Args:
            model_name: Human-readable name for logging
            model_loader_fn: Function that returns loaded model (no arguments)

        Returns:
            Loaded model instance

        Example:
            model = manager.load_model("whisper", lambda: WhisperModel.load())
        """
        # Unload previous model if exists
        if self.current_model is not None:
            if self.verbose:
                print(f"Unloading {self.current_model_name}...")
            self._unload_current_model()

        # Load new model
        if self.verbose:
            print(f"Loading {model_name}...")

        self.current_model = model_loader_fn()
        self.current_model_name = model_name

        # Report VRAM usage
        if self.verbose and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            print(f"  VRAM allocated: {allocated_gb:.2f}GB, reserved: {reserved_gb:.2f}GB")

        return self.current_model

    def unload_current_model(self) -> None:
        """
        Manually unload current model without loading a new one.

        Useful at pipeline end or before long-running non-ML operations.
        """
        if self.current_model is not None:
            if self.verbose:
                print(f"Unloading {self.current_model_name}...")
            self._unload_current_model()

    def _unload_current_model(self) -> None:
        """
        Internal method to properly release model from VRAM.

        Steps:
        1. Move model to CPU (safer than direct deletion)
        2. Delete Python references
        3. Force garbage collection
        4. Clear CUDA cache

        This sequence prevents memory leaks from circular references.
        """
        # Move to CPU first (optional but safer for complex models)
        if hasattr(self.current_model, 'to'):
            try:
                self.current_model.to('cpu')
            except Exception:
                # Some models may not support .to(), skip
                pass

        # Delete references
        del self.current_model
        self.current_model = None
        self.current_model_name = None

        # Force garbage collection (clears Python circular references)
        gc.collect()

        # Clear CUDA cache (releases unused memory back to GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Report VRAM after cleanup
        if self.verbose and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"  VRAM cleared: {allocated_gb:.2f}GB allocated")

    def get_current_model_name(self) -> Optional[str]:
        """Return name of currently loaded model, or None."""
        return self.current_model_name

    def __del__(self):
        """Cleanup on garbage collection."""
        if self.current_model is not None:
            self._unload_current_model()
