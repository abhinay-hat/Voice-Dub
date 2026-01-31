"""
GPU memory monitoring utilities.
Provides real-time VRAM tracking during pipeline execution.
"""
import torch
from typing import Dict


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get detailed GPU memory information in GB.

    Returns:
        dict with keys:
            - allocated: Actually in use by tensors
            - reserved: Allocated by PyTorch (includes cache)
            - free: Available for PyTorch to reserve
            - total: Total GPU memory

    Example:
        mem = get_gpu_memory_info()
        print(f"Using {mem['allocated']:.2f}GB of {mem['total']:.2f}GB")
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}

    # PyTorch's view of memory
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    # Total GPU memory
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Free = total - reserved (reserved includes allocated + cached)
    free = total - reserved

    return {
        'allocated': round(allocated, 2),
        'reserved': round(reserved, 2),
        'free': round(free, 2),
        'total': round(total, 2),
    }


def print_gpu_memory_summary(prefix: str = "") -> None:
    """
    Print formatted GPU memory summary.

    Args:
        prefix: String to prepend to output (e.g., "After loading model: ")

    Example:
        print_gpu_memory_summary("Before inference: ")
        # Output: Before inference: GPU Memory: 8.24GB allocated | 10.50GB reserved | 21.50GB free | 32.00GB total
    """
    mem = get_gpu_memory_info()
    print(f"{prefix}GPU Memory: {mem['allocated']:.2f}GB allocated | "
          f"{mem['reserved']:.2f}GB reserved | "
          f"{mem['free']:.2f}GB free | "
          f"{mem['total']:.2f}GB total")


def get_memory_summary_string() -> str:
    """
    Get memory summary as string (for logging).

    Returns:
        Formatted string with memory info

    Example:
        logger.info(get_memory_summary_string())
    """
    mem = get_gpu_memory_info()
    return (f"GPU Memory: {mem['allocated']:.2f}GB allocated | "
            f"{mem['reserved']:.2f}GB reserved | "
            f"{mem['free']:.2f}GB free")
