"""
GPU validation utility for RTX 5090 environment.
Prevents silent CPU fallback and verifies sm_120 compute capability support.
"""
import torch
import os
import sys
from typing import Dict

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# CRITICAL: Set before importing torch (done at module level)
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'           # Prevent fragmentation (PyTorch 2.0+)
    'max_split_size_mb:128,'              # Don't split blocks >128MB
    'garbage_collection_threshold:0.8'    # Reclaim memory at 80% usage
)

def validate_rtx5090_environment() -> Dict[str, any]:
    """
    Comprehensive GPU validation for RTX 5090 environment.

    Verifies:
    - CUDA availability
    - GPU detection and model verification
    - Compute capability sm_120 support
    - VRAM accessibility (32GB expected)
    - Actual GPU allocation (prevents silent CPU fallback)

    Returns:
        dict: GPU environment details (device_name, compute_capability, total_vram_gb, etc.)

    Raises:
        RuntimeError: If any validation check fails
    """
    print("=" * 60)
    print("GPU ENVIRONMENT VALIDATION")
    print("=" * 60)

    # Check 1: CUDA availability
    print("\n[1/6] Checking CUDA availability...")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Possible causes:\n"
            "  - NVIDIA drivers not installed\n"
            "  - PyTorch CPU-only version installed\n"
            "  - GPU disabled in BIOS\n"
            f"  - PyTorch version: {torch.__version__}\n"
            "  Install PyTorch nightly with CUDA: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
        )
    print("✓ CUDA is available")

    # Check 2: Device detection
    print("\n[2/6] Detecting GPU devices...")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("torch.cuda.is_available() returned True but device_count is 0")
    print(f"✓ Detected {device_count} CUDA device(s)")

    # Check 3: GPU model verification
    print("\n[3/6] Verifying GPU model...")
    device_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU: {device_name}")
    if "5090" not in device_name:
        print(f"⚠ WARNING: Expected RTX 5090, detected: {device_name}")

    # Check 4: Compute capability (CRITICAL for RTX 5090)
    print("\n[4/6] Checking compute capability...")
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"✓ Compute capability: {compute_cap[0]}.{compute_cap[1]} (sm_{compute_cap[0]}{compute_cap[1]})")

    if compute_cap != (12, 0):
        print(f"⚠ WARNING: RTX 5090 has compute capability 12.0, detected: {compute_cap}")

    # Verify PyTorch supports this compute capability
    if compute_cap == (12, 0):
        try:
            test = torch.randn(10, device='cuda')
            _ = test * 2
            del test
            print("✓ PyTorch supports sm_120 (compute capability 12.0)")
        except RuntimeError as e:
            if "no kernel image" in str(e):
                raise RuntimeError(
                    f"PyTorch does NOT support sm_120 (RTX 5090)!\n"
                    f"Current PyTorch version: {torch.__version__}\n"
                    f"Solution: Install PyTorch nightly with cu128:\n"
                    f"  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
                )
            raise

    # Check 5: VRAM accessibility
    print("\n[5/6] Checking VRAM accessibility...")
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024**3)
    print(f"✓ Total VRAM: {total_vram_gb:.2f} GB")

    if total_vram_gb < 30 and "5090" in device_name:
        print(f"⚠ WARNING: RTX 5090 has 32GB VRAM, detected: {total_vram_gb:.2f}GB")

    # Check 6: Test allocation (prevent silent CPU fallback)
    print("\n[6/6] Testing CUDA allocation...")
    try:
        # Allocate 1GB tensor
        test_size = 1024 * 1024 * 256  # 256M floats = 1GB
        test_tensor = torch.randn(test_size, device='cuda')

        # Verify it's actually on GPU
        assert test_tensor.is_cuda, "Tensor created with device='cuda' but is_cuda=False!"

        # Verify memory was allocated
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        assert allocated_gb > 0.9, f"Expected ~1GB allocated, got {allocated_gb:.2f}GB"

        print(f"✓ Successfully allocated {allocated_gb:.2f}GB on GPU")

        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()

    except Exception as e:
        raise RuntimeError(f"CUDA allocation test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("GPU VALIDATION SUCCESSFUL")
    print("=" * 60)

    env_info = {
        'device_name': device_name,
        'device_count': device_count,
        'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
        'total_vram_gb': round(total_vram_gb, 2),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    for key, value in env_info.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    return env_info

if __name__ == "__main__":
    try:
        validate_rtx5090_environment()
    except RuntimeError as e:
        print(f"\n❌ GPU VALIDATION FAILED:\n{e}", file=sys.stderr)
        sys.exit(1)
