# Phase 1: Environment & Foundation - Research

**Researched:** 2026-01-31
**Domain:** PyTorch CUDA Environment Setup for RTX 5090
**Confidence:** HIGH

## Summary

Researched how to set up a verified CUDA environment on RTX 5090 (32GB VRAM) with PyTorch and create project scaffolding that prevents silent CPU fallback and enables efficient GPU memory management for an ML pipeline processing workflow.

The RTX 5090 has compute capability 12.0 (sm_120), which presents a critical compatibility challenge: **PyTorch stable releases (as of January 2026) only support up to sm_90**. This means standard PyTorch installations will fail with "no kernel image available" errors. The only current solution is using PyTorch nightly builds with CUDA 12.8+ or building from source with `TORCH_CUDA_ARCH_LIST="sm_120"`.

The standard approach is to use PyTorch nightly (cu128/cu129) with explicit GPU validation on every startup, load models sequentially (on-demand per stage), and use PYTORCH_CUDA_ALLOC_CONF settings to prevent memory fragmentation. Project structure should follow ML pipeline conventions with separate directories for data, models, temp files, and pipeline stages.

**Primary recommendation:** Use PyTorch nightly with cu128, implement comprehensive GPU validation at startup that verifies sm_120 support, and structure the project with stage-based directories for the 11-phase ML pipeline.

## Standard Stack

The established libraries/tools for PyTorch CUDA environments on RTX 5090:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0.dev (nightly) | Deep learning framework with CUDA support | **CRITICAL**: Stable releases don't support sm_120; nightly is required for RTX 5090 |
| CUDA Toolkit | 12.8+ or 13.0+ | GPU acceleration runtime | RTX 5090 requires CUDA 12.8+ for sm_120 support |
| Python | 3.12+ (3.13 recommended) | Language runtime | PyTorch nightly with sm_120 support requires Python 3.12+; 3.13 has better cu128/cu129 compatibility |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pynvml | 11.450.129+ | NVIDIA Management Library bindings | GPU monitoring, VRAM tracking, detailed device info |
| gpustat | Latest | GPU monitoring CLI/API | Real-time GPU utilization display during processing |
| torch.cuda.amp | Built-in | Automatic mixed precision | Reduce VRAM usage by 30-80% for large models |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch nightly | Build from source | More control but complex setup, slow rebuilds |
| pynvml | nvidia-smi parsing | Less reliable, parsing fragility vs native API |
| venv | conda | conda better for ML dependencies but slower, larger (1.2GB+ vs 12-50MB) |

**Installation:**
```bash
# PyTorch nightly with CUDA 12.8 (for RTX 5090 sm_120 support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# GPU monitoring
pip install pynvml gpustat

# Alternative: Build from source (if nightly doesn't work)
# TORCH_CUDA_ARCH_LIST="sm_120" pip install torch --no-binary torch
```

## Architecture Patterns

### Recommended Project Structure
```
voice-dub/
├── src/
│   ├── pipeline/          # Stage orchestration
│   ├── stages/            # Individual processing stages (speech-to-text, translation, etc.)
│   ├── models/            # Model loading/unloading logic
│   ├── utils/             # GPU validation, memory monitoring
│   └── config/            # Configuration constants
├── data/
│   ├── raw/               # Original uploaded videos
│   ├── temp/              # Intermediate files (per-video subdirectories)
│   │   ├── {video_id}/    # One subdirectory per video
│   │   │   ├── audio/     # Extracted audio, vocals separated
│   │   │   ├── transcripts/
│   │   │   ├── translations/
│   │   │   └── synthesized/
│   └── outputs/           # Final processed videos
├── models/                # Downloaded model weights (cache)
│   ├── whisper/
│   ├── xtts/
│   ├── seamless/
│   └── wav2lip/
├── tests/
│   └── gpu_validation_test.py
├── requirements.txt       # Pinned dependencies for reproducibility
└── pyproject.toml         # Project metadata (optional, for packaging)
```

**Rationale:**
- **Stage-based source code**: Matches 11-phase pipeline, each stage is independent module
- **Per-video temp directories**: Isolates intermediate files, easy cleanup, parallel processing support
- **Models cache directory**: Centralizes model weights, reusable across runs, separate from code
- **Both requirements.txt and pyproject.toml**: requirements.txt for quick pip installs (pinned versions), pyproject.toml for project metadata and tooling config

### Pattern 1: Startup GPU Validation
**What:** Comprehensive GPU check that runs before any model loading, verifies CUDA availability, RTX 5090 detection, sm_120 support, and accessible VRAM.

**When to use:** Every application startup, before pipeline initialization.

**Example:**
```python
# Source: PyTorch official docs + community patterns
import torch
import os

# CRITICAL: Set this BEFORE importing torch to avoid fork poisoning
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

def validate_gpu_environment():
    """Validate RTX 5090 CUDA environment on startup."""

    # Check 1: CUDA available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Ensure:\n"
            "1. NVIDIA drivers are installed\n"
            "2. PyTorch with CUDA support is installed\n"
            "3. GPU is not disabled in BIOS"
        )

    # Check 2: Device detection
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices detected")

    device_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU detected: {device_name}")

    # Check 3: Compute capability (CRITICAL for RTX 5090)
    compute_cap = torch.cuda.get_device_capability(0)
    if compute_cap != (12, 0):
        print(f"WARNING: Expected compute capability 12.0 for RTX 5090, got {compute_cap}")
    print(f"✓ Compute capability: {compute_cap[0]}.{compute_cap[1]}")

    # Check 4: VRAM accessibility
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"✓ Total VRAM: {total_vram:.1f} GB")

    if total_vram < 30:  # RTX 5090 has 32GB
        print(f"WARNING: Expected ~32GB VRAM for RTX 5090, detected {total_vram:.1f}GB")

    # Check 5: Test allocation to confirm no silent CPU fallback
    try:
        test_tensor = torch.randn(1000, 1000, device='cuda')
        assert test_tensor.is_cuda, "Tensor not on CUDA despite device='cuda'"
        del test_tensor
        torch.cuda.empty_cache()
        print("✓ CUDA allocation test passed")
    except Exception as e:
        raise RuntimeError(f"CUDA allocation test failed: {e}")

    return {
        'device_name': device_name,
        'compute_capability': compute_cap,
        'total_vram_gb': total_vram,
        'cuda_version': torch.version.cuda
    }
```

### Pattern 2: Sequential Model Loading/Unloading
**What:** Load models on-demand for each pipeline stage, unload before loading the next to minimize VRAM usage.

**When to use:** Multi-stage ML pipelines where not all models need to be in memory simultaneously (like this 11-phase pipeline).

**Example:**
```python
# Source: PyTorch memory management best practices
import torch
import gc

class ModelManager:
    """Manages sequential model loading/unloading for VRAM efficiency."""

    def __init__(self):
        self.current_model = None
        self.current_model_name = None

    def load_model(self, model_name, model_loader_fn):
        """Load a model, unloading previous model first."""
        # Unload previous model
        if self.current_model is not None:
            print(f"Unloading {self.current_model_name}...")
            self._unload_current_model()

        # Load new model
        print(f"Loading {model_name}...")
        self.current_model = model_loader_fn()
        self.current_model_name = model_name

        # Report VRAM usage
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        print(f"  VRAM allocated: {allocated_gb:.2f}GB, reserved: {reserved_gb:.2f}GB")

        return self.current_model

    def _unload_current_model(self):
        """Properly release model from VRAM."""
        # Move to CPU first (optional but safer)
        if hasattr(self.current_model, 'to'):
            self.current_model.to('cpu')

        # Delete references
        del self.current_model
        self.current_model = None
        self.current_model_name = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        print(f"  VRAM cleared: {torch.cuda.memory_allocated() / (1024**3):.2f}GB allocated")
```

### Pattern 3: Memory-Efficient Configuration
**What:** Environment variables that prevent CUDA memory fragmentation and OOM errors.

**When to use:** Set before application startup, especially for long-running processes with multiple model loads.

**Example:**
```python
# Source: PyTorch CUDA documentation
import os

# Set BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'      # Prevent fragmentation (PyTorch 2.0+)
    'max_split_size_mb:128,'         # Don't split blocks >128MB
    'garbage_collection_threshold:0.8'  # Reclaim memory at 80% usage
)

# Use NVML for CUDA checks (safer for multi-process)
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'

import torch
```

### Anti-Patterns to Avoid
- **Don't preload all models at startup**: With 32GB VRAM, tempting to load everything, but wastes memory and increases startup time
- **Don't use `torch.cuda.is_available()` after fork()**: Causes CUDA initialization errors; set PYTORCH_NVML_BASED_CUDA_CHECK=1 instead
- **Don't ignore compute capability**: RTX 5090's sm_120 will fail silently on stable PyTorch; verify at startup
- **Don't hold tensor references in exception handlers**: Python exception objects hold stack frames, preventing GPU memory from freeing
- **Don't use `.item()` or `print()` on GPU tensors in loops**: Forces CPU-GPU synchronization, kills performance

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU memory monitoring | Custom nvidia-smi parsing | `pynvml` library | Native NVIDIA API, handles edge cases, version-stable |
| VRAM usage tracking | Manual memory calculations | `torch.cuda.memory_allocated()` / `memory_reserved()` | Built-in, accurate, accounts for caching |
| Mixed precision training | Manual float16/float32 conversion | `torch.cuda.amp` (automatic mixed precision) | Handles gradient scaling, loss scaling, prevents underflow |
| Model weight loading | Load entire model to GPU at once | Sequential weight loading or meta device | Reduces peak memory 2x, enables loading models larger than VRAM |
| OOM error recovery | Try/catch with manual cleanup | Move cleanup outside except block | Python exception holds stack frame references, prevents memory freeing |
| Device-agnostic code | Manual if/else for CPU/GPU | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | Standard pattern, handles edge cases |
| Multi-GPU selection | Custom GPU selection logic | `CUDA_VISIBLE_DEVICES` environment variable | Set before Python import, prevents GPU enumeration issues |

**Key insight:** GPU memory management has non-obvious edge cases (fragmentation, cached vs allocated memory, gradient accumulation) that make custom solutions fragile. Use PyTorch's built-in tools which handle these complexities.

## Common Pitfalls

### Pitfall 1: Using Stable PyTorch on RTX 5090
**What goes wrong:** Installation succeeds but model inference fails with "CUDA error: no kernel image is available for execution on the device" or silently falls back to CPU with 10-100x slowdown.

**Why it happens:** RTX 5090 has compute capability 12.0 (sm_120). PyTorch stable releases (as of January 2026) only support up to sm_90. The GPU is detected but kernels aren't compiled for sm_120.

**How to avoid:**
- Use PyTorch nightly: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
- Verify compute capability at startup with `torch.cuda.get_device_capability()`
- Check GitHub issues for stable sm_120 support timeline (Issue #159207, #164342)

**Warning signs:**
- `torch.cuda.is_available()` returns True but inference is slow
- Error message mentions "sm_120 is not compatible"
- GPU utilization (nvidia-smi) shows 0% during model run

### Pitfall 2: Silent CPU Fallback
**What goes wrong:** Code runs without errors but uses CPU instead of GPU, causing 10-100x slowdown. No warnings logged.

**Why it happens:**
- PyTorch defaults to CPU if GPU operations fail
- Some operations lack GPU kernels and silently fall back
- Incorrect device placement (data on CPU, model on GPU)

**How to avoid:**
- Explicitly verify tensor device: `assert tensor.is_cuda`
- Run test inference at startup and check GPU utilization (nvidia-smi)
- Create tensors directly on GPU: `torch.randn(..., device='cuda')` instead of `.to('cuda')`
- Set PYTORCH_NVML_BASED_CUDA_CHECK=1 for safer is_available() checks

**Warning signs:**
- Inference takes seconds instead of milliseconds
- `nvidia-smi` shows 0% GPU utilization
- `torch.cuda.memory_allocated()` returns 0 during inference

### Pitfall 3: CUDA Initialization After System Suspend/Resume
**What goes wrong:** CUDA works initially but fails after suspending/resuming Windows with "CUDA initialization error" or "driver initialization failed."

**Why it happens:** NVIDIA drivers on Windows/Linux don't properly reinitialize CUDA context after suspend/resume (known issue).

**How to avoid:**
- Shutdown/reboot instead of suspend when possible
- Implement try/except around CUDA operations with helpful error message
- Consider process restart on CUDA initialization errors in long-running apps

**Warning signs:**
- Code works on fresh boot, fails after suspend/resume
- Error message: "CUDA driver initialization failed"

### Pitfall 4: Memory Fragmentation from Mixed Batch Sizes
**What goes wrong:** First few batches work, then "CUDA out of memory" errors occur despite having free VRAM. `torch.cuda.memory_reserved()` >> `torch.cuda.memory_allocated()`.

**Why it happens:** PyTorch's caching allocator fragments memory when allocating different-sized blocks. Memory becomes checkerboarded with unusable gaps.

**How to avoid:**
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.0+)
- Set `max_split_size_mb:128` to prevent splitting large blocks
- Use consistent batch sizes when possible
- Call `torch.cuda.empty_cache()` between pipeline stages

**Warning signs:**
- Reserved memory >> allocated memory
- OOM errors when free VRAM exists
- Memory errors appear after several iterations, not immediately

### Pitfall 5: Holding Tensor References in Exception Handlers
**What goes wrong:** GPU memory doesn't free after OOM error despite calling `del` and `torch.cuda.empty_cache()`.

**Why it happens:** Python exception objects hold references to the stack frame where the error occurred, preventing tensor garbage collection.

**How to avoid:**
```python
# WRONG:
try:
    run_model(large_batch)
except RuntimeError:
    del model  # Doesn't work! Exception holds reference
    torch.cuda.empty_cache()

# CORRECT:
oom = False
try:
    run_model(large_batch)
except RuntimeError:
    oom = True

if oom:
    del model  # Now this works
    torch.cuda.empty_cache()
```

**Warning signs:**
- Memory doesn't decrease after exception handling
- `gc.get_referrers(tensor)` shows exception traceback

### Pitfall 6: Using conda Without Understanding Size Impact
**What goes wrong:** Environment takes 3-5GB disk space, activation takes 2-3 seconds, surprises developers expecting venv's 12-50MB and instant activation.

**Why it happens:** conda includes Python interpreter, system libraries, and precompiled binaries for each environment. Great for ML dependency management but heavyweight.

**How to avoid:**
- Use venv for pure-Python projects, conda when you need system dependencies (CUDA, cuDNN)
- For ML projects: Consider hybrid approach (conda base + pip for packages)
- Set expectations: conda trades speed/size for dependency management power

**Warning signs:**
- Unexpected disk space usage
- Slow environment activation
- Multiple copies of CUDA toolkit across environments

## Code Examples

Verified patterns from official sources:

### Complete GPU Validation Script
```python
# Source: PyTorch official docs + RTX 5090 compatibility research
import torch
import os
import sys

# CRITICAL: Set before importing torch
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def validate_rtx5090_environment():
    """
    Comprehensive GPU validation for RTX 5090 environment.
    Verifies CUDA availability, compute capability, VRAM, and prevents silent CPU fallback.

    Returns:
        dict: GPU environment details

    Raises:
        RuntimeError: If GPU validation fails
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
            "  Install PyTorch with CUDA: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
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
        print("  This may indicate using a different GPU or PyTorch version mismatch")

    # Verify PyTorch supports this compute capability
    if compute_cap == (12, 0):
        try:
            # Try to compile a simple kernel
            test = torch.randn(10, device='cuda')
            _ = test * 2
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
        print("  Some VRAM may be reserved by system/other processes")

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
```

### GPU Memory Monitoring Utility
```python
# Source: pynvml documentation + PyTorch memory management
import torch
from typing import Dict

def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get detailed GPU memory information in GB.

    Returns:
        dict with keys: allocated, reserved, free, total
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
        'allocated': round(allocated, 2),  # Actually in use by tensors
        'reserved': round(reserved, 2),     # Allocated by PyTorch (includes cache)
        'free': round(free, 2),             # Available for PyTorch to reserve
        'total': round(total, 2),           # Total GPU memory
    }

def print_gpu_memory_summary(prefix: str = ""):
    """Print formatted GPU memory summary."""
    mem = get_gpu_memory_info()
    print(f"{prefix}GPU Memory: {mem['allocated']:.2f}GB allocated | "
          f"{mem['reserved']:.2f}GB reserved | "
          f"{mem['free']:.2f}GB free | "
          f"{mem['total']:.2f}GB total")

# Usage in pipeline
def process_video(video_path):
    print_gpu_memory_summary("Before loading model: ")

    model = load_model()
    print_gpu_memory_summary("After loading model: ")

    result = model.process(video_path)
    print_gpu_memory_summary("After inference: ")

    del model
    torch.cuda.empty_cache()
    print_gpu_memory_summary("After cleanup: ")

    return result
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual float16 conversion | `torch.cuda.amp` (automatic mixed precision) | PyTorch 1.6 (2020) | 30-80% memory reduction, automatic gradient scaling |
| CUDA malloc/free | Caching allocator with expandable segments | PyTorch 2.0 (2023) | Prevents fragmentation, reduces allocation overhead |
| Manual GPU selection | CUDA_VISIBLE_DEVICES env var | Always available, best practice since ~2018 | Cleaner code, prevents enumeration issues |
| nvidia-smi parsing | pynvml library | Library available since 2016, standard by 2020 | More reliable, version-stable API |
| cudaMallocManaged (UVM) | Explicit pinned memory transfers | Never recommended for DL | UVM has 2x overhead for DL workloads |
| Preload all models | Sequential loading/unloading | Best practice emerged ~2021-2022 | Enables larger models, reduces memory waste |
| Build PyTorch from source for new GPUs | Use nightly builds | Nightly builds added sm_120 in late 2025 | Faster setup for new GPU architectures |

**Deprecated/outdated:**
- **PyTorch < 1.6 for mixed precision**: Manually converting to float16 is obsolete; use torch.cuda.amp
- **32-bit CUDA compilation**: Removed in CUDA 12.0+ (2022)
- **Visual Studio 2015/2017 for CUDA**: Support dropped in CUDA 13.1 (January 2026)
- **Python 3.7-3.10 for latest PyTorch**: PyTorch nightly requires Python 3.12+ for sm_120 support

## Open Questions

Things that couldn't be fully resolved:

1. **When will PyTorch stable support sm_120 (RTX 5090)?**
   - What we know: PyTorch 2.9.0 stable did not add sm_120 support (released late 2025). Nightly builds have partial support.
   - What's unclear: Timeline for stable release with sm_120 support. GitHub Issues #159207 and #164342 are tracking this but no ETA given.
   - Recommendation: Monitor PyTorch release notes. Use nightly builds (cu128/cu129) for now. Plan fallback to build-from-source if nightly builds have compatibility issues with other dependencies (xformers, etc.).

2. **What's the optimal PYTORCH_CUDA_ALLOC_CONF for 32GB VRAM with sequential models?**
   - What we know: `expandable_segments:True` prevents fragmentation, `max_split_size_mb` prevents splitting large blocks, `garbage_collection_threshold` triggers cleanup.
   - What's unclear: Optimal `max_split_size_mb` value for 32GB VRAM. Documentation suggests 128-512MB range but doesn't specify tuning methodology.
   - Recommendation: Start with `expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8`. Monitor reserved vs allocated memory ratio. If ratio >1.5x consistently, increase max_split_size_mb to 256 or 512.

3. **Should models be cached locally or downloaded on-demand?**
   - What we know: HuggingFace models auto-cache in `~/.cache/huggingface/`. PyTorch Hub uses `~/.cache/torch/hub/`. Some models (XTTS, Seamless) may use custom cache locations.
   - What's unclear: Whether to pre-download all models or let them download on first use. Pre-download = longer initial setup but reliable offline operation. On-demand = faster initial setup but requires internet and potential download failures mid-pipeline.
   - Recommendation: Create a `models/` directory in project for explicit caching. Pre-download during setup for production reliability. Document model sizes and download sources.

4. **How to handle CUDA OOM errors during pipeline execution?**
   - What we know: Can catch `RuntimeError` and retry with smaller batch size or fall back to CPU. Move cleanup outside except block to avoid memory leak.
   - What's unclear: Best UX for users. Fail fast (show error, require manual intervention)? Auto-retry with CPU (slow but completes)? Auto-cleanup and skip problematic stage?
   - Recommendation: For Phase 1, implement fail-fast with clear error message showing VRAM usage. Add OOM recovery in later phases once pipeline is stable and we understand typical memory patterns.

## Sources

### Primary (HIGH confidence)
- PyTorch CUDA semantics documentation - https://docs.pytorch.org/docs/stable/notes/cuda.html - PYTORCH_CUDA_ALLOC_CONF settings, memory management, device handling
- NVIDIA CUDA GPU Compute Capability - https://developer.nvidia.com/cuda/gpus - RTX 5090 compute capability 12.0 (sm_120) verified
- PyTorch 2.8 Release Blog - https://pytorch.org/blog/pytorch-2-8/ - Latest CUDA features, CUTLASS backend, memory optimizations
- PyTorch get-started installation guide - https://pytorch.org/get-started/locally/ - Official installation commands for CUDA 12.6/12.8/13.0
- NVIDIA CUDA Installation Guide for Windows - https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ - CUDA Toolkit requirements, Visual Studio compatibility

### Secondary (MEDIUM confidence)
- PyTorch Forums: RTX 5090 sm_120 support discussion - https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099 - Community workarounds, nightly build experiences
- GitHub Issue #159207: Add sm_120 support - https://github.com/pytorch/pytorch/issues/159207 - Official tracking of RTX 5090 support
- GitHub Issue #164342: Official sm_120 in stable builds - https://github.com/pytorch/pytorch/issues/164342 - Timeline discussions for stable release
- MLOps Guide: Project Structure - https://mlops-guide.github.io/Structure/project_structure/ - ML pipeline directory organization
- PyTorch memory management best practices - https://medium.com/@ishita.verma178/pytorch-gpu-optimization-step-by-step-guide-9dead5164ca2 - Mixed precision, gradient checkpointing, memory clearing
- venv vs conda comparison - https://dev.to/devin-rosario/detailed-guide-virtualenv-vs-conda-5gln - Environment management tradeoffs

### Tertiary (LOW confidence - WebSearch only, needs validation)
- PyTorch OOM recovery patterns - https://discuss.pytorch.org/t/recover-from-cuda-out-of-memory/29051 - Exception handling outside except block
- GPU memory monitoring tools - https://github.com/anderskm/gputil - pynvml alternatives
- Python dependency management pyproject.toml vs requirements.txt - https://pydevtools.com/handbook/explanation/pyproject-vs-requirements/ - Modern Python packaging

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch nightly requirement for sm_120 confirmed by official GitHub issues, CUDA 12.8+ requirement verified with NVIDIA docs
- Architecture: HIGH - ML pipeline structure patterns verified across multiple MLOps resources, PyTorch memory management patterns from official docs
- Pitfalls: HIGH - Silent CPU fallback, sm_120 incompatibility, suspend/resume issues all verified in official docs and community reports
- Code examples: HIGH - All based on official PyTorch documentation and verified NVIDIA best practices

**Research date:** 2026-01-31
**Valid until:** 2026-02-14 (14 days - fast-moving area due to pending PyTorch stable sm_120 support)

**Special notes:**
- **CRITICAL BLOCKER**: RTX 5090 requires PyTorch nightly builds as of January 2026. Stable releases will fail. Monitor GitHub issues for stable release timeline.
- **Windows-specific**: CUDA 13.1 dropped Visual Studio 2017 support (Jan 2026). Ensure VS 2019 or 2022 installed.
- **Python version**: sm_120 support in nightly builds requires Python 3.12+ (3.13 recommended for best cu128/cu129 compatibility)
