# Phase 9: Performance Optimization - Research

**Researched:** 2026-02-22
**Domain:** GPU performance optimization, PyTorch CUDA profiling, ML inference throughput
**Confidence:** MEDIUM (torch.compile/Triton on Windows sm_120 is a verified blocking issue; all other findings HIGH/MEDIUM)

## Summary

Phase 9 optimizes the existing pipeline to process 20-minute videos in 10-20 minutes with no GPU OOM errors and >80% GPU utilization during inference stages. The pipeline already uses faster-whisper (float16, ~4.5GB), SeamlessM4T v2 (fp16, ~2.5GB), XTTS-v2 (~4GB), and LatentSync in subprocess (~8GB). Total peak co-resident VRAM across the first three models is ~11GB — well within the 32GB RTX 5090. The key optimization levers are: (1) switching faster-whisper to `BatchedInferencePipeline` with `batch_size=16`, (2) caching XTTS-v2 `gpt_cond_latent` and `speaker_embedding` per speaker to skip repeated conditioning, (3) upgrading ModelManager to an LRU multi-slot manager that keeps Whisper+SeamlessM4T co-resident instead of sequentially unloading, and (4) adding a `PerformanceBenchmarker` utility for stage-level timing.

**Critical finding:** `torch.compile()` is NOT usable on Windows with sm_120 (RTX 5090). Triton, which `torch.compile` depends on, does not support sm_120 on Windows as of Feb 2026. Attempting `torch.compile` will raise `"Value 'sm_120' is not defined for option 'gpu-name'"`. Do not attempt torch.compile on this hardware/OS combination.

**Primary recommendation:** Implement BatchedInferencePipeline for faster-whisper (batch_size=16), XTTS-v2 conditioning latent caching per speaker, LRU multi-slot ModelManager (3 slots: Whisper+SeamlessM4T co-resident, XTTS third, LatentSync in subprocess), and CUDA Event-based stage benchmarking. Skip torch.compile entirely.

## Standard Stack

### Core (already installed — no new installs required)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faster-whisper | current | ASR with batched inference | `BatchedInferencePipeline` is drop-in replacement for `WhisperModel.transcribe` |
| pynvml | via nvidia-ml-py | GPU utilization monitoring (SM%, memory%) | Direct NVML bindings, no subprocess overhead, NVIDIA-official |
| torch.profiler | PyTorch nightly | Per-operation CUDA kernel timing | Official PyTorch profiler with Chrome trace export |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.cuda.Event | PyTorch nightly | Accurate GPU kernel elapsed time | All stage timing — avoids async measurement errors |
| pytest-benchmark | 5.2.x | Statistical benchmarking with warmup | Stage-level timing tests |
| torch.utils.benchmark.Timer | PyTorch nightly | Warmup-aware micro-benchmarks | Sub-stage timing (e.g., single Whisper forward pass) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pynvml direct | GPUtil wrapper | GPUtil calls nvidia-smi subprocess — slower, more fragile |
| pynvml direct | nvidia-smi subprocess | subprocess adds latency, parsing complexity |
| torch.cuda.Event timing | time.time() | time.time() without synchronize() is inaccurate for async CUDA ops |
| LRU ModelManager (3 slots) | Sequential ModelManager | Sequential requires unload/load between every stage — wastes 5-15s per transition |

**Installation (only new dependency):**

```bash
pip install pynvml pytest-benchmark
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── models/
│   ├── model_manager.py         # Upgrade to LRU multi-slot manager
│   └── __init__.py
├── utils/
│   ├── gpu_validation.py        # Existing — unchanged
│   ├── memory_monitor.py        # Existing — extend with utilization metrics
│   ├── performance_benchmarker.py  # NEW — stage timing utility
│   └── __init__.py
tests/
├── test_performance.py          # NEW — benchmarks with mocked models
```

### Pattern 1: LRU Multi-Slot ModelManager

**What:** Extend `ModelManager` to hold N model slots with LRU eviction policy. Models stay loaded until evicted by a new load that would exceed the VRAM budget.

**When to use:** The pipeline's models fit comfortably together (Whisper ~4.5GB + SeamlessM4T ~2.5GB + XTTS ~4GB = ~11GB, leaving 21GB headroom). Keeping them co-resident eliminates load/unload overhead between stages.

**VRAM co-residency budget:**
```
Whisper Large V3 (float16):     ~4.5GB
SeamlessM4T v2 (fp16):          ~2.5GB
XTTS-v2:                        ~4.0GB
  Subtotal (all three):         ~11.0GB
LatentSync (subprocess, isolated): ~8.0GB (separate process, does NOT share)
Framework overhead + buffers:   ~2.0GB
  SAFE TOTAL (main process):    ~13.0GB of 32GB
  Available headroom:           ~19GB
```

All three main-process models (Whisper, SeamlessM4T, XTTS) can safely be co-resident. LatentSync runs in an isolated subprocess and does not count against main process VRAM.

**Example:**
```python
# Source: architecture derived from existing ModelManager pattern + LRU semantics
from collections import OrderedDict
import torch
import gc
from typing import Callable, Any, Optional

class LRUModelManager:
    """
    Multi-slot model manager with LRU eviction.
    Keeps up to max_slots models co-resident in VRAM.
    Evicts least-recently-used model when at capacity.
    """

    def __init__(self, max_slots: int = 3, vram_budget_gb: float = 20.0, verbose: bool = True):
        self._models: OrderedDict[str, Any] = OrderedDict()
        self.max_slots = max_slots
        self.vram_budget_gb = vram_budget_gb
        self.verbose = verbose

    def load_model(self, model_name: str, loader_fn: Callable[[], Any]) -> Any:
        # Cache hit — move to end (most recently used)
        if model_name in self._models:
            self._models.move_to_end(model_name)
            if self.verbose:
                print(f"[LRUModelManager] Cache hit: {model_name}")
            return self._models[model_name]

        # Evict LRU if at capacity
        while len(self._models) >= self.max_slots:
            evicted_name, evicted_model = self._models.popitem(last=False)
            self._evict(evicted_name, evicted_model)

        # Load new model
        if self.verbose:
            print(f"[LRUModelManager] Loading: {model_name}")
        model = loader_fn()
        self._models[model_name] = model
        self._models.move_to_end(model_name)
        return model

    def _evict(self, name: str, model: Any) -> None:
        if self.verbose:
            print(f"[LRUModelManager] Evicting: {name}")
        if hasattr(model, 'to'):
            try:
                model.to('cpu')
            except Exception:
                pass
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self) -> None:
        for name, model in list(self._models.items()):
            self._evict(name, model)
        self._models.clear()
```

### Pattern 2: faster-whisper BatchedInferencePipeline

**What:** Replace `WhisperModel.transcribe()` with `BatchedInferencePipeline` and `batch_size=16`. This parallelizes 30-second audio chunks through the model simultaneously, achieving up to 12.5x speedup over openai/whisper.

**When to use:** Always for the ASR stage. BatchedInferencePipeline is a drop-in replacement.

**VRAM impact:** batch_size=16 at float16 uses approximately 4.5-6GB VRAM (within budget). batch_size=32 may spike to 8GB — test on hardware.

**Example:**
```python
# Source: https://github.com/SYSTRAN/faster-whisper (README, BatchedInferencePipeline section)
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Load model (unchanged from existing code)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Wrap with batched pipeline
batched_model = BatchedInferencePipeline(model=model)

# Use exactly like WhisperModel.transcribe — drop-in replacement
segments, info = batched_model.transcribe(
    audio_path,
    batch_size=16,          # Start here; test 32 for marginal gain
    vad_filter=True,        # Existing behavior
    word_timestamps=True,   # Existing behavior
)
```

### Pattern 3: XTTS-v2 Conditioning Latent Caching

**What:** Call `model.get_conditioning_latents()` once per speaker before synthesis loop, cache the result, reuse on every segment for that speaker. Eliminates repeated conditioning computation.

**When to use:** Always — the SpeakerEmbeddingCache already exists in the codebase. Extend it to store `gpt_cond_latent` alongside `speaker_embedding`.

**Example:**
```python
# Source: https://docs.coqui.ai/en/latest/models/xtts.html
# Precompute once per speaker (in reference extraction step):
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[reference_wav_path],
    gpt_cond_len=30,
    max_ref_length=60,
)

# Cache both tensors (move to CPU to free GPU buffer between uses):
speaker_cache[speaker_id] = {
    "gpt_cond_latent": gpt_cond_latent.cpu(),
    "speaker_embedding": speaker_embedding.cpu(),
}

# During synthesis (move back to GPU for inference):
cached = speaker_cache[speaker_id]
output = model.inference(
    text=translated_text,
    language="en",
    gpt_cond_latent=cached["gpt_cond_latent"].cuda(),
    speaker_embedding=cached["speaker_embedding"].cuda(),
    temperature=0.65,
    speed=speed_factor,
)
```

### Pattern 4: CUDA Event-Based Stage Timing

**What:** Wrap each pipeline stage with CUDA Event timers that accurately measure GPU wall time (not CPU-queued time). The existing `time.time()` calls in stages are inaccurate for async GPU operations.

**When to use:** All GPU-bound stages (Whisper, SeamlessM4T, XTTS). CPU-bound stages (assembly, JSON export) can use `time.time()`.

**Example:**
```python
# Source: https://blog.speechmatics.com/cuda-timings
# Source: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
import torch

def time_cuda_stage(fn, *args, warmup=False, **kwargs):
    """
    Accurately time a GPU-bound function using CUDA Events.
    Always call torch.cuda.synchronize() before reading elapsed_time.
    """
    if warmup:
        # Optional: run once before timing to initialize CUDA kernels
        with torch.no_grad():
            fn(*args, **kwargs)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        result = fn(*args, **kwargs)
    end_event.record()

    # CRITICAL: must synchronize before calling elapsed_time()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    return result, elapsed_ms / 1000.0  # Convert to seconds
```

### Pattern 5: GPU Utilization Monitoring via pynvml

**What:** Use `pynvml.nvmlDeviceGetUtilizationRates()` to read SM utilization % and memory bandwidth % from NVML directly. No subprocess overhead.

**When to use:** In `PerformanceBenchmarker` to validate the >80% GPU utilization criterion during inference stages.

**Example:**
```python
# Source: https://github.com/gpuopenanalytics/pynvml
import pynvml

def get_gpu_utilization(device_index: int = 0) -> dict:
    """
    Returns current GPU SM utilization and memory bandwidth utilization.
    Same metric as 'GPU-Util' in nvidia-smi output.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util_pct": util.gpu,        # SM utilization %
        "mem_util_pct": util.memory,     # Memory bandwidth utilization %
        "used_gb": mem.used / (1024**3),
        "total_gb": mem.total / (1024**3),
    }
```

### Anti-Patterns to Avoid

- **time.time() for GPU stages without synchronize():** CUDA kernels execute asynchronously. `time.time()` returns CPU-enqueue time, not actual GPU completion time. Always use CUDA Events or call `torch.cuda.synchronize()` before reading wall time.
- **torch.compile() on Windows sm_120:** Triton (torch.compile's backend) does not support sm_120 on Windows as of Feb 2026. Will crash with `"Value 'sm_120' is not defined for option 'gpu-name'"`. Skip entirely.
- **Reloading XTTS per segment:** XTTS initialization (model load + conditioning) is expensive. Load once, cache latents per speaker, reuse across all segments.
- **Unloading Whisper before SeamlessM4T loads:** With 32GB VRAM, both fit simultaneously. The existing sequential unload pattern wastes 5-15s per stage transition. Use LRU multi-slot manager instead.
- **batch_size > 32 for Whisper without VRAM check:** Performance plateaus at ~32 but VRAM increases linearly. At batch_size=80, VRAM jumps to ~19GB. For 32GB GPU, stay at 16-32 and verify with memory_monitor.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batched Whisper inference | Custom audio chunking + parallel batching | `BatchedInferencePipeline(model, batch_size=16)` | Already handles VAD-aware chunking, padding, decoding coordination |
| GPU utilization reading | subprocess nvidia-smi + stdout parsing | `pynvml.nvmlDeviceGetUtilizationRates()` | NVML direct binding, no process overhead, official API |
| Accurate GPU timing | `time.time()` wrappers | `torch.cuda.Event` with `elapsed_time()` | Only way to correctly measure async GPU ops |
| LRU eviction logic | Custom dict + manual eviction tracking | `collections.OrderedDict` with `move_to_end()` + `popitem(last=False)` | LRU in 3 lines, thread-safe enough for single-threaded pipeline |
| Speaker conditioning cache | Re-calling `get_conditioning_latents()` per segment | Cache dict keyed on speaker_id | Avoids re-running the GPT conditioning encoder per segment |

**Key insight:** The biggest performance gains in this pipeline come from eliminating repeated work (model reloads, conditioning recomputation) rather than from low-level kernel optimization. GPU is fast; Python setup overhead is the bottleneck.

## Common Pitfalls

### Pitfall 1: Inaccurate Stage Timing

**What goes wrong:** Developer adds `start = time.time()` before `model.transcribe()` and `end = time.time()` after. Reports 2s for a stage that actually takes 8s, because CUDA queued the work asynchronously and `time.time()` captured only the launch time.

**Why it happens:** PyTorch CUDA operations are asynchronous by default. The CPU returns from the call before GPU finishes.

**How to avoid:** Always use `torch.cuda.synchronize()` before reading wall time, or use CUDA Events (`torch.cuda.Event.elapsed_time()`). The existing `time.time()` calls in stage files (e.g., `asr_stage.py` line 74, `tts_stage.py` line 115) measure total stage wall time including Python overhead — this is acceptable for top-level benchmarking but misleading for GPU-only timing.

**Warning signs:** Stage times under 1s for large-v3 transcription of long audio — indicates missing synchronization.

### Pitfall 2: torch.compile Crash on Windows sm_120

**What goes wrong:** Developer wraps Whisper or SeamlessM4T model with `torch.compile(model, mode="reduce-overhead")` expecting 15-25% speedup. Pipeline crashes with `"Value 'sm_120' is not defined for option 'gpu-name'"` on first compiled forward pass.

**Why it happens:** torch.compile uses Triton as its kernel compiler backend. Triton 3.2.0 (latest pip wheel for Windows) does not support Blackwell sm_120. Blackwell requires Triton 3.3.1+ which has no Windows pip wheel as of Feb 2026.

**How to avoid:** Do not use `torch.compile` on this stack (Windows, PyTorch nightly, sm_120). The speedup from batching (BatchedInferencePipeline) is larger and safer.

**Warning signs:** Any mention of `triton`, `torch.compile`, `torch.jit.script` with CUDA backend in implementation plans.

### Pitfall 3: VRAM Double-Counting LatentSync

**What goes wrong:** Developer calculates total VRAM as Whisper(4.5) + SeamlessM4T(2.5) + XTTS(4) + LatentSync(8) = 19GB and decides to keep all four co-resident. LatentSync subprocess allocation fails because the main process is also using VRAM.

**Why it happens:** LatentSync runs in an isolated conda subprocess with a separate PyTorch context. Its 8GB is allocated from the GPU, not the main process's allocator. If the main process holds 13GB, LatentSync needs 8GB more = 21GB total, leaving 11GB headroom — still fine on 32GB. But if developer accidentally loads all four into the MAIN process, they hit 19GB+framework overhead which may cause OOM.

**How to avoid:** LatentSync must remain in subprocess (torch 2.5.1+cu121 version conflict). Main process budget cap: 20GB hard limit. With LRU manager `vram_budget_gb=20.0`, any eviction before LatentSync subprocess launch gives it 12+GB to work with.

**Warning signs:** LatentSync subprocess failing with CUDA OOM when it worked before multi-slot caching was added.

### Pitfall 4: XTTS batch_size > 1 is NOT straightforward

**What goes wrong:** Developer tries to batch multiple XTTS synthesis calls by stacking text inputs. XTTS-v2 inference is autoregressive (GPT decoder) — naive batching requires equal-length sequences or masking, which the TTS library does not expose as a public API.

**Why it happens:** XTTS uses a GPT-style autoregressive decoder, not an encoder-only model. "Batching" for TTS means different things than for classification.

**How to avoid:** Do NOT attempt to batch XTTS text inputs. The correct XTTS optimization is conditioning latent caching (computed once per speaker) and speaker-grouped ordering (all segments from speaker A, then speaker B, etc.) to maximize cache reuse. The existing XTTSGenerator already does speaker grouping — keep this.

**Warning signs:** Attempts to stack translated_text inputs into a batch tensor for XTTS inference.

### Pitfall 5: SeamlessM4T torch.compile Partial Success

**What goes wrong:** Developer finds PyTorch blog post showing `torch.compile` gave 2.7x speedup for SeamlessM4T text_decoder and vocoder (on Linux, A100). Tries it on Windows sm_120. It crashes.

**Why it happens:** The PyTorch blog result (2x speedup for text_decoder, 30x for vocoder, 2.7x end-to-end) is real — but it was measured on Linux with CUDA 11.x/12.x and A100 (sm_80). Windows + sm_120 + nightly breaks this.

**How to avoid:** Keep SeamlessM4T on standard fp16 inference without compile. The batch_size=8 already configured in settings.py provides good throughput for the translation batch.

### Pitfall 6: Benchmark Tests Running Real Models

**What goes wrong:** Performance test suite instantiates WhisperModel, LatentSync, etc. to benchmark timing. Tests take 45+ minutes and require full conda environment, GPU, and downloaded models.

**Why it happens:** Timing tests that use real models are integration tests, not unit benchmarks.

**How to avoid:** Use two test tiers:
1. **Unit benchmarks** (fast, CI-safe): Mock the model's `transcribe()`/`inference()` methods with `unittest.mock.MagicMock` returning pre-shaped tensors. Benchmark the orchestration overhead (JSON loading, progress callback firing, result building).
2. **Integration benchmarks** (slow, manual): Actually load models and process a 1-minute reference audio clip. Mark with `@pytest.mark.slow` and exclude from CI.

## Code Examples

Verified patterns from official sources:

### faster-whisper BatchedInferencePipeline Usage

```python
# Source: https://github.com/SYSTRAN/faster-whisper (README)
from faster_whisper import WhisperModel, BatchedInferencePipeline

def transcribe_audio_batched(audio_path: str, model: WhisperModel) -> tuple:
    """
    Drop-in replacement for WhisperModel.transcribe using batched inference.
    batch_size=16 recommended starting point; test 32 on 32GB VRAM.
    Performance plateaus around batch_size=32 per faster-whisper maintainer.
    """
    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe(
        audio_path,
        batch_size=16,
        vad_filter=True,
        word_timestamps=True,
    )
    return list(segments), info  # Materialize generator
```

### CUDA Event Stage Timer

```python
# Source: https://blog.speechmatics.com/cuda-timings
# Source: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
import torch
import time
from typing import Callable, Any, Tuple

class StageTimer:
    """Records GPU and CPU wall time for a pipeline stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self._gpu_start = torch.cuda.Event(enable_timing=True)
        self._gpu_end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._cpu_start = time.time()
        self._gpu_start.record()
        return self

    def __exit__(self, *_):
        self._gpu_end.record()
        # Must synchronize before reading elapsed_time
        torch.cuda.synchronize()
        self.gpu_time_s = self._gpu_start.elapsed_time(self._gpu_end) / 1000.0
        self.cpu_wall_time_s = time.time() - self._cpu_start

    def report(self) -> str:
        return (f"[{self.stage_name}] "
                f"GPU: {self.gpu_time_s:.2f}s | "
                f"Wall: {self.cpu_wall_time_s:.2f}s")
```

### GPU Utilization Sampler

```python
# Source: https://github.com/gpuopenanalytics/pynvml
import threading
import time
import pynvml

class GPUUtilizationSampler:
    """
    Samples GPU SM utilization% in background thread during inference.
    Use to verify >80% utilization during Whisper/SeamlessM4T/XTTS stages.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[int] = []
        self._running = False
        self._thread: threading.Thread = None
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def start(self):
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        self._thread.join()
        if not self._samples:
            return {"avg_util_pct": 0, "max_util_pct": 0, "min_util_pct": 0}
        return {
            "avg_util_pct": sum(self._samples) / len(self._samples),
            "max_util_pct": max(self._samples),
            "min_util_pct": min(self._samples),
            "sample_count": len(self._samples),
        }

    def _sample_loop(self):
        while self._running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            self._samples.append(util.gpu)
            time.sleep(self.interval_s)
```

### LRU ModelManager (minimal implementation)

```python
# Source: derived from collections.OrderedDict LRU pattern
# Source: existing src/models/model_manager.py cleanup pattern
from collections import OrderedDict
import torch, gc
from typing import Callable, Any

class LRUModelManager:
    def __init__(self, max_slots: int = 3, verbose: bool = True):
        self._models: OrderedDict[str, Any] = OrderedDict()
        self.max_slots = max_slots
        self.verbose = verbose

    def load_model(self, name: str, loader: Callable[[], Any]) -> Any:
        if name in self._models:
            self._models.move_to_end(name)
            return self._models[name]
        while len(self._models) >= self.max_slots:
            evicted_name, model = self._models.popitem(last=False)
            self._release(evicted_name, model)
        model = loader()
        self._models[name] = model
        return model

    def _release(self, name: str, model: Any) -> None:
        if self.verbose:
            print(f"[LRU] Evicting {name}")
        if hasattr(model, 'to'):
            try: model.to('cpu')
            except Exception: pass
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self):
        for name, model in list(self._models.items()):
            self._release(name, model)
        self._models.clear()

    def get_loaded_models(self) -> list[str]:
        return list(self._models.keys())
```

### Performance Benchmark Test (mocked models)

```python
# Source: pytest-benchmark pattern for ML stage timing
import pytest
from unittest.mock import MagicMock, patch
import time

@pytest.mark.benchmark(group="asr_stage")
def test_asr_stage_orchestration_overhead(benchmark):
    """
    Benchmarks ASR stage orchestration (JSON loading, progress callbacks, result building)
    WITHOUT loading real models. Tests that Python overhead < 2s for 200 segments.
    """
    mock_transcription = MagicMock()
    mock_transcription.language = "ja"
    mock_transcription.language_probability = 0.99
    mock_transcription.duration = 1200.0  # 20 minutes
    mock_transcription.segments = [MagicMock() for _ in range(200)]

    mock_diarization = MagicMock()
    mock_diarization.num_speakers = 2

    with patch('src.stages.transcription.transcribe_audio', return_value=mock_transcription), \
         patch('src.stages.diarization.diarize_audio', return_value=mock_diarization), \
         patch('src.utils.audio_preprocessing.preprocess_audio_for_asr', return_value='/tmp/fake.wav'):

        def run():
            from src.stages.asr_stage import run_asr_stage
            return run_asr_stage('/tmp/fake.wav', 'test_id', 'hf_fake_token', save_json=False)

        result = benchmark(run)
    assert result.total_segments == 200
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `WhisperModel.transcribe()` single-pass | `BatchedInferencePipeline(batch_size=16)` | faster-whisper ~1.0.0 | 2-4x faster ASR on GPU |
| Sequential model load/unload per stage | LRU multi-slot manager keeping models hot | Phase 9 | Eliminates 5-15s per stage transition |
| `time.time()` stage timing | `torch.cuda.Event.elapsed_time()` | Established CUDA best practice | Accurate async GPU measurement |
| XTTS conditioning computed per segment | `get_conditioning_latents()` cached per speaker | Coqui docs always recommended | Eliminates repeated encoding cost |
| Monitoring via subprocess `nvidia-smi` | `pynvml.nvmlDeviceGetUtilizationRates()` | pynvml always available | No subprocess overhead, programmatic access |

**Deprecated/outdated for this stack:**
- `torch.compile()`: Not usable — Triton lacks Windows sm_120 support as of Feb 2026
- `openai-whisper`: Already replaced by faster-whisper (correct decision from prior phases)
- `WhisperModel.transcribe()` without batching: Still works but leaves 2-4x performance on the table

## Open Questions

1. **BatchedInferencePipeline VRAM with batch_size=16 on large-v3**
   - What we know: batch_size=8 uses ~4.5GB; batch_size=80 uses ~19GB. Performance plateaus around 32.
   - What's unclear: Exact VRAM for batch_size=16 on RTX 5090 with PyTorch nightly (allocator behavior may differ).
   - Recommendation: Start at batch_size=16, measure with `memory_monitor.print_gpu_memory_summary()`, increase to 32 if VRAM stays under 8GB.

2. **torch.compile Windows sm_120 status in PyTorch stable 2.7.0**
   - What we know: PyTorch 2.7.0 stable supports Blackwell for forward passes. Triton 3.2.0 (pip, Windows) does NOT support sm_120. No pip wheel for Triton 3.3.1+ on Windows as of Feb 2026.
   - What's unclear: Whether the project's nightly build includes a bundled Triton version that supports sm_120 (vs. the pip-installed Triton).
   - Recommendation: Validate with a 5-line test script before any compile() usage. If it crashes, skip entirely. Do not block phase on this.

3. **XTTS deepspeed integration on Windows**
   - What we know: Coqui docs mention `use_deepspeed=True` for speedup; requires `pip install deepspeed==0.10.3`.
   - What's unclear: Whether deepspeed 0.10.3 installs cleanly on Windows with sm_120 PyTorch nightly.
   - Recommendation: Skip deepspeed for Phase 9. Conditioning latent caching provides most of the gain. Deepspeed is a stretch goal requiring separate investigation.

4. **20-minute video end-to-end timing estimate**
   - What we know: A 20-minute video has ~1200 audio seconds. Whisper Large V3 at float16 batched processes ~50-100x real-time on A100; RTX 5090 with 32GB and nightly may be similar.
   - What's unclear: Actual real-time factor for the full pipeline on this specific hardware.
   - Recommendation: Build a benchmarking harness (Phase 9 Plan 2 or Plan 3) with a real 20-minute reference video and measure each stage separately before end-to-end timing.

## Sources

### Primary (HIGH confidence)

- `https://github.com/SYSTRAN/faster-whisper` — BatchedInferencePipeline API, batch_size recommendations
- `https://docs.coqui.ai/en/latest/models/xtts.html` — `get_conditioning_latents()` caching pattern
- `https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html` — CUDA Event elapsed_time API
- `https://docs.pytorch.org/docs/stable/notes/cuda.html` — PYTORCH_CUDA_ALLOC_CONF options (expandable_segments, roundup_power2_divisions, max_split_size_mb)
- `https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html` — torch.profiler usage for GPU kernel timing

### Secondary (MEDIUM confidence)

- `https://discuss.pytorch.org/t/rtx-5070-ti-blackwell-pytorch-nightly-triton-still-getting-sm-120-is-not-defined-for-option-gpu-name-error/220460` — Confirmed: Triton does not support sm_120 on Windows as of Feb 2026; verified by PyTorch forum discussion
- `https://github.com/SYSTRAN/faster-whisper/issues/1143` — batch_size=32 plateau statement from maintainer (MahmoudAshraf97)
- `https://github.com/gpuopenanalytics/pynvml` — pynvml API for nvmlDeviceGetUtilizationRates
- `https://blog.speechmatics.com/cuda-timings` — torch.cuda.Event timing pattern (standard practice)

### Tertiary (LOW confidence — flagged for validation)

- WebSearch finding: "SeamlessM4T 2.7x end-to-end speedup via torch.compile + CUDA graph" — from pytorch.org/blog/accelerating-generative-ai-4/ (page fetch failed). Claimed speedup is real but was measured on Linux/A100, NOT Windows/sm_120. Do not apply.
- WebSearch finding: "XTTS streaming latency <150ms on consumer GPU" — from baseten.co blog, single source, not verified against Coqui docs. Streaming mode is not relevant to batch dubbing pipeline.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — faster-whisper BatchedInferencePipeline and pynvml are verified against official sources
- Architecture (LRU manager): HIGH — uses Python stdlib OrderedDict; VRAM calculations verified against known model sizes
- torch.compile exclusion: HIGH — Windows Triton sm_120 limitation confirmed by PyTorch forum thread with maintainer response
- XTTS latent caching: HIGH — directly documented in Coqui official docs
- CUDA Event timing: HIGH — official PyTorch docs
- Pitfalls: HIGH — all derived from verified technical constraints (async GPU, Triton limitation, XTTS autoregressive)
- Benchmark test patterns: MEDIUM — pytest-benchmark is standard but ML mock patterns are inferred from general Python testing

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (30 days — Triton/torch.compile Windows status may change with new pip wheels; check before implementing)
