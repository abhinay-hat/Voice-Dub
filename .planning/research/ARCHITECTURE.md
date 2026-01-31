# Architecture Research

**Domain:** AI Video Dubbing System
**Researched:** 2026-01-31
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB INTERFACE                             │
│                    (Gradio Web UI)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Upload      │  │  Progress    │  │  Download    │          │
│  │  Component   │  │  Tracking    │  │  Component   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
├─────────┴──────────────────┴──────────────────┴──────────────────┤
│                     BACKEND LAYER                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Task Queue Manager                          │    │
│  │          (Async/Queue-based Processing)                  │    │
│  └─────────────────────┬───────────────────────────────────┘    │
├────────────────────────┴──────────────────────────────────────────┤
│                     PROCESSING PIPELINE                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Video    │→ │  ASR     │→ │  Trans   │→ │   TTS    │→       │
│  │ Extract  │  │ Whisper  │  │ Seamless │  │  XTTS    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                   ↓              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┘              │
│  │ Final    │← │ Lip Sync │← │                                  │
│  │ Merge    │  │ Wav2Lip  │  │                                  │
│  └──────────┘  └──────────┘  └──────────────────────────────────┤
│                      ML MODEL LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  Model Manager   │  │  GPU Scheduler   │                     │
│  │  (Load/Unload)   │  │  (ROCm/PyTorch)  │                     │
│  └──────────────────┘  └──────────────────┘                     │
├──────────────────────────────────────────────────────────────────┤
│                      STORAGE LAYER                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Uploads  │  │Intermediate│ │  Models  │  │ Outputs  │        │
│  │ (Temp)   │  │  Files     │  │ (Cached) │  │ (Temp)   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Web UI Layer** | File upload/download, progress display, parameter configuration | Gradio interface with File components, status updates |
| **Task Queue** | Job management, worker distribution, status tracking | Python asyncio.Queue or RQ (Redis Queue) for simpler architecture |
| **Video Extractor** | Split video into audio + video frames, extract metadata | FFmpeg with Python bindings (ffmpeg-python or subprocess) |
| **ASR Module** | Transcribe audio to text with timestamps | Whisper Large V3 via Hugging Face transformers pipeline |
| **Translation Module** | Translate text to target language | Meta SeamlessM4T-v2 (multitask-UnitY2 architecture) |
| **TTS Module** | Generate voice-cloned speech in target language | XTTS-v2 with 6-second reference audio |
| **Lip Sync Module** | Synchronize video lips to new audio | Wav2Lip (encoder-decoder architecture) |
| **Video Merger** | Combine synced video + dubbed audio | FFmpeg merge with proper codec settings |
| **Model Manager** | Load/unload models to GPU, manage VRAM | PyTorch model loading with lazy initialization |
| **GPU Scheduler** | Allocate GPU resources, handle OOM | ROCm runtime with PyTorch device management |
| **Storage Manager** | Handle temp files, cleanup, caching | Python tempfile module with context managers |

## Recommended Project Structure

```
voice_dub/
├── app.py                    # Gradio web interface entry point
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration (GPU, paths, model settings)
│   └── models.py            # Model paths and versions
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline orchestrator
│   ├── queue_manager.py     # Task queue and worker management
│   └── storage.py           # File storage and cleanup logic
├── models/
│   ├── __init__.py
│   ├── base.py              # Base model class with GPU management
│   ├── asr.py               # Whisper ASR wrapper
│   ├── translation.py       # Seamless translation wrapper
│   ├── tts.py               # XTTS-v2 wrapper
│   └── lipsync.py           # Wav2Lip wrapper
├── processors/
│   ├── __init__.py
│   ├── video_extractor.py   # FFmpeg video/audio splitting
│   ├── audio_processor.py   # Audio normalization, format conversion
│   └── video_merger.py      # FFmpeg video/audio merging
├── utils/
│   ├── __init__.py
│   ├── gpu_utils.py         # ROCm/GPU monitoring and recovery
│   ├── error_handler.py     # Pipeline error handling and recovery
│   └── logger.py            # Structured logging
├── tests/
│   └── ...                  # Unit and integration tests
└── data/
    ├── uploads/             # Temporary upload storage
    ├── intermediate/        # Pipeline intermediate files
    ├── outputs/             # Final output videos
    └── models/              # Downloaded model weights
```

### Structure Rationale

- **app.py:** Single entry point keeps Gradio interface separate from business logic, enabling easier testing and alternative interfaces later
- **core/:** Pipeline orchestration and queue management - the "brain" that coordinates all processing steps
- **models/:** Each ML model wrapped in its own class with consistent interface (load, unload, infer) for easier GPU memory management
- **processors/:** Video/audio operations isolated from ML models, using FFmpeg for all encoding/decoding to keep pipeline consistent
- **utils/:** Cross-cutting concerns (GPU monitoring, error handling, logging) extracted to avoid duplication
- **data/:** Clear separation of upload, intermediate, output, and model files for easier cleanup and debugging

## Architectural Patterns

### Pattern 1: Pipeline as Sequential Stages

**What:** Break the dubbing workflow into discrete, sequential stages where each stage reads from previous output and writes to next input. Each stage is independent and can be tested/debugged in isolation.

**When to use:** When processing is inherently sequential (can't translate before transcribing) and each stage produces intermediate artifacts worth preserving for debugging.

**Trade-offs:**
- PRO: Clear separation of concerns, easy to debug individual stages, can restart from any stage on failure
- PRO: Intermediate files enable manual inspection and quality checks
- CON: More disk I/O, requires cleanup logic for intermediate files
- CON: Can't pipeline stages (GPU sits idle during video extraction)

**Example:**
```python
class DubbingPipeline:
    """Sequential pipeline with stage isolation"""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.stages = [
            VideoExtractionStage(),
            ASRStage(),
            TranslationStage(),
            TTSStage(),
            LipSyncStage(),
            VideoMergeStage()
        ]

    async def process(self, video_path: Path) -> Path:
        """Execute pipeline stages sequentially"""
        intermediate = video_path

        for i, stage in enumerate(self.stages):
            try:
                logger.info(f"Stage {i+1}/{len(self.stages)}: {stage.name}")
                intermediate = await stage.execute(intermediate, self.work_dir)

                # Save checkpoint after each stage
                self._save_checkpoint(stage.name, intermediate)

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                # Can restart from last checkpoint
                raise PipelineError(stage.name, intermediate) from e

        return intermediate
```

### Pattern 2: Lazy Model Loading with Singleton Manager

**What:** Load ML models only when needed and keep them in memory across requests. Use a ModelManager singleton to prevent loading same model multiple times and handle GPU OOM by unloading least-recently-used models.

**When to use:** When multiple large models share limited GPU memory and loading time is significant (5-30 seconds per model).

**Trade-offs:**
- PRO: Massive performance improvement - model stays loaded across requests
- PRO: Smart unloading prevents OOM without manual intervention
- CON: Adds complexity to model lifecycle management
- CON: First request is slower (cold start)

**Example:**
```python
class ModelManager:
    """Singleton for managing ML model lifecycle on GPU"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._access_times = {}
        return cls._instance

    def get_model(self, model_type: str, model_class: type):
        """Get model, loading if necessary"""
        if model_type not in self._models:
            try:
                logger.info(f"Loading {model_type} to GPU")
                self._models[model_type] = model_class()
                self._models[model_type].to('cuda')
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Free least-recently-used model
                    self._unload_lru()
                    # Retry
                    self._models[model_type] = model_class()
                    self._models[model_type].to('cuda')
                else:
                    raise

        self._access_times[model_type] = time.time()
        return self._models[model_type]

    def _unload_lru(self):
        """Unload least recently used model to free GPU memory"""
        if not self._models:
            return

        lru_model = min(self._access_times, key=self._access_times.get)
        logger.info(f"Unloading {lru_model} to free GPU memory")

        del self._models[lru_model]
        del self._access_times[lru_model]

        # Explicit GPU cleanup
        torch.cuda.empty_cache()
        import gc
        gc.collect()
```

### Pattern 3: Producer-Consumer with Async Queue

**What:** Web UI acts as producer adding jobs to queue, async worker processes consume jobs from queue. Enables parallel processing of multiple videos and graceful handling of long-running tasks.

**When to use:** When processing time exceeds acceptable HTTP request timeout (>2 minutes) and you need to handle multiple concurrent users.

**Trade-offs:**
- PRO: Non-blocking UI, can handle concurrent uploads
- PRO: Worker failures don't crash the web server
- PRO: Easy to scale by adding more workers
- CON: More complex than synchronous processing
- CON: Requires job status tracking and updates

**Example:**
```python
import asyncio
from asyncio import Queue

class QueueManager:
    """Async producer-consumer queue for video processing"""

    def __init__(self, num_workers: int = 1):
        self.queue = Queue()
        self.jobs = {}  # job_id -> status
        self.num_workers = num_workers

    async def start_workers(self):
        """Start worker tasks"""
        tasks = []
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker(i))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def _worker(self, worker_id: int):
        """Worker that processes jobs from queue"""
        while True:
            job_id, video_path, params = await self.queue.get()

            try:
                logger.info(f"Worker {worker_id} processing {job_id}")
                self.jobs[job_id]['status'] = 'processing'

                # Process video through pipeline
                pipeline = DubbingPipeline(work_dir=Path(f"/tmp/{job_id}"))
                result = await pipeline.process(video_path)

                self.jobs[job_id]['status'] = 'completed'
                self.jobs[job_id]['result'] = result

            except Exception as e:
                logger.error(f"Worker {worker_id} failed on {job_id}: {e}")
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['error'] = str(e)

            finally:
                self.queue.task_done()

    def submit_job(self, job_id: str, video_path: Path, params: dict):
        """Producer: Add job to queue"""
        self.jobs[job_id] = {
            'status': 'queued',
            'submitted': time.time()
        }
        self.queue.put_nowait((job_id, video_path, params))
        return job_id
```

### Pattern 4: Context Managers for Temp File Cleanup

**What:** Use Python's context managers (with statements) to ensure temporary files are cleaned up even when errors occur. Critical for video processing which generates large intermediate files.

**When to use:** Always. Video processing generates GBs of temporary files that must be cleaned up.

**Trade-offs:**
- PRO: Guaranteed cleanup even on exceptions
- PRO: No memory leaks or disk space issues
- PRO: Explicit lifecycle makes debugging easier
- CON: Requires discipline to use consistently

**Example:**
```python
from contextlib import contextmanager
import tempfile
import shutil

@contextmanager
def temp_workspace(job_id: str, keep_on_error: bool = False):
    """Create temporary workspace, cleanup on exit"""
    work_dir = Path(tempfile.mkdtemp(prefix=f"dubbing_{job_id}_"))

    try:
        logger.info(f"Created workspace: {work_dir}")
        yield work_dir
    except Exception as e:
        if keep_on_error:
            logger.warning(f"Keeping workspace for debugging: {work_dir}")
        raise
    finally:
        if not keep_on_error or not isinstance(e, Exception):
            logger.info(f"Cleaning workspace: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)

# Usage
async def process_video(video_path: Path):
    job_id = str(uuid.uuid4())

    with temp_workspace(job_id, keep_on_error=DEBUG_MODE) as work_dir:
        # All processing happens in temp workspace
        extractor = VideoExtractor(work_dir)
        audio_path = await extractor.extract_audio(video_path)

        asr = ModelManager().get_model('whisper', WhisperModel)
        transcript = await asr.transcribe(audio_path)

        # ... more processing ...

        # Copy final output before cleanup
        final_output = work_dir / 'final.mp4'
        output_path = Path('data/outputs') / f'{job_id}.mp4'
        shutil.copy(final_output, output_path)

        return output_path
    # Workspace automatically cleaned up here
```

## Data Flow

### Request Flow

```
User Upload (Gradio)
    ↓
File saved to data/uploads/
    ↓
Job submitted to Queue → Job ID returned
    ↓
Queue Manager assigns to Worker
    ↓
Worker creates temp workspace
    ↓
┌─────────────── PIPELINE STAGES ───────────────┐
│                                                │
│  Video Extract → audio.wav, video_frames/     │
│       ↓                                        │
│  Whisper ASR → transcript.json (text+timing)  │
│       ↓                                        │
│  Seamless → translation.json (target lang)    │
│       ↓                                        │
│  XTTS-v2 → dubbed_audio.wav (voice cloned)    │
│       ↓                                        │
│  Wav2Lip → synced_video.mp4 (lips matched)    │
│       ↓                                        │
│  FFmpeg Merge → final_output.mp4              │
│                                                │
└────────────────────────────────────────────────┘
    ↓
Copy to data/outputs/{job_id}.mp4
    ↓
Cleanup temp workspace
    ↓
Update job status → 'completed'
    ↓
User downloads from Gradio interface
```

### GPU Memory Flow

```
┌─── ROCm GPU Memory (e.g., 16GB VRAM) ───┐
│                                          │
│  [CUDA Kernels: ~2GB]                   │
│                                          │
│  ┌──────────────────────────────┐       │
│  │  Active Model (LRU managed)  │       │
│  │                              │       │
│  │  Whisper Large V3: ~6GB      │       │
│  │       OR                     │       │
│  │  XTTS-v2: ~4GB               │       │
│  │       OR                     │       │
│  │  Wav2Lip: ~2GB               │       │
│  └──────────────────────────────┘       │
│                                          │
│  [Inference tensors: ~4-6GB]            │
│                                          │
│  [Free buffer: ~2GB]                    │
│                                          │
└──────────────────────────────────────────┘

Loading strategy:
1. Load model on-demand (lazy loading)
2. Keep in memory across requests
3. On OOM: Unload LRU model, retry
4. Explicit cleanup after each stage
```

### Key Data Flows

1. **Upload to Processing:** Gradio's FileData uploads to temp location → copied to managed upload directory → path passed to queue with job metadata

2. **Intermediate File Chain:** Each stage reads from previous stage output, writes to new file in workspace → enables restart from any stage → all cleaned up at end

3. **Model Loading:** First request loads model to GPU (slow) → subsequent requests reuse loaded model (fast) → OOM triggers LRU unload → model reloads when needed again

4. **Progress Updates:** Each stage updates job status in shared dict → Gradio polls job status → displays progress bar with current stage

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **1 user, local GPU** | Current architecture is perfect. Single worker, async queue optional but recommended. Focus on model caching and GPU memory management. |
| **2-5 concurrent users** | Add Redis-backed queue (RQ) instead of asyncio.Queue for persistence. Increase worker count to 2-3. Consider model quantization (FP16/INT8) to fit more models in VRAM. |
| **5-20 concurrent users** | Move to distributed queue (Celery + Redis). Add result backend for job status. Consider multi-GPU support if available. Add rate limiting to prevent queue overflow. |
| **20+ concurrent users** | Split into microservices (separate ASR, TTS, Lip Sync services). Add load balancer. Consider cloud GPU instances (AWS EC2 G5). Add object storage (S3) for videos. Not recommended for single AMD GPU setup. |

### Scaling Priorities

1. **First bottleneck: GPU memory**
   - Happens when multiple large models loaded simultaneously
   - **Fix:** Implement LRU model unloading (Pattern 2), quantize models to FP16, enable sequential processing rather than parallel

2. **Second bottleneck: Processing queue backup**
   - Happens when job submission rate exceeds processing rate (>1 job per 30-60 min)
   - **Fix:** Add queue depth monitoring, implement rate limiting on uploads, add second worker if GPU memory allows, show estimated wait time to users

3. **Third bottleneck: Disk I/O**
   - Happens when reading/writing large video files becomes slower than GPU processing
   - **Fix:** Use SSD for temp storage, enable FFmpeg hardware encoding (VAAPI/AMF for AMD), reduce intermediate file sizes with compressed formats

## Anti-Patterns

### Anti-Pattern 1: Loading All Models at Startup

**What people do:** Load Whisper, Seamless, XTTS, and Wav2Lip models all at once when app starts, keeping them in memory permanently.

**Why it's wrong:**
- Exceeds typical consumer GPU VRAM (16-24GB) by far - these 4 models need ~20GB combined
- Causes immediate OOM crash on startup or first inference
- Wastes memory when models aren't being used (e.g., Wav2Lip idle while ASR running)

**Do this instead:** Implement lazy loading with ModelManager (Pattern 2). Load each model only when its stage executes, unload LRU model if OOM occurs. For 16GB GPU, keep only 1 large model loaded at a time.

### Anti-Pattern 2: Synchronous Gradio Processing

**What people do:** Make Gradio button click directly call the full pipeline synchronously, blocking the UI until processing completes (30-60 minutes).

**Why it's wrong:**
- UI freezes for entire processing time, user can't see progress
- HTTP timeout kills request after 5-10 minutes
- Can't handle multiple concurrent users
- No way to cancel or restart failed jobs

**Do this instead:** Use async queue pattern (Pattern 3). Submit job to queue immediately, return job ID. Poll job status in separate Gradio component to show progress. Processing happens in background worker.

### Anti-Pattern 3: Ignoring Intermediate File Cleanup

**What people do:** Write intermediate files (extracted audio, frames, translations, etc.) to temp directory without cleanup, or rely on manual deletion.

**Why it's wrong:**
- Each 20-min video generates ~5-10GB of intermediate files
- After 10 videos, disk space exhausted
- Hard to debug which files belong to which job
- Leaves sensitive user data on disk indefinitely

**Do this instead:** Use context managers (Pattern 4) to create per-job workspace that's automatically cleaned up. Structure: `/tmp/dubbing_{job_id}/` with all intermediates inside. Clean up in finally block even on errors.

### Anti-Pattern 4: Keeping Full Video in Memory

**What people do:** Load entire video into RAM as numpy array for frame-by-frame processing.

**Why it's wrong:**
- 20-min 1080p video = ~50GB uncompressed in memory
- Even compressed, multi-GB memory usage
- Python's memory management struggles with large arrays
- OOM kills process, loses all progress

**Do this instead:** Use FFmpeg streaming or frame-by-frame reading with cv2.VideoCapture. Process in chunks (e.g., 5-second segments). Write to disk incrementally. Let FFmpeg handle video codec compression.

### Anti-Pattern 5: Not Using torch.inference_mode()

**What people do:** Run model inference without wrapping in inference context, leaving autograd enabled.

**Why it's wrong:**
- PyTorch tracks gradients by default, consuming extra GPU memory
- Slower inference (20-30% overhead)
- Memory accumulates across batches if not careful

**Do this instead:**
```python
with torch.inference_mode():
    output = model(input_tensor)
```
Or use `@torch.inference_mode()` decorator on inference functions. Disables gradient tracking, reduces memory, speeds up inference.

### Anti-Pattern 6: Catching OOM Without Cleanup

**What people do:**
```python
try:
    model = Model().to('cuda')
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM error")  # Exception holds reference to stack
```

**Why it's wrong:** Python exception objects hold references to stack frames where errors were raised, preventing tensor objects from being freed. GPU memory stays allocated even though model failed to load.

**Do this instead:** Move cleanup code outside except clause:
```python
def load_model():
    try:
        return Model().to('cuda')
    except RuntimeError as e:
        if "out of memory" in str(e):
            return None  # Exception released when function returns

model = load_model()
if model is None:
    torch.cuda.empty_cache()
    gc.collect()
    # Retry or unload LRU
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **FFmpeg** | Subprocess calls via Python subprocess or ffmpeg-python library | Use ffmpeg-python for complex filter graphs, subprocess for simple operations. Always check return codes. Enable hardware encoding for AMD (VAAPI or AMF). |
| **ROCm Runtime** | PyTorch detects automatically via torch.cuda API | Verify with `torch.cuda.is_available()` and `torch.version.hip`. Monitor with `torch.cuda.memory_summary()`. |
| **Hugging Face Hub** | transformers library for model loading | Models download to `~/.cache/huggingface/`. Can specify custom cache dir. Use `local_files_only=True` after first download to avoid network calls. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Web UI ↔ Queue Manager** | Direct function calls (same process) | Gradio callback submits job, gets job_id. Separate callback polls status. Keep interface thin - just submit and query. |
| **Queue ↔ Workers** | asyncio.Queue or Redis queue | For single-GPU local setup, asyncio.Queue sufficient. For multi-machine, use Redis/Celery. Workers should be long-lived processes. |
| **Pipeline ↔ Model Manager** | Singleton instance, direct calls | Pipeline stages call `ModelManager().get_model()`. Manager handles loading/unloading transparently. Pipeline doesn't know about GPU memory. |
| **Models ↔ GPU** | PyTorch device API | All models should support `.to(device)` and `.cpu()`. Use `torch.cuda.empty_cache()` after unloading. Never mix CPU and GPU tensors in operations. |
| **FFmpeg ↔ Python** | File-based (input/output paths) | Python writes config, FFmpeg reads files and writes output. No shared memory. Use temp files for intermediate steps. Clean up with context managers. |

## ROCm-Specific Considerations

### AMD GPU Architecture Integration

**PyTorch on ROCm:** PyTorch on ROCm has significantly expanded CI coverage across multiple GPU generations from MI200 Series to MI350 Series. Consumer GPUs supported include Navi 31 (gfx1100/gfx1101) and Navi 44/48 (gfx1200/gfx1201).

**Device Detection:**
```python
import torch

# Verify ROCm availability
assert torch.cuda.is_available(), "ROCm not detected"
print(f"ROCm version: {torch.version.hip}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Performance Optimizations:**
- **Inductor Compiler:** Use `torch.compile()` with ROCm backend for 20-30% speedup on MI300/MI350 series
- **SDPA Optimization:** Scaled Dot-Product Attention leverages Composable Kernel (CK) backend for better performance
- **Mixed Precision:** ROCm supports FP16 and on gfx950 (MI350/MI355) supports ultra-low precision (MXFP8/MXFP4)

**Memory Management:**
```python
# ROCm memory monitoring
print(torch.cuda.memory_summary(device=0, abbreviated=False))

# Clear cache after unloading model
del model
torch.cuda.empty_cache()
import gc
gc.collect()
```

**Known Limitations:**
- CUDA kernels take ~1-2GB of VRAM on first allocation, reducing usable memory
- Some models compiled for CUDA may need architecture adjustments for ROCm
- Check model compatibility with ROCm before selection

## Build Order Recommendations

Based on dependencies and risk, recommended build order:

### Phase 1: Foundation (No ML models yet)
1. **Project structure** - Set up directories, config, logging
2. **FFmpeg video extractor** - Extract audio/video, verify FFmpeg works
3. **FFmpeg video merger** - Merge audio+video, test different codecs
4. **Storage manager** - Temp file handling with context managers
5. **Gradio basic UI** - Upload, download, progress display

**Why first:** Validates toolchain (FFmpeg, Gradio, ROCm) without ML complexity. Can test with simple passthrough (upload → extract → merge → download).

### Phase 2: Single Model Integration
6. **Model manager base class** - GPU loading/unloading pattern
7. **Whisper ASR integration** - First ML model, establishes pattern
8. **Test ASR end-to-end** - Upload → extract → transcribe → display

**Why second:** Establishes model loading pattern with simplest ML task. Whisper is well-documented and robust. Tests GPU memory management with real model.

### Phase 3: Pipeline Expansion
9. **Queue manager** - Async job processing
10. **Translation module** - SeamlessM4T integration
11. **TTS module** - XTTS-v2 integration
12. **Test audio pipeline** - Upload → ASR → translate → TTS → play

**Why third:** Builds out audio pipeline completely before adding video complexity. Can test quality of voice cloning and translation.

### Phase 4: Lip Sync Integration (Highest Risk)
13. **Wav2Lip module** - Lip sync integration
14. **Full pipeline integration** - Connect all stages
15. **Error handling** - OOM recovery, pipeline restart
16. **Performance optimization** - Model quantization, caching

**Why last:** Lip sync is most complex and least documented component. Having working audio pipeline first lets you test audio quality independently. Lip sync failures are easier to debug when you know audio is correct.

### Dependencies
- Video merger depends on video extractor (both use FFmpeg patterns)
- All ML modules depend on model manager
- Queue manager should come before heavy ML workloads (ASR is okay without queue for testing)
- Lip sync depends on TTS (needs dubbed audio as input)
- Full pipeline depends on all modules

## Error Handling Strategy

### GPU OOM Recovery
```python
def safe_model_inference(model_fn, *args, max_retries=2):
    """Wrapper for model inference with OOM recovery"""
    for attempt in range(max_retries):
        try:
            with torch.inference_mode():
                return model_fn(*args)
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < max_retries - 1:
                logger.warning(f"OOM on attempt {attempt+1}, freeing memory")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
                # Unload LRU model if available
                ModelManager()._unload_lru()
            else:
                raise
    raise RuntimeError("Failed after max retries")
```

### Pipeline Stage Failures
- **Strategy:** Save checkpoint after each stage completes
- **On failure:** Keep intermediate files for debugging, allow manual restart from failed stage
- **Implementation:** Each stage writes `.success` marker file when complete, pipeline checks for marker before re-running

### FFmpeg Failures
- **Common causes:** Corrupted video, unsupported codec, insufficient disk space
- **Detection:** Check FFmpeg return code, parse stderr for error messages
- **Recovery:** Validate input file before processing, provide clear error message to user, suggest re-encoding with supported codec

## Performance Bottlenecks & Mitigations

| Bottleneck | Symptoms | Mitigation |
|------------|----------|------------|
| **Model Loading Time** | First request takes 30-60s | Implement model caching (Pattern 2), consider warm-start by loading Whisper at startup |
| **GPU OOM** | RuntimeError during inference | LRU model unloading, reduce batch size, use FP16 precision, process shorter video segments |
| **Disk I/O** | FFmpeg slower than GPU | Use SSD for temp storage, enable hardware video encoding (VAAPI/AMF), reduce intermediate file writes |
| **Queue Backup** | Jobs waiting 30+ minutes | Add second worker if GPU memory allows, implement rate limiting, show wait time estimate |
| **Large Video Files** | Upload/download slow, disk space issues | Implement file size limits (e.g., 500MB), add compression for intermediate files, stream downloads |

## Sources

**AI Video Dubbing Pipeline Architecture:**
- [AI dubbing in 2026: the complete guide for global business and content leaders](https://www.rws.com/blog/ai-dubbing-in-2026/)
- [The Best Open Source AI Models for Dubbing in 2026](https://www.siliconflow.com/articles/en/best-open-source-AI-models-for-dubbing)
- [How Does AI Video Dubbing Work? (Includes Costs) – 2026 Guide](https://www.aistudios.com/how-to-guides/how-does-ai-video-dubbing-work-includes-costs---2026-guide)

**Video Processing Pipeline Architecture:**
- [Computer Vision Pipeline Architecture: A Tutorial | Toptal](https://www.toptal.com/developers/computer-vision/computer-vision-pipeline)
- [Automate Video Analysis by Using Azure Machine Learning and Azure AI Vision](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/analyze-video-computer-vision-machine-learning)
- [Breaking the Bottleneck: GPU-Optimised Video Processing for Deep Learning](https://towardsdatascience.com/breaking-the-bottleneck-gpu-optimised-video-processing-for-deep-learning/)
- [Real-Time Video Processing with AI: Techniques and Best Practices for 2025](https://www.forasoft.com/blog/article/real-time-video-processing-with-ai-best-practices)

**Gradio Backend Architecture:**
- [Gradio Backend Documentation](https://www.gradio.app/guides/backend)
- [Gradio File Access Guide](https://www.gradio.app/guides/file-access)

**PyTorch GPU & ROCm:**
- [PyTorch for AMD ROCm Platform now available as Python package](https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/)
- [AMD highlights ROCm 7.2.2 at CES 2026](https://videocardz.com/newz/amd-highlights-rocm-7-2-2-at-ces-2026-with-ryzen-ai-400-support-and-a-single-windows-plus-linux-release)
- [PyTorch GPU Optimization: Step-by-Step Guide](https://medium.com/@ishita.verma178/pytorch-gpu-optimization-step-by-step-guide-9dead5164ca2)
- [Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)

**Whisper ASR Integration:**
- [Whisper ASR Webservice Guide](https://www.oreateai.com/blog/whisper-asr-webservice-a-guide-to-realtime-speech-recognition-technology-and-applications/10823af80b518a77ded0228513811b88)
- [WhisperS2T: Optimized Speech-to-Text Pipeline](https://github.com/shashikg/WhisperS2T)
- [Whisper Automatic Speech Recognition with Diarization](https://learnopencv.com/automatic-speech-recognition/)

**Async Queue Architecture:**
- [Python asyncio Queue Documentation](https://docs.python.org/3/library/asyncio-queue.html)
- [Python Background Task Processing in 2025](https://danielsarney.com/blog/python-background-task-processing-2025-handling-asynchronous-work-modern-applications/)
- [Developing an Asynchronous Task Queue in Python](https://testdriven.io/blog/developing-an-asynchronous-task-queue-in-python/)

**Wav2Lip Integration:**
- [Wav2Lip GitHub Repository](https://github.com/Rudrabha/Wav2Lip)
- [Building an Advanced Lip-Sync Engine: Wav2Lip Integration with FastAPI](https://medium.com/@ashagillofficial/building-an-advanced-lip-sync-engine-a-deep-dive-into-wav2lip-integration-with-fastapi-d7c031cd27b2)
- [8 Best Open Source Lip-Sync Models in 2026](https://www.pixazo.ai/blog/best-open-source-lip-sync-models)

**GPU Error Handling:**
- [PyTorch FAQ - OOM Handling](https://docs.pytorch.org/docs/stable/notes/faq.html)
- [How to Fix Tensorflow GPU OOM Error](https://saturncloud.io/blog/how-to-fix-tensorflow-gpu-oom-error/)

**FFmpeg Integration:**
- [How to Use FFmpeg with Python in 2026?](https://www.gumlet.com/learn/ffmpeg-python/)
- [ffmpeg-python Documentation](https://kkroening.github.io/ffmpeg-python/)
- [Merge audio and video files with FFmpeg](https://www.mux.com/articles/merge-audio-and-video-files-with-ffmpeg)

**Temporary File Management:**
- [Temporary Files in Python: A Handy Guide to tempfile](https://www.timsanteford.com/posts/temporary-files-in-python-a-handy-guide-to-tempfile/)
- [Python tempfile Module Documentation](https://docs.python.org/3/library/tempfile.html)
- [Python tempfile Module: Practical Patterns and Pitfalls](https://thelinuxcode.com/python-tempfile-module-practical-patterns-pitfalls-and-real-world-use/)

**XTTS-v2 Voice Cloning:**
- [XTTS-v2 on Hugging Face](https://huggingface.co/coqui/XTTS-v2)
- [Streaming real-time text to speech with XTTS V2](https://www.baseten.co/blog/streaming-real-time-text-to-speech-with-xtts-v2/)
- [The Best Open-Source Text-to-Speech Models in 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)

**Meta Seamless Translation:**
- [Seamless: In-Depth Walkthrough of Meta's Translation Models](https://towardsdatascience.com/seamless-in-depth-walkthrough-of-metas-new-open-source-suite-of-translation-models-b3f22fd2834b/)
- [SeamlessM4T-v2 Documentation](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)
- [Seamless Communication GitHub](https://github.com/facebookresearch/seamless_communication)

**ML Pipeline Patterns:**
- [ML Pipeline Architecture Design Patterns](https://neptune.ai/blog/ml-pipeline-architecture-design-patterns)
- [Serving ML Models in Production: Common Patterns](https://www.anyscale.com/blog/serving-ml-models-in-production-common-patterns)

**Batch Processing Architecture:**
- [Work Queues: The Simplest Form of Batch Processing](https://newsletter.systemdesignclassroom.com/p/work-queues-the-simplest-form-of)
- [Automated Video Processing with FFmpeg and Docker](https://img.ly/blog/building-a-production-ready-batch-video-processing-server-with-ffmpeg/)

---
*Architecture research for: AI Video Dubbing System*
*Researched: 2026-01-31*
*Confidence: HIGH - Comprehensive review of 2026 documentation for all major components (Whisper, XTTS-v2, Wav2Lip, SeamlessM4T, ROCm, PyTorch, Gradio, FFmpeg)*
