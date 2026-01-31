# Stack Research

**Domain:** AI-Powered Video Dubbing with Voice Cloning
**Researched:** 2026-01-31
**Confidence:** MEDIUM-HIGH

## Recommended Stack

### Core AI Models

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Whisper Large V3** | 1.55B params | Speech-to-text transcription | State-of-the-art ASR with 10-20% error reduction over V2, supports 99 languages, proven ROCm compatibility via CTranslate2. Requires 1-2GB VRAM for inference. |
| **faster-whisper** | Latest (CTranslate2-based) | Whisper optimization | 4x faster than openai/whisper with same accuracy, lower memory usage. Requires custom ROCm build via community forks. |
| **SeamlessM4T v2** | 2.3B params | Translation with prosody preservation | Meta's state-of-the-art multilingual translation, supports speech-to-speech with UnitY2 architecture. Available via Hugging Face Transformers with PyTorch/ROCm support. |
| **XTTS-v2** | Coqui TTS | Voice cloning + emotion transfer | Zero-shot voice cloning from 6-second sample, supports 17 languages with emotional prosody. ROCm support via YellowRoseCx/XTTS-WebUI-ROCm fork. |
| **Wav2Lip** | Original implementation | Lip synchronization | Battle-tested lip sync model with excellent sync quality. Moderate visual quality but proven reliability. Works with PyTorch/ROCm. |

**Confidence:** HIGH for Whisper, SeamlessM4T, XTTS-v2; MEDIUM for faster-whisper ROCm build, Wav2Lip ROCm compatibility

### Deep Learning Framework

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **PyTorch** | 2.9.1+rocm7.2 | Primary ML framework | Official ROCm 7.2 support, all AI models built on PyTorch. Install from AMD's repo.radeon.com, not PyPI (PyPI has CPU-only variant wheels for ROCm 7.x). |
| **ROCm** | 7.2.0 | AMD GPU compute platform | Production-ready release with PyTorch 2.9 support. ROCm 7.1.0+ required for latest features. Supports RDNA3/RDNA4 consumer GPUs. |
| **Transformers** | Latest (4.50+) | Model loading framework | Hugging Face library with native ROCm support, zero code changes needed. Supports Flash Attention 2, GPTQ/AWQ quantization on AMD. |
| **Optimum-AMD** | Latest | AMD-specific optimizations | Official Hugging Face + AMD partnership, provides ROCm-optimized inference for Transformers and ONNX Runtime. |

**Confidence:** HIGH - All verified via official AMD ROCm documentation

### Video Processing

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **FFmpeg** | Latest with AMF | Video encode/decode/processing | Hardware acceleration via AMD AMF (Advanced Media Framework) on Windows or VAAPI on Linux. Essential for video I/O, audio extraction, final muxing. |
| **OpenCV** | 4.x | Frame extraction/manipulation | Industry standard for computer vision tasks. Used for preprocessing frames before lip sync. Python bindings via opencv-python. |
| **MoviePy** | 2.x (Python 3.9+) | High-level video editing | Pythonic API for cuts, concatenations, compositing. Integrates with FFmpeg backend. Use for timeline manipulation, not frame-by-frame processing. |
| **torchaudio** | 2.9.0+rocm7.2 | Audio processing | PyTorch-native audio I/O and transforms. Required by SeamlessM4T. Matches PyTorch version. |

**Confidence:** HIGH for FFmpeg AMF/VAAPI, OpenCV; MEDIUM for MoviePy (verify ROCm compatibility of dependencies)

### Web Backend

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **FastAPI** | 0.100+ | REST API backend | Modern async framework, automatic OpenAPI docs, excellent for AI model serving. Pairs well with Gradio frontend. Production-ready with uvicorn. |
| **Gradio** | 4.x | Web UI framework | Purpose-built for ML demos with one-line Hugging Face hosting. Best for rapid AI model interfaces. Simpler than Streamlit for this use case. |
| **Uvicorn** | Latest | ASGI server | High-performance async server for FastAPI. Required for production deployment. |
| **Pydantic** | 3.x | Data validation | FastAPI dependency, blazing-fast validation (Rust core in v3). Type-safe request/response models. Python 3.9+ required. |

**Confidence:** HIGH - Standard 2025 stack for AI web apps

### Task Queue & Storage

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Redis** | 7.x | Task queue broker + caching | In-memory data store for job queues. Simple, fast, single dependency. Use with RQ for background processing. |
| **RQ (Redis Queue)** | Latest | Background job processing | Simpler than Celery for small-to-medium projects. Low barrier to entry, scales well. Pure Python, easy debugging. |
| **SQLite** | 3.x (Python stdlib) | Job metadata storage | Zero-config embedded database. Perfect for tracking job status, user uploads, processing history. 99% compatible with PostgreSQL for future migration. |
| **tempfile** | Python stdlib | Temporary file management | Built-in module for secure temporary file handling. Auto-cleanup with context managers. Essential for video upload/processing workflow. |

**Confidence:** HIGH - Battle-tested stack for this scale

**Alternative:** Celery + RabbitMQ if you need complex workflows or distributed processing (overkill for single-GPU local tool).

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **sentencepiece** | Latest | Tokenization for SeamlessM4T | Required dependency for Meta translation models |
| **eSpeak NG** | System package | Phoneme synthesis | Required by Coqui TTS for XTTS-v2 voice cloning |
| **scipy** | Latest | Audio file I/O | Writing .wav files, signal processing utilities |
| **numpy** | Latest | Array operations | Universal dependency for ML/video processing |
| **Pillow** | Latest | Image processing | Frame manipulation for lip sync preprocessing |
| **python-dotenv** | Latest | Environment config | Manage ROCm environment variables (HSA_OVERRIDE_GFX_VERSION) |
| **aiofiles** | Latest | Async file I/O | Non-blocking file operations for FastAPI async handlers |
| **python-multipart** | Latest | File upload handling | FastAPI dependency for multipart/form-data uploads |

**Confidence:** HIGH - Standard dependencies for this pipeline

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **Docker** + AMD Container Toolkit | Containerization | Official ROCm Docker images available (rocm/pytorch:latest). Use --device=/dev/kfd --device=/dev/dri flags for GPU access. |
| **pytest** | Testing | Standard Python testing framework |
| **black** | Code formatting | Opinionated formatter, reduces bikeshedding |
| **ruff** | Linting | Blazing-fast Python linter (Rust-based), replaces flake8/pylint |
| **pre-commit** | Git hooks | Automate formatting/linting before commits |

**Confidence:** HIGH

## Installation

### Prerequisites

```bash
# Verify ROCm installation
rocm-smi

# Check ROCm version
cat /opt/rocm/.info/version

# Set GPU compatibility (if needed for consumer AMD GPUs)
# Example for RX 7000 series (gfx1100):
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Core PyTorch + ROCm (Linux)

```bash
# Install ROCm 7.2 if not already installed
# Follow: https://rocm.docs.amd.com/projects/install-on-linux/

# Install PyTorch 2.9.1 with ROCm 7.2 (Python 3.12 required)
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl

pip install torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
            torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
            torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### AI Model Libraries

```bash
# Hugging Face ecosystem
pip install transformers accelerate datasets sentencepiece
pip install optimum[amd]

# Whisper (faster-whisper requires custom ROCm build)
# Use community fork: https://github.com/davidguttman/whisper-rocm
# Or fall back to standard transformers implementation

# Coqui TTS for XTTS-v2 (install after PyTorch)
pip install coqui-tts

# Install eSpeak NG (system package)
sudo apt-get install espeak-ng

# Wav2Lip (clone repo and install deps)
# git clone https://github.com/Rudrabha/Wav2Lip.git
# cd Wav2Lip && pip install -r requirements.txt
```

### Video & Audio Processing

```bash
# Install FFmpeg with AMD support
sudo apt-get install ffmpeg

# Python video libraries
pip install opencv-python moviepy scipy numpy Pillow
```

### Web Backend

```bash
# FastAPI + Gradio stack
pip install fastapi uvicorn gradio pydantic python-multipart aiofiles

# Task queue
pip install redis rq

# Environment management
pip install python-dotenv
```

### Development Tools

```bash
pip install pytest black ruff pre-commit
```

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|-------------------------|
| **ASR** | Whisper Large V3 | Whisper Medium, Distil-Whisper | Medium for faster processing with slight accuracy tradeoff; Distil-Whisper for 6x speedup if accuracy acceptable |
| **Translation** | SeamlessM4T v2 | NLLB-200, M2M-100 | NLLB for more language pairs (200 vs 100); M2M if SeamlessM4T too heavy |
| **TTS** | XTTS-v2 | StyleTTS2, Bark | StyleTTS2 for better quality (but harder setup); Bark for simpler API (but less control) |
| **Lip Sync** | Wav2Lip | LatentSync, SadTalker | LatentSync for higher visual quality (requires 20GB VRAM, Stable Diffusion-based); SadTalker for talking head generation |
| **Web UI** | Gradio | Streamlit, pure FastAPI | Streamlit for more complex dashboards; pure FastAPI if you want custom React/Vue frontend |
| **Task Queue** | RQ | Celery, Dramatiq | Celery for distributed processing or complex workflows; Dramatiq for message reliability over simplicity |
| **Database** | SQLite | PostgreSQL, MySQL | PostgreSQL when scaling to multi-user production; MySQL if already in ecosystem |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **PyPI PyTorch for ROCm 7.x** | PyTorch variant wheels on PyPI only support ROCm 6.3/6.4. ROCm 7.x installs CPU-only version from PyPI. | AMD's repo.radeon.com wheels (see installation above) |
| **Whisper.cpp for this use case** | C++ implementation harder to integrate with Python pipeline, less ecosystem support for ROCm | faster-whisper (Python + CTranslate2) or Transformers Whisper |
| **OpenAI API / cloud TTS** | Project requirement is local/privacy-focused, no cloud APIs | Open-source models (XTTS-v2, StyleTTS2) |
| **CUDA-specific libraries** | cuDNN, CUDA toolkit, NVIDIA-only libraries won't work on AMD | ROCm equivalents or PyTorch-native ops |
| **LatentSync (for now)** | Requires 20-30GB VRAM for training, complex Stable Diffusion pipeline, overkill for initial implementation | Wav2Lip (proven, lower VRAM, simpler) - Consider LatentSync for v2 |
| **Celery + RabbitMQ** | Over-engineered for single-GPU local tool, adds complexity | RQ + Redis (simpler, sufficient for this scale) |

## Stack Patterns by Variant

**If you have 24GB+ VRAM:**
- Use LatentSync instead of Wav2Lip for superior lip sync visual quality
- Load all models in VRAM simultaneously (parallel processing)
- Consider SeamlessExpressive over SeamlessM4T v2 for better prosody

**If you have 8-12GB VRAM:**
- Use Whisper Medium instead of Large V3
- Sequential model loading (offload to RAM between stages)
- Quantize models with GPTQ/AWQ via Optimum-AMD
- Use faster-whisper's int8 quantization

**If targeting Windows instead of Linux:**
- Use AMD AMF for FFmpeg hardware acceleration (instead of VAAPI)
- Install PyTorch from Windows ROCm wheels: https://repo.radeon.com/rocm/windows/rocm-rel-7.2/
- Requires AMD graphics driver 26.1.1+
- Note: Some community ROCm tools (faster-whisper forks) may be Linux-only

**If you need real-time/streaming instead of batch:**
- Replace Gradio with FastAPI + WebSockets
- Use streaming ASR (Whisper streaming mode or alternative)
- Implement chunked processing pipeline
- Consider SeamlessStreaming for real-time translation

## Version Compatibility Matrix

| PyTorch | ROCm | Python | torchaudio | torchvision | Transformers | Notes |
|---------|------|--------|------------|-------------|--------------|-------|
| 2.9.1 | 7.2.0 | 3.12 | 2.9.0 | 0.24.0 | 4.50+ | **Recommended** - Latest production |
| 2.7.x | 6.4 | 3.11 | 2.7.x | 0.22.x | 4.40+ | Fallback if ROCm 7.2 issues |
| 2.2.1 | 5.7+ | 3.9-3.11 | 2.2.x | 0.17.x | 4.30+ | Legacy - not recommended |

**Critical:** All PyTorch ecosystem packages (torch, torchvision, torchaudio) must match the same ROCm version to avoid runtime errors.

## ROCm-Specific Configuration

### Environment Variables

```bash
# Add to ~/.bashrc or .env file

# GPU compatibility override (example for RX 7900 XTX - gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# PyTorch ROCm optimization
export PYTORCH_ROCM_ARCH=gfx1100  # Adjust for your GPU
export HSA_ENABLE_SDMA=0  # Disable if you encounter stability issues
```

### GPU Architecture Mapping

| GPU Series | Architecture | gfx Target | HSA_OVERRIDE_GFX_VERSION |
|------------|--------------|------------|--------------------------|
| RX 7900 XTX/XT | RDNA3 | gfx1100 | 11.0.0 |
| RX 7800/7700 | RDNA3 | gfx1101 | 11.0.1 |
| RX 6900/6800 | RDNA2 | gfx1030 | 10.3.0 |
| RX 6700 | RDNA2 | gfx1031 | 10.3.1 |
| RX 5700 | RDNA | gfx1010 | 10.1.0 |

Check your GPU: `rocminfo | grep gfx`

### Docker Configuration

```bash
# Pull official ROCm PyTorch image
docker pull rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_2.9.1

# Run with GPU access
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_2.9.1

# Or use AMD Container Toolkit (amd-ctk) for cleaner setup
# Install: https://github.com/ROCm/container-toolkit
```

## Performance Optimization Strategies

### Model Optimization
1. **Quantization:** Use Optimum-AMD with AWQ/GPTQ for 2-4x speedup and 50-75% VRAM reduction
2. **Flash Attention 2:** Enable in Transformers for 2x faster attention (supported on ROCm)
3. **torch.compile:** JIT compilation for 4.5x speedup (Whisper), test with your models
4. **Mixed Precision:** Use FP16/BF16 inference to reduce VRAM and increase throughput

### Pipeline Optimization
1. **Async I/O:** Use aiofiles for non-blocking file operations during web uploads
2. **Batching:** Process multiple audio segments in parallel where possible
3. **Model Caching:** Keep models in VRAM between requests (FastAPI app lifespan events)
4. **Sequential vs Parallel:** Sequential loading for low VRAM; parallel for high VRAM

### FFmpeg Hardware Acceleration
```bash
# AMD AMF encoding (Windows)
ffmpeg -i input.mp4 -c:v h264_amf -b:v 5M output.mp4

# VAAPI encoding (Linux)
ffmpeg -vaapi_device /dev/dri/renderD128 -i input.mp4 -vf 'format=nv12,hwupload' -c:v h264_vaapi output.mp4

# Verify hardware acceleration
ffmpeg -hide_banner -encoders | grep -E 'amf|vaapi'
```

## Known Issues & Workarounds

### faster-whisper ROCm
**Issue:** No official CTranslate2 ROCm package
**Workaround:** Use community fork (davidguttman/whisper-rocm) or fall back to Transformers implementation
**Confidence:** MEDIUM - Community fork tested on ROCm 6.4.3, may need updates for 7.2

### XTTS-v2 ROCm
**Issue:** Coqui TTS deprecated, limited ROCm testing
**Workaround:** Use YellowRoseCx/XTTS-WebUI-ROCm fork, confirmed working on ROCm 6.3
**Confidence:** MEDIUM - May require dependency version pinning

### Consumer GPU Compatibility
**Issue:** ROCm officially supports MI-series GPUs; consumer GPU support varies
**Workaround:** Use HSA_OVERRIDE_GFX_VERSION environment variable
**Confidence:** HIGH - Well-documented workaround, widely used

### Wav2Lip Visual Quality
**Issue:** Moderate visual fidelity, visible artifacts on close-ups
**Workaround:** Use Wav2Lip-HD fork or upscale with Real-ESRGAN post-processing
**Alternative:** LatentSync (requires 20GB VRAM)
**Confidence:** HIGH - Known limitation, documented alternatives

## Migration Path

### Phase 1: MVP Stack (Minimum VRAM)
- PyTorch 2.9.1 + ROCm 7.2
- Whisper Medium (faster-whisper or Transformers)
- SeamlessM4T v2 (quantized with Optimum-AMD)
- XTTS-v2
- Wav2Lip
- Gradio UI
- RQ + Redis
- SQLite

**VRAM:** 8-12GB
**Processing Time Target:** <1 hour for 20-min video

### Phase 2: Production Enhancements
- Upgrade to Whisper Large V3
- Add LatentSync as optional high-quality mode
- FastAPI + Gradio (separate frontend/backend)
- PostgreSQL (if multi-user)
- Docker deployment
- Prometheus metrics

**VRAM:** 16-24GB recommended

### Phase 3: Advanced Features
- Real-time streaming mode
- Batch processing for multiple videos
- Fine-tuned models (XTTS-v2 voice fine-tuning)
- Distributed processing (Celery if needed)

## Sources

### Official Documentation (HIGH Confidence)
- [PyTorch ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html) - PyTorch version verification
- [ROCm 7.2.0 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) - ROCm version features
- [PyTorch on ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html) - Installation guide
- [Hugging Face Optimum-AMD](https://huggingface.co/docs/optimum/en/amd/amdgpu/overview) - AMD GPU optimization
- [Transformers ROCm Support](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/hugging-face-models.html) - Hugging Face integration
- [AMD Container Toolkit](https://rocm.blogs.amd.com/software-tools-optimization/amd-container-toolkit/README.html) - Docker deployment
- [FFmpeg AMF Hardware Acceleration](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/wiki/FFmpeg-and-AMF-HW-Acceleration) - Video encoding

### Model Documentation (HIGH Confidence)
- [Whisper Large V3 - Hugging Face](https://huggingface.co/openai/whisper-large-v3) - Model specs
- [XTTS-v2 - Hugging Face](https://huggingface.co/coqui/XTTS-v2) - Voice cloning model
- [SeamlessM4T v2 - Hugging Face](https://huggingface.co/facebook/seamless-m4t-v2-large) - Translation model
- [Wav2Lip - GitHub](https://github.com/Rudrabha/Wav2Lip) - Lip sync implementation

### Community Resources (MEDIUM Confidence)
- [CTranslate2 ROCm Blog](https://rocm.blogs.amd.com/artificial-intelligence/ctranslate2/README.html) - faster-whisper backend
- [XTTS-WebUI-ROCm](https://github.com/YellowRoseCx/XTTS-WebUI-ROCm) - ROCm fork
- [davidguttman/whisper-rocm](https://github.com/davidguttman/whisper-rocm) - faster-whisper ROCm build
- [LatentSync - ByteDance](https://github.com/bytedance/LatentSync) - Advanced lip sync alternative

### Framework Comparisons (MEDIUM Confidence)
- [Gradio vs Streamlit 2025](https://www.squadbase.dev/en/blog/streamlit-vs-gradio-in-2025-a-framework-comparison-for-ai-apps) - Web UI decision
- [Celery vs RQ 2025](https://generalistprogrammer.com/comparisons/celery-vs-rq) - Task queue comparison
- [VidGear - Video Processing Framework](https://github.com/abhiTronix/vidgear) - Alternative video processing

### Blog Posts & Tutorials (LOW-MEDIUM Confidence)
- [AI Dubbing Pipeline 2025](https://www.siliconflow.com/articles/en/best-open-source-AI-models-for-dubbing) - Architecture overview
- [Python AsyncIO Pipelines](https://johal.in/python-asyncio-patterns-building-scalable-concurrent-web-servers-with-uvicorn-in-2025/) - Concurrency patterns
- [Pydantic v3 2025](https://codemagnet.in/2025/12/15/pydantic-v3-the-new-standard-for-data-validation-in-python-why-everything-changed-in-2025/) - Validation library update

---
*Stack research for: AI-Powered Video Dubbing with Voice Cloning*
*Researched: 2026-01-31*
*Next Step: Use this stack to inform roadmap phase structure and technology selection*
