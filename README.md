# Voice Dub

Transform videos in any language into English-dubbed versions with cloned voices, preserved emotion, and lip synchronization. Runs entirely on local hardware using open-source AI models.

## Features

- **Voice Cloning**: Generates English audio that clones each speaker's voice characteristics
- **Emotion Preservation**: Maintains emotional tone (excited, calm, angry) in dubbed audio
- **Lip Synchronization**: Synchronizes lip movements to English audio
- **Multi-Language Support**: Transcribes and translates from 20-30+ languages
- **Local Processing**: No cloud services or API fees, runs on RTX 5090 GPU

## Hardware Requirements

- **GPU**: NVIDIA RTX 5090 (32GB VRAM) with compute capability 12.0 (sm_120)
- **CUDA**: Version 12.8 or higher
- **Drivers**: Latest NVIDIA drivers supporting CUDA 12.8+
- **OS**: Windows 10/11, Linux (Ubuntu 22.04+), or macOS with CUDA support

## Software Requirements

- **Python**: 3.11 or higher
- **PyTorch**: Nightly build with CUDA 12.8 support
- **CUDA Toolkit**: 12.8 or higher

## Installation

### 1. Install Python 3.11

Download and install Python 3.11+ from [python.org](https://www.python.org/downloads/).

Verify installation:
```bash
python --version
# Should show: Python 3.11.x or higher
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd voice-dub
```

### 3. Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 4. Install Dependencies

**CRITICAL**: RTX 5090 requires PyTorch nightly builds due to sm_120 compute capability support.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch nightly with CUDA 12.8
- GPU monitoring tools (pynvml, gpustat)
- Supporting libraries

**Installation time**: 5-10 minutes depending on internet speed (PyTorch nightly is ~2GB).

### 5. Verify GPU Environment

Run the validation script to confirm RTX 5090 is properly configured:

```bash
python src/utils/gpu_validation.py
```

Expected output:
```
============================================================
GPU ENVIRONMENT VALIDATION
============================================================

[1/6] Checking CUDA availability...
✓ CUDA is available

[2/6] Detecting GPU devices...
✓ Detected 1 CUDA device(s)

[3/6] Verifying GPU model...
✓ GPU: NVIDIA GeForce RTX 5090

[4/6] Checking compute capability...
✓ Compute capability: 12.0 (sm_120)
✓ PyTorch supports sm_120 (compute capability 12.0)

[5/6] Checking VRAM accessibility...
✓ Total VRAM: 31.84 GB

[6/6] Testing CUDA allocation...
✓ Successfully allocated 1.00GB on GPU

============================================================
GPU VALIDATION SUCCESSFUL
============================================================
```

If validation fails, see [Troubleshooting](#troubleshooting) section.

### 6. Run Test Suite

Verify complete environment setup:

```bash
python tests/test_gpu_environment.py
```

All tests should pass with "ALL TESTS PASSED ✓" message.

## Project Structure

```
voice-dub/
├── src/
│   ├── pipeline/          # Stage orchestration (future)
│   ├── stages/            # Individual processing stages (future)
│   ├── models/            # Model loading/unloading logic
│   ├── utils/             # GPU validation, memory monitoring
│   └── config/            # Configuration constants
├── data/
│   ├── raw/               # Original uploaded videos
│   ├── temp/              # Intermediate files
│   └── outputs/           # Final processed videos
├── models/                # Downloaded model weights cache
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Usage

*(To be added in Phase 2+)*

## Development

### GPU Memory Management

This project uses sequential model loading to maximize VRAM efficiency:

```python
from src.models.model_manager import ModelManager

manager = ModelManager()

# Load Whisper for speech recognition
whisper = manager.load_model("whisper", lambda: load_whisper())
transcripts = whisper.transcribe(audio)

# Automatically unloads Whisper before loading SeamlessM4T
seamless = manager.load_model("seamless", lambda: load_seamless())
translations = seamless.translate(transcripts)
```

### Memory Monitoring

Track VRAM usage during development:

```python
from src.utils.memory_monitor import print_gpu_memory_summary

print_gpu_memory_summary("Before model load: ")
model = load_large_model()
print_gpu_memory_summary("After model load: ")
```

## Troubleshooting

### "CUDA is not available"

**Cause**: PyTorch CPU-only version installed, or NVIDIA drivers missing.

**Solution**:
1. Verify NVIDIA drivers installed: `nvidia-smi`
2. Reinstall PyTorch nightly:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

### "no kernel image is available for execution on the device"

**Cause**: PyTorch stable release installed (doesn't support sm_120).

**Solution**: Install PyTorch nightly as shown above.

### "Expected RTX 5090, detected: [other GPU]"

**Cause**: Running on different GPU hardware.

**Impact**: Project designed for RTX 5090's 32GB VRAM. Other GPUs may work but with reduced capacity or performance.

**Solution**: Proceed with caution, may need to adjust batch sizes or model configurations.

### Compute capability is not 12.0

**Cause**: Not running on RTX 5090 hardware.

**Impact**: sm_120 specific optimizations won't apply.

**Solution**: Verify GPU with `nvidia-smi`, ensure correct hardware.

### PyTorch nightly installation fails

**Cause**: Network issues or incompatible Python version.

**Solution**:
1. Verify Python 3.11+: `python --version`
2. Try alternative CUDA version:
   ```bash
   pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu129
   ```
3. Check PyTorch nightly status: https://pytorch.org/get-started/locally/

## Technical Details

### Why PyTorch Nightly?

RTX 5090 has compute capability 12.0 (sm_120). As of January 2026, PyTorch stable releases only support up to sm_90. PyTorch nightly builds include sm_120 support, making them required for RTX 5090.

### Memory Management

CUDA allocator configured with:
- `expandable_segments:True` - Prevents memory fragmentation
- `max_split_size_mb:128` - Avoids splitting large blocks
- `garbage_collection_threshold:0.8` - Reclaims memory at 80% usage

These settings are automatically applied via `src/utils/gpu_validation.py`.

## License

*(To be added)*

## Contributing

*(To be added)*

---

**Status**: Phase 1 Complete - Environment & Foundation established
**Next**: Phase 2 - Video Processing Pipeline
