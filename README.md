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

## HuggingFace Token Setup (Required for Speaker Diarization)

The speech recognition pipeline uses PyAnnote for speaker diarization, which requires a HuggingFace account and API token.

### Setup Steps

1. **Create HuggingFace Account**
   - Go to https://huggingface.co/join
   - Create a free account

2. **Accept Model License**
   - Visit https://huggingface.co/pyannote/speaker-diarization-community-1
   - Click "Agree and access repository" to accept the license
   - **Note**: You must accept this license agreement before the model can be downloaded

3. **Generate API Token**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "voice-dub")
   - Select "Read" permission
   - Click "Generate"
   - Copy the token (starts with `hf_`)

4. **Set Environment Variable**

   **Windows PowerShell:**
   ```powershell
   $env:HUGGINGFACE_TOKEN = "hf_your_token_here"
   ```

   **Windows CMD:**
   ```cmd
   set HUGGINGFACE_TOKEN=hf_your_token_here
   ```

   **Linux/macOS:**
   ```bash
   export HUGGINGFACE_TOKEN="hf_your_token_here"
   ```

   **Or create a `.env` file** in the project root:
   ```
   HUGGINGFACE_TOKEN=hf_your_token_here
   ```

### Troubleshooting

- **"Repository not found" error**: You haven't accepted the model license. Visit the model page and click "Agree and access repository".
- **"Authentication required" error**: Token not set or invalid. Verify your token in HuggingFace settings.
- **Token starts with `hf_`**: This is correct. Older tokens may need to be regenerated.

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

## Pipeline Stages

### Phase 2: Video Processing (Complete)
**Status**: ✅ Complete (2026-01-31)

**Stage module:** `src/stages/video_extraction.py`

**Capabilities:**
- FFmpeg-based video/audio extraction with stream copy (no re-encoding)
- Format normalization (mp4, mkv, avi)
- Codec-specific handling (AAC for MP4, stream copy for MKV, MP3 for AVI)
- Progress callback support for UI integration

**Key features:**
- **Speed**: 10-100x faster than re-encoding (stream copy)
- **Quality**: Preserves exact original video quality
- **Format support**: MP4, MKV, AVI containers
- **Audio**: Extracts to 48kHz WAV for downstream processing

**Usage:**
```python
from src.stages.video_extraction import extract_audio_from_video

# Extract audio from video
audio_path = extract_audio_from_video(
    video_path="input.mp4",
    output_audio_path="extracted_audio.wav",
    progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
)
```

---

### Phase 3: Speech Recognition (Complete)
**Status**: ✅ Complete (2026-01-31)

**Stage module:** `src/stages/asr_stage.py`

**Capabilities:**
- Speech-to-text transcription using Whisper Large V3
- Speaker diarization (2-5 speakers) using PyAnnote
- Temporal alignment merging transcription with speakers
- Word-level timestamps for lip sync precision
- Low-confidence segment flagging for manual review
- JSON export with structured output

**Key features:**
- **Accuracy**: Whisper Large V3 (99+ languages, high accuracy)
- **Diarization**: PyAnnote speaker-diarization-community-1 (better speaker counting)
- **Speed**: faster-whisper (2-4x speedup, 50% VRAM reduction)
- **VAD filtering**: Prevents 80% of hallucinations on silence
- **Memory footprint**: ~4.5GB VRAM (Whisper float16) + ~2GB (PyAnnote)

**Usage:**
```python
from src.stages.asr_stage import run_asr_stage

# Transcribe and diarize audio
result = run_asr_stage(
    audio_path="extracted_audio.wav",
    video_id="video123",
    huggingface_token="hf_your_token_here",
    progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
)

# Access results
print(f"Language: {result.detected_language}")
print(f"Speakers: {result.num_speakers}")
print(f"Segments: {result.total_segments}")

# Iterate over aligned segments
for segment in result.segments:
    print(f"[{segment.speaker}] {segment.text}")
```

**Configuration (in `src/config/settings.py`):**
```python
ASR_SAMPLE_RATE = 16000  # Required by Whisper and pyannote
ASR_CONFIDENCE_THRESHOLD = 0.7  # Flag segments below this confidence
```

**Testing:**
```bash
python tests/test_asr_stage.py
# Validates transcription, diarization, alignment, JSON export
```

**Prerequisites:**
- HuggingFace token (see [HuggingFace Token Setup](#huggingface-token-setup-required-for-speaker-diarization))
- PyAnnote model license accepted

---

### Phase 4: Translation Pipeline (Complete)
**Status**: ✅ Complete (2026-01-31)

**Stage module:** `src/stages/translation_stage.py`

**Capabilities:**
- Neural machine translation from 96 languages to English using Meta SeamlessM4T v2 Large (2.3B parameters)
- Multi-candidate generation with beam search for quality optimization
- Duration-aware translation selection (±10% timing tolerance for lip sync compatibility)
- Context preservation across segments (no isolated sentence translation)
- Speaker-aware translation maintaining conversational coherence
- Automatic chunking with overlap for long videos (>15 minutes)
- Low-confidence segment flagging for manual review (Phase 8)
- Progress callback support for UI integration

**Key features:**
- **Supported languages (96 total)**: Japanese, Korean, Mandarin, Spanish, French, German, Hindi, Arabic, and 88 more
- **Speed-first approach**: 3-candidate beam search balances quality and processing time
- **Context window**: 1024 tokens with 128-token overlap for cross-segment context
- **Memory footprint**: ~6GB VRAM for SeamlessM4T v2 Large
- **Duration validation**: Character-count heuristic (15 chars/second for English speech)

**Usage:**
```python
from src.stages.translation_stage import run_translation_stage

# Translate ASR output to English
result = run_translation_stage(
    asr_json_path="data/temp/video123_transcript.json",
    output_json_path="data/temp/video123_translation.json",
    num_candidates=3,  # Beam width for candidate generation
    progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
)

# Check results
print(f"Translated {result.total_segments} segments")
print(f"Average confidence: {result.avg_confidence:.2f}")
print(f"Flagged for review: {result.flagged_count} segments")

# Access translated segments
for segment in result.segments:
    print(f"[{segment.speaker}] {segment.translated_text}")
```

**Pipeline flow:**
1. Load ASR JSON (speaker-labeled transcript with timestamps)
2. Determine chunking strategy (single batch vs overlapping chunks)
3. Load SeamlessM4T v2 via ModelManager
4. Translate segments with 3-candidate beam search
5. Rank candidates by confidence (60%) + duration fit (40%)
6. Select best translation per segment
7. Validate duration constraints (flag if outside ±10%)
8. Flag low-confidence segments (<70% threshold)
9. Export translated JSON with all metadata
10. Cleanup model and CUDA cache

**Configuration (in `src/config/settings.py`):**
```python
SEAMLESS_MODEL_ID = "facebook/seamless-m4t-v2-large"
TRANSLATION_TARGET_LANGUAGE = "eng"
TRANSLATION_CONFIDENCE_THRESHOLD = 0.7  # Flag segments below this
TRANSLATION_DURATION_TOLERANCE = 0.1  # ±10% duration fit
TRANSLATION_CHARS_PER_SECOND = 15  # English speech rate
TRANSLATION_NUM_CANDIDATES = 3  # Beam width
TRANSLATION_BATCH_SIZE = 8  # Process N segments at a time
TRANSLATION_MAX_CHUNK_TOKENS = 1024  # Chunking threshold
TRANSLATION_OVERLAP_TOKENS = 128  # Context preservation overlap
```

**Testing:**
```bash
python tests/test_translation_stage.py
# Validates multi-language translation, chunking, duration constraints, flagging
```

**Key Decisions:**
- **SeamlessM4T v2 over NLLB-200**: Better multilingual quality, speech-aware translation
- **Beam search (3 candidates) over greedy**: Enables duration-aware selection with minimal speed cost
- **Character-count duration heuristic**: Fast estimation, good enough for ±10% tolerance
- **Full-context batching with overlap**: Prevents pronoun/reference errors from isolated translation
- **Speed-first priority**: 3-candidate beam (not 5-10) for fast 20-minute video processing

---

## Usage

*(Full pipeline integration to be added in Phase 5+)*

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

## AI Models Used

| Model | Purpose | VRAM | Speed | Status |
|-------|---------|------|-------|--------|
| **Whisper Large V3** | Speech-to-text transcription | ~4.5GB (fp16) | 2-4x faster than original | ✅ Integrated |
| - Model: faster-whisper/large-v3 | 99+ languages, word timestamps | - | via faster-whisper | Phase 3 |
| **PyAnnote** | Speaker diarization | ~2GB | Real-time | ✅ Integrated |
| - Model: pyannote/speaker-diarization-community-1 | 2-5 speakers | - | - | Phase 3 |
| **SeamlessM4T v2 Large** | Neural machine translation | ~6GB (fp16) | ~1-2s/segment | ✅ Integrated |
| - Model: facebook/seamless-m4t-v2-large | 96 languages → English | 2.3B params | 3-beam search | Phase 4 |
| **XTTS-v2** | Voice cloning + TTS | ~8GB | TBD | Planned |
| - Model: coqui/XTTS-v2 | Emotion preservation | - | - | Phase 5 |
| **Wav2Lip / LatentSync** | Lip synchronization | ~4GB | TBD | Planned |
| - Model: TBD | Sync lips to English audio | - | - | Phase 6 |

**Total VRAM (sequential loading)**: Peak ~8GB per stage (32GB available on RTX 5090)

## License

*(To be added)*

## Contributing

*(To be added)*

---

## Development Progress

| Phase | Plans | Status | Completed |
|-------|-------|--------|-----------|
| 1. Environment & Foundation | 3/3 | Complete | 2026-01-30 |
| 2. Video Processing Pipeline | 2/2 | Complete | 2026-01-31 |
| 3. Speech Recognition Pipeline | 3/3 | Complete | 2026-01-31 |
| 4. Translation Pipeline | 4/4 | Complete | 2026-01-31 |
| 5. Voice Cloning Pipeline | - | Planned | - |
| 6. Lip Synchronization | - | Planned | - |
| 7. Assembly & Export | - | Planned | - |
| 8. Quality Review UI | - | Planned | - |
| 9. Video Upload UI | - | Planned | - |
| 10. Complete Pipeline Integration | - | Planned | - |
| 11. Production Hardening | - | Planned | - |

**Current Status**: Phase 4 Complete - Translation Pipeline ready
**Next**: Phase 5 - Voice Cloning Pipeline
