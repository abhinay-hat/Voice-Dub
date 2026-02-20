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

### Phase 5: Voice Cloning & TTS (Complete)
**Status**: ✅ Complete (2026-02-02)

**Stage module:** `src/stages/tts_stage.py`

**Capabilities:**
- Voice cloning for each speaker using XTTS-v2 with emotion preservation
- Reference sample extraction from original audio (6-10 second segments)
- Speaker embedding generation and caching for VRAM management
- Duration-matched synthesis with binary search speed adjustment (±5% tolerance)
- Audio quality validation using PESQ and STOI metrics
- Emotion preservation detection via pitch variance ratio analysis
- Batch processing with speaker grouping for efficiency
- Quality-based segment flagging for manual review

**Overview:**
Phase 5 generates English audio from translated text using XTTS-v2 voice cloning. Each speaker's voice is cloned from a 6-10 second reference sample extracted from the original video, preserving voice characteristics and emotional tone.

**Components:**
- **Reference Extractor**: Selects cleanest audio samples per speaker using RMS energy proxy
- **Speaker Embeddings**: XTTS conditioning latents for voice cloning (cached for reuse)
- **XTTS Generator**: Synthesis with duration matching via speed parameter (0.8-1.2x)
- **Quality Validator**: PESQ-based quality assessment with emotion preservation check

**Emotion Preservation:**
XTTS-v2 preserves emotional characteristics through speaker conditioning latents extracted from reference audio. The quality validator checks emotion preservation by comparing pitch variance between generated and reference audio:

- **Pitch variance ratio 0.6-1.5**: Emotion preserved (acceptable range)
- **Pitch variance ratio <0.6 or >1.5**: Emotion lost/exaggerated - flagged for review

**Usage:**
```python
from src.stages.tts_stage import run_tts_stage

# Generate English audio with cloned voices
result = run_tts_stage(
    translation_json_path="data/temp/video_translation.json",
    audio_path="data/temp/video_audio.wav",
    output_dir="data/temp/tts_output",
    progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
)

# Check results
print(f"Synthesized {result.successful_segments}/{result.total_segments} segments")
print(f"Flagged for review: {result.flagged_count}")
print(f"Emotion issues: {result.emotion_flagged_count}")

# Access synthesized audio files
for segment in result.segments:
    print(f"[{segment.speaker}] {segment.audio_path}")
    print(f"  Duration: {segment.actual_duration:.2f}s (target: {segment.original_duration:.2f}s)")
    print(f"  Quality: {'PASS' if segment.quality_passed else 'FAIL'}")
    print(f"  Emotion: {'preserved' if segment.emotion_preserved else 'flagged'}")
```

**Pipeline flow:**
1. Load translation JSON (translated text with timing)
2. Create output directory
3. Extract reference samples per speaker (6-10s, RMS-based selection)
4. Generate speaker embeddings (XTTS conditioning latents)
5. Synthesize all segments with duration matching
   - Group by speaker for efficiency
   - Binary search speed adjustment (0.8-1.2x range)
   - Save individual WAV files per segment
6. Validate audio quality with emotion preservation check
   - PESQ score (perceptual quality)
   - STOI score (intelligibility)
   - Pitch variance ratio (emotion preservation)
7. Build result structure with quality flags
8. Export result JSON (segment manifest)
9. Cleanup model and CUDA cache

**Configuration (in `src/config/settings.py`):**
```python
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_SAMPLE_RATE = 24000  # XTTS native sample rate
TTS_TEMPERATURE = 0.65  # Creativity control (default 0.65)
TTS_DURATION_TOLERANCE = 0.05  # ±5% timing accuracy
TTS_MIN_PESQ_SCORE = 2.5  # Quality threshold (1.0-5.0 scale)
TTS_PESQ_REVIEW_THRESHOLD = 3.0  # Flag for review if below
TTS_REFERENCE_MIN_DURATION = 6.0  # Minimum reference sample
TTS_REFERENCE_MAX_DURATION = 10.0  # Maximum reference sample
TTS_SPEED_MIN = 0.8  # Minimum speed adjustment
TTS_SPEED_MAX = 1.2  # Maximum speed adjustment
```

**Testing:**
```bash
python tests/test_tts_stage.py
# 13 integration tests covering imports, logic, quality validation, emotion preservation
```

**Key Decisions:**
- **XTTS-v2 for voice cloning**: Best open-source model with emotion preservation
- **RMS energy for reference selection**: Fast proxy for audio quality
- **Binary search for duration matching**: Achieves ±5% match in 3-5 attempts
- **PESQ + STOI for quality**: Comprehensive assessment (perceptual + intelligibility)
- **Pitch variance for emotion**: Fast proxy for emotional expression preservation
- **Speaker-grouped batching**: Prevents VRAM exhaustion on videos with 10+ speakers

**Requirements:**
- XTTS-v2 requires ~2GB VRAM for inference
- Reference samples need 6-10 seconds of clean audio per speaker
- **Non-Commercial License**: XTTS-v2 is licensed under CPML for non-commercial use only

---

### Phase 7: Lip Synchronization (Complete)
**Status**: ✅ Complete (2026-02-21)

**Stage module:** `src/stages/lip_sync_stage.py`

**Capabilities:**
- Lip sync with LatentSync 1.6 (primary model, diffusion-based, ~18GB VRAM)
- Wav2Lip GAN fallback when LatentSync fails (OOM, face detection errors, Windows InsightFace issues)
- Automatic 5-minute chunking for long videos (prevents face detection VRAM spikes)
- Audio resampling 48kHz → 16kHz before inference (Whisper tiny encoder requirement)
- DeepCache enabled by default for ~2x speedup
- Frame brightness validation after inference (advisory, never fails stage)
- Multi-speaker awareness with warning when `speakers_detected > 1`

**Overview:**
Phase 7 takes the assembled dubbed video from Phase 6 and synchronizes the English audio to the speaker's lip movements. LatentSync 1.6 runs in an isolated conda environment because it requires `torch==2.5.1+cu121`, which conflicts with the project's PyTorch nightly (`cu128`) required for RTX 5090 sm_120 support.

**Why subprocess isolation:**
LatentSync 1.6 pins `torch==2.5.1+cu121` (CUDA 12.1). The Voice Dub project requires PyTorch nightly (`cu128`) for RTX 5090's compute capability sm_120 — PyTorch stable releases do not support sm_120. Installing both in the same environment causes a PyTorch version conflict that breaks either LatentSync inference or the main pipeline's GPU support. The solution is to run LatentSync in an isolated conda environment (`latentsync`) invoked via Python subprocess. The main pipeline manages all orchestration and file I/O; the subprocess handles only inference.

**Installation (one-time setup):**
```bash
# Create isolated conda environment for LatentSync
conda create -y -n latentsync python=3.10.13
conda run -n latentsync conda install -y -c conda-forge ffmpeg

# Clone LatentSync and install dependencies
git clone https://github.com/bytedance/LatentSync.git models/LatentSync
conda run -n latentsync pip install -r models/LatentSync/requirements.txt

# Download model checkpoints (~3GB)
conda run -n latentsync huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir models/LatentSync/checkpoints
conda run -n latentsync huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir models/LatentSync/checkpoints/whisper

# (Optional) Wav2Lip fallback
git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip
# Download wav2lip_gan.pth from Google Drive link in Wav2Lip README
```

**Usage:**
```python
from pathlib import Path
from src.stages.lip_sync_stage import run_lip_sync_stage

result = run_lip_sync_stage(
    assembled_video_path=Path("data/outputs/assembled_video.mp4"),
    output_dir=Path("data/outputs"),
    inference_steps=20,      # 20 = fast, 40-50 = higher quality
    guidance_scale=1.5,      # LatentSync default
    enable_deepcache=True,   # ~2x speedup
    speakers_detected=1,     # Pass > 1 for multi-speaker videos (best-effort)
    progress_callback=lambda p, s: print(f"Lip sync: {p:.0%} — {s}"),
)

print(f"Model: {result.model_used} v{result.model_version}")
print(f"Fallback used: {result.fallback_used}")
print(f"Output: {result.output_path}")
if result.sync_validation:
    print(f"Sync validation: {result.sync_validation.pass_rate:.0%} frames passed")
```

**Long video handling:**
Videos longer than 5 minutes are automatically split into 5-minute chunks, each processed individually through LatentSync (or Wav2Lip on chunk failure), then concatenated via FFmpeg concat demuxer with stream copy. This prevents face detection VRAM spikes that occur on long continuous segments.

**Multi-speaker limitation:**
LatentSync processes the full frame without per-speaker face routing. When `speakers_detected > 1`, the stage logs a warning and sets `result.multi_speaker_mode = True`, but sync quality may vary per speaker since LatentSync targets a single dominant face. Per-speaker face routing (region-of-interest detection per speaker turn) is Phase 8 territory.

**Sync validation:**
After inference, `validate_lip_sync_output()` samples frames at 1-second intervals and checks mean luma (YAVG brightness) via FFmpeg signalstats as a proxy for face detection quality. A frame with YAVG below 10 is classified as a black/corrupt frame. Results are returned in `result.sync_validation`. A pass_rate below 95% logs an advisory warning but never fails the stage — the output is always returned for downstream processing.

**Key decisions:**
- Subprocess isolation: LatentSync torch 2.5.1+cu121 vs project PyTorch nightly (cu128)
- Audio always resampled 48kHz → 16kHz before LatentSync (Whisper tiny encoder requirement)
- 5-minute chunking: prevents face detection VRAM spikes for long videos
- DeepCache enabled by default: ~2x speedup (requires latest LatentSync main, commit f5040cf+)
- Lightweight validation: frame brightness proxy (SyncNet evaluation planned for Phase 8)
- Per-chunk fallback: Wav2Lip tried per-chunk, not whole-video, when LatentSync fails

**Testing:**
```bash
python tests/test_lip_sync_stage.py
# 21 integration tests, all pass without GPU or LatentSync conda env
```

**Requirements:**
- LatentSync conda environment with `torch==2.5.1+cu121` (isolated from main venv)
- ~18GB VRAM for LatentSync 1.6 inference (falls back to Wav2Lip at ~2-4GB)
- FFmpeg available in PATH for audio resampling, chunking, and concatenation

---

### Phase 6: Audio-Video Assembly (Complete)
**Status**: ✅ Complete (2026-02-03)

**Stage module:** `src/stages/assembly_stage.py`

**Capabilities:**
- Frame-perfect audio-video synchronization for dubbed videos
- Float64 timestamp precision prevents drift accumulation over 20+ minutes
- Sample rate normalization to 48kHz (video production standard)
- Audio segment concatenation with gap handling
- Drift detection at 5-minute intervals (ATSC 45ms tolerance)
- FFmpeg merge with clock drift correction (`async=1`)
- Automatic cleanup of temporary files

**Overview:**
Phase 6 assembles synthesized audio segments from Phase 5 into the final dubbed video. The assembly pipeline ensures frame-perfect synchronization through float64 timestamp precision, 48kHz audio normalization, checkpoint-based drift validation, and FFmpeg's async resampling for clock drift correction.

**Components:**
- **Timestamp Validator**: Float64 precision validation prevents drift accumulation
- **Audio Normalizer**: Sample rate normalization to 48kHz with kaiser_best resampling
- **Audio Concatenator**: Segment concatenation with gap handling (silence padding)
- **Drift Detector**: Sync validation at 5-minute intervals (ATSC 45ms tolerance)
- **Video Merger**: FFmpeg merge with `-af aresample=async=1` for clock drift correction

**Sync Validation:**
The drift detector validates audio-video sync at regular intervals to catch progressive drift before it becomes noticeable:

- **Checkpoints at 5-minute intervals** (300s)
- **ATSC tolerance: 45ms maximum drift** (broadcast standard)
- **Drift > 45ms**: Warning logged, user decides whether to accept
- **Float64 timestamps**: Prevents precision loss over 20+ minute videos

**Usage:**
```python
from src.stages.assembly_stage import run_assembly_stage

# Assemble TTS segments into final dubbed video
result = run_assembly_stage(
    video_path=Path("input_video.mp4"),
    tts_result_path=Path("tts_result.json"),
    output_path=Path("dubbed_video.mp4"),
    video_fps=30.0,
    progress_callback=lambda p, m: print(f"{p*100:.0f}%: {m}")
)

# Check sync quality
print(f"Output: {result.output_path}")
print(f"Drift detected: {result.drift_detected}")
print(f"Max drift: {result.max_drift_ms:.2f}ms")

# Review checkpoints
for cp in result.sync_checkpoints:
    status = "✓" if cp.within_tolerance else "✗"
    print(f"  {status} {cp.timestamp/60:.0f}min: {cp.drift_ms:+.2f}ms")
```

**Pipeline Flow:**
1. (0.05) Load TTS result JSON
2. (0.10) Create TimedSegment objects from TTS output
3. (0.15) Validate timestamp precision (float64)
4. (0.25) Normalize all audio segments to 48kHz
5. (0.45) Concatenate normalized segments (with gap padding)
6. (0.60) Validate sync at 5-minute intervals
7. (0.80) Merge with FFmpeg sync flags (`async=1`)
8. (0.95) Export assembly result JSON
9. (1.0) Cleanup temporary files

**Configuration (in `src/config/settings.py`):**
```python
ASSEMBLY_TARGET_SAMPLE_RATE = 48000   # 48kHz video standard
ASSEMBLY_DRIFT_TOLERANCE_MS = 45.0    # ATSC recommendation
ASSEMBLY_CHECKPOINT_INTERVAL = 300.0  # 5-minute intervals (seconds)
ASSEMBLY_RESAMPLING_QUALITY = 'kaiser_best'  # High-quality sinc interpolation
```

**Testing:**
```bash
python tests/test_assembly_stage.py
# 12 integration tests covering all assembly components
```

**Key Decisions:**
- **Float64 timestamps**: Prevents precision loss over 20+ minute videos (float32 accumulates ~10ms drift per 10 minutes)
- **48kHz target rate**: Video production standard, prevents sample rate drift
- **Kaiser_best resampling**: Highest quality sinc interpolation per librosa research
- **45ms drift tolerance**: ATSC standard for acceptable A/V offset (broadcast guideline)
- **5-minute checkpoints**: Early drift detection for long videos
- **Async resampling**: FFmpeg's `aresample=async=1` corrects clock drift during merge

**Requirements:**
- FFmpeg with aresample filter support
- No GPU required (FFmpeg-based, no ML models)
- Sample rate normalization uses librosa with kaiser_best quality

---

## Usage

*(Full pipeline integration to be added in Phase 6+)*

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
| **XTTS-v2** | Voice cloning + TTS | ~2GB | ~0.5-1s/segment | ✅ Integrated |
| - Model: coqui/XTTS-v2 | Emotion preservation, duration matching | Non-commercial | Speed adjust 0.8-1.2x | Phase 5 |
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
| 5. Voice Cloning & TTS | 4/4 | Complete | 2026-02-02 |
| 6. Audio-Video Assembly | 3/3 | Complete | 2026-02-03 |
| 7. Lip Synchronization | 4/4 | Complete | 2026-02-21 |
| 8. Quality Review UI | - | Planned | - |
| 9. Video Upload UI | - | Planned | - |
| 10. Complete Pipeline Integration | - | Planned | - |
| 11. Production Hardening | - | Planned | - |

**Current Status**: Phase 7 Complete - Lip Synchronization with LatentSync 1.6 and Wav2Lip fallback ready
**Next**: Phase 8 - Quality Review UI
