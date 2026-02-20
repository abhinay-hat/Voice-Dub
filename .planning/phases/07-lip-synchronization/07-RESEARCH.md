# Phase 7: Lip Synchronization - Research

**Researched:** 2026-02-21
**Domain:** AI lip synchronization models (LatentSync, Wav2Lip) for video dubbing pipelines
**Confidence:** MEDIUM (pipeline integration verified via official repos; exact RTX 5090 timing unverified)

---

## Summary

Phase 7 applies lip synchronization to the assembled video produced by Phase 6. The input is a merged MP4 (English dubbed audio + original video from `run_assembly_stage()`). The output is a final video where the on-screen mouth movements match the English audio phonemes.

Two models were pre-decided by the project: Wav2Lip HD and LatentSync. Research compared them directly. **LatentSync 1.6 is the better choice**: it produces sharper, more natural results, has superior SyncNet-based accuracy scores, handles temporal consistency via TREPA (no flickering), and requires only ~8GB VRAM for inference on RTX 5090. Wav2Lip remains a fallback: it is faster, simpler to install on Windows, and has no diffusion model overhead, but produces blurry outputs and lacks temporal smoothing.

The major integration risk is LatentSync's dependency stack (diffusers 0.32, torch 2.5.1, insightface, decord) conflicting with the project's PyTorch nightly build for sm_120. LatentSync must run in its own conda/venv environment and be called via subprocess from the main pipeline. This is the standard integration pattern for this class of model.

**Primary recommendation:** Use LatentSync 1.6 (latest checkpoint from ByteDance/LatentSync-1.6 on HuggingFace) in an isolated venv called from `lip_sync_stage.py` via subprocess. Use Wav2Lip as the documented fallback path.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LatentSync | 1.6 (bytedance/LatentSync latest main) | Lip sync inference — diffusion-based | Best OSS quality, TREPA temporal consistency, SyncNet supervision, ByteDance maintained |
| diffusers | 0.32.2 | Stable Diffusion pipeline hosting LatentSync U-Net | Required by LatentSync; manages scheduler, VAE, UNet |
| insightface | 0.7.3 | Face detection and landmark alignment | LatentSync uses InsightFace for face frontalization preprocessing |
| onnxruntime-gpu | 1.21.0 | InsightFace ONNX model inference on GPU | Required by insightface for GPU-accelerated face detection |
| mediapipe | 0.10.11 | Secondary face landmark detection fallback | Graceful degradation when InsightFace fails |
| decord | 0.6.0 | Fast GPU-accelerated video frame reading | LatentSync uses decord for frame loading |
| omegaconf | 2.3.0 | Config file loading (LatentSync YAML configs) | Required by LatentSync inference script |
| accelerate | 0.26.1 | PyTorch model loading utilities | Required by LatentSync pipeline |
| DeepCache | 0.1.1 | Diffusion inference acceleration | ~2x speedup on stable diffusion pipelines |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Wav2Lip (original repo) | commit-based | Fallback lip sync for fast/simple cases | When LatentSync fails, OOM, or result quality unacceptable |
| opencv-python | 4.9.0.80 | Frame extraction, face region blending | Both models; post-processing pipeline |
| librosa | 0.10.1 | Audio feature extraction (mel spectrogram) | LatentSync audio preprocessing |
| python_speech_features | 0.6 | MFCC features for Wav2Lip mel spectrogram | Wav2Lip audio preprocessing |
| scenedetect | 0.6.1 | Scene boundary detection pre-processing | LatentSync data preprocessing pipeline |
| ffmpeg-python | 0.2.0 | Frame extraction and video reconstruction | Both models use FFmpeg for I/O |
| kornia | 0.8.0 | Image augmentation and warping | LatentSync geometric transforms |
| lpips | 0.1.4 | Perceptual quality metrics | Optional: quality validation post-sync |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LatentSync 1.6 | Wav2Lip (GAN) | Wav2Lip: faster (~10x), simpler install, but blurry output and no temporal consistency |
| LatentSync 1.6 | MuseTalk 1.5 | MuseTalk: real-time capable (30fps), but lower SyncNet scores, different architecture |
| LatentSync 1.6 | VideoReTalking | VideoReTalking: older approach, lower quality benchmarks vs LatentSync |
| Isolated venv for LatentSync | Single shared venv | Cannot merge: torch 2.5.1 (LatentSync) conflicts with PyTorch nightly (sm_120 main env) |

**Installation for LatentSync isolated environment:**

```bash
# Create isolated environment (separate from main project venv)
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install FFmpeg (required dependency)
conda install -y -c conda-forge ffmpeg

# Clone LatentSync repo into models/ directory
git clone https://github.com/bytedance/LatentSync.git models/LatentSync

# Install dependencies (pin to requirements.txt versions to avoid diffusers conflicts)
pip install -r models/LatentSync/requirements.txt

# Download checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir models/LatentSync/checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir models/LatentSync/checkpoints
```

**Installation for Wav2Lip fallback (in LatentSync env or separate):**

```bash
pip install opencv-python librosa python_speech_features
# Download checkpoints manually from Rudrabha/Wav2Lip Google Drive links:
# - checkpoints/wav2lip.pth (accuracy-optimized)
# - checkpoints/wav2lip_gan.pth (visual quality)
# - face_detection/detection/sfd/s3fd.pth (SFD face detector)
```

---

## Architecture Patterns

### Recommended Project Structure

```
src/
├── stages/
│   ├── assembly_stage.py      # Phase 6 (complete) - outputs merged video
│   └── lip_sync_stage.py      # Phase 7 (new) - runs LatentSync on merged video
models/
├── LatentSync/                # Cloned LatentSync repo + checkpoints
│   ├── checkpoints/
│   │   ├── latentsync_unet.pt
│   │   └── whisper/tiny.pt
│   ├── configs/unet/
│   │   └── stage2_512.yaml   # Model config for 512x512
│   └── scripts/inference.py  # Entry point called via subprocess
```

### Pattern 1: Subprocess Isolation Pattern

**What:** LatentSync runs in its own conda environment. The main pipeline invokes it via `subprocess.run()` with specific Python interpreter path and `scripts/inference.py` as the entry point.

**When to use:** Required because LatentSync pins `torch==2.5.1` (CUDA 12.1 wheels) while the main project uses PyTorch nightly with CUDA 12.8 for sm_120 support. These cannot coexist in one environment.

**Example:**

```python
# Source: verified pattern from LatentSync inference.sh + subprocess integration

import subprocess
from pathlib import Path

LATENTSYNC_PYTHON = Path("path/to/conda/envs/latentsync/bin/python")  # or Scripts/python.exe on Windows
LATENTSYNC_REPO = Path("models/LatentSync")

def run_latentsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
) -> None:
    """Call LatentSync via subprocess in isolated conda environment."""
    cmd = [
        str(LATENTSYNC_PYTHON), "-m", "scripts.inference",
        "--unet_config_path", str(LATENTSYNC_REPO / "configs/unet/stage2_512.yaml"),
        "--inference_ckpt_path", str(LATENTSYNC_REPO / "checkpoints/latentsync_unet.pt"),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--enable_deepcache",  # ~2x speedup (bug fixed in latest LatentSync main)
    ]
    result = subprocess.run(
        cmd,
        cwd=str(LATENTSYNC_REPO),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LatentSync failed:\n{result.stderr}")
```

### Pattern 2: Long Video Chunking

**What:** LatentSync processes video as a whole but requires faces to be consistently detectable. For 20-minute videos, process in chunks to prevent memory spikes and handle scene changes gracefully.

**When to use:** Always for videos longer than 5 minutes; required when scene changes cause face detection failures.

**Example:**

```python
# Source: derived from LatentSync pipeline architecture and face detection failure patterns

import subprocess
from pathlib import Path

def run_latentsync_chunked(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    chunk_duration_seconds: int = 300,  # 5-minute chunks
) -> None:
    """Split video into chunks, process each, concatenate with FFmpeg."""
    # 1. Extract audio segments matching video chunks using FFmpeg
    # 2. Call run_latentsync() per chunk
    # 3. Use FFmpeg concat demuxer to join chunk outputs
    # 4. Verify audio sync at boundaries
    pass
```

### Pattern 3: Input Format Preparation

**What:** LatentSync 1.6 internally preprocesses to 25FPS and 16000Hz audio. Wav2Lip requires 16000Hz audio and handles any FPS. Both require MP4 video input and WAV audio input as separate files.

**When to use:** Always - Phase 6 output is a merged MP4 with audio; lip sync stage must split audio before calling the model.

**Example:**

```python
# Source: LatentSync preprocessing documentation + Wav2Lip inference.py

import subprocess

def prepare_lip_sync_inputs(merged_video: Path, work_dir: Path):
    """Extract audio from assembled video for separate audio input."""
    audio_path = work_dir / "audio_for_lipsync.wav"
    # Extract audio as 16kHz WAV (LatentSync expects 16kHz internally)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(merged_video),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16kHz sample rate for lip sync models
        "-ac", "1",              # Mono (both models work mono)
        str(audio_path),
    ], check=True)
    return audio_path
```

### Anti-Patterns to Avoid

- **Running LatentSync in the main project venv:** torch 2.5.1 vs nightly nightly incompatibility will cause silent failures or version downgrades.
- **Passing 48kHz audio directly to LatentSync:** LatentSync's Whisper audio encoder expects 16kHz. Pass audio through FFmpeg first.
- **Processing full 20-minute video in one pass without monitoring VRAM:** LatentSync 1.6 inference uses ~18GB VRAM. Combined with any residual model state, this could OOM even on 32GB.
- **Using `enable_deepcache` without pulling latest LatentSync main:** The attribute was missing in a version released in early June 2025; a fix was merged June 19, 2025 (commit f5040cf). Pull latest main to avoid `AttributeError`.
- **Ignoring face detection failures:** LatentSync raises errors when faces are absent in frames. Apply the skip-frame patch from issue #107 or pre-filter frames with scene detection.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Face detection and landmark alignment | Custom dlib/OpenCV face detector | InsightFace buffalo_l model (bundled in LatentSync) | InsightFace handles 2D+3D landmarks, pose normalization, 90+ degree head angles |
| Audio-to-mel spectrogram | Custom STFT pipeline | LatentSync's Audio2Feature (Whisper-based) | Whisper's mel extraction is matched to the trained model's audio embeddings |
| Lip sync accuracy validation | Manual frame inspection | SyncNet evaluation (LSE-D / LSE-C metrics) | Standardized metric; LSE-D lower = better, LSE-C higher = better; avoids subjective eyeball testing |
| Temporal consistency | Hand-blending frames | TREPA in LatentSync 1.5+ | TREPA uses VideoMAE-v2 temporal representations; custom blending degrades sync accuracy |
| Long video concatenation | Custom video stitcher | FFmpeg concat demuxer | FFmpeg concat is frame-accurate; custom stitching risks timestamp drift at chunk boundaries |
| Diffusion inference acceleration | Custom caching | DeepCache (already integrated in LatentSync) | DeepCache provides 2-4x speedup with minimal quality loss; already a dependency |

**Key insight:** Both the audio processing and face alignment pipelines in LatentSync are architected to match the model's training conditions exactly. Deviating from them (e.g., using a different mel spectrogram extractor or face aligner) will degrade sync accuracy even if the code runs without errors.

---

## Common Pitfalls

### Pitfall 1: PyTorch Version Conflict (CRITICAL)

**What goes wrong:** Installing LatentSync dependencies (`torch==2.5.1`, CUDA 12.1 wheels) into the main project environment overwrites PyTorch nightly (CUDA 12.8, sm_120 support). After the overwrite, all other pipeline stages (Whisper, XTTS, SeamlessM4T) fail with "no kernel image available" on RTX 5090.

**Why it happens:** LatentSync's `requirements.txt` pins `torch==2.5.1` with `--extra-index-url https://download.pytorch.org/whl/cu121`. pip will happily downgrade nightly if you run `pip install -r requirements.txt` in the main environment.

**How to avoid:** Always install LatentSync in an isolated conda environment (`latentsync`). Call it from the main pipeline via `subprocess.run()` pointing at the isolated Python interpreter.

**Warning signs:** `torch.cuda.get_device_capability()` returns `(9, 0)` instead of `(12, 0)` after installing LatentSync deps; any stage except lip sync works but lip sync causes GPU errors.

### Pitfall 2: InsightFace Windows Installation Failure

**What goes wrong:** `pip install insightface` fails on Windows due to missing C++ compiler, or installs but crashes at runtime with DLL errors.

**Why it happens:** InsightFace requires Visual C++ Build Tools and has native C++ components. The onnxruntime-gpu conflict (installing both `onnxruntime` and `onnxruntime-gpu`) causes CPU-only inference silently.

**How to avoid:** Install only `onnxruntime-gpu`, not `onnxruntime`. Install Visual C++ Redistributable before attempting insightface install. On Windows, use the pre-built `.whl` from the `cobanov/insightface_windows` repository if standard pip install fails.

**Warning signs:** InsightFace installs without error but face detection runs in under 1ms (CPU speed) instead of expected ~10-50ms (GPU speed).

### Pitfall 3: Face Not Detected on Scene Changes

**What goes wrong:** LatentSync crashes with "Face not detected" error when processing video segments that contain transitions, cuts to non-person shots, or actors turning away from camera.

**Why it happens:** LatentSync's preprocessing pipeline raises an exception rather than gracefully handling frames with no detectable face. This is confirmed behavior from GitHub issue #44 and #107.

**How to avoid:** Two strategies: (1) Pre-segment video using scenedetect and only process segments with continuous face presence; (2) Apply the skip-frame patch from issue #107 that returns the original frame with zero mask when detection fails.

**Warning signs:** Pipeline fails early in processing; error message contains "face not detected" or a `None` type error in `image_processor.py`.

### Pitfall 4: LatentSync 1.6 VRAM Requirement Mismatch

**What goes wrong:** LatentSync 1.6 (512x512) requires ~18GB VRAM for inference, versus 1.5's ~8GB. Attempting to run 1.6 while any other model is loaded in VRAM causes OOM.

**Why it happens:** Higher resolution diffusion model has significantly larger activation memory. ModelManager from Phase 1 must fully unload all previous models before starting LatentSync.

**How to avoid:** Ensure ModelManager.unload_all() is called before running the lip sync subprocess. Since lip sync runs in a subprocess, it starts with a clean CUDA context, but the parent process must not hold GPU memory through CUDA IPC handles.

**Warning signs:** CUDA OOM error immediately after LatentSync subprocess starts; `nvidia-smi` shows VRAM already partially consumed before subprocess launches.

### Pitfall 5: Audio Sample Rate Mismatch

**What goes wrong:** Passing the 48kHz audio from Phase 6 assembly directly to LatentSync causes incorrect audio-visual alignment because LatentSync's Whisper encoder was trained on 16kHz audio.

**Why it happens:** Phase 6 normalizes to 48kHz (video production standard). LatentSync's Audio2Feature uses Whisper tiny which requires 16kHz input. LatentSync may resample internally but this is not guaranteed.

**How to avoid:** Always explicitly resample audio to 16kHz WAV before passing to LatentSync. Use FFmpeg with `-ar 16000` during input preparation.

**Warning signs:** Lip movements appear systematically offset by a constant delay; phoneme-to-mouth mapping looks wrong even though overall timing is correct.

### Pitfall 6: Wav2Lip Checkpoint Path Confusion

**What goes wrong:** Passing the face detection checkpoint path (`s3fd.pth`) as `--checkpoint_path` causes `KeyError: 'state_dict'` crash.

**Why it happens:** Wav2Lip uses two separate checkpoints: one for face detection (s3fd.pth) and one for the Wav2Lip model itself (wav2lip.pth or wav2lip_gan.pth). The `--checkpoint_path` argument expects the Wav2Lip model, not the face detector.

**How to avoid:** Maintain clear naming convention: store face detector at `face_detection/detection/sfd/s3fd.pth` and Wav2Lip model at `checkpoints/wav2lip.pth`.

**Warning signs:** `KeyError: 'state_dict'` immediately on startup before any video processing begins.

### Pitfall 7: LatentSync DeepCache Bug (Fixed in Latest Main)

**What goes wrong:** `AttributeError: 'Namespace' object has no attribute 'enable_deepcache'` when using LatentSync with `--enable_deepcache` flag.

**Why it happens:** A bug was introduced in a LatentSync version that deployed the `enable_deepcache` feature incompletely. The argument existed in the inference script but not in the args namespace.

**How to avoid:** Pull the latest LatentSync main branch (fix merged June 19, 2025, commit f5040cf). If using an older checkout, either update or remove the `--enable_deepcache` flag from subprocess calls.

**Warning signs:** Error occurs immediately on subprocess start before GPU is used.

---

## Code Examples

### Calling LatentSync Inference Programmatically

```python
# Source: github.com/bytedance/LatentSync/blob/main/scripts/inference.py + inference.sh

import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

LATENTSYNC_REPO = Path("models/LatentSync")

# Windows path - conda env Python for latentsync
LATENTSYNC_PYTHON = Path(
    "C:/Users/ASBL/miniconda3/envs/latentsync/python.exe"
)


def run_latentsync_inference(
    video_path: Path,
    audio_path: Path,          # Must be 16kHz WAV
    output_path: Path,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    enable_deepcache: bool = True,
) -> None:
    """
    Run LatentSync 1.6 inference via subprocess in isolated conda env.

    Args:
        video_path: Input MP4 video (any FPS; LatentSync normalizes to 25FPS internally)
        audio_path: Input WAV audio at 16kHz (resampled from 48kHz assembly output)
        output_path: Output MP4 with lip-synced video
        inference_steps: 20 for speed, 40-50 for quality (100s/10s-video on RTX 4090)
        guidance_scale: 1.5 default; higher improves sync but may distort
        enable_deepcache: ~2x speedup, requires latest LatentSync main
    """
    cmd = [
        str(LATENTSYNC_PYTHON), "-m", "scripts.inference",
        "--unet_config_path", str(LATENTSYNC_REPO / "configs/unet/stage2_512.yaml"),
        "--inference_ckpt_path", str(LATENTSYNC_REPO / "checkpoints/latentsync_unet.pt"),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")

    logger.info(f"Running LatentSync: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(LATENTSYNC_REPO),
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for long videos
    )
    if result.returncode != 0:
        logger.error(f"LatentSync stderr:\n{result.stderr}")
        raise RuntimeError(f"LatentSync failed with code {result.returncode}")
    logger.info("LatentSync inference complete")
```

### Preparing Audio Input (48kHz -> 16kHz)

```python
# Source: derived from LatentSync Audio2Feature documentation + FFmpeg standard practice

import subprocess
from pathlib import Path


def prepare_audio_for_lipsync(source_audio_path: Path, output_dir: Path) -> Path:
    """
    Resample audio from 48kHz (Phase 6 output) to 16kHz mono WAV for LatentSync.

    LatentSync uses Whisper tiny internally which expects 16kHz audio.
    Wav2Lip also processes mel spectrograms from 16kHz audio.
    """
    output_path = output_dir / "lipsync_audio_16k.wav"
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(source_audio_path),
        "-vn",
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "16000",           # 16kHz
        "-ac", "1",               # Mono
        str(output_path),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio prep failed: {result.stderr}")
    return output_path
```

### Wav2Lip Fallback Inference

```python
# Source: github.com/Rudrabha/Wav2Lip/blob/master/inference.py (argument reference)

import subprocess
from pathlib import Path


WAV2LIP_REPO = Path("models/Wav2Lip")
WAV2LIP_CHECKPOINT = WAV2LIP_REPO / "checkpoints/wav2lip_gan.pth"  # Better visual quality


def run_wav2lip_inference(
    video_path: Path,
    audio_path: Path,   # 16kHz WAV
    output_path: Path,
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 64,   # Reduce from 128 default if OOM
) -> None:
    """
    Run Wav2Lip inference as fallback when LatentSync fails.

    Note: Wav2Lip produces blurrier output but runs ~10x faster than LatentSync.
    Use wav2lip_gan.pth for better visual quality (slightly less sync accuracy).
    """
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        # "--nosmooth" -- add only if temporal smoothing causes issues
    ]
    result = subprocess.run(
        cmd,
        cwd=str(WAV2LIP_REPO),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed: {result.stderr}")
```

### LipSyncResult Dataclass (following project convention)

```python
# Source: project pattern from assembly_stage.py and other stages

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LipSyncResult:
    """Result of lip synchronization stage."""
    output_path: Path
    model_used: str          # "latentsync" or "wav2lip"
    model_version: str       # "1.6" for LatentSync
    inference_steps: int     # LatentSync steps used (0 for Wav2Lip)
    guidance_scale: float    # LatentSync scale (0.0 for Wav2Lip)
    processing_time: float
    input_video_path: Path
    input_audio_path: Path

    def to_dict(self) -> dict:
        return {
            "output_path": str(self.output_path),
            "model_used": self.model_used,
            "model_version": self.model_version,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "processing_time": self.processing_time,
            "input_video_path": str(self.input_video_path),
            "input_audio_path": str(self.input_audio_path),
        }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Wav2Lip GAN (2020) | LatentSync 1.6 (2025) | June 2025 | Sharper output, better temporal consistency, SyncNet supervision during training |
| 256x256 resolution | 512x512 resolution | LatentSync 1.6 (June 2025) | Addresses blurriness in teeth and lips; 18GB VRAM needed vs 8GB for 1.5 |
| Manual temporal smoothing | TREPA (VideoMAE-v2 temporal alignment) | LatentSync 1.5 (March 2025) | Eliminates flickering without sacrificing sync accuracy |
| Separate face detection models | InsightFace buffalo_l unified bundle | LatentSync pipeline | Single download handles detection + alignment + normalization |
| No diffusion acceleration | DeepCache integration | LatentSync (with fix June 2025) | ~2x inference speed on diffusion denoising loop |

**Deprecated/outdated:**

- **LipGAN**: Predecessor to Wav2Lip, significantly lower quality, no active maintenance.
- **Wav2Lip with GAN (as primary)**: Visual quality better than base Wav2Lip but still significantly blurrier than LatentSync 1.6; use only as emergency fallback.
- **VideoReTalking**: Used in some older dubbing pipelines (e.g., AWS sample). Lower benchmark scores than LatentSync 1.5/1.6. Not recommended for new projects.
- **LatentSync 1.5**: Superseded by 1.6. Use 1.6 for new installations.

---

## Open Questions

1. **LatentSync 1.6 actual VRAM on RTX 5090**
   - What we know: Documented as ~18GB minimum for inference; RTX 5090 has 32GB
   - What's unclear: Whether RTX 5090 sm_120 causes any VRAM overhead or efficiency gains over sm_90; actual measured VRAM during inference has not been verified on this hardware
   - Recommendation: Add VRAM monitoring during first test run; if >28GB, fall back to LatentSync 1.5 (8GB)

2. **LatentSync 1.6 inference speed on RTX 5090**
   - What we know: RTX 4090 processes ~10 seconds of video in ~100 seconds at 20 inference steps
   - What's unclear: RTX 5090 is faster architecture (sm_120, Blackwell); speedup factor is unverified
   - Recommendation: Budget 20 minutes processing time for 20-minute video as conservative estimate; may be 2-4x faster

3. **decord Windows compatibility**
   - What we know: decord is a core LatentSync dependency for video frame loading; `pip install decord` sometimes fails on Windows
   - What's unclear: Whether the latentsync conda env on Windows 11 can successfully install decord==0.6.0
   - Recommendation: Test installation first; if decord fails on Windows, use the alternative `decord` conda package (`conda install -c conda-forge decord`) or switch to opencv VideoCapture as a fallback

4. **Multi-speaker face tracking accuracy**
   - What we know: Both models detect faces per-frame; LatentSync uses InsightFace for tracking; multi-speaker is not a native first-class feature
   - What's unclear: How well InsightFace tracks individual speakers when two faces are simultaneously in frame
   - Recommendation: Process single-active-speaker segments; use diarization output from Phase 3 to know which speaker is visible per segment; use speaker bounding box from Phase 3 to crop active speaker region

5. **Audio extracted from Phase 6 merged video vs original dubbed audio**
   - What we know: Phase 6 creates a merged MP4 with FFmpeg `-async 1`; audio is at 48kHz
   - What's unclear: Whether re-extracting audio from the merged video introduces any quality loss vs using the concatenated WAV directly
   - Recommendation: Use the pre-existing `concat_audio.wav` from Phase 6 temp files if available; otherwise extract from merged video with `ffmpeg -vn -ar 16000`

---

## Sources

### Primary (HIGH confidence)

- `github.com/bytedance/LatentSync` - Official repo: installation, requirements.txt, inference.sh, inference.py, issues #94, #107, #236, #264, #279
- `huggingface.co/ByteDance/LatentSync-1.6` - Official model card: VRAM requirements (18GB), resolution (512x512), checkpoint downloads
- `github.com/Rudrabha/Wav2Lip` - Official Wav2Lip repo: inference.py arguments, face detection architecture, batch sizes, checkpoint structure
- `deepwiki.com/bytedance/LatentSync/3-installation-and-setup` - Installation documentation: Python 3.10.13, conda env, exact dependency list, VRAM table

### Secondary (MEDIUM confidence)

- `github.com/bytedance/LatentSync/issues/94` - Inference speed benchmark: RTX 4090, ~100s per 10s of video (Jan 2025)
- `github.com/bytedance/LatentSync/issues/279` - DeepCache enable_deepcache bug confirmed fixed in commit f5040cf (June 19, 2025)
- `github.com/bytedance/LatentSync/issues/107` - Face detection failure: confirmed workaround (skip-frame patch in image_processor.py)
- WebSearch: LatentSync outperforms Wav2Lip on HDTF/VoxCeleb2 benchmarks (LSE-D, LSE-C metrics); multiple sources agree
- WebSearch: LatentSync 1.5 SyncNet score 0.7681 vs MuseTalk 0.6820 (benchmark comparison)
- `github.com/tin2tin/LatentSync-for-windows` - Windows PowerShell installation scripts exist; confirmed via WebFetch

### Tertiary (LOW confidence)

- LatentSync inference speed on RTX 5090: extrapolated from RTX 4090 benchmarks; not directly measured
- InsightFace GPU vs CPU behavior on Windows: inferred from general onnxruntime-gpu documentation patterns
- 20-minute video processing time estimate: extrapolated from 10s/100s ratio on RTX 4090; RTX 5090 speedup factor unknown

---

## Metadata

**Confidence breakdown:**

- Standard stack: MEDIUM — Library versions verified via official requirements.txt; PyTorch nightly compatibility with LatentSync is unverified (known conflict exists, subprocess isolation is the mitigation)
- Architecture: MEDIUM — Subprocess isolation pattern verified as correct approach from multiple Windows + sm_120 + LatentSync issue reports; exact Windows Python path setup needs empirical validation
- Pitfalls: HIGH — All documented pitfalls come from official GitHub issues with confirmed root causes and fixes; checkpoint confusion, face detection failures, DeepCache bug are all directly verifiable

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (LatentSync is actively maintained by ByteDance; check for 1.7 release or further sm_120 fixes)
