# Phase 5: Voice Cloning & TTS - Research

**Researched:** 2026-02-02
**Domain:** Voice cloning and text-to-speech synthesis with XTTS-v2
**Confidence:** HIGH

## Summary

Voice Cloning & TTS transforms translated English text into spoken audio that preserves each original speaker's voice characteristics and emotional tone. The established approach uses **XTTS-v2** (Coqui TTS), a multilingual voice cloning model that requires only 6-second reference samples to clone voices across languages while preserving emotional expression and speaking style.

XTTS-v2 integrates directly with PyTorch, loads via the `TTS.api.TTS` wrapper or low-level model API, requires ~2GB VRAM for inference (trivial on RTX 5090), and supports 17 languages including English. The model uses 24kHz audio output and defaults to float16 precision for GPU efficiency. Reference audio is processed once into speaker embeddings (conditioning latents) which are cached and reused across all segments for that speaker.

Critical constraints: XTTS-v2 is licensed under **Coqui Public Model License (CPML)** restricting use to **non-commercial purposes only**. Coqui AI shut down in January 2024, making commercial licensing impossible. This phase assumes the project remains non-commercial or requires alternative TTS solutions for commercial use.

**Primary recommendation:** Use XTTS-v2 via TTS library with sequential model loading through ModelManager. Extract 6-10 second reference samples per speaker from diarized segments (cleanest audio), generate speaker embeddings once per speaker, then synthesize all segments for that speaker in batches. Validate output with duration matching (±5% tolerance), automated quality metrics (PESQ/STOI), and manual spot-checks before proceeding to lip sync.

## Standard Stack

The established libraries/tools for voice cloning with XTTS-v2:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| coqui-tts | 0.22+ | XTTS-v2 voice cloning model | Official implementation, battle-tested in production, 140k+ developers |
| PyTorch | 2.x nightly (CUDA 12.8) | GPU inference framework | Required for RTX 5090 sm_120, project standard |
| torchaudio | Nightly (matches PyTorch) | Audio I/O and preprocessing | Native PyTorch integration, no format conversion overhead |
| pyannote.audio | 3.x | Speaker diarization alignment | Already used in Phase 3 (ASR), extract reference samples from speaker segments |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| librosa | 0.11+ | Audio duration analysis, quality validation | Duration matching calculations, pre/post processing |
| pydub | 0.25+ | Audio format conversion if needed | Converting reference samples to WAV if not already |
| pesq / pesqc2 | Latest | PESQ audio quality scoring | Automated quality validation (MOS prediction) |
| speechmetrics | Latest | Unified interface for STOI, PESQ, etc. | Multi-metric quality assessment wrapper |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| XTTS-v2 | Qwen3-TTS (released Jan 2026) | Newer but unproven at scale, may have better quality but less documentation |
| XTTS-v2 | OpenVoice, Bark, VITS | Lower quality emotion preservation, worse accent handling |
| coqui-tts library | Direct HuggingFace model loading | More control but lose built-in preprocessing, harder integration |
| pesq library | Human MOS scoring | Human MOS is gold standard but costly/slow, use for validation only |

**Installation:**
```bash
# Already installed via requirements.txt (Phase 1)
pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128
pip install coqui-tts  # Includes XTTS-v2
pip install librosa pydub
pip install pesq pesqc2 speechmetrics  # Audio quality validation
# pyannote.audio already installed from Phase 3
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── stages/
│   └── tts_stage.py                    # Main TTS orchestration
├── tts/
│   ├── __init__.py
│   ├── reference_extractor.py          # Extract 6-10s samples from diarized segments
│   ├── xtts_generator.py               # XTTS-v2 synthesis wrapper
│   ├── quality_validator.py            # PESQ/STOI/duration validation
│   └── speaker_embeddings.py           # Cache speaker conditioning latents
├── models/
│   └── model_manager.py                # Existing - use for XTTS model loading
└── config/
    └── settings.py                     # Add TTS-specific settings
```

### Pattern 1: Sequential Model Loading (Existing Pattern)
**What:** Load XTTS-v2 through ModelManager after unloading translation model
**When to use:** Transitioning from Phase 4 (translation) to Phase 5 (TTS)
**Example:**
```python
# Source: Project pattern from src/models/model_manager.py
from src.models.model_manager import ModelManager
from TTS.api import TTS

manager = ModelManager()

# Automatically unloads SeamlessM4T from Phase 4
xtts = manager.load_model("xtts", lambda: TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=True
))

# Generate audio for all segments
# ... synthesis logic ...

# Unload when done (before Phase 6 lip sync)
manager.unload_current_model()
```

### Pattern 2: Speaker Embedding Caching
**What:** Extract speaker embeddings once per speaker, reuse for all segments
**When to use:** Whenever generating audio for a speaker (after first segment)
**Example:**
```python
# Source: https://huggingface.co/coqui/XTTS-v2 official documentation
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Low-level API for conditioning latent caching
config = XttsConfig()
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="path/to/xtts", eval=True)
model.cuda()

# Generate embeddings once per speaker
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path="reference_samples/speaker_01.wav"
)

# Cache these embeddings, reuse for all segments by this speaker
speaker_cache = {
    "speaker_01": (gpt_cond_latent, speaker_embedding)
}

# Synthesize segment using cached embeddings
outputs = model.inference(
    text="Translated English text here",
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.65,  # Default creativity control
    speed=1.0  # Adjust for duration matching
)
```

### Pattern 3: Progress Callback Integration
**What:** Report TTS progress to UI using established project callback pattern
**When to use:** All stage orchestration functions (existing pattern)
**Example:**
```python
# Source: Project pattern from src/stages/asr_stage.py and translation_stage.py
from typing import Callable, Optional

def run_tts_stage(
    translation_json: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Path:
    """
    Generate voice-cloned English audio from translated text.

    Args:
        translation_json: Phase 4 output with translated segments
        output_dir: Where to save audio files
        progress_callback: Optional callback(progress: float, status: str)
                          where progress is 0.0-1.0
    """
    # Default no-op callback
    if progress_callback is None:
        progress_callback = lambda p, s: None

    progress_callback(0.05, "Loading translation results...")
    # ... load JSON ...

    progress_callback(0.10, "Loading XTTS-v2 model...")
    # ... load model via ModelManager ...

    progress_callback(0.15, "Extracting reference samples...")
    # ... extract 6-10s samples per speaker ...

    # Per-speaker progress tracking
    total_speakers = len(unique_speakers)
    for idx, speaker_id in enumerate(unique_speakers):
        speaker_progress = 0.15 + (idx / total_speakers) * 0.70
        progress_callback(speaker_progress, f"Generating audio for {speaker_id}...")
        # ... synthesize all segments for this speaker ...

    progress_callback(0.90, "Validating audio quality...")
    # ... run PESQ/STOI validation ...

    progress_callback(1.0, "TTS complete")
    return output_json_path
```

### Pattern 4: Reference Sample Selection
**What:** Select cleanest 6-10 second audio sample per speaker from diarized segments
**When to use:** Once per speaker, before generating any audio
**Example:**
```python
# Source: Composite from WebSearch results and pyannote alignment
import librosa
import numpy as np

def select_reference_sample(speaker_segments: list, audio_path: Path) -> tuple[Path, float]:
    """
    Select best 6-10 second reference sample for speaker.

    Strategy: Find segment with highest energy (proxy for clean audio),
    minimum 6s duration, extract center portion if longer.

    Returns:
        (reference_audio_path, quality_score)
    """
    audio, sr = librosa.load(audio_path, sr=24000)  # XTTS uses 24kHz

    best_segment = None
    best_score = -1

    for segment in speaker_segments:
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        duration = segment['end'] - segment['start']

        # Skip segments too short
        if duration < 6.0:
            continue

        # Extract audio for this segment
        segment_audio = audio[start_sample:end_sample]

        # Calculate quality score (RMS energy as proxy)
        rms = np.sqrt(np.mean(segment_audio**2))

        if rms > best_score:
            best_score = rms
            best_segment = segment

    # Fallback: concatenate multiple segments if no single 6s segment exists
    if best_segment is None:
        return concatenate_segments(speaker_segments, audio, sr)

    # Extract 6-10 second window from best segment
    start = best_segment['start']
    duration = min(best_segment['end'] - start, 10.0)  # Cap at 10s
    if duration > 6.0:
        # Extract middle portion for stability
        offset = (best_segment['end'] - start - duration) / 2
        start += offset

    # Save reference sample
    ref_path = output_dir / f"reference_{speaker_id}.wav"
    start_sample = int(start * sr)
    end_sample = int((start + duration) * sr)
    reference_audio = audio[start_sample:end_sample]

    # Save as 24kHz WAV (XTTS native format)
    import soundfile as sf
    sf.write(ref_path, reference_audio, sr)

    return ref_path, best_score
```

### Anti-Patterns to Avoid
- **Loading XTTS model multiple times**: Use ModelManager once, reuse model instance for all speakers
- **Regenerating speaker embeddings per segment**: Extract conditioning latents once per speaker, cache and reuse
- **Ignoring duration mismatch**: XTTS has no strict duration control; must validate output and retry with speed parameter
- **Using simplified API for production**: `tts.tts_to_file()` doesn't support speed/temperature control; use low-level `model.inference()` for parameter access
- **Skipping quality validation**: XTTS can produce artifacts (especially for short text); must validate before lip sync
- **Not handling short text**: XTTS tries to generate 3+ seconds even for 1-2 words, causing meaningless sound artifacts

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio quality assessment | Custom MOS scoring system | `pesq`, `pesqc2`, `speechmetrics` | PESQ is ITU-T standard (P.862), predicts human MOS 1-5 scale, widely validated |
| Speaker diarization for reference samples | Custom VAD + clustering | `pyannote.audio` (already in project) | CNN-based segmentation with pre-trained models, extract speaker turns directly |
| Audio duration analysis | Manual sample counting | `librosa.get_duration()` | Handles multiple formats, accurate frame counting, no edge case bugs |
| Audio preprocessing (resampling, mono conversion) | SciPy manual processing | `torchaudio.transforms` or `librosa` | Handles anti-aliasing, preserves quality, tested at scale |
| Speaker embedding extraction | Custom acoustic features | XTTS `get_conditioning_latents()` | Trained end-to-end with synthesis model, guaranteed compatibility |
| Duration matching retry logic | Custom text truncation | XTTS `speed` parameter + iterative search | XTTS uses Tortoise-based autoregressive generation, speed parameter is safe way to adjust |

**Key insight:** Voice cloning quality depends on acoustic model training, not feature engineering. XTTS embeddings are co-trained with synthesis decoder; custom features will reduce quality. Focus effort on reference sample selection and quality validation, not acoustic modeling.

## Common Pitfalls

### Pitfall 1: Short Text Artifacts
**What goes wrong:** XTTS generates strange sounds for 1-2 word text, adding meaningless audio to reach 3+ seconds minimum generation length.
**Why it happens:** Model is trained on conversational speech (typically 3-10 second utterances); autoregressive decoder tries to fill time even with minimal text.
**How to avoid:** Concatenate very short segments (< 3 seconds target) with adjacent same-speaker segments before synthesis. If concatenation impossible, flag for manual review.
**Warning signs:** Generated audio significantly longer than expected (>2x translated text duration), non-speech sounds at end of segment.

### Pitfall 2: Duration Mismatch Causing Lip Sync Failure
**What goes wrong:** Generated audio doesn't match translated text timing (±5% tolerance), causing lip sync desynchronization in Phase 6.
**Why it happens:** XTTS has no strict duration control; output length varies based on prosody, speaking rate inference from reference sample, and temperature sampling.
**How to avoid:**
1. Validate output duration against target (calculate from translation text length × chars_per_second)
2. If outside ±5% tolerance, retry with adjusted `speed` parameter (binary search: 0.8-1.2 range)
3. Flag segments requiring >3 retries for manual review
**Warning signs:** Consistent under/over-generation for specific speakers (reference sample has unusual speaking rate), temperature artifacts causing elongated words.

### Pitfall 3: Reference Sample Quality Degradation
**What goes wrong:** Selected reference sample has background noise, overlapping speech, or poor audio quality, causing all generated segments to sound distorted.
**Why it happens:** Voice cloning transfers acoustic characteristics from reference; noise/distortion in reference appears in all outputs.
**How to avoid:**
1. Calculate SNR or RMS energy for candidate segments
2. Prefer segments with high energy (clear speech) and minimum variation (no shouting/whispering)
3. If all segments are noisy, warn user and potentially fall back to generic TTS voice
4. Consider manual reference sample upload feature for critical projects
**Warning signs:** All segments for a speaker sound muffled/noisy, background noise patterns repeat across different text content.

### Pitfall 4: Emotion/Prosody Loss in Translation
**What goes wrong:** English audio sounds flat/monotone despite original speaker's expressive delivery.
**Why it happens:** XTTS emotion transfer depends on reference sample prosody; if reference is emotionally neutral, all outputs will be. Temperature parameter affects variation but not emotional tone.
**How to avoid:**
1. When selecting reference sample, prefer emotionally expressive segments (not monotone narration)
2. For critical emotional moments, consider selecting reference from similar emotional context in video
3. Document limitation: XTTS cannot inject emotion not present in reference
**Warning signs:** User reports "sounds robotic," lack of energy variation across segments, mismatch between visual expressions and audio tone.

### Pitfall 5: VRAM Leak from Speaker Embedding Accumulation
**What goes wrong:** Memory usage grows with number of speakers, eventually causing OOM errors on long videos with many speakers.
**Why it happens:** Speaker embeddings (conditioning latents) are GPU tensors; caching them without cleanup accumulates VRAM usage.
**How to avoid:**
1. Process speakers sequentially, not all-at-once
2. After completing all segments for a speaker, move embeddings to CPU or delete
3. Use `torch.cuda.empty_cache()` between speakers
4. Monitor VRAM with `print_gpu_memory_summary()` after each speaker
**Warning signs:** Gradual VRAM increase during processing, OOM errors on videos with >10 speakers despite model fitting comfortably.

### Pitfall 6: Non-Commercial License Violation
**What goes wrong:** Deploying application commercially with XTTS-v2 violates CPML license, exposing project to legal risk.
**Why it happens:** XTTS-v2 is "open source" (code available) but NOT open license (CPML restricts commercial use). Coqui AI shut down, no one can grant commercial licenses.
**How to avoid:**
1. Document clearly in all user-facing materials: "Non-commercial use only"
2. If commercial deployment needed, evaluate alternatives (Qwen3-TTS, commercial APIs like ElevenLabs)
3. For research/personal use, CPML is permissive
**Warning signs:** User asks about commercial deployment, integration with paid services, revenue generation.

## Code Examples

Verified patterns from official sources:

### Loading XTTS-v2 via TTS API (Simple)
```python
# Source: https://huggingface.co/coqui/XTTS-v2
from TTS.api import TTS

# Initialize model (downloads automatically on first use, ~2GB)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Simple voice cloning (auto-generates embeddings internally)
tts.tts_to_file(
    text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    file_path="output.wav",
    speaker_wav="/path/to/reference.wav",  # 6-10 second sample
    language="en"
)
```

### Loading XTTS-v2 via Model API (Advanced - Recommended for Production)
```python
# Source: https://huggingface.co/coqui/XTTS-v2
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load configuration and model
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

# Generate speaker embeddings (once per speaker)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path="/path/to/reference.wav"
)

# Synthesize with control over parameters
outputs = model.synthesize(
    text="Translated English text here",
    config=config,
    speaker_wav="/path/to/reference.wav",  # Or use cached embeddings
    gpt_cond_len=3,  # Conditioning length
    language="en",
    temperature=0.65,  # 0.1-1.0, controls creativity (lower = more consistent)
    length_penalty=1.0,  # >1.0 encourages longer output
    repetition_penalty=2.0,  # Reduces repeated phrases
    top_k=50,  # Decoder sampling parameter
    top_p=0.85  # Nucleus sampling
)

# outputs["wav"] contains numpy array of audio samples at 24kHz
```

### Duration Validation and Retry Logic
```python
# Source: Composite from project patterns and XTTS documentation
import librosa

def generate_with_duration_matching(
    model: Xtts,
    text: str,
    target_duration: float,
    speaker_embedding: tuple,
    tolerance: float = 0.05,
    max_retries: int = 3
) -> tuple[np.ndarray, float]:
    """
    Generate audio matching target duration within tolerance.

    Returns:
        (audio_array, actual_duration)
    """
    gpt_cond_latent, speaker_emb = speaker_embedding
    speed = 1.0

    for attempt in range(max_retries):
        # Synthesize with current speed
        outputs = model.inference(
            text=text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_emb,
            temperature=0.65,
            speed=speed
        )

        audio = outputs["wav"]
        actual_duration = len(audio) / 24000  # 24kHz sample rate

        # Check if within tolerance
        duration_error = abs(actual_duration - target_duration) / target_duration
        if duration_error <= tolerance:
            return audio, actual_duration

        # Adjust speed for next attempt (binary search)
        if actual_duration > target_duration:
            speed *= 1.1  # Speed up
        else:
            speed *= 0.9  # Slow down

        # Clamp speed to reasonable range
        speed = max(0.8, min(1.2, speed))

    # Max retries exceeded - return best attempt and flag
    print(f"WARNING: Duration mismatch after {max_retries} retries. "
          f"Target: {target_duration:.2f}s, Actual: {actual_duration:.2f}s")
    return audio, actual_duration
```

### Audio Quality Validation (PESQ)
```python
# Source: https://github.com/ludlows/PESQ
from pesq import pesq
import librosa

def validate_audio_quality(
    generated_audio_path: Path,
    reference_audio_path: Path,
    sample_rate: int = 16000
) -> dict:
    """
    Validate generated audio quality using PESQ and STOI metrics.

    Returns dict with metrics:
        - pesq_score: 1.0-5.0 (MOS-LQO prediction)
        - stoi_score: 0.0-1.0 (intelligibility)
        - duration_match: bool
    """
    # Load audio (PESQ requires 16kHz for wideband mode)
    ref_audio, _ = librosa.load(reference_audio_path, sr=sample_rate)
    gen_audio, _ = librosa.load(generated_audio_path, sr=sample_rate)

    # Ensure same length (truncate longer one)
    min_len = min(len(ref_audio), len(gen_audio))
    ref_audio = ref_audio[:min_len]
    gen_audio = gen_audio[:min_len]

    # Calculate PESQ (wideband mode)
    pesq_score = pesq(sample_rate, ref_audio, gen_audio, 'wb')

    # Note: This example uses reference audio as "clean" signal
    # For TTS, typical approach is to use original audio segment as reference
    # PESQ score interpretation:
    #   < 2.0: Poor quality, likely artifacts
    #   2.0-3.0: Fair quality
    #   3.0-4.0: Good quality
    #   > 4.0: Excellent quality

    return {
        "pesq_score": pesq_score,
        "quality": "excellent" if pesq_score > 4.0 else
                   "good" if pesq_score > 3.0 else
                   "fair" if pesq_score > 2.0 else "poor",
        "accept": pesq_score >= 2.0  # Minimum threshold
    }
```

### Integration with ModelManager
```python
# Source: Project pattern from src/models/model_manager.py
from src.models.model_manager import ModelManager
from TTS.api import TTS

def load_xtts_model(manager: ModelManager) -> TTS:
    """
    Load XTTS-v2 through ModelManager.
    Automatically unloads previous model (e.g., SeamlessM4T from Phase 4).
    """
    def xtts_loader() -> TTS:
        return TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=True  # Use CUDA
        )

    # ModelManager handles unloading previous model
    xtts = manager.load_model("xtts", xtts_loader)
    return xtts

# Usage in TTS stage
manager = ModelManager()
xtts = load_xtts_model(manager)

# ... generate audio for all speakers ...

# Cleanup before next phase
manager.unload_current_model()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual MOS testing | Automated PESQ/STOI + spot-check MOS | 2020-2023 | Faster iteration, still validate critical segments manually |
| Fine-tuning TTS per speaker | Zero-shot cloning with 6s samples | 2023-2024 (XTTS v2) | Eliminates 2-4 hour training, enables real-time pipelines |
| Single-language TTS + accent modeling | Multilingual cloning (17 languages) | 2023 (XTTS v2) | Voice transfers across languages, critical for dubbing |
| Separate emotion detection + injection | Emotion transfer from reference sample | 2023 (XTTS v2) | Simpler pipeline, more natural prosody |
| Fixed speaking rate | Speed parameter for duration matching | 2024 (XTTS updates) | Enables strict timing constraints for lip sync |

**Deprecated/outdated:**
- **XTTS v1**: Superseded by v2 (better quality, +2 languages, speaker interpolation). Use v2 only.
- **Tacotron 2 / WaveGlow**: 2018-era TTS, no voice cloning, poor prosody. XTTS quality far superior.
- **Mozilla TTS**: Coqui forked from Mozilla TTS; original project abandoned. Use Coqui TTS.
- **Fine-tuning for every speaker**: XTTS v2 zero-shot cloning (6s samples) eliminates need. Only fine-tune for perfect celebrity mimicry (out of scope).

**2026 Context:**
- **Coqui AI shutdown (Jan 2024)**: Company closed, but open-source TTS library and XTTS-v2 weights remain available. Community maintains forks.
- **Qwen3-TTS (Jan 2026)**: New competitor from Alibaba, supports 3s voice cloning and multi-speaker. Too new to be battle-tested; XTTS-v2 is safer choice.
- **Commercial alternatives**: ElevenLabs, Resemble AI offer better quality but require API costs and internet connectivity. XTTS-v2 runs locally, critical for privacy and offline use.

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal temperature parameter for emotion preservation**
   - What we know: Temperature 0.65 is default, higher = more variation. Official docs recommend 0.65 for balance.
   - What's unclear: Whether increasing temperature (e.g., 0.75-0.85) improves emotional expressiveness or just adds instability.
   - Recommendation: Start with 0.65 (default), add temperature as tunable parameter in settings.py for experimentation. Monitor for artifacts (repetition, unnatural pauses) at higher values.

2. **Multi-speaker interpolation use case**
   - What we know: XTTS-v2 supports multiple speaker references with interpolation between voices.
   - What's unclear: Practical use case in dubbing pipeline (all speakers are distinct individuals, no interpolation needed).
   - Recommendation: Ignore feature for Phase 5. Potential future use: averaging multiple reference samples from same speaker for more robust embeddings.

3. **PESQ threshold for acceptable quality**
   - What we know: PESQ scores 1.0-5.0 (MOS prediction), >3.0 = "good quality." Research uses various thresholds (2.0-3.5) depending on application.
   - What's unclear: Appropriate threshold for dubbed video (balance quality vs. failure rate).
   - Recommendation: Start with PESQ >= 2.5 as minimum (fair quality), flag 2.5-3.0 for manual review, auto-accept >3.0. Collect user feedback to refine threshold.

4. **Reference sample selection for emotional diversity**
   - What we know: Reference sample's emotion transfers to all generated segments. Neutral reference = neutral outputs.
   - What's unclear: Whether to select different reference samples per emotional context (happy scene = happy reference) or use single "average" reference per speaker.
   - Recommendation: Phase 5 MVP uses single reference per speaker (simplicity). Document as limitation, consider scene-adaptive reference selection in future phase if users report emotional flatness.

5. **Handling speakers with insufficient clean audio**
   - What we know: Minimum 6s clean audio required. Noisy/short segments degrade quality.
   - What's unclear: Best fallback strategy when no speaker has 6s clean audio (concatenate short segments? use generic voice? fail with user warning?).
   - Recommendation: Implement concatenation of shorter segments (3-5s each) as fallback. If total <6s, warn user and offer: (1) proceed with low-quality voice, (2) manually upload reference sample, (3) skip speaker (use generic voice or text-only).

## Sources

### Primary (HIGH confidence)
- [Coqui XTTS-v2 Official Documentation](https://docs.coqui.ai/en/latest/models/xtts.html) - Architecture, API usage, parameters
- [XTTS-v2 Hugging Face Model Card](https://huggingface.co/coqui/XTTS-v2) - Installation, usage examples, supported languages
- [coqui-tts PyPI Package](https://pypi.org/project/coqui-tts/) - Official Python package documentation
- [pyannote.audio GitHub](https://github.com/pyannote/pyannote-audio) - Speaker diarization for reference extraction
- [pesq PyPI Package](https://pypi.org/project/pesq/) - PESQ audio quality metric
- [librosa Documentation](https://librosa.org/doc/0.11.0/tutorial.html) - Audio processing and duration analysis

### Secondary (MEDIUM confidence)
- [The Best Open-Source Text-to-Speech Models in 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models) - XTTS-v2 positioning in 2026 landscape
- [Coqui TTS Review - Brutally Honest Analysis 2026](https://qcall.ai/coqui-tts-review) - Limitations and quality assessment
- [XTTS License After Shutdown Discussion](https://github.com/coqui-ai/TTS/issues/3490) - CPML licensing implications post-shutdown
- [Qwen3-TTS Complete 2026 Guide](https://dev.to/czmilo/qwen3-tts-the-complete-2026-guide-to-open-source-voice-cloning-and-ai-speech-generation-1in6) - Alternative TTS comparison
- [speechmetrics GitHub](https://github.com/aliutkus/speechmetrics) - Unified quality metrics wrapper

### Tertiary (LOW confidence - verify before use)
- WebSearch findings on VRAM requirements (2.1GB VRAM, 4-5GB RAM) - Confirmed by multiple sources but not official docs
- WebSearch findings on speed parameter usage - Confirmed in discussions but limited official documentation
- WebSearch findings on short text artifacts - Reported by users but no official acknowledgment

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official documentation, established library ecosystem, project already uses PyTorch/CUDA
- Architecture: HIGH - Official API examples, existing project patterns (ModelManager, callbacks), clear integration path
- Pitfalls: MEDIUM-HIGH - Some from official issues/discussions, others inferred from TTS fundamentals (duration control, quality validation)
- Quality validation: MEDIUM - PESQ/STOI well-documented, but optimal thresholds application-specific
- Licensing: HIGH - CPML license text clear, Coqui shutdown well-documented

**Research date:** 2026-02-02
**Valid until:** 2026-04-02 (60 days - TTS model landscape stable, Coqui shutdown permanent, library updates infrequent)

**Critical constraints:**
- **Non-commercial license (CPML)**: XTTS-v2 cannot be used commercially without license (unavailable post-Coqui shutdown)
- **No strict duration control**: XTTS has no native "generate exactly N seconds" parameter; requires iterative retry with speed adjustment
- **3-second minimum generation**: Short text (<3s target) produces artifacts; requires segment concatenation or special handling
- **Reference sample quality dependency**: Voice cloning quality ceiling determined by reference audio; cannot improve beyond reference clarity/expressiveness
