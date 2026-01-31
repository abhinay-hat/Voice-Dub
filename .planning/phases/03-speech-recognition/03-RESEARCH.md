# Phase 3: Speech Recognition - Research

**Researched:** 2026-01-31
**Domain:** Automatic Speech Recognition (ASR) with Speaker Diarization
**Confidence:** HIGH

## Summary

This research investigates implementing speech recognition with speaker diarization for the Voice Dub project. The standard approach combines **faster-whisper** (optimized Whisper implementation) with **pyannote.audio** (speaker diarization), using temporal overlap alignment to merge transcripts with speaker labels.

Whisper Large V3 provides state-of-the-art transcription with 99-language support and word-level timestamps. PyAnnote.audio's speaker-diarization-community-1 model delivers accurate speaker segmentation with 2-5 speaker detection. The key challenge is aligning their outputs: Whisper produces word-level timestamps, pyannote produces speaker segments, and they must be matched based on temporal overlap.

Critical findings: Use faster-whisper (not openai-whisper) for 2-4x speedup with identical accuracy, enable built-in VAD to prevent hallucinations on silence, use word-level timestamps for lip sync requirements, and implement confidence thresholding via logprobs. The project's existing ModelManager pattern perfectly accommodates sequential loading (Whisper 6GB → pyannote 2-4GB → unload both).

**Primary recommendation:** Use faster-whisper with built-in VAD for transcription, pyannote speaker-diarization-community-1 for diarization, and temporal overlap alignment to merge outputs into structured JSON with word-level timestamps, speaker IDs, and confidence scores.

## Standard Stack

The established libraries/tools for ASR with speaker diarization:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faster-whisper | Latest (4.x) | Speech-to-text transcription | 2-4x faster than openai-whisper with identical accuracy, CTranslate2 optimization, built-in VAD, lower VRAM (~4.5GB vs ~10GB) |
| pyannote.audio | 4.0.3+ | Speaker diarization | State-of-the-art speaker segmentation, PyTorch native, community-1 model significantly outperforms legacy 3.1 pipeline |
| torch | PyTorch nightly | ML framework | Already required for RTX 5090 sm_120 support, used by both libraries |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ffmpeg-python | 0.2.0+ | Audio preprocessing | Convert to 16kHz mono PCM WAV (required by both Whisper and pyannote) |
| ctranslate2 | Latest | Inference engine | Automatically installed by faster-whisper, requires CUDA 12 + cuDNN 9 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| faster-whisper | openai-whisper | Original implementation is 2-4x slower, uses 2x VRAM, no built-in VAD. Only advantage: official OpenAI code |
| faster-whisper | WhisperX | WhisperX bundles Whisper + pyannote + alignment, but less control over pipeline stages and memory management. Good for prototyping, not production |
| pyannote speaker-diarization-community-1 | speaker-diarization-3.1 | Legacy 3.1 pipeline has worse speaker counting/assignment. Community-1 shows significant improvements |
| pyannote | WhisperX diarization | WhisperX uses pyannote under the hood, no advantage to bundling |

**Installation:**
```bash
# Faster-whisper (requires CUDA 12 + cuDNN 9 - already available for RTX 5090)
pip install faster-whisper

# PyAnnote audio + HuggingFace hub for model downloads
pip install pyannote.audio

# FFmpeg for audio preprocessing (already in requirements.txt)
pip install ffmpeg-python
```

**Note:** HuggingFace token required for pyannote models (user will provide when needed).

## Architecture Patterns

### Recommended Project Structure
```
src/
├── stages/
│   ├── asr_stage.py           # Main ASR pipeline (transcription + diarization)
│   └── __init__.py
├── utils/
│   ├── audio_preprocessing.py  # FFmpeg resampling to 16kHz mono
│   ├── timestamp_alignment.py  # Whisper + pyannote temporal overlap matching
│   └── ...
├── models/                     # Existing ModelManager
└── config/                     # Existing settings
```

### Pattern 1: Sequential Model Loading for ASR
**What:** Load Whisper for transcription, unload, then load pyannote for diarization
**When to use:** When processing single audio file through ASR pipeline
**Example:**
```python
# Source: Established pattern from existing src/models/model_manager.py
from src.models.model_manager import ModelManager
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

manager = ModelManager()

# Step 1: Load Whisper for transcription (~4.5GB VRAM)
whisper_model = manager.load_model(
    "whisper-large-v3",
    lambda: WhisperModel("large-v3", device="cuda", compute_type="float16")
)
segments, info = whisper_model.transcribe(
    audio_path,
    word_timestamps=True,  # CRITICAL for lip sync (Phase 7)
    vad_filter=True,       # Prevents hallucinations on silence
    beam_size=5
)

# Step 2: Unload Whisper, load pyannote for diarization (~2-4GB VRAM)
diarization_pipeline = manager.load_model(
    "pyannote-diarization",
    lambda: Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        use_auth_token=HUGGINGFACE_TOKEN
    ).to(torch.device("cuda"))
)
diarization = diarization_pipeline(audio_path, min_speakers=2, max_speakers=5)

# Step 3: Unload pyannote when done
manager.unload_current_model()
```

### Pattern 2: Audio Preprocessing Pipeline
**What:** Convert any audio format to 16kHz mono PCM WAV before ASR
**When to use:** Always, before passing to Whisper or pyannote
**Example:**
```python
# Source: https://github.com/openai/whisper (audio processing requirements)
# Both Whisper and pyannote require 16kHz mono audio
import ffmpeg

def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Convert audio to 16kHz mono PCM WAV.

    Both Whisper and pyannote automatically resample, but doing it
    once upfront is more efficient than each model doing it separately.
    """
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            acodec='pcm_s16le',  # 16-bit signed PCM
            ar=16000,            # 16kHz sample rate
            ac=1,                # Mono channel
            f='wav'              # WAV container
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path
```

### Pattern 3: Timestamp Alignment via Temporal Overlap
**What:** Match Whisper word timestamps to pyannote speaker segments based on greatest overlap
**When to use:** To assign speaker IDs to Whisper transcription words
**Example:**
```python
# Source: https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/
# Temporal intersection approach - match by greatest overlap

def align_timestamps(whisper_segments, diarization):
    """
    Align Whisper word-level timestamps with pyannote speaker segments.

    For each Whisper word, find the pyannote speaker segment with
    greatest temporal overlap.
    """
    aligned_transcript = []

    for segment in whisper_segments:
        for word_info in segment.words:
            word_start = word_info.start
            word_end = word_info.end
            word_text = word_info.word
            word_confidence = word_info.probability

            # Find speaker with maximum overlap
            best_speaker = None
            max_overlap = 0

            for turn, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(word_start, turn.start)
                overlap_end = min(word_end, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            aligned_transcript.append({
                "text": word_text,
                "start": word_start,
                "end": word_end,
                "speaker": best_speaker,
                "confidence": word_confidence
            })

    return aligned_transcript
```

### Pattern 4: Confidence-Based Flagging
**What:** Flag low-confidence segments using Whisper's logprobs/probability scores
**When to use:** To identify segments needing user review (success criterion: <70% flagged)
**Example:**
```python
# Source: https://github.com/SYSTRAN/faster-whisper word-level probabilities
# faster-whisper provides word.probability from logprobs

def flag_low_confidence_segments(aligned_transcript, threshold=0.7):
    """
    Flag segments below confidence threshold for user review.

    Whisper's word.probability is exp(avg_logprob) - already normalized.
    """
    flagged_segments = []

    for word_data in aligned_transcript:
        if word_data["confidence"] < threshold:
            word_data["needs_review"] = True
            flagged_segments.append(word_data)
        else:
            word_data["needs_review"] = False

    return aligned_transcript, flagged_segments
```

### Pattern 5: Output JSON Structure
**What:** Standardized transcript format for downstream stages (translation, voice cloning)
**When to use:** Save transcription results to data/temp/{video_id}_transcript.json
**Example:**
```python
# Source: Phase context - downstream requirements for translation and voice cloning

output_format = {
    "metadata": {
        "video_id": "abc123",
        "duration": 120.5,
        "detected_language": "ja",  # ISO 639-1 code
        "num_speakers": 2,
        "processing_time": 12.3
    },
    "segments": [
        {
            "id": 0,
            "text": "こんにちは",
            "start": 0.0,
            "end": 1.2,
            "speaker": "SPEAKER_00",
            "confidence": 0.95,
            "needs_review": False,
            "words": [
                {
                    "word": "こんにちは",
                    "start": 0.0,
                    "end": 1.2,
                    "confidence": 0.95
                }
            ]
        }
    ],
    "flagged_count": 0,
    "flagged_segments": []
}
```

### Anti-Patterns to Avoid

- **Don't use openai-whisper:** slower, higher VRAM, no VAD. Use faster-whisper instead.
- **Don't skip VAD:** Whisper hallucinates on silence. Always enable vad_filter=True.
- **Don't use sentence-level timestamps:** Success criteria requires 0.1s accuracy for lip sync (Phase 7). Use word_timestamps=True.
- **Don't hardcode num_speakers:** Use min_speakers/max_speakers bounds (2-5) instead. Let pyannote detect actual count.
- **Don't skip audio preprocessing:** Both models expect 16kHz mono. Preprocessing once is more efficient than each model resampling.
- **Don't ignore compute_type:** Use float16 on GPU to halve VRAM usage with negligible accuracy loss.
- **Don't process stereo audio directly for calls:** If input is stereo with one speaker per channel (e.g., customer service calls), split channels first. Guarantees 100% speaker attribution and is 30-50% faster than pyannote.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Word-level timestamp alignment | Custom DTW or forced alignment | faster-whisper with word_timestamps=True | Whisper's internal alignment is already accurate to 0.01s precision, far better than needed (0.1s requirement) |
| Speaker diarization | Custom embedding + clustering | pyannote speaker-diarization-community-1 | Speaker diarization requires voice activity detection, speaker embeddings, clustering, and overlap detection - each has edge cases. PyAnnote is state-of-the-art |
| Voice Activity Detection | Amplitude thresholding or RMS | faster-whisper built-in VAD (Silero-VAD) | Simple VAD misses soft speech, false triggers on music. Silero-VAD is ML-based and battle-tested |
| Timestamp overlap calculation | Manual interval intersection | Use existing algorithms (e.g., max(start_a, start_b), min(end_a, end_b)) | Off-by-one errors are common. Temporal overlap is a solved problem |
| Audio resampling | Custom interpolation | FFmpeg resampling | Audio resampling has anti-aliasing, filter design, and edge case requirements. FFmpeg handles all formats |
| Confidence score calculation | Custom logprob aggregation | faster-whisper word.probability | Whisper already provides normalized probabilities. Don't recompute from raw logprobs |
| Language detection | Custom classifier | Whisper's built-in detect_language() | Whisper trained on 99 languages, detection is more accurate than custom models |

**Key insight:** ASR and speaker diarization are research-heavy domains with subtle edge cases (overlapping speech, background noise, accents, silence handling). Use battle-tested implementations instead of custom solutions. The only custom code needed is alignment logic (temporal overlap matching).

## Common Pitfalls

### Pitfall 1: Whisper Hallucinations on Silence
**What goes wrong:** Whisper generates phantom text when processing silent audio (e.g., "Thank you for watching" at video end).
**Why it happens:** Whisper was trained with weak supervision - learned to generate common phrases when uncertain.
**How to avoid:** Enable VAD filtering with `vad_filter=True` in faster-whisper. This removes non-speech segments before transcription.
**Warning signs:** Repeated generic phrases ("Thank you", "Please subscribe") in transcripts, text appearing during silent sections.

**Source:** [Calm-Whisper research](https://arxiv.org/html/2505.12969v1) shows 80% hallucination reduction with VAD.

### Pitfall 2: PyTorch Memory Replacement During Installation
**What goes wrong:** Installing faster-whisper uninstalls CUDA PyTorch and replaces with CPU-only version, breaking GPU support.
**Why it happens:** faster-whisper's ctranslate2 dependency has conflicting PyTorch requirements.
**How to avoid:** Install PyTorch with CUDA support FIRST via conda, then install faster-whisper via pip.
**Warning signs:** "CUDA not available" errors after installation, even though PyTorch was working before.

**Correct installation order:**
```bash
# 1. Install PyTorch with CUDA first (already done in this project)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 2. Install faster-whisper (won't replace PyTorch)
pip install faster-whisper
```

**Source:** [GitHub Issue #930](https://github.com/SYSTRAN/faster-whisper/issues/930)

### Pitfall 3: Word-Level Timestamps Explode Memory
**What goes wrong:** Enabling `return_timestamps="word"` with large batch sizes causes OOM errors (>20GB VRAM vs <7GB for sentence-level).
**Why it happens:** Word-level timestamps require more intermediate activations during beam search.
**How to avoid:** Use word_timestamps=True but reduce batch_size if processing very long audio. For this project (1-30min videos, 32GB VRAM), not an issue.
**Warning signs:** OOM errors when word_timestamps=True works fine when False.

**Source:** [Transformers Issue #27834](https://github.com/huggingface/transformers/issues/27834)

### Pitfall 4: Pyannote Speaker Confusion
**What goes wrong:** Pyannote assigns same speaker ID to different speakers, or splits one speaker into multiple IDs.
**Why it happens:** Speakers with similar pitch/accents confuse the embedding model. Also happens with overlapping speech.
**How to avoid:** Use speaker-diarization-community-1 (not 3.1 legacy). Provide min_speakers/max_speakers hints if known. For post-processing, implement speaker ID consistency checks.
**Warning signs:** Transcript alternates speakers mid-sentence, speaker count doesn't match actual video.

**Source:** [PyAnnote Community-1 announcement](https://www.pyannote.ai/blog/community-1) - "significant reductions in speaker confusion"

### Pitfall 5: Language Auto-Detection on Mixed-Language Audio
**What goes wrong:** Whisper detects language from first 30 seconds, then uses that language for entire audio. Fails on code-switching videos.
**Why it happens:** Whisper assumes single language per audio file.
**How to avoid:** For this project, assume single-language input (success criteria tests 6 languages separately, not mixed). If code-switching needed later, segment audio by language first (out of scope for Phase 3).
**Warning signs:** Transcription accuracy drops partway through video, transcript switches to wrong language.

**Source:** [Whisper Discussion #49](https://github.com/openai/whisper/discussions/49)

### Pitfall 6: HuggingFace Token Not Provided
**What goes wrong:** Pyannote models are gated - require accepting license and providing HuggingFace token. Script crashes with authentication error.
**Why it happens:** Pyannote uses gated models to track usage and enforce licensing.
**How to avoid:** Document HuggingFace token requirement in setup instructions. User must create account, accept model agreement, and provide token.
**Warning signs:** "Repository not found" or "authentication required" errors when loading pyannote pipeline.

**Source:** [PyAnnote Installation Docs](https://github.com/pyannote/pyannote-audio)

### Pitfall 7: Temporal Alignment Edge Cases
**What goes wrong:** Words at speaker boundaries get assigned to wrong speaker, or no speaker at all.
**Why it happens:** Whisper and pyannote have slight timestamp disagreements (~0.1-0.3s). Word might overlap two speaker segments equally.
**How to avoid:** Use "greatest overlap" heuristic (not first match). If overlap is zero, assign to nearest speaker segment temporally.
**Warning signs:** Some words have speaker=None, speaker IDs flip mid-word, boundary words misattributed.

**Solution pattern:**
```python
# If no overlap found, assign to nearest speaker segment
if best_speaker is None:
    # Find temporally closest speaker segment
    min_distance = float('inf')
    for turn, speaker in diarization.itertracks(yield_label=True):
        distance = min(abs(word_start - turn.end), abs(word_end - turn.start))
        if distance < min_distance:
            min_distance = distance
            best_speaker = speaker
```

## Code Examples

Verified patterns from official sources:

### Complete ASR Pipeline
```python
# Source: Synthesized from faster-whisper and pyannote official examples
# Combined with existing project ModelManager pattern

from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
import json

from src.models.model_manager import ModelManager
from src.config.settings import TEMP_DATA_DIR, MODELS_DIR

def transcribe_with_diarization(
    audio_path: str,
    video_id: str,
    huggingface_token: str
) -> dict:
    """
    Complete ASR pipeline: transcription + speaker diarization.

    Args:
        audio_path: Path to 16kHz mono WAV file
        video_id: Unique identifier for output file
        huggingface_token: HuggingFace API token for pyannote models

    Returns:
        dict: Complete transcript with speaker labels and confidence scores
    """
    manager = ModelManager(verbose=True)

    # Step 1: Transcription with Whisper
    print("Loading Whisper Large V3...")
    whisper = manager.load_model(
        "whisper-large-v3",
        lambda: WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
            download_root=str(MODELS_DIR)
        )
    )

    print(f"Transcribing {audio_path}...")
    segments, info = whisper.transcribe(
        audio_path,
        word_timestamps=True,    # Word-level for lip sync
        vad_filter=True,         # Prevent hallucinations
        beam_size=5,             # Balance accuracy vs speed
        language=None            # Auto-detect
    )

    # Convert generator to list (needed for alignment)
    segments_list = list(segments)
    detected_language = info.language

    # Step 2: Speaker Diarization with pyannote
    print("Loading pyannote diarization...")
    diarization = manager.load_model(
        "pyannote-diarization",
        lambda: Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=huggingface_token
        ).to(torch.device("cuda"))
    )

    print("Detecting speakers...")
    speaker_segments = diarization(
        audio_path,
        min_speakers=2,
        max_speakers=5
    )

    # Step 3: Align timestamps
    print("Aligning timestamps with speakers...")
    aligned_words = []

    for segment in segments_list:
        if not hasattr(segment, 'words'):
            continue

        for word in segment.words:
            # Find speaker with maximum overlap
            best_speaker = None
            max_overlap = 0

            for turn, speaker in speaker_segments.itertracks(yield_label=True):
                overlap_start = max(word.start, turn.start)
                overlap_end = min(word.end, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            # If no overlap, assign to nearest speaker
            if best_speaker is None:
                min_distance = float('inf')
                for turn, speaker in speaker_segments.itertracks(yield_label=True):
                    distance = min(
                        abs(word.start - turn.end),
                        abs(word.end - turn.start)
                    )
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = speaker

            aligned_words.append({
                "word": word.word,
                "start": round(word.start, 2),
                "end": round(word.end, 2),
                "speaker": best_speaker if best_speaker else "UNKNOWN",
                "confidence": round(word.probability, 3),
                "needs_review": word.probability < 0.7
            })

    # Step 4: Group into segments by speaker
    segments_output = []
    current_segment = None

    for word_data in aligned_words:
        if current_segment is None or current_segment["speaker"] != word_data["speaker"]:
            # Start new segment
            if current_segment is not None:
                segments_output.append(current_segment)

            current_segment = {
                "id": len(segments_output),
                "text": word_data["word"],
                "start": word_data["start"],
                "end": word_data["end"],
                "speaker": word_data["speaker"],
                "confidence": word_data["confidence"],
                "needs_review": word_data["needs_review"],
                "words": [word_data]
            }
        else:
            # Continue current segment
            current_segment["text"] += word_data["word"]
            current_segment["end"] = word_data["end"]
            current_segment["confidence"] = min(
                current_segment["confidence"],
                word_data["confidence"]
            )
            current_segment["needs_review"] = (
                current_segment["needs_review"] or word_data["needs_review"]
            )
            current_segment["words"].append(word_data)

    # Add final segment
    if current_segment is not None:
        segments_output.append(current_segment)

    # Step 5: Build output
    num_speakers = len(set(w["speaker"] for w in aligned_words))
    flagged_segments = [s for s in segments_output if s["needs_review"]]

    output = {
        "metadata": {
            "video_id": video_id,
            "duration": round(segments_list[-1].end, 2) if segments_list else 0,
            "detected_language": detected_language,
            "num_speakers": num_speakers,
            "total_segments": len(segments_output)
        },
        "segments": segments_output,
        "flagged_count": len(flagged_segments),
        "flagged_segments": [s["id"] for s in flagged_segments]
    }

    # Step 6: Save to file
    output_path = TEMP_DATA_DIR / f"{video_id}_transcript.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to {output_path}")

    # Step 7: Cleanup
    manager.unload_current_model()

    return output
```

### Audio Preprocessing
```python
# Source: https://github.com/openai/whisper audio requirements
# FFmpeg preprocessing to 16kHz mono

import ffmpeg
from pathlib import Path

def preprocess_audio_for_asr(input_path: str, output_dir: str = None) -> str:
    """
    Convert audio to 16kHz mono PCM WAV for Whisper and pyannote.

    Both models require 16kHz mono. Preprocessing once is more efficient
    than each model resampling separately.

    Args:
        input_path: Path to input audio (any format supported by FFmpeg)
        output_dir: Directory for output file (default: same as input)

    Returns:
        str: Path to preprocessed WAV file
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_16khz_mono.wav"

    try:
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',  # 16-bit signed PCM
                ar=16000,            # 16kHz sample rate (required)
                ac=1,                # Mono channel (required)
                f='wav'              # WAV container
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Preprocessed audio saved to {output_path}")
        return str(output_path)

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg preprocessing failed: {error_msg}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| openai-whisper library | faster-whisper with CTranslate2 | 2023-2024 | 2-4x speedup, 50% VRAM reduction, built-in VAD - no accuracy loss |
| PyAnnote speaker-diarization-3.0/3.1 | speaker-diarization-community-1 | Late 2025 | Significant improvements in speaker counting and assignment, reduced speaker confusion |
| Manual Whisper timestamp extraction | word_timestamps=True parameter | Whisper v3 (2023) | Built-in word-level timestamps with 0.01s precision, no custom alignment needed |
| No hallucination prevention | VAD filtering (Silero-VAD) | 2024-2025 | 80% reduction in hallucinations on silence, critical for production use |
| Sentence-level timestamps | Word-level timestamps | Whisper Large V3 | Enables precise lip sync alignment (0.1s accuracy requirement), critical for Phase 7 |

**Deprecated/outdated:**
- **openai-whisper for production:** Slower, higher VRAM, no VAD. Use faster-whisper instead.
- **PyAnnote 3.1 pipeline:** Legacy model with worse speaker assignment. Use community-1.
- **WhisperX for controlled pipelines:** Bundles everything, less control over memory management. Use faster-whisper + pyannote separately with ModelManager.
- **Custom forced alignment:** Whisper's built-in word timestamps are already accurate enough. Don't use Montreal Forced Aligner or wav2vec2 alignment unless sub-0.01s precision needed.

## Open Questions

Things that couldn't be fully resolved:

1. **Exact pyannote VRAM requirements**
   - What we know: Smaller than Whisper, estimates suggest 2-4GB for community-1 model
   - What's unclear: Exact VRAM usage for speaker-diarization-community-1 on RTX 5090
   - Recommendation: Profile during implementation with memory_monitor.py. With 32GB VRAM and ModelManager sequential loading, this is not a blocker.

2. **faster-whisper memory leak status**
   - What we know: GitHub issues reported gradual memory growth on very long audio (5+ hours) in 2024
   - What's unclear: Whether latest faster-whisper 4.x resolves the leak
   - Recommendation: For 1-30min videos (success criteria), not a concern. If leak exists, ModelManager's cleanup handles it. Monitor with memory_monitor.py during testing.

3. **Code-switching handling**
   - What we know: Whisper detects language from first 30s, assumes single language. Research shows poor performance on intra-sentence code-switching.
   - What's unclear: Whether success criteria requires code-switching support ("transcribes 20-30 languages" - ambiguous if this means mixed languages in one video)
   - Recommendation: Implement single-language assumption. If code-switching needed, handle in future phase with VAD-based segmentation + per-segment language detection.

4. **Optimal beam_size for RTX 5090**
   - What we know: beam_size=5 is default, higher values improve accuracy but increase latency. Success criterion is <10s for 1-min audio.
   - What's unclear: Optimal beam_size for RTX 5090 to balance accuracy and speed
   - Recommendation: Start with beam_size=5 (default). Profile with 1-min test audio. If under 5s, try beam_size=10 for better accuracy. If over 8s, reduce to beam_size=3.

## Sources

### Primary (HIGH confidence)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper) - Official repository, API documentation
- [Whisper Large V3 Model Card](https://huggingface.co/openai/whisper-large-v3) - Model specifications, VRAM requirements, performance metrics
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) - Installation, API, performance benchmarks
- [PyAnnote.audio GitHub](https://github.com/pyannote/pyannote-audio) - Version 4.0.3 documentation, installation, API usage
- [PyAnnote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - Model specifications, output format

### Secondary (MEDIUM confidence)
- [Whisper and Pyannote Integration Guide](https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/) - Temporal overlap alignment algorithm
- [WhisperX GitHub](https://github.com/m-bain/whisperX) - Timestamp accuracy improvements, batching patterns
- [Building Custom ASR Pipeline](https://medium.com/@rafaelgalle1/building-a-custom-scalable-audio-transcription-pipeline-whisper-pyannote-ffmpeg-d0f03f884330) - Audio preprocessing best practices
- [Calm-Whisper Research](https://arxiv.org/html/2505.12969v1) - VAD hallucination reduction (80%)
- [PyAnnote Community-1 Announcement](https://www.pyannote.ai/blog/community-1) - Speaker confusion improvements

### Tertiary (LOW confidence)
- [Whisper Speaker Diarization Tutorial 2026](https://brasstranscripts.com/blog/whisper-speaker-diarization-guide) - Integration patterns (not verified with official docs)
- [Best Speaker Diarization Models Compared](https://brasstranscripts.com/blog/speaker-diarization-models-comparison) - Performance comparisons (third-party benchmarks)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - faster-whisper and pyannote.audio are verified as industry standard, official documentation confirms capabilities
- Architecture: HIGH - Temporal overlap alignment is well-documented, ModelManager pattern already proven in Phase 1
- Pitfalls: MEDIUM - Most pitfalls verified via GitHub issues and official docs, some based on community reports

**Research date:** 2026-01-31
**Valid until:** 2026-03-02 (30 days) - ASR domain is stable, but library versions update frequently
