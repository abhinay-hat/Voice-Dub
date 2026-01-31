# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Watch any video content in English while preserving the original speaker's voice characteristics and emotional expression, without relying on cloud services or paying API fees.
**Current focus:** Phase 3: Speech Recognition

## Current Position

Phase: 3 of 11 (Speech Recognition)
Plan: 1 of 3 (complete)
Status: In progress
Last activity: 2026-01-31 — Completed 03-01-PLAN.md (Audio Preprocessing & Whisper Transcription)

Progress: [███░░░░░░░] 36%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 65 minutes
- Total execution time: 6.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 3     | 358m  | 119m     |
| 02    | 2     | 7m    | 3.5m     |
| 03    | 1     | 3m    | 3m       |

**Recent Trend:**
- Last 5 plans: 01-03 (15m), 02-01 (4m), 02-02 (3m), 03-01 (3m)
- Trend: Foundation-building plans very fast (3-4m), initial research/setup plans slow (337m)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- XTTS-v2 for voice cloning (best open-source with emotion preservation)
- Meta Seamless for translation (preserves vocal expression)
- Whisper Large V3 for speech-to-text (99+ languages, high accuracy)
- Gradio for web UI (simplest for friends to use)
- PyTorch + CUDA stack (RTX 5090 native support)
- Wav2Lip HD or LatentSync for lip sync (good quality with acceptable speed)

**CRITICAL HARDWARE UPDATE:**
Research assumed AMD GPU/ROCm, but actual hardware is RTX 5090 (32GB VRAM) + CUDA. This means:
- Native PyTorch CUDA support (simpler than ROCm)
- Better model compatibility (no community forks needed)
- Faster processing (10-20 min vs 1 hour for 20-min video)
- Multiple models can load simultaneously (32GB VRAM)
- Update stack to use standard CUDA implementations

**Phase 01-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| cuda-128-nightly | Use PyTorch nightly with CUDA 12.8 | Required for RTX 5090 sm_120 support | ✅ Implemented |
| comprehensive-validation | 6-step GPU validation with allocation test | Prevents silent CPU fallback | ✅ Implemented |
| python-311 | Use Python 3.11 instead of 3.13 | Fully compatible, already available | ✅ Implemented |

**Phase 01-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| sequential-model-loading | Sequential model loading pattern | Prevents VRAM exhaustion on 32GB RTX 5090 | ✅ Implemented |
| 4-step-cleanup | 4-step model cleanup protocol | Ensures complete VRAM release without leaks | ✅ Implemented |
| stage-based-directory-structure | Stage-based src/ organization | Maps to 11-phase pipeline architecture | ✅ Implemented |

**Phase 01-03 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| comprehensive-test-suite | 4-test environment validation suite | Validates GPU, memory, model loading, env vars | ✅ Implemented |
| hardware-requirements-first | Front-load RTX 5090 requirements in README | Users know compatibility immediately | ✅ Implemented |
| sm120-troubleshooting | Dedicated troubleshooting for sm_120 issues | Users can self-solve common RTX 5090 problems | ✅ Implemented |

**Phase 02-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| ffmpeg-python-wrapper | Use ffmpeg-python over subprocess or MoviePy | 40-100x faster for I/O, type-safe API | ✅ Implemented |
| format-normalization | Normalize container formats to mp4/mkv/avi | Simpler downstream logic vs FFmpeg's complex strings | ✅ Implemented |
| context-manager-temps | Use context managers for temp file management | Prevents disk space leaks from orphaned files | ✅ Implemented |

**Phase 02-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| stream-copy-extraction | Use stream copy for video extraction (no re-encoding) | Preserves exact video quality and 10-100x speedup | ✅ Implemented |
| format-specific-codecs | Select audio codec based on output container format | MP4 needs AAC, MKV supports copy, AVI needs MP3 | ✅ Implemented |
| progress-callback-pattern | Use progress callback for UI integration | Enables Gradio progress bars without tight coupling | ✅ Implemented |

**Phase 03-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| faster-whisper-over-openai | Use faster-whisper instead of openai-whisper | 2-4x speedup, 50% VRAM reduction, built-in VAD | ✅ Implemented |
| word-level-timestamps | Enable word_timestamps=True | Required for lip sync precision (Phase 7) | ✅ Implemented |
| vad-filtering | Enable vad_filter=True | Prevents 80% of hallucinations on silence | ✅ Implemented |
| float16-compute | Use compute_type=float16 | Halves VRAM usage (~4.5GB vs ~10GB) | ✅ Implemented |
| 16khz-preprocessing | Preprocess audio to 16kHz mono | Both models require it, do once not twice | ✅ Implemented |

### Pending Todos

None yet.

### Blockers/Concerns

~~**Phase 1:** Hardware update requires stack validation (PyTorch CUDA not ROCm, standard model implementations not ROCm forks)~~ ✅ RESOLVED (01-01)
- PyTorch nightly 2.11.0.dev20260130+cu128 installed
- RTX 5090 sm_120 support verified
- GPU validation utility confirms 31.84 GB VRAM accessible

**Phase 1 Complete:** Foundation established, ready for Phase 2 (Video Processing Pipeline)
- ✅ RTX 5090 environment validated with sm_120 compute capability
- ✅ Sequential model loading prevents VRAM exhaustion
- ✅ Memory monitoring utilities track VRAM usage
- ✅ Project structure established
- ✅ Automated test suite validates environment
- ✅ Setup documentation enables reproduction

## Session Continuity

Last session: 2026-01-31 (plan 03-01 execution)
Stopped at: Completed 03-01-PLAN.md - Audio preprocessing and Whisper transcription with word-level timestamps
Resume file: None
Next: Plan 03-02 - Speaker diarization with pyannote
