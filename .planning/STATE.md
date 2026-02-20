# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-31)

**Core value:** Watch any video content in English while preserving the original speaker's voice characteristics and emotional expression, without relying on cloud services or paying API fees.
**Current focus:** Phase 7: Lip Synchronization (executing)

## Current Position

Phase: 7 of 11 (Lip Synchronization)
Plan: 2 of 4 (07-01 and 07-02 complete)
Status: In progress — Plans 07-01 and 07-02 complete (wave 1 done); 07-03 and 07-04 pending
Last activity: 2026-02-20 — Completed 07-01: LatentSync conda env, checkpoints, audio_prep, and latentsync_runner modules

Progress: [█████████░] 86%

## Performance Metrics

**Velocity:**
- Total plans completed: 19
- Average duration: 27 minutes
- Total execution time: 9.8 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01    | 3     | 358m  | 119m     |
| 02    | 2     | 7m    | 3.5m     |
| 03    | 3     | 12m   | 4m       |
| 04    | 4     | 126m  | 31.5m    |
| 05    | 4     | 19m   | 4.8m     |
| 06    | 3     | 21m   | 7m       |

**Recent Trend:**
- Last 5 plans: 06-01 (3m), 06-02 (6m), 06-03 (12m), 07-01 (18m)
- Trend: Phase 7 in progress; 07-01 longer due to Miniconda install + checkpoint downloads

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

**Phase 03-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| speaker-diarization-community-1 | Use pyannote/speaker-diarization-community-1 | Better speaker counting vs legacy 3.1 pipeline | ✅ Implemented |
| temporal-overlap-alignment | Use temporal overlap for speaker-word matching | More accurate speaker attribution for boundary words | ✅ Implemented |
| nearest-speaker-fallback | Assign words to nearest speaker when no overlap | Prevents "UNKNOWN" speakers from timestamp mismatches | ✅ Implemented |
| speaker-contiguous-grouping | Group consecutive words by speaker into segments | Structured output for downstream stages | ✅ Implemented |

**Phase 03-03 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| json-transcript-format | Export ASR results to JSON with nested structure | Enables downstream stages to consume structured transcripts | ✅ Implemented |
| progress-callback-pattern | Support optional progress callbacks | Enables responsive Gradio UI integration | ✅ Implemented |
| automatic-cleanup | Cleanup models and temp files after ASR | Prevents VRAM/disk leaks in sequential pipeline | ✅ Implemented |

**Phase 04-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| seamless-fp16 | Use fp16 (float16) for SeamlessM4T inference | Halves VRAM usage (~2.5GB vs ~5GB) with negligible quality impact | ✅ Implemented |
| greedy-decoding-first | Implement greedy decoding before beam search | Simpler baseline, beam search added in 04-02 for multi-candidate | ✅ Implemented |
| lazy-model-loading | Defer model loading until first translate call | Allows translator instantiation without immediate VRAM allocation | ✅ Implemented |
| sentencepiece-upgrade | Upgrade sentencepiece to 0.2.1 for protobuf compatibility | Fixes "Descriptors cannot be created directly" error | ✅ Implemented |

**Phase 04-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| character-count-duration | Use character-count heuristic (15 chars/sec) for duration estimation | Instant calculation vs linguistic analysis overhead | ✅ Implemented |
| linear-penalty-scoring | Linear penalty within tolerance, 0.0 outside | Gradual degradation for acceptable translations, hard cutoff for invalid | ✅ Implemented |
| 60-40-weight-split | Default 60% confidence / 40% duration weights | Speed-first priority with duration constraints for downstream stages | ✅ Implemented |
| detailed-validation-dict | Return dict with all metadata from validate_duration() | Enables debugging and transparent decision-making | ✅ Implemented |

**Phase 04-03 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| beam-width-3 | Beam width defaults to 3 candidates | Balances quality (diverse options) vs speed (3x compute) | ✅ Implemented |
| batch-size-8 | Batch size of 8 segments for GPU processing | Balances throughput and VRAM usage, 2-4x speedup | ✅ Implemented |
| chunk-1024-overlap-128 | 1024 token max chunk with 128 token overlap | Preserves conversational context for long videos | ✅ Implemented |
| later-chunk-wins | Later chunks override earlier for overlapped segments | More context from subsequent segments improves quality | ✅ Implemented |
| compute-transition-scores | Use compute_transition_scores for confidence | Official Transformers method for per-token log probabilities | ✅ Implemented |

**Phase 04-04 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| stage-orchestration-pattern | Stage orchestration mirrors ASR pattern | Consistency across pipeline stages for maintainability | ✅ Implemented |
| conditional-chunking-1024 | Conditional chunking at 1024 token threshold | Automatic strategy selection based on transcript length | ✅ Implemented |
| duration-validation-10pct | Duration validation flags segments outside ±10% | Ensures lip sync compatibility with timing constraints | ✅ Implemented |
| confidence-threshold-07 | Confidence flagging threshold at 0.7 | Balances quality and coverage for Phase 8 review | ✅ Implemented |
| utf8-json-export | JSON export uses ensure_ascii=False | Preserves non-English characters in original text | ✅ Implemented |

**Phase 05-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| rms-energy-proxy | Use RMS energy as proxy for audio quality | Fast reference selection via instant calculation | ✅ Implemented |
| segment-concatenation-fallback | Concatenate short segments when no 6s+ segment exists | Enables voice cloning for fragmented speech patterns | ✅ Implemented |
| gpu-cpu-cache-management | Support GPU/CPU movement for speaker embeddings | Prevents OOM on videos with 10+ speakers | ✅ Implemented |
| tts-temperature-065 | Default temperature 0.65 for synthesis | Balances consistency and expressiveness per XTTS docs | ✅ Implemented |

**Phase 05-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| low-level-synthesizer-api | Use model.synthesizer.tts() for speed parameter access | Enables precise duration control via speed adjustment (0.8-1.2x range) | ✅ Implemented |
| binary-search-speed-matching | Binary search for speed parameter within 0.8-1.2 range | Achieves ±5% duration match in 3-5 attempts for 90%+ segments | ✅ Implemented |
| speaker-grouped-batching | Group segments by speaker for batch processing | Processes videos with 10+ speakers without VRAM exhaustion | ✅ Implemented |
| 20-percent-failure-threshold | Raise BatchSynthesisError if >20% segments fail | Batch continues for isolated failures, alerts user for widespread problems | ✅ Implemented |

**Phase 05-03 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| pesq-stoi-combination | Use PESQ + STOI for comprehensive quality assessment | Combines perceptual quality (PESQ) with intelligibility (STOI) | ✅ Implemented |
| pitch-variance-emotion-proxy | Use pitch variance ratio as emotion preservation proxy | Detects flat/monotone (ratio < 0.6) or exaggerated (ratio > 1.5) output | ✅ Implemented |
| duration-only-fallback | Support duration-only validation without reference | Enables quality checking when reference audio unavailable | ✅ Implemented |
| pesq-quality-tiers | Four-tier PESQ classification | Maps MOS scores to human-readable categories for UI/logging | ✅ Implemented |

**Phase 05-04 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| stage-orchestration-pattern | Stage orchestration mirrors ASR/translation pattern | Consistency across pipeline stages for maintainability | ✅ Implemented |
| explicit-quality-failure-handling | Quality validation with TTSStageFailed exception | Stage fails if >50% segments rejected, prevents bad audio from proceeding | ✅ Implemented |
| emotion-preservation-tracking | Track emotion preservation separately from quality_passed | Enables UI to show emotion flags without failing stage | ✅ Implemented |
| generate-speaker-embeddings-helper | Use generate_speaker_embeddings() helper function | Cleaner API, handles XTTS model loading correctly | ✅ Implemented |

**Phase 06-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| float64-timestamps | Use float64 precision for all timestamps | Prevents precision loss over 20+ minute videos | ✅ Implemented |
| 48khz-target-rate | Normalize all audio to 48kHz | DVD/broadcast standard, prevents sample rate drift | ✅ Implemented |
| kaiser-best-resampling | Use librosa kaiser_best for resampling | Highest quality sinc interpolation per research | ✅ Implemented |
| 45ms-drift-tolerance | Set 45ms as drift tolerance threshold | ATSC standard for acceptable A/V offset | ✅ Implemented |
| 5min-checkpoints | Validate sync at 5-minute intervals | Early drift detection for long videos | ✅ Implemented |
| frame-boundary-alignment | Align timestamps to video frame boundaries | Prevents sub-frame jitter accumulation | ✅ Implemented |

**Phase 06-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| dict-async-workaround | Use `**{'async': 1}` for aresample parameter | Enables FFmpeg async parameter without Python syntax error | ✅ Implemented |
| checkpoint-actual-duration | Calculate actual duration by summing segments up to checkpoint | Accurate drift measurement at any checkpoint timestamp | ✅ Implemented |
| format-specific-codecs | Provide different codec configs per output format | Ensures compatibility and optimal quality per format | ✅ Implemented |
| compatibility-validation | Validate inputs with ffprobe before merge | User-friendly error messages for missing streams | ✅ Implemented |

**Phase 07-01 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| subprocess-isolation | LatentSync runs in isolated conda env (torch 2.5.1+cu121) via subprocess | Prevents PyTorch version conflict with RTX 5090 nightly (cu128) | ✅ Implemented |
| ffmpeg-resampling-not-librosa | Use FFmpeg for 48->16kHz conversion | Format-identical to LatentSync's own preprocessing; avoids extra dependency | ✅ Implemented |
| deepcache-enabled-by-default | enable_deepcache=True default in run_latentsync_inference() | DeepCache fix (f5040cf) confirmed in repo; ~2x speedup at negligible quality cost | ✅ Implemented |
| env-var-override | LATENTSYNC_PYTHON_PATH env var overrides hardcoded path | Portability across machines without code changes | ✅ Implemented |

**Phase 07-02 Decisions:**

| ID | Title | Impact | Status |
|----|-------|--------|--------|
| wav2lip-gan-pth-guard | ValueError if s3fd.pth passed as checkpoint_path | Prevents KeyError: state_dict crash from wrong checkpoint file | ✅ Implemented |
| accurate-seek-reencode | Use -ss -t after -i for chunk splitting (re-encode) | Prevents GOP misalignment that inflates concat duration (12s→17.3s with stream copy) | ✅ Implemented |
| 5min-chunk-default | 300s default chunk duration for both LatentSync and Wav2Lip | Prevents face detection VRAM spikes on long videos | ✅ Implemented |

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

**Phase 2 Complete:** Video processing pipeline ready
- ✅ FFmpeg-based video/audio extraction with stream copy
- ✅ Format normalization and codec selection
- ✅ Progress callback pattern for UI integration

**Phase 3 Complete:** Speech recognition pipeline ready
- ✅ Whisper Large V3 transcription with word-level timestamps
- ✅ Pyannote speaker diarization (2-5 speakers)
- ✅ Temporal alignment merging transcription with speakers
- ✅ Complete ASR stage orchestration with JSON export
- ✅ HuggingFace token documentation for users
- ⚠️ Requires user setup: pyannote.audio installation + HuggingFace token

**Phase 4 Complete:** Translation pipeline (4/4 plans complete, 2026-01-31)
- ✅ SeamlessM4T v2 translator wrapper with fp16 optimization (04-01)
- ✅ Duration validation and candidate ranking (04-02)
- ✅ Multi-candidate beam search and context chunking (04-03)
- ✅ Translation stage integration with JSON I/O (04-04)

**Phase 5 Complete:** Voice Cloning & TTS (4/4 plans complete, 2026-02-02)
- ✅ Reference sample extraction with RMS-based selection (05-01)
- ✅ Speaker embedding cache with GPU/CPU management (05-01)
- ✅ TTS configuration parameters (05-01)
- ✅ XTTS synthesis wrapper with duration matching (05-02)
- ✅ Binary search speed adjustment for ±5% duration tolerance (05-02)
- ✅ Speaker-grouped batch processing (05-02)
- ✅ Audio quality validation with PESQ and STOI (05-03)
- ✅ Emotion preservation via pitch variance ratio (05-03)
- ✅ TTS stage orchestration with quality validation (05-04)
- ✅ Comprehensive integration tests (13 tests) (05-04)
- ✅ Phase 5 documentation in README (05-04)

**Phase 6 Complete:** Audio-Video Assembly (3/3 plans complete, 2026-02-03)
- ✅ Assembly infrastructure with timestamp validation (06-01)
- ✅ Float64 precision for drift prevention (06-01)
- ✅ 48kHz sample rate normalization with kaiser_best (06-01)
- ✅ Audio concatenation with gap padding (06-01)
- ✅ Checkpoint-based drift detection at 5-minute intervals (06-02)
- ✅ FFmpeg merge with aresample async=1 for drift correction (06-02)
- ✅ Complete assembly module API with 15 public exports (06-02)
- ✅ Assembly stage orchestration with run_assembly_stage() (06-03)
- ✅ 12-function integration test suite (479 lines, 100% pass rate) (06-03)
- ✅ Complete Phase 6 documentation in README (06-03)

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 07-01-PLAN.md — LatentSync conda env, checkpoints, audio_prep, latentsync_runner
Resume file: None
Next: Execute 07-03 (stage orchestration, depends on 07-01 and 07-02)
