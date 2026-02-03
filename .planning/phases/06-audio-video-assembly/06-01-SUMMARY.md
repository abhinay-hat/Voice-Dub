---
phase: 06-audio-video-assembly
plan: 01
subsystem: audio-processing
tags: [audio-normalization, timestamp-validation, librosa, numpy, float64, 48khz, drift-prevention]

# Dependency graph
requires:
  - phase: 05-voice-cloning-tts
    provides: TTS segments at 24kHz sample rate
  - phase: 03-speech-recognition
    provides: Word-level timestamps from ASR
provides:
  - Float64 timestamp validation infrastructure
  - 48kHz sample rate normalization with kaiser_best sinc interpolation
  - Audio segment concatenation with gap padding
  - Assembly configuration constants (48kHz, 45ms tolerance, 300s intervals)
affects: [06-02-drift-detection, 06-03-video-merge, 07-lip-sync]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Float64 timestamp precision throughout pipeline
    - High-quality kaiser_best sinc resampling for sample rate normalization
    - TimedSegment dataclass for precise audio segment metadata
    - Gap-aware audio concatenation with silence padding

key-files:
  created:
    - src/assembly/__init__.py
    - src/assembly/timestamp_validator.py
    - src/assembly/audio_normalizer.py
    - src/assembly/audio_concatenator.py
  modified:
    - src/config/settings.py

key-decisions:
  - "Use float64 timestamps throughout pipeline to prevent precision loss over 20+ minute videos"
  - "Normalize all audio to 48kHz (DVD/broadcast standard) before assembly"
  - "Use librosa kaiser_best resampling for highest quality sinc interpolation"
  - "Set 45ms drift tolerance per ATSC recommendation for audio-video offset"
  - "Implement 5-minute checkpoint intervals for drift validation"

patterns-established:
  - "TimedSegment dataclass: Audio segment with float64 timestamps, speaker_id, and audio_path"
  - "to_frame_boundary() method: Align timestamps to nearest video frame boundaries"
  - "normalize_sample_rate(): Idempotent resampling (returns original if already at target rate)"
  - "Gap padding: Insert silence when concatenating segments with gaps > 1μs"

# Metrics
duration: 3min
completed: 2026-02-03
---

# Phase 6 Plan 1: Audio-Video Assembly Infrastructure Summary

**Float64 timestamp validation, 48kHz sample rate normalization with kaiser_best sinc interpolation, and gap-aware audio concatenation for drift-free synchronization**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-03T10:40:44Z
- **Completed:** 2026-02-03T10:43:22Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Float64 timestamp precision infrastructure prevents drift over 20+ minute videos
- 48kHz normalization using kaiser_best sinc interpolation (research-validated highest quality)
- TimedSegment dataclass with frame boundary alignment for video sync
- Audio concatenation with automatic silence padding for gaps
- Assembly constants (48kHz standard, 45ms ATSC tolerance, 300s checkpoints)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add assembly constants to settings.py** - `2985336` (feat)
2. **Task 2: Create timestamp_validator.py with TimedSegment dataclass** - `204e0a4` (feat)
3. **Task 3: Create audio_normalizer.py and audio_concatenator.py** - `b200d3b` (feat)

## Files Created/Modified

**Created:**
- `src/assembly/__init__.py` - Assembly module exports and documentation
- `src/assembly/timestamp_validator.py` - Float64 timestamp validation, TimedSegment dataclass with frame alignment
- `src/assembly/audio_normalizer.py` - Sample rate normalization to 48kHz using kaiser_best resampling
- `src/assembly/audio_concatenator.py` - Audio segment concatenation with gap padding

**Modified:**
- `src/config/settings.py` - Added ASSEMBLY_TARGET_SAMPLE_RATE (48000), ASSEMBLY_DRIFT_TOLERANCE_MS (45.0), ASSEMBLY_CHECKPOINT_INTERVAL (300.0), ASSEMBLY_RESAMPLING_QUALITY ('kaiser_best')

## Decisions Made

**Float64 precision for all timestamps:**
- Prevents precision loss over 20+ minute videos (research pitfall #2)
- Python's native float is 64-bit, provides ~15 decimal digits
- NumPy float64 used for duration calculations to maintain precision

**48kHz as target sample rate:**
- DVD/broadcast production standard (research confirmed)
- XTTS outputs 24kHz, Whisper preprocesses to 16kHz - normalization needed
- Consistent rate prevents progressive drift from sample rate mismatch

**kaiser_best resampling quality:**
- Research confirmed as highest quality sinc interpolation
- Prevents aliasing artifacts vs linear interpolation
- Uses band-limited sinc function for clean frequency response

**45ms drift tolerance:**
- ATSC standard for acceptable audio-video offset
- Human perception threshold for A/V sync issues
- Applied at 5-minute checkpoint intervals (300s)

**Frame boundary alignment:**
- TimedSegment.to_frame_boundary() rounds to nearest frame
- Prevents sub-frame jitter accumulation over hundreds of segments
- Critical for lip sync (Phase 7) where 1 frame (33ms at 30fps) is noticeable

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all modules implemented smoothly with existing dependencies (librosa, soundfile, numpy already installed from previous phases).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 6 Plan 2 (Drift Detection):**
- TimedSegment dataclass provides precise timestamp metadata
- validate_timestamps_precision() ensures float64 consistency
- 48kHz normalization infrastructure ready for batch processing

**Ready for Phase 6 Plan 3 (Video Merge):**
- Audio normalization ensures consistent sample rate for FFmpeg
- Concatenation with gap padding maintains timeline alignment
- Assembly constants define merge parameters (tolerance, checkpoints)

**Blockers:** None

**Concerns:** None - foundation components are complete and ready for integration into assembly stage orchestrator.

---
*Phase: 06-audio-video-assembly*
*Completed: 2026-02-03*
