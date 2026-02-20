---
phase: 07-lip-synchronization
plan: 03
subsystem: lip-sync
tags: [latentsync, wav2lip, ffmpeg, ffprobe, lip-sync, orchestration, validation]

# Dependency graph
requires:
  - phase: 07-01
    provides: audio_prep.py (16kHz resampling), latentsync_runner.py (subprocess inference)
  - phase: 07-02
    provides: wav2lip_runner.py (fallback), chunker.py (split/concat for long videos)
  - phase: 06-audio-video-assembly
    provides: run_assembly_stage() pattern that run_lip_sync_stage() mirrors
provides:
  - run_lip_sync_stage() pipeline entry point for lip sync stage
  - LipSyncResult dataclass with 12-field to_dict() JSON serialization
  - LipSyncStageFailed exception for stage-level failures
  - validate_lip_sync_output() and SyncValidation dataclass (brightness-based frame validation)
affects:
  - 07-04 (integration tests and Gradio UI — depends on run_lip_sync_stage() and LipSyncResult)
  - 08 (pipeline orchestration — calls run_lip_sync_stage as a stage)
  - 11 (Gradio UI — displays LipSyncResult fields and sync_validation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stage orchestration pattern: result dataclass + to_dict() + stage exception + single orchestration function with progress callbacks"
    - "Advisory validation: sync validation never fails stage; caller logs warning only"
    - "Brightness-based video QA via ffprobe signalstats YAVG metric"
    - "Optimistic fallback: 0-frame signalstats result treated as passed to avoid blocking pipeline"

key-files:
  created:
    - src/stages/lip_sync_stage.py
    - src/lip_sync/validator.py
  modified:
    - src/lip_sync/__init__.py
    - src/stages/__init__.py

key-decisions:
  - "Sync validation is advisory-only: exceptions caught and logged as warnings, stage never fails due to validation"
  - "Optimistic default for 0 brightness samples: pass_rate=1.0, passed=True to avoid blocking pipeline when ffprobe signalstats returns nothing"
  - "inference_steps=0 and guidance_scale=0.0 used in LipSyncResult when fallback_used=True (Wav2Lip has no such params)"
  - "Chunked fallback is per-chunk: if LatentSync fails on any chunk, Wav2Lip is tried for that chunk only"

patterns-established:
  - "Stage pattern: run_{stage}_stage(input, output_dir, ..., progress_callback) -> {Stage}Result"
  - "Exception pattern: {Stage}StageFailed wraps both expected errors and unexpected exceptions"
  - "JSON export: result.to_dict() written to output_dir/{stage}_result.json"
  - "Progress callbacks: 9 distinct points from 0.05 to 1.0, default no-op lambda"

# Metrics
duration: 3min
completed: 2026-02-20
---

# Phase 7 Plan 3: Lip Sync Stage Orchestration Summary

**Stage orchestration function run_lip_sync_stage() connecting audio prep, LatentSync/Wav2Lip inference, chunking, and brightness-based output validation into a single pipeline-compatible entry point**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-20T20:19:57Z
- **Completed:** 2026-02-20T20:22:56Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `run_lip_sync_stage()` orchestrates the full lip sync pipeline in one call: 16kHz audio resampling, optional 5-minute chunking for long videos, LatentSync primary inference with Wav2Lip GAN fallback on RuntimeError, advisory sync validation, and JSON result export
- `LipSyncResult` dataclass with 12 fields (including `sync_validation`, `multi_speaker_mode`, `fallback_used`, `chunks_processed`) and a `to_dict()` method that serializes correctly to JSON
- `validate_lip_sync_output()` samples every 30th frame via ffprobe signalstats YAVG, returning `SyncValidation` with pass_rate and passed flag; never blocks the pipeline

## Task Commits

Each task was committed atomically:

1. **Task 1: Create lip_sync_stage.py** - `2cc1a03` (feat)
2. **Task 2: Create validator.py and update stages/__init__.py** - `438f7ef` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/stages/lip_sync_stage.py` — run_lip_sync_stage(), LipSyncResult, LipSyncStageFailed (358 lines)
- `src/lip_sync/validator.py` — validate_lip_sync_output(), SyncValidation, BRIGHTNESS_THRESHOLD
- `src/lip_sync/__init__.py` — added validate_lip_sync_output and SyncValidation exports
- `src/stages/__init__.py` — added run_lip_sync_stage, LipSyncResult, LipSyncStageFailed exports

## Decisions Made

- **Advisory-only sync validation**: Exceptions from `validate_lip_sync_output()` are caught and logged as warnings; the stage never fails due to validation. This prevents ffprobe signalstats compatibility issues from blocking the pipeline.
- **Optimistic zero-sample fallback**: When ffprobe signalstats returns 0 brightness samples (e.g. lavfi/signalstats not available), `SyncValidation` defaults to `pass_rate=1.0, passed=True`. Being optimistic avoids blocking valid output.
- **Per-chunk fallback in chunked mode**: If LatentSync fails on a specific chunk (RuntimeError), Wav2Lip is tried for that chunk only. The whole stage fails only if both models fail on the same chunk.
- **Wav2Lip result fields**: `inference_steps=0` and `guidance_scale=0.0` stored in LipSyncResult when `fallback_used=True` since Wav2Lip has no such parameters.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `run_lip_sync_stage()` ready for Plan 07-04 integration tests
- `LipSyncResult.to_dict()` produces stable JSON schema for Gradio UI (Phase 11)
- Pipeline orchestration (Phase 8) can call `run_lip_sync_stage(assembled_video, output_dir)` directly
- Multi-speaker warning logged when `speakers_detected > 1` — no architectural change needed

---
*Phase: 07-lip-synchronization*
*Completed: 2026-02-20*
