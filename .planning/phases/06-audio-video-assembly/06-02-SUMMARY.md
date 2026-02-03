# Phase 6 Plan 2: Drift Detection & Video Merging Summary

**One-liner:** Checkpoint-based drift detection at 5-minute intervals with FFmpeg aresample async=1 for progressive drift prevention

---

## Plan Reference

- **Phase:** 06-audio-video-assembly
- **Plan:** 02
- **Subsystem:** assembly
- **Tags:** drift-detection, ffmpeg, sync-validation, audio-video-merge

## Execution Summary

**Duration:** 6 minutes
**Started:** 2026-02-03T10:49:31Z
**Completed:** 2026-02-03T10:55:18Z
**Tasks completed:** 3/3
**Files modified:** 3

## Accomplishments

### Core Deliverables

**1. Drift Detection Infrastructure**
- `SyncCheckpoint` dataclass with timestamp, frame_number, audio_sample, drift_ms fields
- `DriftValidationResult` dataclass with is_synced, checkpoints, max_drift_ms aggregation
- `validate_sync_at_intervals()` validates sync at 5-minute checkpoints (300s default)
- Uses 45ms ATSC tolerance for drift detection (ASSEMBLY_DRIFT_TOLERANCE_MS)
- Float64 precision throughout for sub-millisecond accuracy over 20+ minute videos
- `check_segment_continuity()` validates chronological ordering and detects overlaps

**2. FFmpeg Video Merger with Sync Flags**
- `MergeResult` dataclass tracking output_path, codecs, async_correction flag, merge_duration
- `merge_with_sync_validation()` applies modern FFmpeg sync pattern:
  - Uses `.filter('aresample', **{'async': 1})` (replaces deprecated `-async 1`)
  - Explicit stream mapping prevents auto-selection issues
  - Stream copy for video (vcodec='copy') - no re-encoding, preserves quality
  - Format-specific audio codecs (AAC for MP4, copy for MKV, MP3 for AVI)
- `get_merge_config()` provides optimal codec settings per format
- `validate_audio_video_compatibility()` validates inputs before merge using ffprobe

**3. Complete Assembly Module API**
- Updated `src/assembly/__init__.py` with 15 public exports:
  - Timestamp validation: TimedSegment, validate_timestamps_precision, ensure_float64
  - Audio normalization: normalize_sample_rate, validate_sample_rate, batch_normalize
  - Concatenation: concatenate_audio_segments, get_total_duration
  - Drift detection: SyncCheckpoint, DriftValidationResult, validate_sync_at_intervals, check_segment_continuity
  - Video merging: MergeResult, merge_with_sync_validation, get_merge_config, validate_audio_video_compatibility
- Comprehensive docstring for module usage

## Files Created/Modified

### Created
- `src/assembly/drift_detector.py` (279 lines) - Checkpoint-based sync validation
- `src/assembly/video_merger.py` (280 lines) - FFmpeg merge with drift correction

### Modified
- `src/assembly/__init__.py` - Added drift detector and video merger exports

## Key Technical Details

### Drift Detection Algorithm
- Generates checkpoints at interval_seconds (5 minutes) throughout video
- For each checkpoint, calculates actual cumulative duration from segments
- Compares actual vs expected timestamp, calculates drift in milliseconds
- Flags checkpoints exceeding ±45ms tolerance (ATSC recommendation)
- Returns comprehensive validation result with all checkpoint data

### FFmpeg Sync Pattern
- Modern approach: `-af aresample=async=1` instead of deprecated `-async 1`
- Async correction stretches/compresses audio to match video clock
- Prevents progressive drift from clock mismatch over long durations
- Explicit stream mapping: `ffmpeg.output(video_input.video, audio_stream, ...)`
- MP4 optimization: `movflags='faststart'` for streaming playback

### Architecture Integration
- Drift detector uses TimedSegment from timestamp_validator
- Video merger validates compatibility before merge (ffprobe checks)
- All components use float64 precision from Phase 6 Plan 1
- Sample rate validation ensures 48kHz throughout (ASSEMBLY_TARGET_SAMPLE_RATE)

## Testing & Verification

All verification criteria passed:

1. **Drift detector test:** Created segments totaling 10 minutes, validated at 5-minute intervals
   - Result: Synced=True, max_drift=0.00ms for perfectly aligned segments
   - Checkpoints: 300s and 600s, both with 0ms drift

2. **Video merger inspection:** Verified aresample filter present in source code
   - Pattern: `.filter('aresample', **{'async': 1})` found
   - Uses dict unpacking to pass 'async' parameter (Python keyword workaround)

3. **Module exports test:** Successfully imported all 15 public symbols
   - TimedSegment, SyncCheckpoint, MergeResult all importable
   - No import errors

## Decisions Made

| ID | Title | Rationale | Impact |
|----|-------|-----------|--------|
| dict-async-workaround | Use `**{'async': 1}` for aresample parameter | 'async' is Python keyword, can't use as named argument | Enables FFmpeg async parameter without syntax error |
| checkpoint-actual-duration | Calculate actual duration by summing segments up to checkpoint | Original algorithm only accumulated completed segments | Accurate drift measurement at any checkpoint timestamp |
| format-specific-codecs | Provide different codec configs per output format | MP4 needs AAC, MKV supports copy, AVI needs MP3 | Ensures compatibility and optimal quality per format |
| compatibility-validation | Validate inputs with ffprobe before merge | Prevents cryptic FFmpeg errors during merge | User-friendly error messages for missing streams |

## Deviations from Plan

**1. [Rule 1 - Bug] Fixed cumulative duration calculation in drift detector**
- **Found during:** Task 1 verification testing
- **Issue:** Initial algorithm only accumulated segments that ended before checkpoint, missing partial segments that span the checkpoint
- **Fix:** Changed to iterate all segments and sum durations up to checkpoint timestamp, handling partial segments
- **Files modified:** src/assembly/drift_detector.py
- **Verification:** Test with 2 segments (0-300s, 300-600s) now shows 0ms drift at both checkpoints
- **Commit:** cfe9578 (part of Task 1 commit)

**2. [Rule 1 - Bug] Fixed Python keyword conflict with 'async' parameter**
- **Found during:** Task 2 verification testing
- **Issue:** `audio_stream.filter('aresample', async=1)` caused SyntaxError - 'async' is reserved keyword
- **Fix:** Used dict unpacking pattern: `.filter('aresample', **{'async': 1})`
- **Files modified:** src/assembly/video_merger.py
- **Verification:** Import and source inspection both pass
- **Commit:** a3e41b5 (part of Task 2 commit)

---

**Total deviations:** 2 auto-fixed bugs (both Rule 1)
**Impact on plan:** Both fixes were necessary for correct operation, no scope creep.

## Next Phase Readiness

**Phase 6 Plan 3 Prerequisites:**
- ✅ Drift detection validates sync at checkpoints
- ✅ Video merger provides FFmpeg merge with drift correction
- ✅ Assembly module exports complete API
- ✅ All components use float64 precision
- ✅ Sample rate validation ensures 48kHz consistency

**Potential concerns:**
- None identified. Infrastructure ready for assembly stage orchestration (Plan 3).

**Dependencies for downstream work:**
- Plan 3 will integrate these components into assembly_stage.py orchestrator
- Pattern will mirror ASR/Translation/TTS stage patterns from Phases 3-5

## Dependency Graph

### Requires (builds on)
- 06-01: Assembly infrastructure (TimedSegment, float64 validation, 48kHz normalization)
- 02-02: Video processing (ffmpeg-python patterns, format detection)

### Provides (delivers)
- Drift detection at checkpoint intervals for long videos
- FFmpeg merge with modern sync flags
- Complete assembly module API

### Affects (impacts)
- 06-03: Assembly stage orchestration (will use drift detector and video merger)
- 07-*: Lip sync phases (depend on drift-free audio-video alignment)

## Tech Stack Updates

### Added
- None (uses existing ffmpeg-python, numpy, logging)

### Patterns Established
- Checkpoint-based validation for progressive drift detection
- Dict unpacking for Python keyword conflicts in ffmpeg-python
- FFmpeg aresample filter for clock drift correction (modern pattern)
- Format-specific codec configuration via lookup table

## Performance Notes

- **Plan execution:** 6 minutes (Task 1: ~3m, Task 2: ~2m, Task 3: ~1m)
- **Drift validation overhead:** Negligible - O(segments × checkpoints) calculation
- **FFmpeg merge:** Stream copy for video = fast, no re-encoding overhead
- **Sample rate validation:** Uses soundfile.info() - instant metadata read

## Lessons Learned

1. **FFmpeg keyword conflicts:** ffmpeg-python uses kwargs that can collide with Python reserved words ('async'). Use dict unpacking as workaround.

2. **Segment accumulation logic:** Checkpoint validation requires summing all segment durations up to checkpoint, not just completed segments. Partial segments must be handled.

3. **Float64 precision propagation:** Explicit np.float64() conversions prevent drift from implicit float32 downcasting in calculations.

## Next Steps

Ready for **06-03-PLAN.md - Assembly Stage Orchestration**
- Integrate drift detector into assembly pipeline
- Integrate video merger as final stage
- Create assembly_stage.py orchestrator
- Add progress callbacks for UI integration
- Test complete assembly flow with real audio segments
