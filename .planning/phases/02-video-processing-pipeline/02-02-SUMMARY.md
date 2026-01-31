---
phase: 02
plan: 02
subsystem: video-processing-core
tags: [ffmpeg, audio-extraction, video-extraction, stream-merging, pipeline-orchestration]
dependencies:
  requires:
    - "02-01: FFmpeg foundation and video probing utilities"
    - "01-02: Project structure and temporary file management"
  provides:
    - "Audio extraction to WAV format (48kHz, 16-bit PCM, stereo)"
    - "Video stream extraction with stream copy (no re-encoding)"
    - "Audio-video stream merging with format-appropriate codecs"
    - "End-to-end pipeline orchestration with automatic cleanup"
  affects:
    - "02-03: Gradio interface will use process_video() for backend processing"
    - "Phase 3+: ML models will be inserted between extraction and merging"
tech-stack:
  added: []
  patterns:
    - "Stream copy extraction (vcodec='copy') for lossless video preservation"
    - "Format-specific codec selection (AAC for MP4, copy for MKV)"
    - "Pipeline orchestration with progress callbacks"
    - "TempFileManager context manager for automatic cleanup"
key-files:
  created:
    - src/video_processing/extractor.py: "Audio/video stream extraction from source video"
    - src/video_processing/merger.py: "Stream recombination into output video"
    - src/video_processing/pipeline.py: "End-to-end video processing orchestration"
  modified: []
decisions:
  - id: stream-copy-extraction
    title: Use stream copy for video extraction (no re-encoding)
    rationale: "Preserves exact video quality and provides 10-100x speedup vs re-encoding"
    impact: "Video stream never degraded during processing, only audio is modified"
  - id: format-specific-codecs
    title: Select audio codec based on output container format
    rationale: "MP4 requires AAC, MKV supports any codec (use copy), AVI needs MP3"
    impact: "Optimal codec automatically selected, prevents 'codec not supported' errors"
  - id: progress-callback-pattern
    title: Use progress callback for UI integration
    rationale: "Enables Gradio progress bars without tight coupling to UI framework"
    impact: "Pipeline can be used from CLI, tests, or GUI with same interface"
metrics:
  duration: "3 minutes"
  completed: "2026-01-31"
---

# Phase 2 Plan 2: Video Processing Core Summary

**One-liner:** Audio/video extraction with stream copy preservation, format-aware stream merging, and pipeline orchestration with automatic temp file cleanup and progress tracking.

## What Was Built

This plan implemented the core video processing modules that extract, preserve, and recombine video streams:

1. **Extractor Module**: Extracts audio to WAV (48kHz, 16-bit PCM) and video stream with no re-encoding
2. **Merger Module**: Recombines video and audio with format-appropriate codecs (AAC for MP4, copy for MKV)
3. **Pipeline Orchestrator**: End-to-end workflow with extraction, passthrough, merge, and automatic cleanup

### Key Components

**src/video_processing/extractor.py**
- `extract_audio()`: Extracts audio stream to WAV format (PCM 16-bit signed little-endian, 48kHz, stereo)
- `extract_video_stream()`: Extracts video stream without audio using stream copy (no re-encoding)
- `extract_streams()`: Convenience function that extracts both audio and video to temp directory
- `ExtractionResult`: Dataclass containing extracted paths and metadata

**src/video_processing/merger.py**
- `merge_audio_video()`: Merges video stream with new/processed audio stream
- `MergeConfig`: Dataclass for video/audio codec configuration
- `get_optimal_merge_config()`: Returns optimal codec config based on container formats
- `validate_merge_inputs()`: Validates video and audio files before merge

**src/video_processing/pipeline.py**
- `process_video()`: Main orchestration function (validate -> extract -> passthrough -> merge -> cleanup)
- `ProcessingResult`: Dataclass with output path, metadata, and processing metrics
- `validate_processing_environment()`: Checks FFmpeg availability and temp directory writability

## Verification Results

✓ All three modules import without errors
✓ No circular import errors
✓ Type hints present on all public functions
✓ Docstrings explain parameters and return values
✓ Extractor uses stream copy for video (no quality loss)
✓ Merger selects appropriate codecs per container format (AAC for MP4, copy for MKV)
✓ Pipeline uses TempFileManager for automatic cleanup
✓ Progress callback pattern enables UI integration

## Testing Performed

1. **Import Verification**: Confirmed all modules import successfully with expected exports
2. **Type Safety**: Verified type hints and dataclass definitions compile correctly
3. **Documentation**: Checked docstrings document parameters, returns, and exceptions

Full integration testing with actual video files will occur in Plan 02-03.

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

| ID | Decision | Rationale | Impact |
|----|----------|-----------|--------|
| stream-copy-extraction | Use stream copy for video extraction (no re-encoding) | Preserves exact video quality and provides 10-100x speedup vs re-encoding | Video stream never degraded during processing, only audio is modified |
| format-specific-codecs | Select audio codec based on output container format | MP4 requires AAC, MKV supports any codec (use copy), AVI needs MP3 for compatibility | Optimal codec automatically selected via get_optimal_merge_config(), prevents 'codec not supported in container' errors |
| progress-callback-pattern | Use progress callback for UI integration | Enables Gradio progress bars without tight coupling to UI framework | Pipeline can be used from CLI, tests, or GUI with same Callable interface |

## Commits

| Commit | Type | Description | Files |
|--------|------|-------------|-------|
| 7d93a2b | feat | Create audio/video extractor module | src/video_processing/extractor.py |
| 5a297a0 | feat | Create audio-video merger module | src/video_processing/merger.py |
| 620c683 | feat | Create pipeline orchestrator | src/video_processing/pipeline.py |

## Next Phase Readiness

**Ready for Plan 02-03**: Gradio interface and integration testing

This plan delivered all required video processing infrastructure:
- Audio can be extracted to WAV format (48kHz, 16-bit PCM, stereo)
- Video stream preserved with stream copy (no re-encoding, no quality loss)
- Output format matches input format (MP4 -> MP4, MKV -> MKV)
- Codecs selected automatically based on container format
- Pipeline orchestrates full flow with automatic cleanup
- Progress callback ready for Gradio integration

**Blockers/Concerns**: None

**Recommendations for Next Plan**:
1. Create Gradio interface that uses `process_video()` as backend
2. Test with actual video files (MP4, MKV, AVI) to verify stream extraction/merging
3. Verify temp file cleanup works correctly (check temp directory after processing)
4. Test progress callback integration with Gradio's `gr.Progress()`
5. Handle edge cases: videos without audio streams, variable frame rate videos

## Notes

- **Stream copy efficiency**: Using `vcodec='copy'` avoids re-encoding, providing 10-100x speedup compared to re-encoding
- **Format-aware codec selection**: `get_optimal_merge_config()` prevents common pitfall of codec/container mismatch
- **Round-trip validation**: Pipeline currently does extract -> passthrough -> merge to validate toolchain. Future phases will insert ML processing between extraction and merging:
  - Phase 3: Speech recognition (Whisper)
  - Phase 4: Translation (SeamlessM4T)
  - Phase 5: Voice cloning (XTTS-v2)
  - Phase 6: Lip sync (Wav2Lip/LatentSync)
- **Error handling**: All functions validate inputs and provide descriptive error messages by parsing FFmpeg stderr
- **Memory efficiency**: No video loading into memory - FFmpeg handles streaming directly from disk
