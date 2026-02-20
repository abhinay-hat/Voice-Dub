---
phase: 07-lip-synchronization
plan: 02
subsystem: lip-sync
tags: [wav2lip, ffmpeg, video-chunking, subprocess, lip-sync, fallback]

# Dependency graph
requires:
  - phase: 06-audio-video-assembly
    provides: FFmpeg concat demuxer pattern (reused in concatenate_video_chunks)
provides:
  - Wav2Lip GAN fallback inference subprocess wrapper (wav2lip_runner.py)
  - 5-minute video chunking and reassembly for long-video processing (chunker.py)
  - Safety guard preventing s3fd.pth/wav2lip_gan.pth confusion
affects: [07-lip-synchronization/07-03-PLAN.md, stage orchestration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "subprocess wrapper pattern for external ML repo inference scripts"
    - "FFmpeg accurate seek with re-encode for clean chunk keyframe alignment"
    - "FFmpeg concat demuxer with -c copy for lossless reassembly"

key-files:
  created:
    - src/lip_sync/wav2lip_runner.py
    - src/lip_sync/chunker.py
    - src/lip_sync/__init__.py
  modified: []

key-decisions:
  - "wav2lip_gan.pth safety guard: ValueError if s3fd.pth passed as checkpoint_path"
  - "accurate-seek-reencode: -ss -t after -i (not before) for video chunk splitting to prevent GOP duration inflation"
  - "5-minute default chunk duration: prevents VRAM spikes in face detection for both LatentSync and Wav2Lip"

patterns-established:
  - "Subprocess runner pattern: validate inputs -> build cmd list -> subprocess.run(cwd=repo) -> check returncode"
  - "Chunker pattern: split_video_into_chunks -> [process each chunk independently] -> concatenate_video_chunks"

# Metrics
duration: 3min
completed: 2026-02-20
---

# Phase 7 Plan 02: Wav2Lip Fallback Runner and Video Chunker Summary

**Wav2Lip GAN subprocess wrapper with s3fd.pth safety guard, and 5-minute FFmpeg video chunker using accurate seek re-encode for clean concatenation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-20T19:59:07Z
- **Completed:** 2026-02-20T20:02:41Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Wav2Lip GAN inference wrapper that invokes `models/Wav2Lip/inference.py` via subprocess with `wav2lip_gan.pth` checkpoint
- Safety guard that raises `ValueError` immediately if `s3fd.pth` (face detector) is passed as checkpoint - prevents the `KeyError: 'state_dict'` crash
- Video chunker using accurate FFmpeg seek (`-ss -t` after `-i`) with libx264 re-encode for clean GOP boundaries
- `concatenate_video_chunks` using FFmpeg concat demuxer with `-c copy` for lossless reassembly
- `VideoChunk` dataclass tracking index, start/end seconds, and chunk file paths
- `src/lip_sync/__init__.py` exporting all public symbols for Plan 07-03 stage orchestration

## Task Commits

Each task was committed atomically:

1. **Task 1: Create wav2lip_runner.py with checkpoint path safety guard** - `508160c` (feat)
2. **Task 2: Create chunker.py for 5-minute video splitting and reassembly** - `40ef9e3` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/lip_sync/wav2lip_runner.py` - Wav2Lip GAN subprocess wrapper with s3fd.pth guard, FileNotFoundError for missing repo/checkpoint
- `src/lip_sync/chunker.py` - VideoChunk dataclass, split_video_into_chunks (accurate seek), concatenate_video_chunks (concat demuxer), get_video_duration (ffprobe)
- `src/lip_sync/__init__.py` - Module init exporting wav2lip_runner and chunker public API

## Decisions Made
- **wav2lip_gan.pth over wav2lip.pth**: GAN version has better visual quality, more appropriate for fallback scenarios
- **Accurate seek for chunking**: `-ss -t` placed after `-i` forces re-encode from exact frame, preventing GOP misalignment that inflated concat duration from 12s to 17.3s with stream copy
- **5-minute default chunk duration**: Both LatentSync and Wav2Lip work reliably on 5-minute segments; prevents face detection memory spikes on long videos
- **No `--nosmooth` flag**: Wav2Lip temporal smoothing improves visual quality and should not be disabled for fallback use

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed video chunk duration inflation with FFmpeg stream copy**
- **Found during:** Task 2 (chunker.py verification)
- **Issue:** Using `-ss` before `-i` with `-c copy` caused GOP misalignment - a 12-second video split into 3 chunks and re-concatenated produced a 17.3-second output instead of 12.0 seconds. This would cause lip sync drift across chunk boundaries.
- **Fix:** Changed chunk extraction from `-ss [start] -i [video] -t [dur] -c copy` to `-i [video] -ss [start] -t [dur] -c:v libx264 -c:a pcm_s16le`. Accurate seek with re-encode creates clean keyframe boundaries.
- **Files modified:** `src/lip_sync/chunker.py`
- **Verification:** 12-second synthetic video now splits into 3 chunks and concatenates back to exactly 12.0 seconds
- **Committed in:** `40ef9e3` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix essential for correct lip sync - duration inflation across chunk boundaries would cause A/V drift. No scope creep.

## Issues Encountered
- FFmpeg stream copy (`-c copy`) with fast seek (`-ss` before `-i`) inflates chunk duration due to GOP keyframe misalignment. The lavfi synthetic test video exposed this clearly: 3 chunks of 5s/5s/2s concatenated to 17.3s instead of 12s. Accurate seek with re-encode resolves this entirely.

## User Setup Required
None - no external service configuration required.

The Wav2Lip runner references `models/Wav2Lip/` which requires:
1. `git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip`
2. Download `wav2lip_gan.pth` from Wav2Lip GitHub README (Google Drive link)

These are documented in the runner's module docstring and error messages. No automated setup needed from this plan.

## Next Phase Readiness
- Plan 07-03 (stage orchestration) can now import from `src.lip_sync.wav2lip_runner` and `src.lip_sync.chunker`
- Plan 07-01 (LatentSync runner) executes in parallel; when complete, its exports should be added to `src/lip_sync/__init__.py`
- Both `run_wav2lip_inference` and `split_video_into_chunks` / `concatenate_video_chunks` have the exact signatures Plan 07-03 needs

---
*Phase: 07-lip-synchronization*
*Completed: 2026-02-20*
