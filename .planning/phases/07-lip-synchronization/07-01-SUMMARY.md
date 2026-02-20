---
phase: 07-lip-synchronization
plan: 01
subsystem: lip-sync
tags: [latentsync, conda, pytorch, subprocess-isolation, ffmpeg, audio-prep, whisper]

# Dependency graph
requires:
  - phase: 06-audio-video-assembly
    provides: 48kHz assembled audio that needs downsampling to 16kHz for lip sync models
provides:
  - LatentSync 1.6 conda environment isolated from main nightly PyTorch env
  - models/LatentSync cloned repo with stage2_512.yaml config and all checkpoints
  - src/lip_sync/audio_prep.py: FFmpeg-based 48kHz -> 16kHz mono WAV resampler
  - src/lip_sync/latentsync_runner.py: subprocess wrapper for LatentSync isolated inference
  - src/lip_sync/__init__.py: unified public API for entire lip_sync module
affects:
  - 07-02 (smoke test plan uses audio_prep and latentsync_runner directly)
  - 07-03 (stage orchestration imports run_latentsync_inference and prepare_audio_for_lipsync)
  - 07-04 (integration tests depend on module public API)

# Tech tracking
tech-stack:
  added:
    - miniconda3 (conda package manager, installed fresh)
    - conda env latentsync with Python 3.10.13
    - torch 2.5.1+cu121 (LatentSync's required version, isolated from main env)
    - LatentSync 1.6 (ByteDance diffusion-based lip sync model)
    - DeepCache 0.1.1 (2x inference speedup for LatentSync)
    - insightface 0.7.3 (face detection for LatentSync)
    - onnxruntime-gpu 1.21.0
    - mediapipe 0.10.11
    - diffusers 0.32.2
    - decord 0.6.0
  patterns:
    - Subprocess isolation pattern: incompatible PyTorch versions run in separate conda envs called via subprocess
    - LATENTSYNC_PYTHON_PATH env var override for portability across machines
    - FFmpeg-based audio resampling (not librosa) to match LatentSync's own preprocessing expectations

key-files:
  created:
    - src/lip_sync/audio_prep.py
    - src/lip_sync/latentsync_runner.py
  modified:
    - src/lip_sync/__init__.py

key-decisions:
  - "subprocess-isolation: LatentSync pins torch==2.5.1+cu121; run via conda subprocess to avoid contaminating RTX 5090 nightly env"
  - "ffmpeg-resampling-not-librosa: Use FFmpeg for 48->16kHz conversion to match LatentSync preprocessing expectations exactly"
  - "deepcache-enabled-by-default: enable_deepcache=True default since commit f5040cf is present in cloned repo"
  - "env-var-override: LATENTSYNC_PYTHON_PATH allows portability without code changes"
  - "20-steps-default: inference_steps=20 balances speed/quality for smoke testing; 40-50 for production"

patterns-established:
  - "Subprocess isolation: incompatible ML model dependencies run in separate conda envs via subprocess.run(), never imported directly"
  - "Pre-flight validation: runner validates all required files exist before invoking subprocess"
  - "Stderr tail logging: only last 2000 chars of stderr logged to prevent log flooding on failure"

# Metrics
duration: 18min
completed: 2026-02-20
---

# Phase 7 Plan 1: LatentSync Environment Setup Summary

**LatentSync 1.6 isolated in conda env (torch 2.5.1+cu121) with subprocess runner module, checkpoints downloaded, and FFmpeg-based 48kHz->16kHz audio prep confirmed working**

## Performance

- **Duration:** 18 minutes
- **Started:** 2026-02-20T19:57:29Z
- **Completed:** 2026-02-20T20:16:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Installed Miniconda3 fresh (not pre-existing on machine), accepted TOS, created `latentsync` conda env with Python 3.10.13
- Cloned LatentSync repo at commit a229c39 with DeepCache fix (f5040cf) in history; installed all 26 requirements including torch==2.5.1+cu121 and DeepCache==0.1.1
- Downloaded LatentSync 1.6 checkpoints: `latentsync_unet.pt` and `whisper/tiny.pt` via huggingface-cli
- Confirmed PyTorch isolation: latentsync=2.5.1+cu121 vs main env=2.11.0.dev20260130+cu128 (RTX 5090 nightly)
- Created `audio_prep.py` with FFmpeg resampler verified to produce 16kHz WAV from 48kHz input (ffprobe confirmed)
- Created `latentsync_runner.py` with subprocess wrapper, FileNotFoundError pre-flight validation, DeepCache flag support
- Updated `__init__.py` to expose unified public API: all six exports from audio_prep + latentsync_runner + wav2lip_runner + chunker

## Task Commits

Each task was committed atomically:

1. **Task 1 + Task 2: Create lip_sync module with LatentSync runner and audio prep** - `d193d44` (feat)

**Plan metadata:** (to be committed with docs commit)

## Files Created/Modified

- `src/lip_sync/audio_prep.py` - FFmpeg-based 48kHz->16kHz mono WAV resampler for lip sync models
- `src/lip_sync/latentsync_runner.py` - Subprocess wrapper for LatentSync 1.6 isolated conda inference
- `src/lip_sync/__init__.py` - Unified public API; updated to include LatentSync + audio_prep exports alongside pre-existing wav2lip_runner and chunker exports

## Decisions Made

- **subprocess-isolation**: LatentSync pins torch==2.5.1 (CUDA 12.1) which would destroy the RTX 5090's CUDA 12.8 nightly environment. Subprocess via conda env is the only safe integration path.
- **ffmpeg-resampling-not-librosa**: Use FFmpeg for 48->16kHz conversion. LatentSync's own preprocessing uses FFmpeg, so the output is format-identical. Librosa would add an unnecessary dependency with slightly different interpolation.
- **deepcache-enabled-by-default**: enable_deepcache=True because commit f5040cf is confirmed present in the cloned repo, providing ~2x inference speedup at negligible quality cost.
- **env-var-override**: `LATENTSYNC_PYTHON_PATH` environment variable makes the runner portable across machines without code changes. Default hardcoded to `C:/Users/ASBL/miniconda3/envs/latentsync/python.exe`.
- **20-steps-default**: inference_steps=20 as default for smoke testing speed; production runs can use 40-50 for higher quality.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Miniconda3 not installed on machine**
- **Found during:** Task 1 (conda environment creation)
- **Issue:** Plan assumed `C:/Users/ASBL/miniconda3` existed. The path did not exist - conda was not installed anywhere on the system.
- **Fix:** Downloaded Miniconda3-latest-Windows-x86_64.exe (94MB, already present at /c/tmp/), installed silently to `C:/Users/ASBL/miniconda3`, accepted TOS for all three default channels.
- **Files modified:** None (system-level installation, outside repo)
- **Verification:** `conda --version` returned `conda 25.11.1`; `conda env list` showed base env
- **Committed in:** d193d44 (documented in commit message)

**2. [Rule 1 - Bug] Pre-existing `__init__.py` had different API than plan specified**
- **Found during:** Task 2 (creating `__init__.py`)
- **Issue:** `__init__.py` already existed with imports from `wav2lip_runner` and `chunker` (both pre-written by a previous planning session). The plan spec for `__init__.py` only included LatentSync exports, which would have deleted the existing Wav2Lip API.
- **Fix:** Updated `__init__.py` to include ALL exports: audio_prep + latentsync_runner (new) + wav2lip_runner + chunker (pre-existing). Used relative imports (`from .module`) instead of absolute imports.
- **Files modified:** src/lip_sync/__init__.py
- **Verification:** `from src.lip_sync import prepare_audio_for_lipsync, run_latentsync_inference, LATENTSYNC_PYTHON, LATENTSYNC_REPO` returns `Imports OK`
- **Committed in:** d193d44

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. Conda install unblocked Task 1. __init__.py fix preserved existing wav2lip_runner/chunker API that Plans 07-02 through 07-04 depend on.

## Issues Encountered

- Miniconda not installed on the machine - downloaded and installed silently during Task 1 execution (see Deviations).
- The `__init__.py` had conflicting pre-existing content - merged rather than replaced (see Deviations).

## User Setup Required

None - no external service configuration required. The conda environment and checkpoints are set up programmatically.

**Note for users on other machines:** Set `LATENTSYNC_PYTHON_PATH` environment variable if miniconda3 is installed at a different path than `C:/Users/ASBL/miniconda3/envs/latentsync/python.exe`.

## Next Phase Readiness

- **Ready for Plan 07-02 (Smoke Test):** `run_latentsync_inference()` and `prepare_audio_for_lipsync()` are importable, conda env exists, checkpoints downloaded
- **Ready for Plan 07-03 (Stage Orchestration):** Public API established, LATENTSYNC_REPO path resolves correctly
- **Potential concern:** LatentSync 1.6 requires ~18GB VRAM. RTX 5090 has 32GB so this should be fine, but monitor during smoke test.
- **Potential concern:** InsightFace Windows installation sometimes fails. If smoke test fails on face detection, Wav2Lip fallback in wav2lip_runner.py is ready.

---
*Phase: 07-lip-synchronization*
*Completed: 2026-02-20*
