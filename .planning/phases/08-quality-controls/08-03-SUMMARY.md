---
phase: 08-quality-controls
plan: 03
subsystem: ui
tags: [gradio, generator, pipeline-runner, cancellation, progress, gradio-update, threading]

# Dependency graph
requires:
  - phase: 08-01
    provides: "Gradio Blocks app scaffold with upload_section, review_section, processing_section, output_section columns"
  - phase: 08-02
    provides: "validate_asr_output, validate_translation_output, validate_tts_output, validate_lip_sync_output in src/ui/validators.py"
  - phase: 03-03
    provides: "run_asr_stage(audio_path, video_id, huggingface_token, progress_callback, save_json) -> ASRResult"
  - phase: 04-04
    provides: "run_translation_stage(asr_json_path, output_json_path) -> TranslationResult"
  - phase: 05-04
    provides: "run_tts_stage(translation_json_path, audio_path) -> TTSResult with output_dir/tts_result.json"
  - phase: 06-03
    provides: "run_assembly_stage(video_path, tts_result_path, output_path) -> AssemblyResult"
  - phase: 07-03
    provides: "run_lip_sync_stage(assembled_video_path, output_dir) -> LipSyncResult"
provides:
  - "run_asr_ui() generator: video validation, audio extraction, ASR, transcript DataFrame"
  - "run_full_pipeline() generator: Translation -> TTS -> Assembly -> Lip Sync with progress and cancellation"
  - "cancel_pipeline(): sets threading.Event to cancel between stages"
  - "_cancel_event: module-level threading.Event for cancellation state"
  - "_write_edited_transcript(): DataFrame -> ASR JSON serializer for translation stage input"
affects: ["08-04", "plan-08-event-wiring"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy imports inside generators: stage modules imported inside functions, not at module level"
    - "Generator-based Gradio streaming: each yield updates UI state in real time"
    - "Module-level threading.Event for cross-generator cancellation"
    - "TYPE_CHECKING imports for type hints without runtime ML library loading"

key-files:
  created:
    - src/ui/pipeline_runner.py
  modified: []

key-decisions:
  - "Lazy imports inside generator bodies: all stage modules (asr_stage, translation_stage, tts_stage, assembly_stage, lip_sync_stage) imported inside run_asr_ui() and run_full_pipeline() — not at module level. Prevents cascading pyannote/TTS/InsightFace imports when pipeline_runner is loaded in test or UI context."
  - "TYPE_CHECKING pattern for stage result types: matches validators.py (08-02) convention for type hints without runtime imports"
  - "run_assembly_stage uses tts_result_path=Path(tts_result.output_dir)/tts_result.json: assembly stage reads segment paths from JSON, not from segments list in memory"
  - "run_lip_sync_stage uses assembled_video_path and output_dir: audio extracted internally from assembled video — no separate audio_path needed"
  - "run_translation_stage called with asr_json_path + output_json_path: no video_id param (reads from JSON); output_json_path must be explicit to populate result.output_path"
  - "_write_edited_transcript sets detected_language='und': original language not preserved in DataFrame; SeamlessM4T re-detects language from source text if needed"

patterns-established:
  - "Generator pattern: yield 6-tuples with gr.update(visible=...) to control column visibility from inside generator"
  - "Cancel-check pattern: if _cancel_event.is_set() before every stage with yield + return"
  - "Validation-gate pattern: call validate_*_output() after each stage, yield error and return on (False, msg)"
  - "CUDA cleanup in every finally: gc.collect() + torch.cuda.empty_cache() guards both success and error paths"

# Metrics
duration: 3min
completed: 2026-02-21
---

# Phase 8 Plan 03: Pipeline Runner Summary

**Generator-based Gradio pipeline runner connecting all 5 dubbing stages with real-time progress yields, per-stage validation gates, and threading.Event cancellation — lazy imports keep module load free of ML libraries**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-21T05:54:50Z
- **Completed:** 2026-02-21T05:57:43Z
- **Tasks:** 2 (plus research task 1 with no commit)
- **Files modified:** 1

## Accomplishments

- `run_asr_ui()` generator: validates video, extracts audio, runs ASR stage, returns transcript DataFrame with 6 UI update yields
- `run_full_pipeline()` generator: Translation -> TTS -> Assembly -> Lip Sync with 5 cancellation checkpoints and stage validation before each step
- `cancel_pipeline()` sets module-level `_cancel_event`, stages detect it at every boundary
- `_write_edited_transcript()` serializes DataFrame edits back to ASR JSON format for translation stage

## Task Commits

Each task was committed atomically:

1. **Task 1: Research (read-only)** - no commit
2. **Task 2: run_asr_ui generator** - `d20e2d7` (feat)
3. **Task 3: run_full_pipeline + cancel_pipeline** - `779803a` (feat — empty commit; both tasks implemented in the same file write)

**Plan metadata:** see docs commit below

## Files Created/Modified

- `src/ui/pipeline_runner.py` (510 lines) — Generator functions, cancel_pipeline(), _cancel_event, _write_edited_transcript()

## Decisions Made

**lazy-stage-imports** — Stage modules (asr_stage, translation_stage, tts_stage, assembly_stage, lip_sync_stage) are imported lazily inside the generator functions, not at module level. This prevents cascading pyannote/TTS/InsightFace/assembly imports when pipeline_runner.py is loaded. Matches the TYPE_CHECKING pattern established in validators.py (08-02).

**assembly-tts-result-json** — `run_assembly_stage` is called with `tts_result_path` pointing to the `tts_result.json` file written by run_tts_stage (at `tts_result.output_dir/tts_result.json`). The assembly stage reads audio segment paths from this JSON — it does NOT accept a segments list in memory.

**lip-sync-assembled-video-path** — `run_lip_sync_stage` is called with `assembled_video_path` and `output_dir`. It extracts the 16kHz audio from the assembled video internally via `prepare_audio_for_lipsync`. No separate audio_path is needed.

**translation-explicit-output-path** — `run_translation_stage` has no `video_id` parameter; it reads video_id from the JSON. `output_json_path` must be passed explicitly to populate `result.output_path` for the next stage.

**detected-language-und** — `_write_edited_transcript` writes `"detected_language": "und"` because the original language is not preserved in the DataFrame. SeamlessM4T will detect language from the source text segments.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Lazy imports to prevent pyannote/TTS import failure at module load**

- **Found during:** Task 2 verification
- **Issue:** `from src.stages.asr_stage import run_asr_stage` at module level caused `ModuleNotFoundError: No module named 'pyannote'` because pyannote is not installed in the standard venv (requires separate setup per Phase 3 docs). Same issue would affect TTS, assembly, and lip sync stages.
- **Fix:** Moved all stage module imports inside generator function bodies (lazy import pattern). Added `TYPE_CHECKING` block for type hint-only imports — same pattern established in validators.py (08-02 decision `type-checking-imports`).
- **Files modified:** `src/ui/pipeline_runner.py`
- **Verification:** `python -c "from src.ui.pipeline_runner import run_asr_ui"` succeeds without pyannote installed
- **Committed in:** d20e2d7

**2. [Rule 1 - Bug] Corrected run_assembly_stage API (wrong in plan spec)**

- **Found during:** Task 1 (API research) / Task 3 implementation
- **Issue:** Plan spec showed `run_assembly_stage(video_path, tts_output_dir, video_id, segments)`. Actual API is `run_assembly_stage(video_path: Path, tts_result_path: Path, output_path: Path, video_fps, progress_callback)` — reads from TTS result JSON, not from segments in memory.
- **Fix:** Used `tts_result_path=Path(tts_result.output_dir) / "tts_result.json"` and `output_path=TEMP_DATA_DIR / f"{video_id}_assembled.mp4"`.
- **Files modified:** `src/ui/pipeline_runner.py`
- **Committed in:** 779803a

**3. [Rule 1 - Bug] Corrected run_lip_sync_stage API (wrong in plan spec)**

- **Found during:** Task 1 (API research) / Task 3 implementation
- **Issue:** Plan spec showed `run_lip_sync_stage(video_path, audio_path, video_id)`. Actual API is `run_lip_sync_stage(assembled_video_path: Path, output_dir: Path, ...)` — audio extracted internally, no video_id parameter.
- **Fix:** Used `assembled_video_path=assembly_result.output_path` and `output_dir=TEMP_DATA_DIR / f"{video_id}_lipsync"`.
- **Files modified:** `src/ui/pipeline_runner.py`
- **Committed in:** 779803a

**4. [Rule 1 - Bug] Corrected run_translation_stage API (different param name)**

- **Found during:** Task 1 (API research) / Task 3 implementation
- **Issue:** Plan spec showed parameter `transcript_json_path`. Actual parameter is `asr_json_path`. Also no `video_id` parameter — read from JSON. `output_json_path` must be passed explicitly to populate `result.output_path`.
- **Fix:** Used `asr_json_path=asr_json_path, output_json_path=translation_json_path`.
- **Files modified:** `src/ui/pipeline_runner.py`
- **Committed in:** 779803a

**5. [Rule 1 - Bug] extract_audio not exported from src.video_processing (only in extractor submodule)**

- **Found during:** Task 1 (API research)
- **Issue:** Plan spec showed `from src.video_processing import extract_audio`. Actual `src/video_processing/__init__.py` does NOT export `extract_audio` — only `probe_video`, `get_video_info`, `detect_container_format`, `validate_video_file`, `process_video`, `ProcessingResult`.
- **Fix:** Used `from src.video_processing.extractor import extract_audio` (lazy import inside generator).
- **Files modified:** `src/ui/pipeline_runner.py`
- **Committed in:** d20e2d7

---

**Total deviations:** 5 auto-fixed (1 blocking — lazy imports; 4 incorrect API specs in plan)
**Impact on plan:** All auto-fixes necessary for correct operation. No scope creep. Stage logic and validation gates match plan spec exactly.

## Issues Encountered

None beyond the API deviations documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `src/ui/pipeline_runner.py` complete with all exports: `run_asr_ui`, `run_full_pipeline`, `cancel_pipeline`, `_cancel_event`
- Ready for 08-04: event wiring that connects `start_asr_btn.click()` to `run_asr_ui` and `proceed_btn.click()` to `run_full_pipeline`
- No blockers

---
*Phase: 08-quality-controls*
*Completed: 2026-02-21*
