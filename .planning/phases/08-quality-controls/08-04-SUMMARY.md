---
phase: 08-quality-controls
plan: 04
subsystem: ui
tags: [gradio, event-handlers, integration-tests, wiring, streaming, cancellation, queue]

# Dependency graph
requires:
  - phase: 08-01
    provides: "Gradio Blocks scaffold with upload_col, review_col, processing_col, output_col components"
  - phase: 08-02
    provides: "validators.py with validate_asr/translation/tts/lip_sync_output; clip_preview.py with extract_preview_clip"
  - phase: 08-03
    provides: "pipeline_runner.py: run_asr_ui, run_full_pipeline, cancel_pipeline, _cancel_event generators"
provides:
  - "Fully wired src/ui/app.py: 7 event handlers connecting all buttons to pipeline_runner and clip_preview"
  - "demo.queue(default_concurrency_limit=1) for single-GPU RTX 5090 stability"
  - "22 integration tests covering app structure, generator validation, cancel behavior, JSON serialization"
  - "Human-verified: UI launches at localhost:7860, error handling works without crash"
affects: ["09-packaging", "phase-09", "final-integration"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gradio queue at app level: demo.queue(default_concurrency_limit=1) caps streaming generators for RTX 5090"
    - "Event handle store: asr_event and pipeline_event stored as variables so cancel_btn can reference via cancels=[pipeline_event]"
    - "Wrapper functions for Gradio compatibility: show_video_info, preview_clip, go_back, restart wrap lower-level APIs"

key-files:
  created:
    - tests/test_ui_integration.py
  modified:
    - src/ui/app.py

key-decisions:
  - "demo.queue(default_concurrency_limit=1) added inside create_app() to limit concurrent generator execution to 1 (RTX 5090 single-GPU stability)"
  - "cancel_btn.click uses cancels=[pipeline_event] Gradio mechanism alongside _cancel_event.set() for dual-path cancellation"
  - "22 integration tests rather than 6 minimum: tests cover 6-tuple yield structure, None token, confidence flagging, and duration computation edge cases"

patterns-established:
  - "Event variable retention: store .click() return values (asr_event, pipeline_event) to enable cancels=[...] reference"
  - "Wrapper-function isolation: UI-facing wrapper functions (show_video_info, preview_clip, go_back, restart) keep event handler signatures clean"

# Metrics
duration: ~2min
completed: 2026-02-22
---

# Phase 8 Plan 04: Event Handler Wiring Summary

**All 7 Gradio event handlers wired in app.py connecting UI scaffold to pipeline_runner generators, with 22 integration tests and human-verified browser UI at localhost:7860**

## Performance

- **Duration:** ~2 min (commits span 2026-02-21T06:03:42Z to 2026-02-21T06:05:05Z)
- **Started:** 2026-02-21T06:03:42Z
- **Completed:** 2026-02-22 (human checkpoint approved)
- **Tasks:** 2 auto + 1 checkpoint (human-verify)
- **Files modified:** 2

## Accomplishments

- `video_input.change` -> `show_video_info`: displays resolution, duration, format, audio presence on upload
- `start_asr_btn.click` -> `run_asr_ui`: streaming generator yielding 6-tuples (upload_col, review_col, transcript_df, state, status_text, asr_status_visible)
- `preview_btn.click` -> `preview_clip`: wraps `extract_preview_clip` with state-driven video path lookup
- `proceed_btn.click` -> `run_full_pipeline`: streaming generator with 6 UI outputs including stage status and progress
- `cancel_btn.click` -> `cancel_pipeline` with `cancels=[pipeline_event]` for dual Gradio + threading.Event cancellation
- `back_to_upload_btn.click` -> `go_back`: returns to Step 1 preserving video in state
- `process_another_btn.click` -> `restart`: returns to Step 1 clearing all state
- `demo.queue(default_concurrency_limit=1)` limits concurrent streaming generators to 1 for RTX 5090 stability
- 22 integration tests covering app structure, generator validation, cancel behavior, JSON serialization, validator importability

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire event handlers** - `3c9f195` (feat)
2. **Task 2: Write integration tests** - `9e87147` (test)
3. **Task 3: Human UI verification** - checkpoint approved (no commit)

**Plan metadata:** committed below (docs commit)

## Files Created/Modified

- `src/ui/app.py` (+142 lines) — 7 event handlers, 4 wrapper functions, demo.queue() call
- `tests/test_ui_integration.py` (314 lines, 22 tests) — integration tests for wiring and pipeline runner behavior

## Decisions Made

**demo.queue at app level** — `demo.queue(default_concurrency_limit=1)` called inside `create_app()` before `return demo`. Caps concurrent generator executions to 1 so the RTX 5090 never runs two ML pipelines simultaneously.

**cancels=[pipeline_event]** — `proceed_btn.click()` return value stored as `pipeline_event` so the cancel button can use Gradio's native `cancels=[pipeline_event]` mechanism in addition to setting `_cancel_event`. This provides two cancellation paths: Gradio interrupt (immediate) and stage-boundary check (graceful).

**22 tests not 6 minimum** — Extended test suite covers: 6-tuple yield structure validation, `None` token rejection (separate from empty string), confidence flagging threshold behavior, duration computation from Start/End columns, and `detected_language="und"` serialization.

## Deviations from Plan

None - plan executed exactly as written. Event handler signatures matched component variable names from 08-01. No API deviations encountered (those were resolved in 08-03).

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `src/ui/app.py` fully wired: all buttons connected, cancellation functional, error handling in place
- `tests/test_ui_integration.py` provides 22 passing tests (no GPU required)
- Phase 8 complete: 4/4 plans done
- Human-verified: UI launches at localhost:7860, Step 1 visible, error messages shown on bad input without crash
- Ready for Phase 9 (packaging/deployment) or end-to-end integration testing

---
*Phase: 08-quality-controls*
*Completed: 2026-02-22*
