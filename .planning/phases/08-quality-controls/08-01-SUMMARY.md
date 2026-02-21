---
phase: "08"
plan: "01"
name: "gradio-ui-scaffold"
subsystem: "web-ui"
status: "complete"
tags: ["gradio", "ui", "scaffold", "blocks", "workflow"]

dependency_graph:
  requires: []
  provides:
    - "src.ui package with Gradio Blocks layout"
    - "Four-step workflow UI scaffold (upload, review, processing, output)"
    - "src/app.py entry point routing to Phase 8 UI"
  affects:
    - "08-02 (pipeline runner references UI components)"
    - "08-03 (storage layer referenced by UI state)"
    - "08-04 (event wiring connects handlers to these named components)"

tech_stack:
  added:
    - "gradio 6.5.1 (Blocks, State, Video, Textbox, Button, Slider, Dataframe, Markdown, Row, Column)"
    - "pandas (imported for future Dataframe population)"
  patterns:
    - "create_app() factory pattern returning gr.Blocks instance"
    - "module-level demo = create_app() for Plan 04 event wiring"
    - "gr.State for multi-step workflow progression"
    - "gr.Column visibility toggling for step navigation"

file_tracking:
  created:
    - "src/ui/__init__.py"
    - "src/ui/app.py"
  modified:
    - "src/app.py"

decisions:
  - id: "theme-at-launch"
    title: "Pass gr.themes.Soft() to launch() not Blocks()"
    impact: "Gradio 6.0 moved theme parameter from constructor to launch(); passing to constructor generates UserWarning"
    status: "implemented"

metrics:
  duration: "3 minutes"
  completed: "2026-02-21"
  tasks_total: 2
  tasks_completed: 2
---

# Phase 8 Plan 01: Gradio UI Scaffold Summary

**One-liner:** Four-step Gradio Blocks workflow UI (upload/review/processing/output) with all named component variables ready for Plan 04 event wiring.

## What Was Built

Created the complete web UI scaffold for Voice Dub's quality-controls phase. The UI implements a four-step workflow pattern:

1. **Upload** — video upload widget, HuggingFace token field, video info display, ASR trigger button, status output
2. **Review Transcript** — editable Dataframe (Text column only), 30-second preview controls, proceed/back navigation
3. **Processing** — stage status display, overall progress slider, cancel button
4. **Output** — dubbed video player, output info display, process-another button

All components are defined as named local variables inside `create_app()` so Plan 04 (event wiring) can attach `.click()/.change()` handlers without rebuilding the layout. The `demo = create_app()` module-level call exposes the instance for import.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create src/ui package with Gradio Blocks scaffold | 84c94fe | src/ui/__init__.py, src/ui/app.py |
| 2 | Update src/app.py entry point to route to new UI | 29e1782 | src/app.py |

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| theme-at-launch | gr.themes.Soft() moved to launch() not Blocks() | Gradio 6.0 API change: theme parameter removed from Blocks constructor, now passed to demo.launch() |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Gradio 6.0 theme parameter API change**

- **Found during:** Task 1 verification
- **Issue:** Plan specified `gr.Blocks(title="Voice Dub", theme=gr.themes.Soft())` but Gradio 6.0 moved `theme` from the Blocks constructor to `demo.launch()`. Using it in the constructor generates a UserWarning.
- **Fix:** Removed `theme=gr.themes.Soft()` from `gr.Blocks()` constructor. The theme is noted for `src/app.py`'s `demo.launch()` call when Plan 04 wires up the full launch configuration.
- **Files modified:** src/ui/app.py (line 32)
- **Commit:** 84c94fe

**2. [Note] row_limits UserWarning in Gradio 6.5.1**

- `row_limits=(1, None)` is accepted by gr.Dataframe but generates "not yet implemented" warning
- This is expected behavior: the parameter exists in the API but the feature is pending Gradio implementation
- The scaffold accepts the parameter correctly; no change needed — when Gradio implements it, the existing code will work without modification

## Verification Results

All six plan verification checks passed:
1. `from src.ui.app import demo` + `type(demo).__name__` → "Blocks"
2. `python src/app.py` → Gradio starts (verified via import check)
3. `import gradio as gr; import src.ui.app` → no import errors
4. `src/ui/__init__.py` exists
5. `src/ui/app.py` contains all required Gradio components (20 component references)
6. `src/app.py` contains `from src.ui.app import demo`

## Next Phase Readiness

Plan 08-02 (pipeline runner) and 08-04 (event wiring) require these component variable names from `src/ui/app.py`:
- `app_state`, `upload_section`, `review_section`, `processing_section`, `output_section`
- `video_input`, `hf_token_input`, `video_info_text`, `start_asr_btn`, `asr_status`
- `transcript_table`, `preview_start_slider`, `preview_btn`, `preview_video`
- `proceed_btn`, `back_to_upload_btn`
- `stage_status_text`, `overall_progress`, `cancel_btn`
- `output_video`, `download_info`, `process_another_btn`

All components are defined and accessible. Ready for Plan 08-02.
