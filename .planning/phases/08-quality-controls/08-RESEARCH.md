# Phase 8: Quality Controls - Research

**Researched:** 2026-02-21
**Domain:** Gradio Web UI + Pipeline Quality Gates + Real-time Progress Tracking
**Confidence:** HIGH

## Summary

Phase 8 is the first time a web UI is built for this project. It introduces Gradio 6.6.0 as the UI layer wrapping the existing pipeline stages (Phases 1-7). The scope is significant: this is simultaneously the web UI construction AND the quality control layer. These two concerns are deeply intertwined — quality controls (transcript editing, clip preview, stage validation, cancel) are only meaningful when there is a UI to interact with.

The standard approach is a Gradio `Blocks` application with a multi-step workflow pattern: video upload -> ASR results + transcript editing -> confirm/preview clip -> full pipeline run with streaming progress -> output. Each step is implemented as a separate UI section with visibility toggled by gr.State. Pipeline cancellation uses generator-function yield patterns combined with a `threading.Event` stop flag. Stage validation happens between pipeline stages by yielding validation results before proceeding. Clip preview uses ffmpeg-python to extract 10-30 second segments.

A critical hardware-specific issue has been identified: Gradio's queue system (which uses a thread pool on the backend) can cause CUDA operations to hang or silently fall back to CPU on Windows with RTX 5090/Blackwell GPUs (GitHub issue #12492). The workaround is `queue=False` on GPU-heavy event handlers, or restructuring as generator functions which are handled differently. This must be addressed in every plan that touches GPU inference.

**Primary recommendation:** Use Gradio 6.6.0 Blocks with generator-based pipeline functions, threading.Event cancellation, gr.Dataframe for transcript editing, gr.Progress for stage progress, and `concurrency_limit=1` with `queue=False` on GPU-heavy event handlers.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 6.6.0 | Web UI framework | Only UI framework decided; Python-native, no JS knowledge required, fast iteration |
| ffmpeg-python | 0.2.0+ | Clip extraction for preview | Already in requirements.txt, zero new dependencies |
| threading (stdlib) | Python 3.10+ | Cancellation stop flag | Standard library, no overhead, works with Gradio generator pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.x | Transcript dataframe editing | gr.Dataframe requires pandas DataFrame as value |
| json (stdlib) | Python 3.10+ | Read/write stage output JSON | Pipeline stages already output structured JSON |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Gradio Blocks | gr.Interface | Interface is simpler but cannot do multi-step workflows or conditional visibility |
| gr.Dataframe | gr.Textbox (JSON) | Textbox is harder for users to edit structured data; Dataframe gives column-by-column editing |
| threading.Event | asyncio.Event | asyncio approach requires all Gradio handlers to be async def; threading.Event works with regular def functions and Gradio's threadpool |

**Installation:**
```bash
pip install gradio>=6.6.0 pandas>=2.0.0
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── ui/
│   ├── __init__.py
│   ├── app.py               # Main Gradio app: gr.Blocks definition, launch()
│   ├── pipeline_runner.py   # Generator function wrapping pipeline stages
│   └── validators.py        # Stage output validation functions
├── stages/                  # Existing: asr_stage.py, translation_stage.py, etc.
├── pipeline/
│   └── orchestrator.py      # Coordinates stage calls (if not existing)
└── config/
    └── settings.py          # Already exists
```

### Pattern 1: Multi-Step Workflow via Visibility Toggling
**What:** Show different UI sections based on pipeline progress. Each section is a gr.Column/gr.Row with visible=True/False controlled by returning new component instances from event handlers.
**When to use:** For every workflow transition (upload -> review -> processing -> output)
**Example:**
```python
# Source: https://www.gradio.app/guides/blocks-and-event-listeners
# Source: https://www.gradio.app/guides/controlling-layout

import gradio as gr

with gr.Blocks() as demo:
    # Section 1: Upload (shown first)
    with gr.Column(visible=True) as upload_section:
        video_input = gr.Video(label="Upload Video")
        extract_btn = gr.Button("Extract & Transcribe", variant="primary")

    # Section 2: Transcript Review (hidden until ASR completes)
    with gr.Column(visible=False) as review_section:
        transcript_df = gr.Dataframe(
            headers=["ID", "Speaker", "Start", "End", "Text", "Confidence"],
            datatype=["number", "str", "number", "number", "str", "number"],
            interactive=True,
            static_columns=[0, 1, 2, 3, 5],  # Only Text column is editable
            label="Edit Transcript (Text column is editable)"
        )
        preview_btn = gr.Button("Preview 30s Clip", variant="secondary")
        proceed_btn = gr.Button("Start Full Processing", variant="primary")

    # Section 3: Processing (hidden until user confirms)
    with gr.Column(visible=False) as processing_section:
        progress_status = gr.Textbox(label="Stage", interactive=False)
        cancel_btn = gr.Button("Cancel Processing", variant="stop")

    # Section 4: Output (hidden until complete)
    with gr.Column(visible=False) as output_section:
        output_video = gr.Video(label="Dubbed Video")

    def show_review_section(transcript_data):
        return (
            gr.Column(visible=False),   # hide upload
            gr.Column(visible=True),    # show review
            transcript_data
        )

    extract_btn.click(
        fn=run_asr,
        inputs=[video_input],
        outputs=[upload_section, review_section, transcript_df],
        queue=False  # CRITICAL: GPU operations on Windows RTX 5090
    )
```

### Pattern 2: Generator-Based Pipeline with gr.Progress
**What:** Pipeline stages run inside a generator function that yields progress updates. Gradio calls the generator repeatedly, updating UI between yields.
**When to use:** For the main pipeline run (transcription, translation, TTS, lip sync). This is the correct way to get real-time progress in Gradio.
**Example:**
```python
# Source: https://www.gradio.app/guides/progress-bars
# Source: https://www.gradio.app/guides/streaming-outputs

import gradio as gr
import threading

_cancel_event = threading.Event()

def run_pipeline(video_path, transcript_df, progress=gr.Progress()):
    """Generator-based pipeline runner with progress tracking."""
    global _cancel_event
    _cancel_event.clear()  # Reset cancel flag

    progress(0, desc="Starting pipeline...")
    yield gr.Textbox(value="Initializing..."), gr.Video(value=None)

    # Stage 1: Translation
    if _cancel_event.is_set():
        yield gr.Textbox(value="Cancelled"), gr.Video(value=None)
        return
    progress(0.2, desc="Translating...")
    yield gr.Textbox(value="Stage: Translating (20%)"), gr.Video(value=None)
    translation_result = run_translation(transcript_df)

    # Stage validation after translation
    if not validate_translation(translation_result):
        yield gr.Textbox(value="Translation validation failed - check output"), gr.Video(value=None)
        return

    # Stage 2: Voice Cloning
    if _cancel_event.is_set():
        yield gr.Textbox(value="Cancelled"), gr.Video(value=None)
        return
    progress(0.4, desc="Cloning voice...")
    yield gr.Textbox(value="Stage: Voice Cloning (40%)"), gr.Video(value=None)
    tts_result = run_tts(translation_result)

    # ... continue for each stage
    progress(1.0, desc="Complete!")
    yield gr.Textbox(value="Complete!"), gr.Video(value=output_path)

def cancel_pipeline():
    global _cancel_event
    _cancel_event.set()
    return gr.Textbox(value="Cancellation requested...")
```

**IMPORTANT - Per-user cancel:** For a local single-user app, a global threading.Event is acceptable. For multi-user scenarios, use gr.State to hold a per-session cancel event.

### Pattern 3: Transcript Editing with gr.Dataframe
**What:** Load ASR output into gr.Dataframe with most columns static (read-only) and only the Text column editable. User edits text and the corrected dataframe is passed to the next stage.
**When to use:** QC-01 — user edits transcript before dubbing.
**Example:**
```python
# Source: https://www.gradio.app/docs/gradio/dataframe
import pandas as pd
import gradio as gr

def load_transcript_to_dataframe(transcript_json: dict) -> pd.DataFrame:
    """Convert ASR output JSON to pandas DataFrame for gr.Dataframe."""
    rows = []
    for segment in transcript_json["segments"]:
        rows.append({
            "ID": segment["id"],
            "Speaker": segment["speaker"],
            "Start": round(segment["start"], 2),
            "End": round(segment["end"], 2),
            "Text": segment["text"],
            "Confidence": round(segment["confidence"], 3),
        })
    return pd.DataFrame(rows)

transcript_df = gr.Dataframe(
    headers=["ID", "Speaker", "Start", "End", "Text", "Confidence"],
    datatype=["number", "str", "number", "number", "str", "number"],
    interactive=True,
    static_columns=[0, 1, 2, 3, 5],  # ID, Speaker, Start, End, Confidence are read-only
    wrap=True,                         # Wrap long text in cells
    label="Transcript (edit Text column only)"
)

def get_edited_segments(df: pd.DataFrame) -> list[dict]:
    """Convert edited dataframe back to segment list for pipeline."""
    return [
        {
            "id": int(row["ID"]),
            "speaker": row["Speaker"],
            "start": float(row["Start"]),
            "end": float(row["End"]),
            "text": row["Text"],  # User-edited value
            "confidence": float(row["Confidence"]),
        }
        for _, row in df.iterrows()
    ]
```

### Pattern 4: Clip Preview with ffmpeg-python
**What:** Extract 10-30 second preview clip from original video at user-specified time, output as gr.Video.
**When to use:** QC-02 — user wants to preview a section before committing to full processing.
**Example:**
```python
# Source: https://www.clipcat.com/blog/a-beginners-guide-to-ffmpeg-python-how-to-trim-videos/
# Source: ffmpeg-python official docs (already in project)
import ffmpeg
from pathlib import Path
import tempfile

def extract_preview_clip(
    video_path: str,
    start_seconds: float = 0.0,
    duration_seconds: float = 30.0
) -> str:
    """
    Extract a preview clip from video. Returns path to temporary clip.

    Uses c='copy' for speed (keyframe-accurate). For frame-accurate cuts,
    remove c='copy' but expect 5-10x slower extraction.
    """
    output_path = str(Path(tempfile.mkdtemp()) / "preview.mp4")
    (
        ffmpeg
        .input(video_path, ss=start_seconds)
        .output(output_path, t=duration_seconds, c='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path
```

### Pattern 5: Stage Validation Before Proceeding
**What:** After each pipeline stage, validate the output before calling the next stage. Return validation errors to the UI if any stage fails.
**When to use:** QC-04 — system validates each stage output.
**Example:**
```python
# Source: Project's existing settings.py thresholds

def validate_asr_output(transcript: dict) -> tuple[bool, str]:
    """
    Validate ASR output before proceeding to translation.
    Returns (is_valid, error_message).
    """
    if not transcript.get("segments"):
        return False, "No speech detected in video"

    total_segments = len(transcript["segments"])
    flagged_count = transcript.get("flagged_count", 0)
    flagged_ratio = flagged_count / total_segments if total_segments > 0 else 0

    if flagged_ratio > 0.5:
        return False, f"Too many low-confidence segments ({flagged_count}/{total_segments}). Consider re-recording with less background noise."

    return True, ""

def validate_translation_output(translation: dict) -> tuple[bool, str]:
    """Validate translation duration fit before TTS."""
    segments = translation.get("segments", [])
    mismatched = [s for s in segments if abs(s.get("duration_ratio", 1.0) - 1.0) > 0.3]
    if len(mismatched) > len(segments) * 0.2:
        return False, f"{len(mismatched)} segments have severe duration mismatch (>30%). TTS may not sync properly."
    return True, ""
```

### Pattern 6: CUDA-Safe Gradio Event Handlers
**What:** Prevent CUDA hangs on Windows RTX 5090 by setting queue=False on GPU-heavy event handlers, or by restructuring as generator functions.
**When to use:** EVERY event handler that calls GPU inference (ASR, translation, TTS, lip sync).
**Example:**
```python
# Source: https://github.com/gradio-app/gradio/issues/12492
# WRONG - will hang on Windows RTX 5090 with Gradio's queue:
btn.click(fn=gpu_inference_fn, inputs=[...], outputs=[...])

# CORRECT - option 1: disable queue for GPU handler:
btn.click(fn=gpu_inference_fn, inputs=[...], outputs=[...], queue=False)

# CORRECT - option 2: use generator (Gradio handles generators differently):
def gpu_inference_generator():
    yield "Starting..."
    result = gpu_inference_fn()
    yield result

btn.click(fn=gpu_inference_generator, inputs=[...], outputs=[...])
```

### Anti-Patterns to Avoid
- **Global state for multi-user:** A global `_cancel_event` breaks concurrent users. For this single-user local app it is acceptable. Document the limitation.
- **queue=True on GPU handlers:** Causes CUDA hangs on Windows RTX 5090. Always use `queue=False` or generators.
- **Synchronous full pipeline:** Do not run the entire pipeline as a single blocking function. Users see nothing and assume it crashed. Use generators to yield intermediate results.
- **gr.Interface instead of gr.Blocks:** Interface cannot do conditional component visibility. Blocks is mandatory for this multi-step workflow.
- **Re-encoding preview clips unnecessarily:** Use `c='copy'` for preview clips to avoid 5-10x slowdown. Accept slight keyframe inaccuracy (usually <0.5 seconds).
- **Editable Confidence column:** Make confidence column static in the dataframe. Users should not edit confidence scores — only transcript text.
- **gr.update() syntax:** Gradio 6 uses direct component instantiation for updates (e.g., `return gr.Column(visible=True)`), not `gr.update()`. The old syntax may still work but is not idiomatic Gradio 6.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress display | Custom HTML/JS progress bar | gr.Progress() parameter | Built-in to Gradio, integrates with queue system, shows percentage + description |
| Transcript table editing | Custom HTML table editor | gr.Dataframe(interactive=True, static_columns=[...]) | Editing, sorting, validation built-in; handles pandas DataFrames natively |
| Real-time status updates | WebSocket custom server | Generator function with yield | Gradio's built-in streaming handles the transport layer |
| Cancel button | Custom signal handler | threading.Event + cancels=[event] | Gradio's cancels= stops the generator cleanly at the next yield point |
| Clip preview extraction | Frame-by-frame Python decoding | ffmpeg-python with ss+t parameters | Already in requirements.txt; single line of code; handles all codecs |
| Video display | HTML5 video player setup | gr.Video(value=path) | Handles codec compatibility, browser serving, temporary file cleanup |
| Session isolation | Database per user | gr.State() | Per-session state without persistence complexity; auto-cleans on tab close |

**Key insight:** Gradio handles the entire transport layer (WebSockets, HTTP, file serving, CORS) automatically. Every custom solution for these problems adds complexity with no benefit for a local single-user app.

## Common Pitfalls

### Pitfall 1: CUDA Hang with Gradio Queue on Windows RTX 5090
**What goes wrong:** GPU operations freeze indefinitely when Gradio's queue system runs the callback in a thread pool worker. The UI appears to hang with no error message.
**Why it happens:** Gradio's queue runs Python functions in a thread pool. On Windows with Blackwell architecture GPUs, CUDA context initialization in non-main threads causes deadlocks.
**How to avoid:** Set `queue=False` on every event handler that calls GPU inference. Or restructure as generator functions.
**Warning signs:** UI hangs after clicking a button that triggers GPU inference; no error in console; works fine when run as plain Python outside Gradio.
**Source:** [GitHub issue #12492](https://github.com/gradio-app/gradio/issues/12492) - confirmed on RTX 5090 Windows

### Pitfall 2: Cancel Does Not Stop Currently-Running Synchronous Functions
**What goes wrong:** User clicks cancel but pipeline continues running to completion.
**Why it happens:** Gradio's `cancels=` parameter only interrupts generator functions (between yield points) or queued-but-not-yet-started tasks. It cannot stop a running synchronous function.
**How to avoid:** Structure ALL long-running pipeline code as generators. Check `_cancel_event.is_set()` at each stage boundary. Only synchronous code that completes quickly (< 1 second) should be non-generator.
**Warning signs:** Cancel button does nothing; job continues after cancel is clicked.
**Source:** [GitHub issue #6724](https://github.com/gradio-app/gradio/issues/6724), verified Gradio docs

### Pitfall 3: Cancels + .then() Chain Breaks in Gradio 5.13.1
**What goes wrong:** After a cancel event fires, `.then()` chained events do not execute. UI gets stuck in an intermediate state.
**Why it happens:** Known bug in Gradio (issue #10432, #10251) where `cancels=` interferes with `.then()` event chains.
**How to avoid:** Avoid complex `.then()` chains when cancel is involved. Use a single generator function for the main pipeline rather than chained event handlers. Use `gr.State` to track UI step rather than chaining.
**Warning signs:** Cancel button fires but expected UI updates after cancel (like showing a "Cancelled" message) never appear.
**Source:** [GitHub issue #10432](https://github.com/gradio-app/gradio/issues/10432), [issue #10251](https://github.com/gradio-app/gradio/issues/10251)

### Pitfall 4: gr.Dataframe Row/Column Count API Changed in Gradio 6
**What goes wrong:** Code using `row_count=(5, "dynamic")` or `col_count=(6, "fixed")` tuple syntax raises errors or warnings.
**Why it happens:** Gradio 6 replaced these tuple parameters with separate `row_limits` and `column_limits` parameters.
**How to avoid:** Use the new Gradio 6 API: `row_limits=(None, None)` for dynamic rows, `column_limits=(6, 6)` for fixed columns.
**Warning signs:** Deprecation warnings or TypeError when creating gr.Dataframe with tuple row_count/col_count.
**Source:** [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide)

### Pitfall 5: Video Preview Files Filling Disk
**What goes wrong:** Each preview clip generates a temporary .mp4 file. After many previews, temp directory fills up.
**Why it happens:** ffmpeg-python writes to disk; Python temp files are not automatically cleaned up.
**How to avoid:** Write preview clips to `data/temp/` directory which is already managed by the project. OR use Python's `tempfile.TemporaryDirectory()` context manager and delete after gr.Video serves the file. For a local app, periodic manual cleanup is acceptable.
**Warning signs:** Disk space shrinking over time; many `preview_*.mp4` files accumulating.

### Pitfall 6: Gradio 6 Blocks.queue() Parameters Changed
**What goes wrong:** Code using `demo.queue(concurrency_count=...)` fails because `concurrency_count` was removed.
**Why it happens:** Gradio 4+ moved concurrency control to individual event listeners via `concurrency_limit=` parameter.
**How to avoid:** Use `concurrency_limit=1` on individual event handlers: `btn.click(fn=..., concurrency_limit=1)`. For shared GPU access across multiple buttons, use `concurrency_id="gpu_queue"`.
**Warning signs:** TypeError or unexpected behavior when setting queue parameters; multiple pipeline runs overlapping.
**Source:** [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide)

### Pitfall 7: Pipeline Stages Not Cleaned Up on Cancel
**What goes wrong:** User cancels during TTS/lip sync. VRAM from the last-loaded model is never freed. Next run OOMs.
**Why it happens:** When a generator is cancelled, Python does not run code after the last yield. CUDA memory is never cleared.
**How to avoid:** Wrap each stage in try/finally in the generator. In the finally block, call `manager.unload_current_model()` and `torch.cuda.empty_cache()`. This runs even when the generator is cancelled.
**Warning signs:** Gradual VRAM accumulation across cancelled runs; OOM errors on second/third pipeline attempt.

## Code Examples

Verified patterns from official sources:

### Complete Gradio App Structure
```python
# Source: https://www.gradio.app/guides/blocks-and-event-listeners
# Source: https://www.gradio.app/guides/controlling-layout
# Gradio 6.6.0 Blocks pattern for multi-step pipeline UI

import gradio as gr
import pandas as pd
import threading
from pathlib import Path

# Per-app cancel flag (single-user local app)
_cancel_event = threading.Event()

with gr.Blocks(title="Voice Dub") as demo:
    # Application-level state
    app_state = gr.State({
        "video_path": None,
        "transcript": None,
        "step": "upload"
    })

    # Step 1: Upload
    with gr.Column(visible=True) as upload_col:
        gr.Markdown("## Step 1: Upload Video")
        video_upload = gr.Video(label="Source Video", interactive=True)
        hf_token_input = gr.Textbox(
            label="HuggingFace Token",
            type="password",
            placeholder="Required for speaker diarization"
        )
        start_asr_btn = gr.Button("Extract & Transcribe", variant="primary", size="lg")

    # Step 2: Review Transcript
    with gr.Column(visible=False) as review_col:
        gr.Markdown("## Step 2: Review & Edit Transcript")
        transcript_table = gr.Dataframe(
            headers=["ID", "Speaker", "Start", "End", "Text", "Confidence"],
            datatype=["number", "str", "number", "number", "str", "number"],
            interactive=True,
            static_columns=[0, 1, 2, 3, 5],
            wrap=True,
            label="Transcript (edit Text column only)"
        )
        with gr.Row():
            preview_start = gr.Slider(0, 600, value=0, step=1, label="Preview Start (seconds)")
            preview_btn = gr.Button("Preview 30s Clip", variant="secondary")
        preview_video = gr.Video(label="Preview Clip", visible=False)
        proceed_btn = gr.Button("Start Full Dubbing", variant="primary", size="lg")

    # Step 3: Processing
    with gr.Column(visible=False) as processing_col:
        gr.Markdown("## Step 3: Processing")
        stage_status = gr.Textbox(label="Current Stage", interactive=False)
        cancel_btn = gr.Button("Cancel Processing", variant="stop")

    # Step 4: Output
    with gr.Column(visible=False) as output_col:
        gr.Markdown("## Step 4: Complete")
        output_video = gr.Video(label="Dubbed Video")
        restart_btn = gr.Button("Process Another Video", variant="secondary")

    # Wire up events (see pipeline_runner.py for fn implementations)
    asr_event = start_asr_btn.click(
        fn=run_asr_stage,
        inputs=[video_upload, hf_token_input, app_state],
        outputs=[upload_col, review_col, transcript_table, app_state],
        queue=False  # CRITICAL: GPU on Windows RTX 5090
    )

    preview_btn.click(
        fn=extract_preview,
        inputs=[app_state, preview_start],
        outputs=[preview_video],
        queue=False
    )

    pipeline_event = proceed_btn.click(
        fn=run_full_pipeline,  # Must be a generator function
        inputs=[transcript_table, app_state],
        outputs=[review_col, processing_col, stage_status, output_col, output_video]
    )

    cancel_btn.click(
        fn=cancel_pipeline,
        inputs=None,
        outputs=[stage_status],
        cancels=[pipeline_event]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False  # Local only
)
```

### Generator-Based Pipeline Runner
```python
# Source: https://www.gradio.app/guides/progress-bars
# Source: https://www.gradio.app/guides/streaming-outputs

import gradio as gr
import threading
import torch
import gc

_cancel_event = threading.Event()

def run_full_pipeline(transcript_df, app_state, progress=gr.Progress()):
    """
    Generator function wrapping all pipeline stages.
    Yields UI updates between stages for real-time progress.
    MUST be a generator (uses yield) for cancellation to work.
    """
    global _cancel_event
    _cancel_event.clear()

    video_path = app_state["video_path"]

    # Show processing section
    yield (
        gr.Column(visible=False),    # hide review
        gr.Column(visible=True),     # show processing
        "Initializing...",           # stage_status
        gr.Column(visible=False),    # hide output
        gr.Video(value=None)         # output_video
    )

    try:
        # Stage: Translation
        if _cancel_event.is_set():
            yield gr.Column(visible=False), gr.Column(visible=True), "Cancelled by user", gr.Column(visible=False), gr.Video(value=None)
            return
        progress(0.0, desc="Translating segments...")
        yield gr.Column(visible=False), gr.Column(visible=True), "Translating (0%)...", gr.Column(visible=False), gr.Video(value=None)

        segments = dataframe_to_segments(transcript_df)
        translation_result = run_translation(segments)  # Stage 4

        # Validate translation
        is_valid, error_msg = validate_translation_output(translation_result)
        if not is_valid:
            yield gr.Column(visible=False), gr.Column(visible=True), f"Validation failed: {error_msg}", gr.Column(visible=False), gr.Video(value=None)
            return
        progress(0.25, desc="Translation complete")

        # Stage: Voice Cloning
        if _cancel_event.is_set():
            return
        progress(0.25, desc="Cloning voice...")
        yield gr.Column(visible=False), gr.Column(visible=True), "Voice Cloning (25%)...", gr.Column(visible=False), gr.Video(value=None)
        tts_result = run_tts(translation_result)  # Stage 5

        # Stage: Assembly
        if _cancel_event.is_set():
            return
        progress(0.5, desc="Assembling audio...")
        yield gr.Column(visible=False), gr.Column(visible=True), "Assembling (50%)...", gr.Column(visible=False), gr.Video(value=None)
        assembled = run_assembly(video_path, tts_result)  # Stage 6

        # Stage: Lip Sync
        if _cancel_event.is_set():
            return
        progress(0.75, desc="Synchronizing lips...")
        yield gr.Column(visible=False), gr.Column(visible=True), "Lip Sync (75%)...", gr.Column(visible=False), gr.Video(value=None)
        final_output = run_lip_sync(assembled)  # Stage 7

        progress(1.0, desc="Complete!")
        yield (
            gr.Column(visible=False),    # hide review
            gr.Column(visible=False),    # hide processing
            "Complete!",
            gr.Column(visible=True),     # show output
            gr.Video(value=final_output)
        )

    finally:
        # CRITICAL: Clean up VRAM even if cancelled
        from src.models.model_manager import ModelManager
        # ModelManager cleanup (if a model is still loaded)
        gc.collect()
        torch.cuda.empty_cache()

def cancel_pipeline():
    global _cancel_event
    _cancel_event.set()
    return "Cancellation requested - stopping after current operation..."
```

### Stage Validation Functions
```python
# Source: Established from project's settings.py thresholds

from src.config.settings import (
    ASR_CONFIDENCE_THRESHOLD,
    TRANSLATION_DURATION_TOLERANCE,
    TTS_PESQ_REVIEW_THRESHOLD
)

def validate_asr_output(transcript: dict) -> tuple[bool, str]:
    """Validate ASR stage output. Returns (is_valid, message)."""
    if not transcript.get("segments"):
        return False, "No speech segments detected. Is the video silent?"

    segments = transcript["segments"]
    low_conf = [s for s in segments if s.get("confidence", 1.0) < ASR_CONFIDENCE_THRESHOLD]
    ratio = len(low_conf) / len(segments)

    if ratio > 0.5:
        return False, f"Warning: {len(low_conf)}/{len(segments)} segments have low confidence. Consider reviewing before proceeding."

    return True, f"ASR complete: {len(segments)} segments, {len(low_conf)} flagged for review."

def validate_translation_output(translation: dict) -> tuple[bool, str]:
    """Validate translation duration fit."""
    segments = translation.get("segments", [])
    if not segments:
        return False, "Translation produced no output segments."
    # Check duration mismatch
    tolerance = TRANSLATION_DURATION_TOLERANCE
    mismatched = [
        s for s in segments
        if abs(s.get("duration_ratio", 1.0) - 1.0) > 0.3
    ]
    if len(mismatched) > len(segments) * 0.3:
        return False, f"{len(mismatched)} segments have >30% duration mismatch. Sync quality may be poor."
    return True, f"Translation complete: {len(segments)} segments."

def validate_tts_output(tts_result: dict) -> tuple[bool, str]:
    """Validate TTS audio quality."""
    segments = tts_result.get("segments", [])
    if not segments:
        return False, "TTS produced no audio segments."
    failed = [s for s in segments if s.get("pesq_score", 3.0) < TTS_PESQ_REVIEW_THRESHOLD]
    if failed:
        return True, f"Warning: {len(failed)} segments below quality threshold. Review recommended."
    return True, "TTS complete: all segments pass quality check."
```

### Clip Preview Extraction
```python
# Source: https://www.clipcat.com/blog/a-beginners-guide-to-ffmpeg-python-how-to-trim-videos/
# Source: ffmpeg-python already in requirements.txt

import ffmpeg
from pathlib import Path

from src.config.settings import TEMP_DATA_DIR

def extract_preview_clip(video_path: str, start_seconds: float = 0.0, duration: float = 30.0) -> str:
    """
    Extract a preview clip for QC-02 preview requirement.

    Uses stream copy (c='copy') for speed. Cuts at nearest keyframe,
    which may be up to 2 seconds off from requested start. Acceptable for preview.

    Returns path to temporary preview file.
    """
    output_path = str(TEMP_DATA_DIR / "preview_clip.mp4")
    (
        ffmpeg
        .input(video_path, ss=start_seconds)
        .output(output_path, t=duration, c='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Gradio `gr.Interface` | `gr.Blocks` for complex UIs | Gradio 3.x | Blocks allows conditional visibility, multi-step workflows, custom layouts |
| `gr.update()` for component updates | Return `gr.ComponentName(...)` directly | Gradio 4.x → 6.x | Cleaner API; gr.update() still works but is not idiomatic in Gradio 6 |
| `row_count=(n, "dynamic")` tuple in Dataframe | Separate `row_limits=(min, max)` parameter | Gradio 6.0 | Breaking change; old tuple syntax deprecated |
| Queue-based cancellation | Generator function yield-based cancellation | Gradio 3.x+ | Cancellation only works cleanly with generators; synchronous functions cannot be interrupted |
| `demo.queue(concurrency_count=N)` | `btn.click(..., concurrency_limit=N)` | Gradio 4.x | Per-event concurrency control is more granular |

**Deprecated/outdated:**
- **`gr.update()`**: Still functional but not idiomatic in Gradio 6. Return component instances directly.
- **`row_count`/`col_count` tuples in gr.Dataframe**: Replaced by `row_limits`/`column_limits` in Gradio 6.
- **`concurrency_count` in `demo.queue()`**: Replaced by per-event `concurrency_limit` parameter.
- **Tuple format for gr.Video subtitles**: Deprecated in Gradio 6; use `subtitles` parameter directly.

## Open Questions

1. **CUDA queue hang fix status in Gradio 6.6.0**
   - What we know: Issue #12492 reported hang on Windows RTX 5090 with queue=True. PR #12556 was merged to fix the threading issue.
   - What's unclear: Whether the fix is included in the current 6.6.0 release (released 2026-02-17). The issue may or may not be resolved.
   - Recommendation: Use `queue=False` defensively on all GPU handlers regardless. The downside (no queuing) is irrelevant for a single-user local app. Remove `queue=False` only after testing confirms no hang.

2. **Per-user cancel event for future multi-user support**
   - What we know: Global `threading.Event` is fine for single-user local app. For multi-user, `gr.State` holding a per-session event is needed.
   - What's unclear: Whether the project will ever serve multiple concurrent users.
   - Recommendation: Implement with global event now (simplest). Document the limitation. Refactor to gr.State if multi-user ever needed.

3. **Gradio 6 Windows file path handling**
   - What we know: Gradio serves uploaded files from a temp cache. Video components return file paths as strings.
   - What's unclear: Whether Windows-style paths (`C:\...`) are returned correctly or need normalization to forward slashes for FFmpeg.
   - Recommendation: Always normalize paths with `Path(video_path).as_posix()` or use `str(Path(video_path))` consistently before passing to ffmpeg-python.

4. **HuggingFace token input security**
   - What we know: Pyannote requires HF token. gr.Textbox(type="password") masks input in the UI.
   - What's unclear: Whether storing the token in gr.State is secure for a local app.
   - Recommendation: For a local app (friends-only), gr.State is acceptable. Do not log or persist the token to disk in the UI layer. The token is already handled by pyannote's pipeline loading in Phase 3.

## Sources

### Primary (HIGH confidence)
- [Gradio 6.6.0 Official Docs - Progress](https://www.gradio.app/docs/gradio/progress) - gr.Progress API, track_tqdm parameter
- [Gradio Official Docs - Dataframe](https://www.gradio.app/docs/gradio/dataframe) - static_columns, datatype, row_limits API
- [Gradio Official Docs - Video](https://www.gradio.app/docs/gradio/video) - streaming, playback_position, events
- [Gradio Official Docs - Button](https://www.gradio.app/docs/gradio/button) - variant="stop", cancels parameter
- [Gradio Official Docs - State](https://www.gradio.app/docs/gradio/state) - session isolation, time_to_live
- [Gradio Guides - Streaming Outputs](https://www.gradio.app/guides/streaming-outputs) - generator function pattern, yield
- [Gradio Guides - Progress Bars](https://www.gradio.app/guides/progress-bars) - gr.Progress() usage
- [Gradio Guides - Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) - event chaining, cancels=, .then()
- [Gradio Guides - Controlling Layout](https://www.gradio.app/guides/controlling-layout) - Row, Column, visible
- [Gradio Guides - Queuing](https://www.gradio.app/guides/queuing) - concurrency_limit, concurrency_id
- [Gradio 6 Migration Guide](https://www.gradio.app/main/guides/gradio-6-migration-guide) - breaking changes, row_count deprecation
- [Gradio - Maximum Performance](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance) - GPU concurrency_limit=1 guidance
- [PyPI - gradio 6.6.0](https://pypi.org/project/gradio/) - requires Python >=3.10, released 2026-02-17
- [Gradio HighlightedText Docs](https://www.gradio.app/docs/gradio/highlightedtext) - confidence visualization format

### Secondary (MEDIUM confidence)
- [GitHub Issue #12492](https://github.com/gradio-app/gradio/issues/12492) - CUDA hang on Windows RTX 5090 with queue=True, verified with PR fix #12556
- [GitHub Issue #6724](https://github.com/gradio-app/gradio/issues/6724) - Cancel external thread pattern, threading.Event via gr.State
- [GitHub Issue #10432](https://github.com/gradio-app/gradio/issues/10432) - cancels= breaks .then() chains bug
- [ffmpeg-python clip extraction](https://www.clipcat.com/blog/a-beginners-guide-to-ffmpeg-python-how-to-trim-videos/) - ss + t parameters, c='copy' pattern

### Tertiary (LOW confidence)
- WebSearch results on Gradio pipeline patterns - community blog posts, not individually verified against official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Gradio 6.6.0 confirmed on PyPI (2026-02-17), official docs confirm all APIs used
- Architecture: HIGH - All patterns verified directly against official Gradio docs and GitHub issues
- Pitfalls: HIGH for CUDA hang (confirmed GitHub issue with hardware match), HIGH for cancel limitations (confirmed in docs), MEDIUM for disk fill and cleanup

**Research date:** 2026-02-21
**Valid until:** 2026-03-23 (30 days) - Gradio is fast-moving (weekly releases), re-verify Gradio version before implementing
