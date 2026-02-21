"""
Voice Dub - Gradio web UI scaffold.

Four-step workflow layout:
  Step 1: Upload Video (ASR extraction)
  Step 2: Review & Edit Transcript
  Step 3: Processing (dubbing pipeline)
  Step 4: Output (download dubbed video)

Event handlers wired in Plan 04 — connects UI to pipeline_runner and clip_preview.
"""
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

from src.ui.pipeline_runner import run_asr_ui, run_full_pipeline, cancel_pipeline
from src.ui.clip_preview import extract_preview_clip
from src.video_processing import get_video_info, validate_video_file
from src.config.settings import TEMP_DATA_DIR


def show_video_info(video_path: str) -> str:
    """Return formatted video metadata for the info text box."""
    if not video_path:
        return ""
    is_valid, err = validate_video_file(video_path)
    if not is_valid:
        return f"Invalid: {err}"
    try:
        info = get_video_info(video_path)
        dur = int(info.duration)
        h, m, s = dur // 3600, (dur % 3600) // 60, dur % 60
        return (
            f"Resolution: {info.width}x{info.height}\n"
            f"Duration: {h:02d}:{m:02d}:{s:02d}\n"
            f"Format: {info.container_format.upper()}\n"
            f"Audio: {'Yes' if info.has_audio else 'No'}"
        )
    except Exception as exc:
        return f"Could not read video info: {exc}"


def preview_clip(app_state: dict, start_seconds: float):
    """Extract a 30-second preview clip starting at start_seconds."""
    video_path = app_state.get("video_path") if app_state else None
    if not video_path:
        return gr.update(visible=False)
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = str(TEMP_DATA_DIR / f"preview_{uuid.uuid4().hex[:6]}.mp4")
    try:
        extract_preview_clip(video_path, float(start_seconds), out, duration=30.0)
        return gr.update(value=out, visible=True)
    except Exception:
        return gr.update(visible=False)


def go_back(state: dict):
    """Return to Step 1 without clearing the uploaded video."""
    return gr.update(visible=True), gr.update(visible=False), state


def restart():
    """Return to Step 1 with fully cleared state."""
    empty_state = {
        "step": "upload",
        "video_path": None,
        "video_id": None,
        "audio_path": None,
        "asr_output_path": None,
    }
    return gr.update(visible=True), gr.update(visible=False), empty_state


def create_app() -> gr.Blocks:
    """
    Build and return the Voice Dub Gradio Blocks application.

    All component references are stored as local variables so event handlers
    can be attached without rebuilding the layout.

    Returns:
        gr.Blocks: Configured Gradio application instance.
    """
    with gr.Blocks(title="Voice Dub") as demo:

        gr.Markdown("# Voice Dub")
        gr.Markdown(
            "Dub any video into English while preserving the original speaker's voice "
            "and emotional expression — entirely on local hardware."
        )

        # ------------------------------------------------------------------ #
        # Shared application state                                             #
        # ------------------------------------------------------------------ #
        app_state = gr.State(
            {
                "step": "upload",
                "video_path": None,
                "video_id": None,
                "audio_path": None,
                "asr_output_path": None,
            }
        )

        # ------------------------------------------------------------------ #
        # Step 1 — Upload Video                                                #
        # ------------------------------------------------------------------ #
        with gr.Column(visible=True) as upload_section:
            gr.Markdown("## Step 1: Upload Video")

            video_input = gr.Video(
                label="Source Video",
                sources=["upload"],
                interactive=True,
            )

            hf_token_input = gr.Textbox(
                label="HuggingFace Token (required for speaker diarization)",
                type="password",
                placeholder="hf_...",
            )

            video_info_text = gr.Textbox(
                label="Video Info",
                interactive=False,
                lines=5,
            )

            start_asr_btn = gr.Button(
                "Extract & Transcribe",
                variant="primary",
                size="lg",
            )

            asr_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                visible=False,
            )

        # ------------------------------------------------------------------ #
        # Step 2 — Review & Edit Transcript                                    #
        # ------------------------------------------------------------------ #
        with gr.Column(visible=False) as review_section:
            gr.Markdown("## Step 2: Review & Edit Transcript")
            gr.Markdown(
                "Edit the **Text** column to fix transcription errors. "
                "Other columns are read-only."
            )

            transcript_table = gr.Dataframe(
                headers=["ID", "Speaker", "Start", "End", "Text", "Confidence"],
                datatype=["number", "str", "number", "number", "str", "number"],
                interactive=True,
                static_columns=[0, 1, 2, 3, 5],
                wrap=True,
                label="Transcript (edit Text column only)",
                row_limits=(1, None),
            )

            gr.Markdown("### Preview a 30-second clip before full render:")

            with gr.Row():
                preview_start_slider = gr.Slider(
                    minimum=0,
                    maximum=600,
                    value=0,
                    step=1,
                    label="Preview start (seconds)",
                )
                preview_btn = gr.Button(
                    "Preview 30s Clip",
                    variant="secondary",
                )

            preview_video = gr.Video(
                label="Preview Clip",
                visible=False,
            )

            proceed_btn = gr.Button(
                "Start Full Dubbing",
                variant="primary",
                size="lg",
            )

            back_to_upload_btn = gr.Button(
                "Back to Upload",
                variant="secondary",
            )

        # ------------------------------------------------------------------ #
        # Step 3 — Processing                                                  #
        # ------------------------------------------------------------------ #
        with gr.Column(visible=False) as processing_section:
            gr.Markdown("## Step 3: Processing")

            stage_status_text = gr.Textbox(
                label="Current Stage",
                interactive=False,
                lines=3,
            )

            overall_progress = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                label="Overall Progress",
                interactive=False,
            )

            cancel_btn = gr.Button(
                "Cancel Processing",
                variant="stop",
                size="lg",
            )

        # ------------------------------------------------------------------ #
        # Step 4 — Output                                                      #
        # ------------------------------------------------------------------ #
        with gr.Column(visible=False) as output_section:
            gr.Markdown("## Step 4: Complete")

            output_video = gr.Video(label="Dubbed Video")

            download_info = gr.Textbox(
                label="Output Info",
                interactive=False,
                lines=3,
            )

            process_another_btn = gr.Button(
                "Process Another Video",
                variant="secondary",
            )

        # ------------------------------------------------------------------ #
        # Event handlers                                                       #
        # ------------------------------------------------------------------ #

        # 1. Video info on upload
        video_input.change(
            fn=show_video_info,
            inputs=video_input,
            outputs=video_info_text,
            queue=False,
        )

        # 2. ASR button — streaming generator (6-tuple yield)
        # run_asr_ui yields:
        #   (upload_col_visible, review_col_visible, transcript_df, app_state,
        #    status_text, asr_status_visible)
        asr_event = start_asr_btn.click(
            fn=run_asr_ui,
            inputs=[video_input, hf_token_input, app_state],
            outputs=[
                upload_section,
                review_section,
                transcript_table,
                app_state,
                asr_status,
                asr_status,
            ],
        )

        # 3. Preview clip button
        preview_btn.click(
            fn=preview_clip,
            inputs=[app_state, preview_start_slider],
            outputs=preview_video,
            queue=False,
        )

        # 4. Full pipeline button — streaming generator (6-tuple yield)
        # run_full_pipeline yields:
        #   (review_col_visible, processing_col_visible, status_text,
        #    progress_value, output_col_visible, output_video_path)
        pipeline_event = proceed_btn.click(
            fn=run_full_pipeline,
            inputs=[transcript_table, app_state],
            outputs=[
                review_section,
                processing_section,
                stage_status_text,
                overall_progress,
                output_section,
                output_video,
            ],
        )

        # 5. Cancel button
        cancel_btn.click(
            fn=cancel_pipeline,
            inputs=None,
            outputs=stage_status_text,
            cancels=[pipeline_event],
        )

        # 6. Back to upload button
        back_to_upload_btn.click(
            fn=go_back,
            inputs=app_state,
            outputs=[upload_section, review_section, app_state],
            queue=False,
        )

        # 7. Process another video button
        process_another_btn.click(
            fn=restart,
            inputs=None,
            outputs=[upload_section, output_section, app_state],
            queue=False,
        )

        # Enable Gradio queue (required for streaming generators)
        demo.queue(default_concurrency_limit=1)

    return demo


# Module-level demo instance consumed by src/app.py
demo = create_app()
