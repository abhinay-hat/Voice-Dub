"""
Voice Dub - Gradio web UI scaffold.

Four-step workflow layout:
  Step 1: Upload Video (ASR extraction)
  Step 2: Review & Edit Transcript
  Step 3: Processing (dubbing pipeline)
  Step 4: Output (download dubbed video)

Event handler stubs are connected in Plan 04 (event wiring).
"""
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

# Module-level pipeline event tracker (cancel event lives in pipeline_runner.py)
_pipeline_event = None


def create_app() -> gr.Blocks:
    """
    Build and return the Voice Dub Gradio Blocks application.

    All component references are stored as local variables so Plan 04
    can attach .click()/.change() event handlers without rebuilding the layout.

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

    return demo


# Module-level demo instance consumed by src/app.py and Plan 04 event wiring
demo = create_app()
