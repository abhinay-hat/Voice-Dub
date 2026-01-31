"""
Gradio web interface for Voice Dub video processing.
Provides upload, processing, and download capabilities for video dubbing pipeline.
"""
from pathlib import Path
import sys

# Add project root to Python path to support src package imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from src.video_processing import process_video, validate_video_file, get_video_info


def display_video_info(video_path: str) -> str:
    """
    Display video metadata when video is uploaded.

    Args:
        video_path: Path to uploaded video file

    Returns:
        str: Formatted video information string
    """
    if not video_path:
        return ""

    try:
        # Validate video file
        is_valid, error = validate_video_file(video_path)
        if not is_valid:
            return f"❌ Invalid video: {error}"

        # Get video metadata
        info = get_video_info(video_path)

        # Format duration as HH:MM:SS
        duration_seconds = int(info.duration)
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Format video info
        return (
            f"✓ Valid video\n"
            f"Resolution: {info.width}x{info.height}\n"
            f"Codec: {info.codec.upper()}\n"
            f"FPS: {info.fps:.2f}\n"
            f"Duration: {duration_str}\n"
            f"Format: {info.container_format.upper()}\n"
            f"Audio: {'Yes' if info.has_audio else 'No'}"
        )
    except Exception as e:
        return f"❌ Error reading video: {str(e)}"


def process_video_ui(video_path: str, progress=gr.Progress()) -> tuple[str, str]:
    """
    Process video through the pipeline with progress tracking.

    Args:
        video_path: Path to uploaded video file
        progress: Gradio progress tracker

    Returns:
        tuple[str, str]: (output_video_path, status_message)
    """
    if not video_path:
        return None, "❌ No video uploaded"

    try:
        # Validate input
        is_valid, error = validate_video_file(video_path)
        if not is_valid:
            return None, f"❌ Invalid video: {error}"

        # Define progress callback that maps to Gradio's Progress
        def progress_callback(pct: float, desc: str):
            progress(pct, desc=desc)

        # Process video
        result = process_video(
            video_path,
            progress_callback=progress_callback
        )

        # Return output path and success status
        status = (
            f"✓ Processing complete!\n"
            f"Duration: {result.duration:.1f}s\n"
            f"Processing time: {result.processing_time:.1f}s\n"
            f"Resolution: {result.video_info.width}x{result.video_info.height}\n"
            f"Output: {result.output_path.name}"
        )

        return str(result.output_path), status

    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        return None, error_msg


# Create Gradio interface
with gr.Blocks(title="Voice Dub - Video Processing") as demo:
    gr.Markdown("# Voice Dub - Video Processing")
    gr.Markdown(
        "Upload a video to test the processing pipeline. "
        "Currently performs round-trip (extract audio/video, merge back) "
        "to validate FFmpeg toolchain."
    )

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(
                sources=["upload"],
                label="Upload Video",
                format="mp4"  # Browser playback format
            )
            video_info = gr.Textbox(
                label="Video Info",
                interactive=False,
                lines=7
            )
            process_btn = gr.Button("Process Video", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Processed Video")
            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=7
            )

    # Event handlers
    input_video.change(
        fn=display_video_info,
        inputs=input_video,
        outputs=video_info
    )

    process_btn.click(
        fn=process_video_ui,
        inputs=input_video,
        outputs=[output_video, status]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Accessible on network (for friends)
        server_port=7860,
        max_file_size="500mb"  # Reasonable limit for testing
    )
