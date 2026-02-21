"""
Clip preview extractor for Voice Dub UI (QC-02).

Extracts a short preview clip from a video using FFmpeg stream copy (fast,
no re-encoding). Used by the Gradio UI so users can review a 30-second
sample before committing to the full pipeline run.
"""
import subprocess
from pathlib import Path


def extract_preview_clip(
    video_path: str,
    start_seconds: float,
    output_path: str,
    duration: float = 30.0,
) -> str:
    """
    Extract a preview clip from a video using FFmpeg.

    Uses stream copy (-c copy) for speed — no re-encoding of the source.
    This means the clip may not start on a keyframe; for UI preview purposes
    this is acceptable. For frame-accurate extraction, re-encoding is needed.

    Args:
        video_path: Absolute path to the source video file.
        start_seconds: Start time in seconds. Clamped to 0 if negative.
        output_path: Absolute path where the output clip will be written.
            The parent directory must exist.
        duration: Clip duration in seconds. Default 30 seconds.

    Returns:
        output_path on success (allows chaining).

    Raises:
        ValueError: If video_path does not exist.
        RuntimeError: If FFmpeg returns a non-zero exit code (encoding failed,
            invalid input, output directory missing, etc.).

    Example:
        >>> clip = extract_preview_clip(
        ...     video_path="/data/raw/movie.mp4",
        ...     start_seconds=120.0,
        ...     output_path="/tmp/preview.mp4",
        ... )
        >>> print(clip)  # "/tmp/preview.mp4"
    """
    if not Path(video_path).exists():
        raise ValueError(f"Video file not found: {video_path}")

    # Clamp negative start times to zero
    start_seconds = max(0.0, start_seconds)

    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",                        # Overwrite output without prompting
            "-ss", str(start_seconds),   # Seek to start (placed before -i for fast seek)
            "-i", video_path,            # Input file
            "-t", str(duration),         # Duration of output clip
            "-c", "copy",                # Stream copy: no re-encoding (fast)
            output_path,                 # Output file
        ],
        capture_output=True,
    )

    if proc.returncode != 0:
        error_text = proc.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed with exit code {proc.returncode}: {error_text}"
        )

    return output_path
