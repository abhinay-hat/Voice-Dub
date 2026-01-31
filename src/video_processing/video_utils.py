"""
Video metadata probing and validation utilities using FFmpeg.
Provides functions for extracting video information and validating video files.
"""
from pathlib import Path
from typing import NamedTuple, Union
import ffmpeg


class VideoInfo(NamedTuple):
    """Structured video metadata extracted from FFmpeg probe."""
    width: int
    height: int
    duration: float  # seconds
    codec: str
    fps: float
    container_format: str  # normalized: "mp4", "mkv", or "avi"
    has_audio: bool


def probe_video(video_path: Union[str, Path]) -> dict:
    """
    Probe video file using FFmpeg and return raw metadata.

    Args:
        video_path: Path to video file

    Returns:
        dict: Raw FFmpeg probe data containing streams and format information

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg cannot probe the file (corrupted, unsupported format)

    Example:
        >>> probe = probe_video("video.mp4")
        >>> duration = float(probe['format']['duration'])
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        probe_data = ffmpeg.probe(str(video_path))
        return probe_data
    except ffmpeg.Error as e:
        # Extract stderr for more helpful error message
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise ffmpeg.Error(
            f"Failed to probe video file '{video_path}': {stderr}"
        ) from e


def get_video_info(video_path: Union[str, Path]) -> VideoInfo:
    """
    Extract structured video metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo: Named tuple with width, height, duration, codec, fps,
                   container_format, has_audio

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg cannot probe the file
        ValueError: If video stream is missing from file

    Example:
        >>> info = get_video_info("video.mp4")
        >>> print(f"{info.width}x{info.height} @ {info.fps:.2f}fps")
    """
    probe_data = probe_video(video_path)

    # Extract video stream
    video_stream = next(
        (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
        None
    )

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Extract audio stream presence
    has_audio = any(
        stream['codec_type'] == 'audio'
        for stream in probe_data['streams']
    )

    # Parse frame rate (handles fractional format like "30000/1001" -> 29.97)
    fps_str = video_stream.get('r_frame_rate', '0/1')
    try:
        numerator, denominator = map(int, fps_str.split('/'))
        fps = numerator / denominator if denominator != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # Get container format and normalize it
    format_name = probe_data['format']['format_name']
    container_format = detect_container_format(video_path)

    return VideoInfo(
        width=int(video_stream['width']),
        height=int(video_stream['height']),
        duration=float(probe_data['format']['duration']),
        codec=video_stream['codec_name'],
        fps=fps,
        container_format=container_format,
        has_audio=has_audio
    )


def detect_container_format(video_path: Union[str, Path]) -> str:
    """
    Detect and normalize video container format.

    FFmpeg returns complex format names like "mov,mp4,m4a,3gp,3g2,mj2".
    This function normalizes to simple format strings: "mp4", "mkv", or "avi".

    Args:
        video_path: Path to video file

    Returns:
        str: Normalized format ("mp4", "mkv", or "avi")

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg cannot probe the file
        ValueError: If format is not supported (not MP4/MKV/AVI)

    Example:
        >>> detect_container_format("video.mp4")
        'mp4'
        >>> detect_container_format("video.mkv")
        'mkv'
    """
    probe_data = probe_video(video_path)
    format_name = probe_data['format']['format_name']

    # Map FFmpeg format names to normalized format strings
    # FFmpeg returns comma-separated list of compatible formats
    format_lower = format_name.lower()

    if 'mp4' in format_lower or 'mov' in format_lower:
        return 'mp4'
    elif 'matroska' in format_lower or 'webm' in format_lower:
        return 'mkv'
    elif 'avi' in format_lower:
        return 'avi'
    else:
        raise ValueError(
            f"Unsupported video format: {format_name}. "
            f"Supported formats: MP4, MKV, AVI"
        )


def validate_video_file(video_path: Union[str, Path]) -> tuple[bool, str]:
    """
    Validate video file for processing.

    Checks:
    - File exists
    - Has video stream
    - Container format is supported (MP4/MKV/AVI)

    Args:
        video_path: Path to video file

    Returns:
        tuple[bool, str]: (is_valid, error_message)
                         Returns (True, "") if valid
                         Returns (False, "error description") if invalid

    Example:
        >>> valid, error = validate_video_file("video.mp4")
        >>> if not valid:
        ...     print(f"Invalid video: {error}")
    """
    video_path = Path(video_path)

    # Check file exists
    if not video_path.exists():
        return False, f"File not found: {video_path}"

    # Check file is readable
    if not video_path.is_file():
        return False, f"Path is not a file: {video_path}"

    try:
        # Probe video metadata
        probe_data = probe_video(video_path)

        # Check for video stream
        video_stream = next(
            (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
            None
        )
        if video_stream is None:
            return False, "No video stream found in file"

        # Check format is supported
        try:
            container_format = detect_container_format(video_path)
        except ValueError as e:
            return False, str(e)

        # All checks passed
        return True, ""

    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        return False, f"FFmpeg probe failed: {stderr}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
