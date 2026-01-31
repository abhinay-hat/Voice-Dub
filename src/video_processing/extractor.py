"""
Video stream extraction utilities using FFmpeg.
Provides functions for extracting audio and video streams from video files.
"""
from pathlib import Path
from typing import Union
from dataclasses import dataclass
import ffmpeg

from .video_utils import get_video_info, VideoInfo


@dataclass
class ExtractionResult:
    """Result of stream extraction containing both audio and video paths."""
    audio_path: Path
    video_path: Path
    duration: float  # seconds
    sample_rate: int
    video_info: VideoInfo


def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 48000
) -> Path:
    """
    Extract audio stream from video to WAV format (PCM 16-bit signed little-endian).

    Args:
        video_path: Path to input video file
        output_path: Path where WAV file should be saved
        sample_rate: Audio sample rate in Hz (default: 48000 - video production standard)

    Returns:
        Path: Path to extracted audio file (same as output_path)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg fails to extract audio
        ValueError: If video has no audio stream

    Example:
        >>> extract_audio("video.mp4", "audio.wav", sample_rate=48000)
        Path("audio.wav")
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    # Validate input file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check for audio stream presence
    try:
        probe_data = ffmpeg.probe(str(video_path))
        has_audio = any(
            stream['codec_type'] == 'audio'
            for stream in probe_data['streams']
        )
        if not has_audio:
            raise ValueError(f"No audio stream found in video: {video_path}")
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise ffmpeg.Error(
            f"Failed to probe video for audio stream: {stderr}"
        ) from e

    # Extract audio stream
    try:
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(
            stream.audio,
            str(output_path),
            acodec='pcm_s16le',  # PCM 16-bit signed little-endian
            ar=sample_rate,      # Sample rate (48kHz default)
            ac=2                 # Stereo (2 channels)
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise ffmpeg.Error(
            f"Failed to extract audio from '{video_path}': {stderr}"
        ) from e

    return output_path


def extract_video_stream(
    video_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Path:
    """
    Extract video stream without audio (for later recombination).

    Uses stream copy to avoid re-encoding - fast and lossless.

    Args:
        video_path: Path to input video file
        output_path: Path where video-only file should be saved

    Returns:
        Path: Path to extracted video file (same as output_path)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg fails to extract video stream

    Example:
        >>> extract_video_stream("video.mp4", "video_only.mp4")
        Path("video_only.mp4")
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    # Validate input file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract video stream with audio removed
    try:
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(
            stream,
            str(output_path),
            vcodec='copy',  # Copy video stream without re-encoding (fast, lossless)
            an=None         # Remove audio stream
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise ffmpeg.Error(
            f"Failed to extract video stream from '{video_path}': {stderr}"
        ) from e

    return output_path


def extract_streams(
    video_path: Union[str, Path],
    temp_dir: Path
) -> ExtractionResult:
    """
    Convenience function to extract both audio and video streams.

    Extracts:
    - Audio to {temp_dir}/audio.wav (48kHz, 16-bit PCM, stereo)
    - Video to {temp_dir}/video_only.{ext} (stream copy, no audio)

    Args:
        video_path: Path to input video file
        temp_dir: Directory for temporary extracted files

    Returns:
        ExtractionResult: Container with audio_path, video_path, and metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        ffmpeg.Error: If FFmpeg fails during extraction
        ValueError: If video has no audio stream

    Example:
        >>> with TempFileManager() as temp:
        ...     result = extract_streams("video.mp4", temp.temp_dir)
        ...     print(f"Audio: {result.audio_path}, Video: {result.video_path}")
    """
    input_video_path = Path(video_path)
    temp_dir = Path(temp_dir)

    # Get video metadata
    video_info = get_video_info(input_video_path)

    # Determine video-only output extension based on container format
    video_ext_map = {
        'mp4': 'mp4',
        'mkv': 'mkv',
        'avi': 'avi'
    }
    video_ext = video_ext_map.get(video_info.container_format, 'mp4')

    # Define output paths
    audio_output_path = temp_dir / 'audio.wav'
    video_output_path = temp_dir / f'video_only.{video_ext}'

    # Extract audio stream
    extract_audio(input_video_path, audio_output_path, sample_rate=48000)

    # Extract video stream
    extract_video_stream(input_video_path, video_output_path)

    return ExtractionResult(
        audio_path=audio_output_path,
        video_path=video_output_path,
        duration=video_info.duration,
        sample_rate=48000,
        video_info=video_info
    )
