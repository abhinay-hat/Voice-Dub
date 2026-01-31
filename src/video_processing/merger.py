"""
Audio-video stream merging utilities using FFmpeg.
Provides functions for recombining processed audio with video streams.
"""
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
import ffmpeg


@dataclass
class MergeConfig:
    """Configuration for audio-video merging."""
    video_codec: str = "copy"  # Stream copy by default (fast, lossless)
    audio_codec: str = "aac"   # AAC for MP4 compatibility
    audio_bitrate: str = "192k"  # Standard quality
    container_format: Optional[str] = None  # Auto-detect from output path


def merge_audio_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[MergeConfig] = None
) -> Path:
    """
    Merge video stream with new/processed audio stream.

    Args:
        video_path: Path to video-only file (or file with audio to be replaced)
        audio_path: Path to audio file (WAV, MP3, AAC, etc.)
        output_path: Path for merged output video
        config: Merge configuration (if None, uses default config)

    Returns:
        Path: Path to merged output file (same as output_path)

    Raises:
        FileNotFoundError: If video or audio file doesn't exist
        ffmpeg.Error: If FFmpeg fails during merge

    Example:
        >>> merge_audio_video("video_only.mp4", "audio.wav", "output.mp4")
        Path("output.mp4")
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Validate inputs exist
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use default config if none provided
    if config is None:
        # Detect output format from extension
        output_ext = output_path.suffix.lower().lstrip('.')
        if output_ext == 'mkv':
            config = MergeConfig(video_codec='copy', audio_codec='copy')
        else:
            # MP4, AVI, or other - use AAC for compatibility
            config = MergeConfig(video_codec='copy', audio_codec='aac', audio_bitrate='192k')

    # Validate merge inputs
    is_valid, error_msg = validate_merge_inputs(video_path, audio_path)
    if not is_valid:
        raise ValueError(f"Merge validation failed: {error_msg}")

    # Merge streams
    try:
        video_input = ffmpeg.input(str(video_path))
        audio_input = ffmpeg.input(str(audio_path))

        # Explicitly map video and audio streams
        stream = ffmpeg.output(
            video_input.video,
            audio_input.audio,
            str(output_path),
            vcodec=config.video_codec,
            acodec=config.audio_codec,
            audio_bitrate=config.audio_bitrate if config.audio_codec != 'copy' else None
        )

        ffmpeg.run(stream, quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise ffmpeg.Error(
            f"Failed to merge video and audio: {stderr}"
        ) from e

    return output_path


def get_optimal_merge_config(input_format: str, output_format: str) -> MergeConfig:
    """
    Get optimal merge configuration based on container formats.

    Args:
        input_format: Input video format ("mp4", "mkv", or "avi")
        output_format: Output video format ("mp4", "mkv", or "avi")

    Returns:
        MergeConfig: Optimal configuration for the format combination

    Example:
        >>> config = get_optimal_merge_config("mp4", "mp4")
        >>> # Returns MergeConfig(video_codec='copy', audio_codec='aac')
    """
    # MP4 output: video copy, audio AAC (required for compatibility)
    if output_format == 'mp4':
        return MergeConfig(
            video_codec='copy',
            audio_codec='aac',
            audio_bitrate='192k',
            container_format='mp4'
        )

    # MKV output: both copy (MKV supports almost any codec)
    elif output_format == 'mkv':
        return MergeConfig(
            video_codec='copy',
            audio_codec='copy',  # No re-encoding needed for MKV
            audio_bitrate='192k',  # Unused when codec is 'copy'
            container_format='mkv'
        )

    # AVI output: video copy, audio mp3 (for compatibility)
    elif output_format == 'avi':
        return MergeConfig(
            video_codec='copy',
            audio_codec='mp3',
            audio_bitrate='192k',
            container_format='avi'
        )

    else:
        # Default to MP4 settings for unknown formats
        return MergeConfig(
            video_codec='copy',
            audio_codec='aac',
            audio_bitrate='192k',
            container_format=output_format
        )


def validate_merge_inputs(
    video_path: Union[str, Path],
    audio_path: Union[str, Path]
) -> tuple[bool, str]:
    """
    Validate inputs for audio-video merging.

    Checks:
    - Both files exist
    - Video file has video stream
    - Audio file has audio stream

    Args:
        video_path: Path to video file
        audio_path: Path to audio file

    Returns:
        tuple[bool, str]: (is_valid, error_message)
                         Returns (True, "") if valid
                         Returns (False, "error description") if invalid

    Example:
        >>> valid, error = validate_merge_inputs("video.mp4", "audio.wav")
        >>> if not valid:
        ...     print(f"Validation failed: {error}")
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    # Check files exist
    if not video_path.exists():
        return False, f"Video file not found: {video_path}"
    if not audio_path.exists():
        return False, f"Audio file not found: {audio_path}"

    # Check video has video stream
    try:
        video_probe = ffmpeg.probe(str(video_path))
        has_video = any(
            stream['codec_type'] == 'video'
            for stream in video_probe['streams']
        )
        if not has_video:
            return False, f"No video stream found in: {video_path}"
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        return False, f"Failed to probe video file: {stderr}"

    # Check audio has audio stream
    try:
        audio_probe = ffmpeg.probe(str(audio_path))
        has_audio = any(
            stream['codec_type'] == 'audio'
            for stream in audio_probe['streams']
        )
        if not has_audio:
            return False, f"No audio stream found in: {audio_path}"
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        return False, f"Failed to probe audio file: {stderr}"

    # All checks passed
    return True, ""
