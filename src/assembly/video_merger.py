"""
Video-audio merging with explicit FFmpeg sync flags.

Uses modern FFmpeg best practices with explicit stream mapping and aresample
async correction to prevent progressive drift over long videos.

Example:
    >>> from pathlib import Path
    >>> from src.assembly.video_merger import merge_with_sync_validation
    >>> result = merge_with_sync_validation(
    ...     video_path=Path('video_only.mp4'),
    ...     audio_path=Path('dubbed_audio_48k.wav'),
    ...     output_path=Path('final_dubbed.mp4')
    ... )
    >>> print(f'Output: {result.output_path}')
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import time
import ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of audio-video merge operation.

    Attributes:
        output_path: Path to merged output file
        video_codec: Video codec used (e.g., 'copy')
        audio_codec: Audio codec used (e.g., 'aac')
        async_correction: Whether aresample async=1 was applied
        merge_duration: Time to complete merge in seconds
    """
    output_path: Path
    video_codec: str
    audio_codec: str
    async_correction: bool
    merge_duration: float


def get_merge_config(output_format: str) -> Dict[str, str]:
    """Get optimal codec configuration for output format.

    Different container formats support different codecs. This function
    returns the best codec configuration for each format.

    Args:
        output_format: Output format extension (e.g., 'mp4', 'mkv', 'avi')

    Returns:
        Dict with 'vcodec', 'acodec', and optionally 'audio_bitrate' keys

    Example:
        >>> config = get_merge_config('mp4')
        >>> print(config)
        {'vcodec': 'copy', 'acodec': 'aac', 'audio_bitrate': '192k'}
    """
    format_lower = output_format.lower().lstrip('.')

    configs = {
        'mp4': {
            'vcodec': 'copy',
            'acodec': 'aac',
            'audio_bitrate': '192k'
        },
        'mkv': {
            'vcodec': 'copy',
            'acodec': 'copy'  # MKV supports raw PCM
        },
        'avi': {
            'vcodec': 'copy',
            'acodec': 'mp3',
            'audio_bitrate': '192k'
        }
    }

    return configs.get(format_lower, configs['mp4'])  # Default to MP4


def validate_audio_video_compatibility(
    video_path: Path,
    audio_path: Path
) -> Tuple[bool, List[str]]:
    """Validate audio and video files are compatible for merging.

    Uses ffprobe to check stream types and sample rates.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file

    Returns:
        Tuple of (is_compatible: bool, warnings: List[str])

    Example:
        >>> video = Path('video_only.mp4')
        >>> audio = Path('dubbed_48k.wav')
        >>> compatible, warnings = validate_audio_video_compatibility(video, audio)
        >>> if not compatible:
        ...     print(f'Incompatible: {warnings}')
    """
    warnings = []
    is_compatible = True

    # Check video file exists and has video stream
    if not video_path.exists():
        warnings.append(f"Video file not found: {video_path}")
        return False, warnings

    try:
        video_probe = ffmpeg.probe(str(video_path))
        has_video_stream = any(
            stream['codec_type'] == 'video'
            for stream in video_probe.get('streams', [])
        )
        if not has_video_stream:
            warnings.append(f"No video stream found in {video_path.name}")
            is_compatible = False
    except ffmpeg.Error as e:
        warnings.append(f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}")
        is_compatible = False

    # Check audio file exists and has audio stream
    if not audio_path.exists():
        warnings.append(f"Audio file not found: {audio_path}")
        return False, warnings

    try:
        audio_probe = ffmpeg.probe(str(audio_path))
        audio_streams = [
            stream for stream in audio_probe.get('streams', [])
            if stream['codec_type'] == 'audio'
        ]

        if not audio_streams:
            warnings.append(f"No audio stream found in {audio_path.name}")
            is_compatible = False
        else:
            # Check sample rate (should be 48kHz for video production)
            sample_rate = int(audio_streams[0].get('sample_rate', 0))
            if sample_rate != 48000:
                warnings.append(
                    f"Audio sample rate is {sample_rate}Hz (expected 48000Hz). "
                    f"Consider resampling for best quality."
                )
                # Not a failure, just a warning

    except ffmpeg.Error as e:
        warnings.append(f"Failed to probe audio: {e.stderr.decode() if e.stderr else str(e)}")
        is_compatible = False

    return is_compatible, warnings


def merge_with_sync_validation(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    async_correction: bool = True
) -> MergeResult:
    """Merge video and audio with explicit synchronization flags.

    Uses modern FFmpeg best practices:
    - Explicit stream mapping (not auto-selection)
    - aresample filter for clock drift correction (replaces deprecated -async)
    - Stream copy for video (no re-encoding)
    - Format-appropriate audio codec (AAC for MP4, etc.)

    Args:
        video_path: Video-only file (from Phase 2 extractor)
        audio_path: Dubbed audio at 48kHz (after normalization)
        output_path: Final output video path
        async_correction: Enable clock drift correction (default: True)

    Returns:
        MergeResult with merge details

    Raises:
        FileNotFoundError: If input files don't exist
        ffmpeg.Error: If merge fails

    Example:
        >>> result = merge_with_sync_validation(
        ...     video_path=Path('data/temp/video_only.mp4'),
        ...     audio_path=Path('data/temp/dubbed_48k.wav'),
        ...     output_path=Path('data/outputs/final_dubbed.mp4')
        ... )
        >>> print(f'Merged in {result.merge_duration:.2f}s')
    """
    start_time = time.time()

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Validate inputs
    logger.info(f"Merging video {video_path.name} with audio {audio_path.name}")

    is_compatible, warnings = validate_audio_video_compatibility(video_path, audio_path)
    if not is_compatible:
        raise ValueError(f"Incompatible inputs: {'; '.join(warnings)}")

    for warning in warnings:
        logger.warning(warning)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get codec configuration for output format
    output_format = output_path.suffix
    config = get_merge_config(output_format)
    logger.debug(f"Using codec config for {output_format}: {config}")

    # Build FFmpeg pipeline
    video_input = ffmpeg.input(str(video_path))
    audio_input = ffmpeg.input(str(audio_path))

    # Build audio filter chain
    audio_stream = audio_input.audio

    if async_correction:
        # aresample with async=1 replaces deprecated -async flag
        # Corrects for clock drift between audio and video
        # Research: This is the modern approach (FFmpeg 4.0+)
        # Note: Use **{} to pass 'async' parameter (Python keyword)
        audio_stream = audio_stream.filter('aresample', **{'async': 1})
        logger.debug("Applying aresample async=1 for drift correction")

    # Explicit stream mapping:
    # - video_input.video: First video stream from video file
    # - audio_stream: Audio with optional drift correction
    output_kwargs = {
        'vcodec': config['vcodec'],
        'acodec': config['acodec'],
        'map_metadata': 0,  # Copy metadata from first input
    }

    # Add audio bitrate if specified in config
    if 'audio_bitrate' in config:
        output_kwargs['audio_bitrate'] = config['audio_bitrate']

    # Add MP4-specific optimization
    if output_format.lower() in ['.mp4', 'mp4']:
        output_kwargs['movflags'] = 'faststart'  # Enable streaming playback

    stream = ffmpeg.output(
        video_input.video,
        audio_stream,
        str(output_path),
        **output_kwargs
    )

    # Run merge
    logger.info(f"Running FFmpeg merge to {output_path.name}...")
    try:
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg merge failed: {error_msg}")
        raise

    merge_duration = time.time() - start_time

    logger.info(
        f"Merge complete: {output_path.name} "
        f"({merge_duration:.2f}s, async_correction={async_correction})"
    )

    return MergeResult(
        output_path=output_path,
        video_codec=config['vcodec'],
        audio_codec=config['acodec'],
        async_correction=async_correction,
        merge_duration=merge_duration
    )
