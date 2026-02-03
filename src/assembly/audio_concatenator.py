"""
Audio segment concatenation with gap padding for drift-free assembly.

Concatenates audio segments with proper gap handling to maintain precise timing
alignment with video. Inserts silence padding for gaps between segments to ensure
the final audio matches expected timestamps.

Example:
    >>> from pathlib import Path
    >>> from src.assembly.timestamp_validator import TimedSegment
    >>> from src.assembly.audio_concatenator import concatenate_audio_segments
    >>> segments = [
    ...     TimedSegment(0.0, 5.5, 'seg1.wav', 'S1'),
    ...     TimedSegment(5.5, 12.3, 'seg2.wav', 'S2'),
    ... ]
    >>> output = concatenate_audio_segments(segments, Path('final_audio.wav'))
"""

import logging
from pathlib import Path
from typing import List
import numpy as np
import librosa
import soundfile as sf

from src.assembly.timestamp_validator import TimedSegment, ensure_float64
from src.assembly.audio_normalizer import validate_sample_rate
from src.config.settings import ASSEMBLY_TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


def concatenate_audio_segments(
    segments: List[TimedSegment],
    output_path: Path,
    target_sr: int = ASSEMBLY_TARGET_SAMPLE_RATE
) -> Path:
    """Concatenate audio segments with gap padding into single audio file.

    Loads each segment, validates sample rates match, handles gaps between segments
    by inserting silence padding, and writes the concatenated result. This ensures
    the final audio timeline matches the expected timestamps for drift-free sync.

    Args:
        segments: List of TimedSegment objects (already normalized to target_sr)
        output_path: Path where concatenated audio will be written
        target_sr: Expected sample rate for all segments (default: 48000)

    Returns:
        Path to concatenated audio file

    Raises:
        ValueError: If segments have mismatched sample rates or invalid ordering
        FileNotFoundError: If any segment audio file doesn't exist

    Example:
        >>> segments = [
        ...     TimedSegment(0.0, 5.5, 'data/temp/seg1_48k.wav', 'S1'),
        ...     TimedSegment(5.5, 12.3, 'data/temp/seg2_48k.wav', 'S2'),
        ... ]
        >>> output = concatenate_audio_segments(segments, Path('data/outputs/final.wav'))
    """
    if not segments:
        raise ValueError("Empty segments list provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Concatenating {len(segments)} audio segments to {output_path.name}")

    # Validate all segments have same sample rate
    logger.debug("Validating sample rates...")
    for i, seg in enumerate(segments):
        audio_path = Path(seg.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Segment {i} audio not found: {audio_path}")

        matches, actual_sr = validate_sample_rate(audio_path, target_sr)
        if not matches:
            raise ValueError(
                f"Segment {i} has sample rate {actual_sr}Hz, expected {target_sr}Hz. "
                f"Run normalize_sample_rate() first."
            )

    # Build concatenated audio with gap handling
    concatenated_audio = []
    current_time = 0.0  # Track current position in output timeline

    for i, seg in enumerate(segments):
        seg_start = ensure_float64(seg.start)
        seg_end = ensure_float64(seg.end)

        # Check for gap from current position to segment start
        gap = seg_start - current_time
        if gap > 1e-6:  # More than 1 microsecond (floating point tolerance)
            # Insert silence padding for the gap
            gap_samples = int(np.round(gap * target_sr))
            silence = np.zeros(gap_samples, dtype=np.float32)
            concatenated_audio.append(silence)
            logger.debug(f"Inserted {gap:.3f}s silence before segment {i}")
        elif gap < -1e-6:  # Negative gap = overlap
            logger.warning(
                f"Segment {i} overlaps previous segment by {abs(gap):.3f}s. "
                f"This may cause audio artifacts."
            )

        # Load and append segment audio
        audio, loaded_sr = librosa.load(seg.audio_path, sr=target_sr, mono=True)
        concatenated_audio.append(audio)

        # Update current position
        current_time = seg_end
        logger.debug(f"Added segment {i}: {seg.audio_path} ({seg.duration:.3f}s)")

    # Concatenate all audio chunks
    if not concatenated_audio:
        raise ValueError("No audio data to concatenate")

    final_audio = np.concatenate(concatenated_audio)

    # Write concatenated audio
    sf.write(
        output_path,
        final_audio,
        target_sr,
        subtype='PCM_16'
    )

    duration = len(final_audio) / target_sr
    logger.info(
        f"Concatenation complete: {output_path.name} "
        f"({duration:.2f}s, {len(segments)} segments)"
    )

    return output_path


def get_total_duration(segments: List[TimedSegment]) -> np.float64:
    """Calculate total duration of all segments.

    Returns the end time of the last segment, which represents the total
    duration of the concatenated audio timeline.

    Args:
        segments: List of TimedSegment objects

    Returns:
        Total duration in seconds as np.float64

    Example:
        >>> segments = [
        ...     TimedSegment(0.0, 5.5, 'seg1.wav', 'S1'),
        ...     TimedSegment(5.5, 12.3, 'seg2.wav', 'S2'),
        ... ]
        >>> duration = get_total_duration(segments)
        >>> print(f"Total: {duration:.1f}s")
        Total: 12.3s
    """
    if not segments:
        return np.float64(0.0)

    # Find the maximum end time across all segments
    max_end = max(ensure_float64(seg.end) for seg in segments)
    return max_end
