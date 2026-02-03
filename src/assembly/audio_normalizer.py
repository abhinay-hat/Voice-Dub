"""
Audio sample rate normalization for drift-free audio-video assembly.

Normalizes all audio segments to 48kHz (DVD/broadcast standard) using high-quality
kaiser_best sinc interpolation. Research shows this provides best quality for
video production workflows.

Example:
    >>> from pathlib import Path
    >>> from src.assembly.audio_normalizer import normalize_sample_rate
    >>> normalized_path = normalize_sample_rate(Path('segment_22k.wav'))
    >>> # Creates segment_22k_48k.wav at 48kHz
"""

import logging
from pathlib import Path
from typing import List, Tuple
import librosa
import soundfile as sf
import numpy as np

from src.config.settings import ASSEMBLY_TARGET_SAMPLE_RATE, ASSEMBLY_RESAMPLING_QUALITY

logger = logging.getLogger(__name__)


def normalize_sample_rate(
    audio_path: Path,
    target_sr: int = ASSEMBLY_TARGET_SAMPLE_RATE,
    quality: str = ASSEMBLY_RESAMPLING_QUALITY
) -> Path:
    """Normalize audio file to target sample rate using high-quality resampling.

    Loads audio at its native sample rate, resamples if needed using kaiser_best
    sinc interpolation, and writes to a new file. If already at target sample rate,
    returns the original path without processing.

    Args:
        audio_path: Path to input audio file
        target_sr: Target sample rate in Hz (default: 48000)
        quality: Resampling quality - 'kaiser_best', 'kaiser_fast', 'scipy', etc.
                 (default: 'kaiser_best' from settings)

    Returns:
        Path to normalized audio file (may be original if already at target_sr)

    Example:
        >>> audio_path = Path('data/temp/segment_S1_22050.wav')
        >>> normalized = normalize_sample_rate(audio_path)
        >>> # Returns Path('data/temp/segment_S1_22050_48k.wav') at 48kHz
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio at native sample rate
    audio, original_sr = librosa.load(audio_path, sr=None, mono=True)

    # Check if already at target sample rate
    if original_sr == target_sr:
        logger.debug(f"Audio already at {target_sr}Hz: {audio_path.name}")
        return audio_path

    # Resample using high-quality sinc interpolation
    logger.info(f"Resampling {audio_path.name}: {original_sr}Hz -> {target_sr}Hz (quality: {quality})")
    audio_resampled = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type=quality
    )

    # Write normalized audio to new file
    output_path = audio_path.parent / f"{audio_path.stem}_48k.wav"
    sf.write(
        output_path,
        audio_resampled,
        target_sr,
        subtype='PCM_16'  # 16-bit PCM for broad compatibility
    )

    logger.debug(f"Written normalized audio: {output_path.name} ({len(audio_resampled)/target_sr:.2f}s)")
    return output_path


def validate_sample_rate(
    audio_path: Path,
    expected_sr: int = ASSEMBLY_TARGET_SAMPLE_RATE
) -> Tuple[bool, int]:
    """Validate that audio file matches expected sample rate.

    Uses soundfile.info() to efficiently check sample rate without loading
    the entire audio file into memory.

    Args:
        audio_path: Path to audio file
        expected_sr: Expected sample rate in Hz (default: 48000)

    Returns:
        Tuple of (matches_expected: bool, actual_sample_rate: int)

    Example:
        >>> audio_path = Path('segment_48k.wav')
        >>> matches, actual_sr = validate_sample_rate(audio_path, 48000)
        >>> print(f"Matches: {matches}, Actual: {actual_sr}Hz")
        Matches: True, Actual: 48000Hz
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    info = sf.info(audio_path)
    actual_sr = info.samplerate
    matches = (actual_sr == expected_sr)

    if not matches:
        logger.debug(
            f"Sample rate mismatch for {audio_path.name}: "
            f"expected {expected_sr}Hz, got {actual_sr}Hz"
        )

    return matches, actual_sr


def batch_normalize(
    audio_paths: List[Path],
    target_sr: int = ASSEMBLY_TARGET_SAMPLE_RATE,
    quality: str = ASSEMBLY_RESAMPLING_QUALITY
) -> List[Path]:
    """Normalize multiple audio files to target sample rate.

    Processes a batch of audio files, resampling each to the target sample rate.
    Handles mixed sample rates in the batch - only resamples files that need it.

    Args:
        audio_paths: List of paths to audio files
        target_sr: Target sample rate in Hz (default: 48000)
        quality: Resampling quality (default: 'kaiser_best' from settings)

    Returns:
        List of paths to normalized audio files (in same order as input)

    Example:
        >>> paths = [Path('seg1_22k.wav'), Path('seg2_16k.wav'), Path('seg3_48k.wav')]
        >>> normalized = batch_normalize(paths)
        >>> # Returns [Path('seg1_22k_48k.wav'), Path('seg2_16k_48k.wav'), Path('seg3_48k.wav')]
    """
    if not audio_paths:
        logger.warning("Empty audio_paths list provided to batch_normalize")
        return []

    normalized_paths = []
    total = len(audio_paths)

    logger.info(f"Batch normalizing {total} audio files to {target_sr}Hz")

    for i, audio_path in enumerate(audio_paths, 1):
        try:
            normalized_path = normalize_sample_rate(audio_path, target_sr, quality)
            normalized_paths.append(normalized_path)

            if i % 10 == 0 or i == total:
                logger.debug(f"Progress: {i}/{total} files normalized")

        except Exception as e:
            logger.error(f"Failed to normalize {audio_path.name}: {e}")
            raise

    logger.info(f"Batch normalization complete: {total} files processed")
    return normalized_paths
