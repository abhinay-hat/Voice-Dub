"""
Audio quality validation for TTS outputs using PESQ and STOI metrics.

This module provides quality assessment for generated speech audio to ensure
adequate clarity, intelligibility, and timing before proceeding to lip sync.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from ..config.settings import (
    TTS_MIN_PESQ_SCORE,
    TTS_PESQ_REVIEW_THRESHOLD,
    TTS_DURATION_TOLERANCE
)


@dataclass
class AudioQualityResult:
    """Quality validation result for a single audio segment."""
    segment_id: int
    audio_path: str

    # Duration metrics
    target_duration: float
    actual_duration: float
    duration_error: float  # Percentage error
    duration_valid: bool  # Within tolerance

    # PESQ score (1.0-5.0, higher = better)
    pesq_score: Optional[float]  # None if calculation failed
    pesq_quality: str  # "excellent", "good", "fair", "poor"

    # STOI score (0.0-1.0, higher = more intelligible)
    stoi_score: Optional[float]  # None if calculation failed

    # Emotion preservation proxy (pitch variance ratio)
    pitch_variance_ratio: Optional[float]  # Ratio of generated/reference pitch variance

    # Overall assessment
    passes_quality: bool  # True if acceptable for lip sync
    flagged_for_review: bool  # True if marginal quality
    rejection_reason: Optional[str]  # Why it failed (if applicable)


class QualityValidator:
    """Validates TTS audio quality using PESQ and STOI metrics."""

    def __init__(
        self,
        min_pesq: float = TTS_MIN_PESQ_SCORE,
        review_pesq: float = TTS_PESQ_REVIEW_THRESHOLD,
        duration_tolerance: float = TTS_DURATION_TOLERANCE
    ):
        """
        Initialize validator with quality thresholds.

        Args:
            min_pesq: Minimum PESQ score to pass (default 2.5)
            review_pesq: Flag for review if below (default 3.0)
            duration_tolerance: Acceptable duration error (default 0.05 = 5%)
        """
        self.min_pesq = min_pesq
        self.review_pesq = review_pesq
        self.duration_tolerance = duration_tolerance

    def validate_single(
        self,
        audio_path: Path,
        reference_path: Optional[Path],
        segment_id: int,
        target_duration: float
    ) -> AudioQualityResult:
        """
        Validate a single audio segment with full quality metrics.

        Args:
            audio_path: Path to generated audio file
            reference_path: Path to reference audio (speaker sample) for comparison
            segment_id: Segment identifier
            target_duration: Expected duration in seconds

        Returns:
            AudioQualityResult with all quality metrics
        """
        # Load generated audio
        generated_audio, sr = load_audio_for_validation(audio_path, target_sr=16000)
        actual_duration = len(generated_audio) / sr

        # Calculate duration metrics
        duration_error = abs(actual_duration - target_duration) / target_duration
        duration_valid = duration_error <= self.duration_tolerance

        # Initialize quality metrics
        pesq_score = None
        stoi_score = None
        pitch_variance_ratio = None
        pesq_quality = "unknown"
        passes_quality = False
        flagged_for_review = False
        rejection_reason = None

        # Check if audio is silent
        if flag_silent_audio(generated_audio):
            rejection_reason = "Audio is silent or too quiet"
            pesq_quality = "poor"
            return AudioQualityResult(
                segment_id=segment_id,
                audio_path=str(audio_path),
                target_duration=target_duration,
                actual_duration=actual_duration,
                duration_error=duration_error,
                duration_valid=duration_valid,
                pesq_score=pesq_score,
                pesq_quality=pesq_quality,
                stoi_score=stoi_score,
                pitch_variance_ratio=pitch_variance_ratio,
                passes_quality=False,
                flagged_for_review=False,
                rejection_reason=rejection_reason
            )

        # If reference provided, calculate PESQ, STOI, and emotion preservation
        if reference_path and reference_path.exists():
            reference_audio, _ = load_audio_for_validation(reference_path, target_sr=16000)

            # Ensure same length for comparison (pad/trim)
            min_length = min(len(generated_audio), len(reference_audio))
            generated_audio_cmp = generated_audio[:min_length]
            reference_audio_cmp = reference_audio[:min_length]

            # Calculate PESQ (wideband mode for 16kHz)
            try:
                pesq_score = pesq(16000, reference_audio_cmp, generated_audio_cmp, 'wb')
                pesq_quality = self._classify_pesq(pesq_score)
            except Exception as e:
                warnings.warn(f"PESQ calculation failed: {e}")
                pesq_score = None
                pesq_quality = "unknown"

            # Calculate STOI
            try:
                stoi_score = stoi(reference_audio_cmp, generated_audio_cmp, 16000)
            except Exception as e:
                warnings.warn(f"STOI calculation failed: {e}")
                stoi_score = None

            # Calculate emotion preservation
            pitch_variance_ratio = validate_emotion_preservation(
                generated_audio, reference_audio, sample_rate=16000
            )

        # Determine overall quality assessment
        if pesq_score is not None:
            if pesq_score < self.min_pesq:
                passes_quality = False
                rejection_reason = f"PESQ score {pesq_score:.2f} below minimum {self.min_pesq}"
            elif pesq_score < self.review_pesq:
                passes_quality = True
                flagged_for_review = True
            else:
                passes_quality = True
        else:
            # No PESQ available, use duration only
            passes_quality = duration_valid
            if not duration_valid:
                rejection_reason = f"Duration error {duration_error*100:.1f}% exceeds {self.duration_tolerance*100:.0f}%"

        # Check emotion preservation
        if pitch_variance_ratio is not None:
            if pitch_variance_ratio < 0.6 or pitch_variance_ratio > 1.5:
                flagged_for_review = True
                if not rejection_reason:
                    rejection_reason = f"Emotion preservation marginal (pitch ratio {pitch_variance_ratio:.2f})"

        return AudioQualityResult(
            segment_id=segment_id,
            audio_path=str(audio_path),
            target_duration=target_duration,
            actual_duration=actual_duration,
            duration_error=duration_error,
            duration_valid=duration_valid,
            pesq_score=pesq_score,
            pesq_quality=pesq_quality,
            stoi_score=stoi_score,
            pitch_variance_ratio=pitch_variance_ratio,
            passes_quality=passes_quality,
            flagged_for_review=flagged_for_review,
            rejection_reason=rejection_reason
        )

    def validate_duration_only(
        self,
        audio_path: Path,
        segment_id: int,
        target_duration: float
    ) -> AudioQualityResult:
        """
        Simplified validation when no reference available - duration check only.

        Args:
            audio_path: Path to generated audio file
            segment_id: Segment identifier
            target_duration: Expected duration in seconds

        Returns:
            AudioQualityResult with duration metrics only
        """
        # Get actual duration
        actual_duration = get_audio_duration(audio_path)

        # Calculate duration metrics
        duration_error = abs(actual_duration - target_duration) / target_duration
        duration_valid = duration_error <= self.duration_tolerance

        # Load audio to check for silence
        generated_audio, _ = load_audio_for_validation(audio_path, target_sr=16000)
        is_silent = flag_silent_audio(generated_audio)

        rejection_reason = None
        if not duration_valid:
            rejection_reason = f"Duration error {duration_error*100:.1f}% exceeds {self.duration_tolerance*100:.0f}%"
        elif is_silent:
            rejection_reason = "Audio is silent or too quiet"

        passes_quality = duration_valid and not is_silent

        return AudioQualityResult(
            segment_id=segment_id,
            audio_path=str(audio_path),
            target_duration=target_duration,
            actual_duration=actual_duration,
            duration_error=duration_error,
            duration_valid=duration_valid,
            pesq_score=None,
            pesq_quality="unknown",
            stoi_score=None,
            pitch_variance_ratio=None,
            passes_quality=passes_quality,
            flagged_for_review=False,
            rejection_reason=rejection_reason
        )

    def validate_batch(
        self,
        synthesis_results: list[dict],
        reference_dir: Optional[Path] = None
    ) -> tuple[list[AudioQualityResult], dict]:
        """
        Validate all synthesized segments in batch.

        Args:
            synthesis_results: List of dicts with keys: segment_id, audio_path,
                               target_duration, speaker_id
            reference_dir: Directory containing reference_{speaker_id}.wav files

        Returns:
            Tuple of (results_list, summary_dict)
            summary_dict contains: total, passed, flagged, rejected, emotion_flags
        """
        results = []

        for result in synthesis_results:
            segment_id = result['segment_id']
            audio_path = Path(result['audio_path'])
            target_duration = result['target_duration']
            speaker_id = result.get('speaker_id')

            # Look for reference file if directory provided
            reference_path = None
            if reference_dir and speaker_id is not None:
                reference_path = reference_dir / f"reference_{speaker_id}.wav"
                if not reference_path.exists():
                    reference_path = None

            # Validate with or without reference
            if reference_path:
                quality_result = self.validate_single(
                    audio_path, reference_path, segment_id, target_duration
                )
            else:
                quality_result = self.validate_duration_only(
                    audio_path, segment_id, target_duration
                )

            results.append(quality_result)

        # Generate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.passes_quality and not r.flagged_for_review)
        flagged = sum(1 for r in results if r.flagged_for_review)
        rejected = sum(1 for r in results if not r.passes_quality)
        emotion_flags = sum(
            1 for r in results
            if r.pitch_variance_ratio is not None
            and (r.pitch_variance_ratio < 0.6 or r.pitch_variance_ratio > 1.5)
        )

        summary = {
            'total': total,
            'passed': passed,
            'flagged': flagged,
            'rejected': rejected,
            'emotion_flags': emotion_flags,
            'pass_rate': passed / total if total > 0 else 0.0
        }

        return results, summary

    def _classify_pesq(self, score: float) -> str:
        """Classify PESQ score into quality tier."""
        if score >= 4.0:
            return "excellent"
        elif score >= 3.0:
            return "good"
        elif score >= 2.5:
            return "fair"
        else:
            return "poor"


# Audio file utilities

def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or 0.0 if file unreadable
    """
    try:
        return librosa.get_duration(path=str(audio_path))
    except Exception as e:
        warnings.warn(f"Could not read audio duration from {audio_path}: {e}")
        return 0.0


def load_audio_for_validation(audio_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio at specified sample rate for validation.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 16000 for PESQ/STOI)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load and resample to target sample rate
    audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

    # Normalize amplitude to [-1, 1] range
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio, sr


def calculate_rms_energy(audio: np.ndarray) -> float:
    """
    Calculate RMS energy of audio signal.

    Args:
        audio: Audio signal as numpy array

    Returns:
        RMS energy value
    """
    return np.sqrt(np.mean(audio ** 2))


def flag_silent_audio(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if audio is mostly silent.

    Args:
        audio: Audio signal as numpy array
        threshold: RMS threshold below which audio is considered silent

    Returns:
        True if audio is silent or too quiet
    """
    rms = calculate_rms_energy(audio)
    return rms < threshold


def extract_pitch_contour(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract F0 (fundamental frequency) contour from audio.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Audio sample rate

    Returns:
        Pitch contour as numpy array (contains NaN for unvoiced regions)
    """
    # Use pyin for pitch tracking (more robust than piptrack)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sample_rate
    )

    return f0


def validate_emotion_preservation(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int = 16000
) -> Optional[float]:
    """
    Validate emotion preservation by comparing pitch dynamics.

    This uses pitch variance ratio as a proxy for emotional expression preservation.
    A ratio close to 1.0 indicates similar pitch dynamics (emotion preserved).

    Args:
        generated_audio: Generated TTS audio
        reference_audio: Reference speaker sample
        sample_rate: Audio sample rate

    Returns:
        Pitch variance ratio (generated/reference), or None if extraction failed
    """
    try:
        # Extract pitch contours
        gen_f0 = extract_pitch_contour(generated_audio, sample_rate)
        ref_f0 = extract_pitch_contour(reference_audio, sample_rate)

        # Calculate variance for voiced regions only (exclude NaN)
        gen_f0_voiced = gen_f0[~np.isnan(gen_f0)]
        ref_f0_voiced = ref_f0[~np.isnan(ref_f0)]

        # Need sufficient voiced frames for meaningful comparison
        if len(gen_f0_voiced) < 10 or len(ref_f0_voiced) < 10:
            warnings.warn("Insufficient voiced frames for emotion preservation check")
            return None

        gen_variance = np.var(gen_f0_voiced)
        ref_variance = np.var(ref_f0_voiced)

        # Avoid division by zero
        if ref_variance < 1e-6:
            warnings.warn("Reference audio has near-zero pitch variance")
            return None

        ratio = gen_variance / ref_variance
        return ratio

    except Exception as e:
        warnings.warn(f"Emotion preservation check failed: {e}")
        return None


# Convenience alias for single validation
def validate_audio_quality(
    audio_path: Path,
    reference_path: Optional[Path],
    segment_id: int,
    target_duration: float,
    min_pesq: float = TTS_MIN_PESQ_SCORE,
    review_pesq: float = TTS_PESQ_REVIEW_THRESHOLD,
    duration_tolerance: float = TTS_DURATION_TOLERANCE
) -> AudioQualityResult:
    """
    Convenience function for single audio quality validation.

    Args:
        audio_path: Path to generated audio file
        reference_path: Path to reference audio (speaker sample)
        segment_id: Segment identifier
        target_duration: Expected duration in seconds
        min_pesq: Minimum acceptable PESQ score
        review_pesq: Flag for review threshold
        duration_tolerance: Acceptable duration error

    Returns:
        AudioQualityResult with quality metrics
    """
    validator = QualityValidator(min_pesq, review_pesq, duration_tolerance)
    return validator.validate_single(audio_path, reference_path, segment_id, target_duration)
