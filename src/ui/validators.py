"""
Stage output validators for Voice Dub quality controls (QC-04).

These are pure functions that inspect pipeline stage results and return a
(bool, str) tuple: (success, human-readable message).

Used by the Gradio UI to show status after each pipeline stage and decide
whether to allow the user to proceed.

All functions work with actual stage result dataclasses from their respective
stage modules, but accept any object with matching attributes (enabling
mock-based testing without GPU hardware).

Import strategy: Stage result types are imported under TYPE_CHECKING only.
This prevents cascading ML library imports (pyannote, torch, TTS, etc.) when
validators are loaded in a test environment or lightweight UI context. The
functions use duck typing (getattr with defaults) so they work on any object
with matching attributes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # Only imported during static analysis (mypy, pyright) — never at runtime.
    # This avoids pulling in pyannote, torch, TTS, etc. just to call validators.
    from src.stages.asr_stage import ASRResult
    from src.stages.translation_stage import TranslationResult
    from src.stages.tts_stage import TTSResult
    from src.stages.lip_sync_stage import LipSyncResult

from src.config.settings import ASR_CONFIDENCE_THRESHOLD

# Confidence threshold for ASR low-quality flag (0.7)
_ASR_LOW_CONFIDENCE_THRESHOLD = ASR_CONFIDENCE_THRESHOLD


def validate_asr_output(result: "ASRResult | None") -> Tuple[bool, str]:
    """
    Validate ASR stage output before proceeding to translation.

    Args:
        result: ASRResult from run_asr_stage(), or None if stage failed.

    Returns:
        (True, summary_msg)  — proceed to translation
        (False, error_msg)   — stage must be re-run or input changed

    Spec:
        - None        -> (False, "ASR stage did not return a result.")
        - 0 segments  -> (False, "No speech segments detected in video.")
        - >50% low    -> (True, "... Warning: X/N segments have low confidence ...")
        - otherwise   -> (True, "ASR complete: N segments, M speaker(s).")
    """
    if result is None:
        return False, "ASR stage did not return a result."

    if not result.segments:
        return False, "No speech segments detected in video."

    n = len(result.segments)
    low_confidence = sum(
        1 for s in result.segments
        if getattr(s, "confidence", 1.0) < _ASR_LOW_CONFIDENCE_THRESHOLD
    )

    # Count unique speakers
    speakers = {getattr(s, "speaker", "UNKNOWN") for s in result.segments}
    speaker_count = len(speakers)

    if low_confidence / n > 0.5:
        return True, (
            f"ASR complete: {n} segments, {speaker_count} speaker(s). "
            f"Warning: {low_confidence}/{n} segments have low confidence "
            f"— review flagged rows."
        )

    return True, f"ASR complete: {n} segments, {speaker_count} speaker(s)."


def validate_translation_output(result: "TranslationResult | None") -> Tuple[bool, str]:
    """
    Validate translation stage output before proceeding to TTS.

    Args:
        result: TranslationResult from run_translation_stage(), or None.

    Returns:
        (True, summary_msg)  — proceed to TTS
        (False, error_msg)   — stage must be re-run or source changed

    Spec:
        - None                     -> (False, "Translation stage did not return a result.")
        - 0 segments               -> (False, "No translated segments in result.")
        - >20% invalid duration    -> (False, "Translation failed: X/N segments exceed ...")
        - >0 but ≤20% invalid      -> (True, "... Warning: X segments exceed timing tolerance.")
        - all valid                -> (True, "Translation complete: N segments translated.")
    """
    if result is None:
        return False, "Translation stage did not return a result."

    if not result.segments:
        return False, "No translated segments in result."

    n = len(result.segments)
    invalid = sum(
        1 for s in result.segments
        if not getattr(s, "is_valid_duration", True)
    )

    if invalid / n > 0.2:
        return False, (
            f"Translation failed: {invalid}/{n} segments exceed timing tolerance. "
            "Processing stopped."
        )

    if invalid > 0:
        return True, (
            f"Translation complete: {n} segments translated. "
            f"Warning: {invalid} segments exceed timing tolerance."
        )

    return True, f"Translation complete: {n} segments translated."


def validate_tts_output(result: "TTSResult | None") -> Tuple[bool, str]:
    """
    Validate TTS stage output before assembly.

    Args:
        result: TTSResult from run_tts_stage(), or None.

    Returns:
        (True, summary_msg)  — proceed to assembly
        (False, error_msg)   — stage must be re-run or parameters adjusted

    Spec:
        - None                              -> (False, "TTS stage did not return a result.")
        - total_segments == 0               -> (False, "TTS stage produced no segments.")
        - failed_segments / total > 0.5     -> (False, "Voice cloning failed: >50% ...")
        - emotion_flagged_count > 0         -> (True, "... Warning: X segments flagged ...")
        - all passed                        -> (True, "Voice cloning complete: N segments synthesized.")

    Note: TTSResult.failed_segments is an int (count of failed segments),
    not a list. TTSResult.emotion_flagged_count tracks emotion preservation issues.
    """
    if result is None:
        return False, "TTS stage did not return a result."

    total = getattr(result, "total_segments", 0)
    if total == 0:
        return False, "TTS stage produced no segments."

    # failed_segments is an int count on TTSResult (not a list)
    failed = getattr(result, "failed_segments", 0)

    if failed / total > 0.5:
        return False, (
            f"Voice cloning failed: >50% of segments rejected ({failed}/{total}). "
            "Processing stopped."
        )

    # emotion_flagged_count tracks pitch variance issues (Phase 5 decision)
    emotion_flagged = getattr(result, "emotion_flagged_count", 0)
    if emotion_flagged > 0:
        return True, (
            f"Voice cloning complete: {total} segments. "
            f"Warning: {emotion_flagged} segments flagged for review."
        )

    return True, f"Voice cloning complete: {total} segments synthesized."


def validate_lip_sync_output(result: "LipSyncResult | None") -> Tuple[bool, str]:
    """
    Validate lip sync stage output.

    Args:
        result: LipSyncResult from run_lip_sync_stage(), or None.

    Returns:
        (True, summary_msg)  — lip sync succeeded
        (False, error_msg)   — sync validation failed

    Spec:
        - None                                  -> (False, "Lip sync stage did not return a result.")
        - sync_validation.passed is False       -> (False, "Lip sync validation failed: X.X% frames passed.")
        - fallback_used=True                    -> (True, "Lip sync complete (Wav2Lip fallback used). ...")
        - normal success                        -> (True, "Lip sync complete: {model} in {time:.1f}s.")

    Note: SyncValidation has no .message field. The failure message is derived
    from sync_validation.pass_rate (float 0.0-1.0) formatted as percentage.
    When sync_validation is None (validator skipped), the stage is treated as
    success (advisory-only design from Phase 7).
    """
    if result is None:
        return False, "Lip sync stage did not return a result."

    sync_val = getattr(result, "sync_validation", None)
    if sync_val is not None and getattr(sync_val, "passed", True) is False:
        pass_rate = getattr(sync_val, "pass_rate", 0.0)
        return False, f"Lip sync validation failed: {pass_rate:.1%} frames passed."

    if getattr(result, "fallback_used", False):
        return True, "Lip sync complete (Wav2Lip fallback used). Review output quality."

    model = getattr(result, "model_used", "Unknown")
    duration = getattr(result, "processing_time", 0.0)
    return True, f"Lip sync complete: {model} in {duration:.1f}s."
