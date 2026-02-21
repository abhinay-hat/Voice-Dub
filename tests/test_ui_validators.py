"""
TDD test suite for stage validators and clip preview extractor.

Tests written BEFORE implementation (RED phase) following the structure of
actual stage result dataclasses from src/stages/*.py.

Real field structures (confirmed from source):
  - ASRResult.segments: List[AlignedSegment], each with .confidence (float), .speaker (str)
  - TranslationResult.segments: List[TranslatedSegment], each with .is_valid_duration (bool)
  - TTSResult.failed_segments: int (count, not list), .segments: List[SynthesizedSegment]
    each SynthesizedSegment has .flagged_for_review (bool)
  - LipSyncResult: .sync_validation (Optional[SyncValidation]), .fallback_used (bool),
    .model_used (str), .processing_time (float)
  - SyncValidation: .passed (bool), .pass_rate (float)
"""
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper factories for mock stage results
# ---------------------------------------------------------------------------

def make_asr_segment(confidence=0.9, speaker="S1", text="hello"):
    """Create mock AlignedSegment matching actual dataclass fields."""
    return SimpleNamespace(confidence=confidence, speaker=speaker, text=text)


def make_asr_result(segments):
    """Create mock ASRResult matching actual dataclass fields."""
    return SimpleNamespace(segments=segments)


def make_translation_segment(is_valid_duration=True):
    """Create mock TranslatedSegment matching actual dataclass fields."""
    return SimpleNamespace(is_valid_duration=is_valid_duration)


def make_translation_result(segments):
    """Create mock TranslationResult matching actual dataclass fields."""
    return SimpleNamespace(segments=segments)


def make_tts_segment(flagged_for_review=False, quality_passed=True):
    """Create mock SynthesizedSegment matching actual dataclass fields."""
    return SimpleNamespace(flagged_for_review=flagged_for_review, quality_passed=quality_passed)


def make_tts_result(total_segments=3, failed_segments=0, flagged_count=0,
                    emotion_flagged_count=0, segments=None):
    """Create mock TTSResult matching actual dataclass fields.

    NOTE: TTSResult.failed_segments is an int (count), NOT a list.
    """
    if segments is None:
        segments = [make_tts_segment() for _ in range(total_segments)]
    return SimpleNamespace(
        total_segments=total_segments,
        failed_segments=failed_segments,
        flagged_count=flagged_count,
        emotion_flagged_count=emotion_flagged_count,
        segments=segments,
    )


def make_sync_validation(passed=True, pass_rate=0.98):
    """Create mock SyncValidation matching actual dataclass fields."""
    return SimpleNamespace(passed=passed, pass_rate=pass_rate)


def make_lip_sync_result(sync_validation=None, fallback_used=False,
                         model_used="latentsync", processing_time=120.0,
                         output_path="/out.mp4"):
    """Create mock LipSyncResult matching actual dataclass fields."""
    return SimpleNamespace(
        sync_validation=sync_validation,
        fallback_used=fallback_used,
        model_used=model_used,
        processing_time=processing_time,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Tests: validate_asr_output
# ---------------------------------------------------------------------------

class TestValidateASROutput:
    """Tests for validate_asr_output() — covers all spec branches."""

    def test_none_input_returns_false(self):
        from src.ui.validators import validate_asr_output
        result, msg = validate_asr_output(None)
        assert result is False
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_empty_segments_returns_false(self):
        from src.ui.validators import validate_asr_output
        mock_result = make_asr_result(segments=[])
        result, msg = validate_asr_output(mock_result)
        assert result is False
        assert isinstance(msg, str)

    def test_all_high_confidence_returns_true(self):
        """All segments above 0.7 threshold → success with segment count in message."""
        from src.ui.validators import validate_asr_output
        segs = [make_asr_segment(confidence=0.95, speaker="S1"),
                make_asr_segment(confidence=0.92, speaker="S1")]
        mock_result = make_asr_result(segments=segs)
        result, msg = validate_asr_output(mock_result)
        assert result is True
        assert isinstance(msg, str)
        # Segment count must appear in the message
        assert "2" in msg

    def test_majority_low_confidence_returns_true_with_warning(self):
        """2 low (0.3), 1 high (0.9) = 66% low > 50% threshold → True + warning."""
        from src.ui.validators import validate_asr_output
        segs = [
            make_asr_segment(confidence=0.3, speaker="S1"),
            make_asr_segment(confidence=0.3, speaker="S1"),
            make_asr_segment(confidence=0.9, speaker="S1"),
        ]
        mock_result = make_asr_result(segments=segs)
        result, msg = validate_asr_output(mock_result)
        assert result is True
        assert "warning" in msg.lower() or "low confidence" in msg.lower()

    def test_exactly_50_percent_low_confidence_no_warning(self):
        """Exactly 50% low does NOT trigger warning (threshold is >50%)."""
        from src.ui.validators import validate_asr_output
        segs = [
            make_asr_segment(confidence=0.3, speaker="S1"),
            make_asr_segment(confidence=0.95, speaker="S1"),
        ]
        mock_result = make_asr_result(segments=segs)
        result, msg = validate_asr_output(mock_result)
        assert result is True

    def test_message_includes_speaker_count(self):
        """Success message should include speaker count (2 distinct speakers)."""
        from src.ui.validators import validate_asr_output
        segs = [
            make_asr_segment(confidence=0.9, speaker="SPEAKER_00"),
            make_asr_segment(confidence=0.85, speaker="SPEAKER_01"),
        ]
        mock_result = make_asr_result(segments=segs)
        result, msg = validate_asr_output(mock_result)
        assert result is True
        # Should mention 2 speakers
        assert "2" in msg


# ---------------------------------------------------------------------------
# Tests: validate_translation_output
# ---------------------------------------------------------------------------

class TestValidateTranslationOutput:
    """Tests for validate_translation_output() — covers all spec branches."""

    def test_none_input_returns_false(self):
        from src.ui.validators import validate_translation_output
        result, msg = validate_translation_output(None)
        assert result is False
        assert isinstance(msg, str)

    def test_no_segments_returns_false(self):
        from src.ui.validators import validate_translation_output
        mock = make_translation_result(segments=[])
        result, msg = validate_translation_output(mock)
        assert result is False

    def test_high_invalid_duration_returns_false(self):
        """3/5 = 60% invalid > 20% threshold → fail."""
        from src.ui.validators import validate_translation_output
        segs = [
            make_translation_segment(is_valid_duration=False),
            make_translation_segment(is_valid_duration=False),
            make_translation_segment(is_valid_duration=False),
            make_translation_segment(is_valid_duration=True),
            make_translation_segment(is_valid_duration=True),
        ]
        mock = make_translation_result(segments=segs)
        result, msg = validate_translation_output(mock)
        assert result is False
        # Should mention segment count info
        assert isinstance(msg, str)

    def test_all_valid_duration_returns_true(self):
        from src.ui.validators import validate_translation_output
        segs = [make_translation_segment(is_valid_duration=True) for _ in range(3)]
        mock = make_translation_result(segments=segs)
        result, msg = validate_translation_output(mock)
        assert result is True
        assert "3" in msg

    def test_some_invalid_within_threshold_returns_true_with_warning(self):
        """1/10 = 10% invalid ≤ 20% threshold → True + warning."""
        from src.ui.validators import validate_translation_output
        segs = [make_translation_segment(is_valid_duration=True) for _ in range(9)]
        segs.append(make_translation_segment(is_valid_duration=False))
        mock = make_translation_result(segments=segs)
        result, msg = validate_translation_output(mock)
        assert result is True
        assert "warning" in msg.lower() or "timing" in msg.lower()

    def test_exactly_20_percent_invalid_passes(self):
        """Exactly 20% (2/10) is not > 20%, so should still pass."""
        from src.ui.validators import validate_translation_output
        segs = [make_translation_segment(is_valid_duration=True) for _ in range(8)]
        segs += [make_translation_segment(is_valid_duration=False) for _ in range(2)]
        mock = make_translation_result(segments=segs)
        result, msg = validate_translation_output(mock)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: validate_tts_output
# ---------------------------------------------------------------------------

class TestValidateTTSOutput:
    """Tests for validate_tts_output() — covers all spec branches.

    TTSResult.failed_segments is an int (count), not a list.
    """

    def test_none_input_returns_false(self):
        from src.ui.validators import validate_tts_output
        result, msg = validate_tts_output(None)
        assert result is False
        assert isinstance(msg, str)

    def test_majority_failed_returns_false(self):
        """failed_segments=3, total_segments=5 → 60% > 50% → fail."""
        from src.ui.validators import validate_tts_output
        mock = make_tts_result(total_segments=5, failed_segments=3)
        result, msg = validate_tts_output(mock)
        assert result is False

    def test_exactly_50_percent_failed_passes(self):
        """failed_segments=2, total=4 → exactly 50% → not > 50% → pass."""
        from src.ui.validators import validate_tts_output
        mock = make_tts_result(total_segments=4, failed_segments=2)
        result, msg = validate_tts_output(mock)
        assert result is True

    def test_all_passed_returns_true(self):
        """No failures → True with synthesized count in message."""
        from src.ui.validators import validate_tts_output
        mock = make_tts_result(total_segments=3, failed_segments=0)
        result, msg = validate_tts_output(mock)
        assert result is True
        assert "3" in msg

    def test_some_flagged_returns_true_with_warning(self):
        """emotion_flagged_count > 0 → True + warning mentioning flagged count."""
        from src.ui.validators import validate_tts_output
        mock = make_tts_result(total_segments=5, failed_segments=0,
                               emotion_flagged_count=2)
        result, msg = validate_tts_output(mock)
        assert result is True
        assert "warning" in msg.lower() or "flagged" in msg.lower() or "2" in msg

    def test_zero_total_segments_returns_false(self):
        """No segments at all → fail."""
        from src.ui.validators import validate_tts_output
        mock = make_tts_result(total_segments=0, failed_segments=0, segments=[])
        result, msg = validate_tts_output(mock)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: validate_lip_sync_output
# ---------------------------------------------------------------------------

class TestValidateLipSyncOutput:
    """Tests for validate_lip_sync_output() — covers all spec branches."""

    def test_none_input_returns_false(self):
        from src.ui.validators import validate_lip_sync_output
        result, msg = validate_lip_sync_output(None)
        assert result is False
        assert isinstance(msg, str)

    def test_failed_sync_validation_returns_false(self):
        """sync_validation.passed=False → (False, msg with pass_rate %)."""
        from src.ui.validators import validate_lip_sync_output
        sync_val = make_sync_validation(passed=False, pass_rate=0.3)
        mock = make_lip_sync_result(sync_validation=sync_val)
        result, msg = validate_lip_sync_output(mock)
        assert result is False
        # Message must mention the pass_rate as percentage (30.0% = 0.3 * 100)
        assert "30.0%" in msg or "30%" in msg

    def test_valid_result_returns_true(self):
        """sync_validation.passed=True, fallback_used=False → success."""
        from src.ui.validators import validate_lip_sync_output
        sync_val = make_sync_validation(passed=True, pass_rate=0.98)
        mock = make_lip_sync_result(sync_validation=sync_val,
                                    model_used="latentsync",
                                    processing_time=120.0)
        result, msg = validate_lip_sync_output(mock)
        assert result is True
        assert isinstance(msg, str)

    def test_fallback_used_returns_true_with_note(self):
        """fallback_used=True → (True, msg mentioning fallback or wav2lip)."""
        from src.ui.validators import validate_lip_sync_output
        mock = make_lip_sync_result(sync_validation=None, fallback_used=True,
                                    model_used="wav2lip", processing_time=90.0)
        result, msg = validate_lip_sync_output(mock)
        assert result is True
        assert "fallback" in msg.lower() or "wav2lip" in msg.lower()

    def test_no_sync_validation_no_fallback_returns_true(self):
        """sync_validation=None, fallback_used=False → normal success."""
        from src.ui.validators import validate_lip_sync_output
        mock = make_lip_sync_result(sync_validation=None, fallback_used=False,
                                    model_used="latentsync", processing_time=45.0)
        result, msg = validate_lip_sync_output(mock)
        assert result is True

    def test_success_message_includes_model_and_duration(self):
        """Normal success message should include model name and duration."""
        from src.ui.validators import validate_lip_sync_output
        sync_val = make_sync_validation(passed=True, pass_rate=0.99)
        mock = make_lip_sync_result(sync_validation=sync_val,
                                    model_used="latentsync",
                                    processing_time=87.5)
        result, msg = validate_lip_sync_output(mock)
        assert result is True
        assert "latentsync" in msg.lower() or "87" in msg


# ---------------------------------------------------------------------------
# Tests: extract_preview_clip
# ---------------------------------------------------------------------------

class TestExtractPreviewClip:
    """Tests for extract_preview_clip() — uses subprocess mocking."""

    def test_missing_file_raises_value_error(self):
        """Non-existent video_path raises ValueError before calling ffmpeg."""
        from src.ui.clip_preview import extract_preview_clip
        with pytest.raises(ValueError, match="not found"):
            extract_preview_clip("/nonexistent/video.mp4", 0.0, "/tmp/out.mp4")

    def test_ffmpeg_failure_raises_runtime_error(self):
        """Non-zero ffmpeg return code raises RuntimeError."""
        from src.ui.clip_preview import extract_preview_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = b"ffmpeg: invalid option"
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(RuntimeError):
                    extract_preview_clip(tmp, 0.0, "/tmp/out.mp4")
        finally:
            os.unlink(tmp)

    def test_negative_start_clamped_to_zero(self):
        """start_seconds=-10.0 must be clamped to 0.0 in the ffmpeg command."""
        from src.ui.clip_preview import extract_preview_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            mock_result = MagicMock()
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                extract_preview_clip(tmp, -10.0, "/tmp/out.mp4")
                args = mock_run.call_args[0][0]
                ss_idx = args.index("-ss")
                assert float(args[ss_idx + 1]) >= 0.0, (
                    f"Expected -ss >= 0 but got {args[ss_idx + 1]}"
                )
        finally:
            os.unlink(tmp)

    def test_successful_clip_returns_output_path(self):
        """Successful ffmpeg call returns the output_path string."""
        from src.ui.clip_preview import extract_preview_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            mock_result = MagicMock()
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result):
                result = extract_preview_clip(tmp, 30.0, "/tmp/out.mp4")
                assert result == "/tmp/out.mp4"
        finally:
            os.unlink(tmp)

    def test_default_duration_is_30_seconds(self):
        """Default duration parameter is 30.0 seconds."""
        from src.ui.clip_preview import extract_preview_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            mock_result = MagicMock()
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                extract_preview_clip(tmp, 0.0, "/tmp/out.mp4")
                args = mock_run.call_args[0][0]
                t_idx = args.index("-t")
                assert float(args[t_idx + 1]) == 30.0
        finally:
            os.unlink(tmp)

    def test_custom_duration_passed_to_ffmpeg(self):
        """Custom duration=60.0 should appear as -t 60.0 in ffmpeg args."""
        from src.ui.clip_preview import extract_preview_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            mock_result = MagicMock()
            mock_result.returncode = 0
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                extract_preview_clip(tmp, 0.0, "/tmp/out.mp4", duration=60.0)
                args = mock_run.call_args[0][0]
                t_idx = args.index("-t")
                assert float(args[t_idx + 1]) == 60.0
        finally:
            os.unlink(tmp)
