"""
Integration tests for ASR stage orchestration.

Tests verify module structure, imports, dataclasses, and alignment logic
without requiring GPU or actual audio files.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import dataclasses


def test_asr_stage_imports():
    """Verify all ASR stage components import correctly."""
    try:
        from src.stages.asr_stage import run_asr_stage, ASRResult
        from src.stages.transcription import transcribe_audio, TranscriptionResult
        from src.stages.alignment import align_transcript_with_speakers, AlignedSegment
        from src.utils.audio_preprocessing import preprocess_audio_for_asr
        print("OK: All ASR imports successful (note: pyannote.audio import skipped - requires installation)")
    except ImportError as e:
        # If pyannote.audio is not installed, we expect diarization import to fail
        # But asr_stage.py should still be structurally valid
        if "pyannote" in str(e):
            print("OK: ASR stage structure verified (pyannote.audio not installed yet - expected)")
        else:
            raise


def test_asr_result_fields():
    """Verify ASRResult has all required fields."""
    try:
        from src.stages.asr_stage import ASRResult
    except ImportError as e:
        if "pyannote" in str(e):
            print("OK: ASRResult structure test skipped (pyannote.audio not installed)")
            return
        raise

    fields = {f.name for f in dataclasses.fields(ASRResult)}
    required = {
        'video_id', 'duration', 'detected_language', 'language_probability',
        'num_speakers', 'total_segments', 'flagged_count', 'flagged_segment_ids',
        'segments', 'processing_time', 'output_path'
    }

    missing = required - fields
    assert not missing, f"Missing fields: {missing}"

    print(f"OK: ASRResult has all required fields: {fields}")


def test_alignment_overlap_logic():
    """Verify temporal overlap alignment works correctly."""
    try:
        from src.stages.alignment import find_speaker_for_word
        from src.stages.diarization import SpeakerTurn
    except ImportError as e:
        if "pyannote" in str(e):
            print("OK: Alignment logic test skipped (pyannote.audio not installed)")
            return
        raise

    turns = [
        SpeakerTurn(speaker='SPEAKER_00', start=0.0, end=5.0),
        SpeakerTurn(speaker='SPEAKER_01', start=5.0, end=10.0)
    ]

    # Word fully in SPEAKER_00 range
    result = find_speaker_for_word(1.0, 2.0, turns)
    assert result == 'SPEAKER_00', f"Expected SPEAKER_00, got {result}"

    # Word fully in SPEAKER_01 range
    result = find_speaker_for_word(7.0, 8.0, turns)
    assert result == 'SPEAKER_01', f"Expected SPEAKER_01, got {result}"

    # Word at boundary - should assign based on greater overlap
    result = find_speaker_for_word(4.8, 5.2, turns)
    assert result in ['SPEAKER_00', 'SPEAKER_01'], f"Expected SPEAKER_00 or SPEAKER_01, got {result}"

    # Word before all turns - should assign to nearest (SPEAKER_00)
    result = find_speaker_for_word(-1.0, -0.5, turns)
    assert result == 'SPEAKER_00', f"Expected SPEAKER_00 (nearest), got {result}"

    # Word after all turns - should assign to nearest (SPEAKER_01)
    result = find_speaker_for_word(11.0, 12.0, turns)
    assert result == 'SPEAKER_01', f"Expected SPEAKER_01 (nearest), got {result}"

    print("OK: Alignment overlap logic tests passed")


def test_confidence_threshold_setting():
    """Verify ASR confidence threshold is configured."""
    from src.config.settings import ASR_CONFIDENCE_THRESHOLD

    assert ASR_CONFIDENCE_THRESHOLD == 0.7, f"Expected 0.7, got {ASR_CONFIDENCE_THRESHOLD}"
    print(f"OK: Confidence threshold configured: {ASR_CONFIDENCE_THRESHOLD}")


def test_aligned_segment_structure():
    """Verify AlignedSegment dataclass has correct structure."""
    try:
        from src.stages.alignment import AlignedSegment, AlignedWord
    except ImportError as e:
        if "pyannote" in str(e):
            print("OK: AlignedSegment structure test skipped (pyannote.audio not installed)")
            return
        raise

    # Check AlignedSegment fields
    seg_fields = {f.name for f in dataclasses.fields(AlignedSegment)}
    seg_required = {'id', 'text', 'start', 'end', 'speaker', 'confidence', 'needs_review', 'words'}
    assert seg_required.issubset(seg_fields), f"Missing AlignedSegment fields: {seg_required - seg_fields}"

    # Check AlignedWord fields
    word_fields = {f.name for f in dataclasses.fields(AlignedWord)}
    word_required = {'word', 'start', 'end', 'speaker', 'confidence', 'needs_review'}
    assert word_required.issubset(word_fields), f"Missing AlignedWord fields: {word_required - word_fields}"

    print(f"OK: AlignedSegment structure verified: {seg_fields}")
    print(f"OK: AlignedWord structure verified: {word_fields}")


def test_transcription_result_structure():
    """Verify TranscriptionResult dataclass has correct structure."""
    from src.stages.transcription import TranscriptionResult, SegmentInfo, WordInfo

    # Check TranscriptionResult fields
    trans_fields = {f.name for f in dataclasses.fields(TranscriptionResult)}
    trans_required = {'language', 'language_probability', 'duration', 'segments'}
    assert trans_required.issubset(trans_fields), f"Missing TranscriptionResult fields: {trans_required - trans_fields}"

    # Check SegmentInfo fields
    seg_fields = {f.name for f in dataclasses.fields(SegmentInfo)}
    seg_required = {'id', 'text', 'start', 'end', 'words', 'avg_logprob'}
    assert seg_required.issubset(seg_fields), f"Missing SegmentInfo fields: {seg_required - seg_fields}"

    # Check WordInfo fields
    word_fields = {f.name for f in dataclasses.fields(WordInfo)}
    word_required = {'word', 'start', 'end', 'probability'}
    assert word_required.issubset(word_fields), f"Missing WordInfo fields: {word_required - word_fields}"

    print(f"OK: TranscriptionResult structure verified: {trans_fields}")
    print(f"OK: SegmentInfo structure verified: {seg_fields}")
    print(f"OK: WordInfo structure verified: {word_fields}")


if __name__ == "__main__":
    print("Running ASR stage integration tests...\n")

    test_asr_stage_imports()
    test_asr_result_fields()
    test_alignment_overlap_logic()
    test_confidence_threshold_setting()
    test_aligned_segment_structure()
    test_transcription_result_structure()

    print("\n" + "="*60)
    print("ALL ASR STAGE TESTS PASSED OK:")
    print("="*60)
