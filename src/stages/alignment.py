"""
Temporal alignment between Whisper transcription and pyannote speaker diarization.
Matches word timestamps to speaker segments using overlap calculation.
"""
from dataclasses import dataclass
from typing import List

from src.stages.transcription import TranscriptionResult, WordInfo, SegmentInfo
from src.stages.diarization import DiarizationResult, SpeakerTurn
from src.config.settings import ASR_CONFIDENCE_THRESHOLD


@dataclass
class AlignedWord:
    """Word with speaker assignment and review flag."""
    word: str
    start: float
    end: float
    speaker: str
    confidence: float
    needs_review: bool


@dataclass
class AlignedSegment:
    """Segment of consecutive words from same speaker."""
    id: int
    text: str
    start: float
    end: float
    speaker: str
    confidence: float
    needs_review: bool
    words: List[AlignedWord]


def find_speaker_for_word(
    word_start: float,
    word_end: float,
    turns: List[SpeakerTurn]
) -> str:
    """
    Find speaker for a word based on temporal overlap with speaker turns.

    Uses greatest overlap heuristic. If no overlap exists (word between speaker turns),
    assigns to nearest speaker temporally.

    Args:
        word_start: Word start time in seconds
        word_end: Word end time in seconds
        turns: List of speaker turns with temporal boundaries

    Returns:
        str: Speaker label with maximum overlap, or "UNKNOWN" if no speakers

    Example:
        >>> turns = [SpeakerTurn('A', 0.0, 5.0), SpeakerTurn('B', 5.0, 10.0)]
        >>> find_speaker_for_word(2.0, 3.0, turns)
        'A'
        >>> find_speaker_for_word(6.0, 7.0, turns)
        'B'
    """
    best_speaker = None
    max_overlap = 0.0

    # Find speaker with maximum temporal overlap
    for turn in turns:
        # Calculate overlap duration using interval intersection
        overlap_start = max(word_start, turn.start)
        overlap_end = min(word_end, turn.end)
        overlap_duration = max(0.0, overlap_end - overlap_start)

        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = turn.speaker

    # If no overlap found, assign to nearest speaker temporally
    if best_speaker is None and turns:
        min_distance = float('inf')
        for turn in turns:
            # Distance is minimum of distance to turn start or end
            distance = min(
                abs(word_start - turn.end),
                abs(word_end - turn.start)
            )
            if distance < min_distance:
                min_distance = distance
                best_speaker = turn.speaker

    # Return speaker or UNKNOWN if still not found
    return best_speaker if best_speaker else "UNKNOWN"


def align_transcript_with_speakers(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
    confidence_threshold: float = None
) -> List[AlignedSegment]:
    """
    Align Whisper transcription words with pyannote speaker segments.

    For each word, finds speaker with maximum temporal overlap. Groups consecutive
    words by speaker into segments. Flags low-confidence words for user review.

    Args:
        transcription: Whisper transcription result with word-level timestamps
        diarization: Pyannote diarization result with speaker turns
        confidence_threshold: Confidence below which to flag for review (default: ASR_CONFIDENCE_THRESHOLD)

    Returns:
        List[AlignedSegment]: Segments grouped by speaker with word-level details

    Example:
        >>> result = align_transcript_with_speakers(transcript, diarization)
        >>> print(f"{len(result)} segments, {sum(s.needs_review for s in result)} flagged")
        >>> for segment in result:
        ...     print(f"{segment.speaker}: {segment.text}")
    """
    if confidence_threshold is None:
        confidence_threshold = ASR_CONFIDENCE_THRESHOLD

    # Align all words with speakers
    aligned_words = []
    total_words = sum(len(segment.words) for segment in transcription.segments)

    print(f"Aligning {total_words} words with speaker segments...")

    for segment in transcription.segments:
        for word_info in segment.words:
            # Find speaker with maximum overlap
            speaker = find_speaker_for_word(
                word_info.start,
                word_info.end,
                diarization.turns
            )

            # Create aligned word with review flag
            aligned_word = AlignedWord(
                word=word_info.word,
                start=round(word_info.start, 2),
                end=round(word_info.end, 2),
                speaker=speaker,
                confidence=round(word_info.probability, 3),
                needs_review=(word_info.probability < confidence_threshold)
            )
            aligned_words.append(aligned_word)

    # Group consecutive words by speaker into segments
    aligned_segments = []
    current_segment = None

    for word_data in aligned_words:
        # Start new segment if speaker changed or first word
        if current_segment is None or current_segment['speaker'] != word_data.speaker:
            # Save previous segment if exists
            if current_segment is not None:
                aligned_segments.append(_build_aligned_segment(
                    segment_id=len(aligned_segments),
                    segment_data=current_segment
                ))

            # Start new segment
            current_segment = {
                'speaker': word_data.speaker,
                'words': [word_data],
                'start': word_data.start,
                'end': word_data.end
            }
        else:
            # Continue current segment
            current_segment['words'].append(word_data)
            current_segment['end'] = word_data.end

    # Add final segment
    if current_segment is not None:
        aligned_segments.append(_build_aligned_segment(
            segment_id=len(aligned_segments),
            segment_data=current_segment
        ))

    # Count flagged segments
    flagged_count = sum(1 for seg in aligned_segments if seg.needs_review)

    print(f"Alignment complete: {len(aligned_segments)} segments, {flagged_count} flagged for review")

    return aligned_segments


def _build_aligned_segment(segment_id: int, segment_data: dict) -> AlignedSegment:
    """
    Build AlignedSegment from accumulated segment data.

    Internal helper for align_transcript_with_speakers.

    Args:
        segment_id: Unique segment identifier
        segment_data: Dict with speaker, words, start, end

    Returns:
        AlignedSegment: Structured segment with aggregated metadata
    """
    words = segment_data['words']

    # Concatenate word texts with spacing
    text = ''.join(word.word for word in words)

    # Segment confidence is minimum of word confidences
    min_confidence = min(word.confidence for word in words)

    # Segment needs review if any word needs review
    needs_review = any(word.needs_review for word in words)

    return AlignedSegment(
        id=segment_id,
        text=text,
        start=segment_data['start'],
        end=segment_data['end'],
        speaker=segment_data['speaker'],
        confidence=round(min_confidence, 3),
        needs_review=needs_review,
        words=words
    )
