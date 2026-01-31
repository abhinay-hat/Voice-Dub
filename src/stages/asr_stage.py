"""
Complete ASR stage orchestration module.
Orchestrates transcription, diarization, and alignment into a single pipeline.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable
import json
import time

from src.models.model_manager import ModelManager
from src.stages.transcription import transcribe_audio, TranscriptionResult
from src.stages.diarization import diarize_audio, DiarizationResult
from src.stages.alignment import align_transcript_with_speakers, AlignedSegment, AlignedWord
from src.utils.audio_preprocessing import preprocess_audio_for_asr
from src.config.settings import TEMP_DATA_DIR, ASR_CONFIDENCE_THRESHOLD


@dataclass
class ASRResult:
    """Complete ASR pipeline result with metadata and aligned segments."""
    video_id: str
    duration: float
    detected_language: str
    language_probability: float
    num_speakers: int
    total_segments: int
    flagged_count: int
    flagged_segment_ids: List[int]
    segments: List[AlignedSegment]
    processing_time: float
    output_path: Optional[str] = None


def run_asr_stage(
    audio_path: str,
    video_id: str,
    huggingface_token: str,
    progress_callback: Callable[[float, str], None] = None,
    save_json: bool = True
) -> ASRResult:
    """
    Complete ASR pipeline: transcription + diarization + alignment + JSON export.

    Orchestrates the entire speech recognition pipeline:
    1. Preprocess audio to 16kHz mono
    2. Transcribe with Whisper Large V3 (word timestamps + VAD)
    3. Diarize speakers with pyannote
    4. Align transcription words with speaker segments
    5. Flag low-confidence segments for review
    6. Export to JSON

    Args:
        audio_path: Path to audio file (any format - will be preprocessed)
        video_id: Unique identifier for this video (used in output filename)
        huggingface_token: HuggingFace API token for pyannote model access
        progress_callback: Optional callback(progress: float, status: str) for UI updates
        save_json: Whether to save result to JSON file (default: True)

    Returns:
        ASRResult: Complete result with segments, speakers, confidence flags, and metadata

    Example:
        >>> result = run_asr_stage(
        ...     audio_path="video_audio.wav",
        ...     video_id="video123",
        ...     huggingface_token="hf_..."
        ... )
        >>> print(f"Language: {result.detected_language}")
        >>> print(f"Speakers: {result.num_speakers}")
        >>> print(f"Flagged: {result.flagged_count}/{result.total_segments}")
    """
    # Start timer for performance tracking
    start_time = time.time()

    # Create model manager for sequential loading
    model_manager = ModelManager(verbose=True)

    # Default progress callback to no-op if None
    if progress_callback is None:
        progress_callback = lambda progress, status: None

    preprocessed_audio_path = None

    try:
        # Step 1: Preprocess audio to 16kHz mono
        progress_callback(0.05, "Preprocessing audio to 16kHz mono...")
        print("Preprocessing audio to 16kHz mono...")
        preprocessed_audio_path = preprocess_audio_for_asr(audio_path)

        # Step 2: Transcription with Whisper Large V3
        progress_callback(0.10, "Transcribing with Whisper Large V3...")
        print("Transcribing with Whisper Large V3...")
        transcription: TranscriptionResult = transcribe_audio(
            preprocessed_audio_path,
            model_manager
        )
        detected_language = transcription.language
        language_probability = transcription.language_probability
        duration = transcription.duration

        # Step 3: Speaker diarization with pyannote
        progress_callback(0.45, "Detecting speakers with pyannote...")
        print("Detecting speakers with pyannote...")
        diarization: DiarizationResult = diarize_audio(
            preprocessed_audio_path,
            huggingface_token,
            model_manager
        )
        num_speakers = diarization.num_speakers

        # Step 4: Align transcription with speakers
        progress_callback(0.80, "Aligning transcription with speakers...")
        print("Aligning transcription with speakers...")
        aligned_segments: List[AlignedSegment] = align_transcript_with_speakers(
            transcription,
            diarization
        )

        # Count flagged segments (needs_review=True)
        flagged_segment_ids = [seg.id for seg in aligned_segments if seg.needs_review]
        flagged_count = len(flagged_segment_ids)

        # Step 5: Build result
        progress_callback(0.95, "Building ASR result...")
        processing_time = time.time() - start_time

        result = ASRResult(
            video_id=video_id,
            duration=duration,
            detected_language=detected_language,
            language_probability=language_probability,
            num_speakers=num_speakers,
            total_segments=len(aligned_segments),
            flagged_count=flagged_count,
            flagged_segment_ids=flagged_segment_ids,
            segments=aligned_segments,
            processing_time=processing_time
        )

        # Step 6: Save JSON if requested
        if save_json:
            progress_callback(0.98, "Saving JSON output...")
            TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = TEMP_DATA_DIR / f"{video_id}_transcript.json"

            # Convert result to JSON-serializable dict
            result_dict = _asr_result_to_dict(result)

            # Write JSON with proper encoding for non-English characters
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

            result.output_path = str(output_path)
            print(f"Transcript saved to: {output_path}")

        # Summary message
        print(f"ASR complete: {len(aligned_segments)} segments, {num_speakers} speakers, {flagged_count} flagged for review")

        progress_callback(1.0, "ASR complete")

        return result

    finally:
        # Step 7: Cleanup - unload models and delete temp files
        model_manager.unload_current_model()

        # Delete preprocessed 16kHz WAV file
        if preprocessed_audio_path and Path(preprocessed_audio_path).exists():
            try:
                Path(preprocessed_audio_path).unlink()
                print(f"Cleaned up temp file: {preprocessed_audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete temp file {preprocessed_audio_path}: {e}")


def _asr_result_to_dict(result: ASRResult) -> dict:
    """
    Convert ASRResult to JSON-serializable dictionary.

    Handles nested dataclasses (AlignedSegment, AlignedWord) by converting
    them to dictionaries recursively.

    Args:
        result: ASRResult instance to serialize

    Returns:
        dict: JSON-serializable dictionary representation
    """
    # Convert top-level ASRResult
    result_dict = asdict(result)

    # Segments are already converted by asdict (it handles nested dataclasses)
    # But we need to ensure AlignedSegment and AlignedWord are properly serialized
    # asdict() handles this automatically by recursively converting dataclasses

    return result_dict
