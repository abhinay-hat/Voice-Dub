"""
Whisper-based transcription with word-level timestamps.
Provides speech-to-text conversion using faster-whisper with VAD filtering.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from faster_whisper import WhisperModel

from src.models.model_manager import ModelManager
from src.utils.audio_preprocessing import preprocess_audio_for_asr
from src.config.settings import MODELS_DIR


@dataclass
class WordInfo:
    """Word-level timing and confidence information."""
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float


@dataclass
class SegmentInfo:
    """Segment-level transcription with word details."""
    id: int
    text: str
    start: float
    end: float
    words: List[WordInfo]
    avg_logprob: float


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    language: str
    language_probability: float
    duration: float
    segments: List[SegmentInfo]


def transcribe_audio(
    audio_path: str,
    model_manager: Optional[ModelManager] = None
) -> TranscriptionResult:
    """
    Transcribe audio using Whisper Large V3 with word-level timestamps.

    Uses faster-whisper for 2-4x speedup vs openai-whisper with identical accuracy.
    Includes VAD (Voice Activity Detection) filtering to prevent hallucinations on silence.

    Args:
        audio_path: Path to audio file (any format - will be preprocessed to 16kHz mono)
        model_manager: Optional ModelManager for sequential loading (created if None)

    Returns:
        TranscriptionResult: Transcription with segments, words, timestamps, and language

    Example:
        >>> result = transcribe_audio("video_audio.wav")
        >>> print(f"Language: {result.language}")
        >>> print(f"Segments: {len(result.segments)}")
        >>> for segment in result.segments:
        ...     print(f"{segment.start:.2f}s: {segment.text}")
    """
    # Create model manager if not provided (allows reuse across pipeline)
    if model_manager is None:
        model_manager = ModelManager(verbose=True)

    # Preprocess audio to 16kHz mono WAV (required by Whisper)
    print("Preprocessing audio to 16kHz mono...")
    preprocessed_audio = preprocess_audio_for_asr(audio_path)

    # Load Whisper Large V3 via ModelManager
    print("Loading Whisper Large V3...")
    whisper_model = model_manager.load_model(
        "whisper-large-v3",
        lambda: WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",  # Halves VRAM (~4.5GB vs ~10GB) with negligible accuracy loss
            download_root=str(MODELS_DIR)
        )
    )

    # Transcribe with word-level timestamps and VAD filtering
    print("Transcribing audio...")
    segments_generator, info = whisper_model.transcribe(
        preprocessed_audio,
        word_timestamps=True,  # CRITICAL: enables word-level timing for lip sync (Phase 7)
        vad_filter=True,       # CRITICAL: prevents hallucinations on silence (80% reduction)
        beam_size=5,           # Balance accuracy vs speed
        language=None          # Auto-detect language from first 30 seconds
    )

    # Convert generator to list (needed for multiple iterations and counting)
    segments_list = list(segments_generator)

    # Build structured result
    segments_output = []

    for i, segment in enumerate(segments_list):
        # Extract word-level information
        word_list = []
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                word_list.append(WordInfo(
                    word=word.word,
                    start=round(word.start, 2),
                    end=round(word.end, 2),
                    probability=round(word.probability, 3)
                ))

        # Build segment info
        segments_output.append(SegmentInfo(
            id=i,
            text=segment.text,
            start=round(segment.start, 2),
            end=round(segment.end, 2),
            words=word_list,
            avg_logprob=round(segment.avg_logprob, 3)
        ))

    # Calculate duration from last segment
    duration = segments_output[-1].end if segments_output else 0.0

    print(f"Transcription complete: {len(segments_output)} segments, language: {info.language}")

    # DO NOT unload model - leave that to caller or next model load
    # This allows chaining with diarization in the same pipeline

    return TranscriptionResult(
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=duration,
        segments=segments_output
    )
