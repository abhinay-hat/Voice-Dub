"""
Speaker diarization using pyannote.audio.
Detects and labels speakers in audio with temporal segments.
"""
from dataclasses import dataclass
from typing import List, Optional
import torch

from pyannote.audio import Pipeline

from src.models.model_manager import ModelManager


@dataclass
class SpeakerTurn:
    """Speaker turn with temporal boundaries."""
    speaker: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class DiarizationResult:
    """Complete diarization result with speaker segments."""
    num_speakers: int
    turns: List[SpeakerTurn]
    duration: float


def diarize_audio(
    audio_path: str,
    huggingface_token: str,
    model_manager: Optional[ModelManager] = None,
    min_speakers: int = 2,
    max_speakers: int = 5
) -> DiarizationResult:
    """
    Detect and label speakers in audio using pyannote speaker-diarization-community-1.

    Uses state-of-the-art speaker diarization model to segment audio by speaker.
    Requires HuggingFace authentication token and user acceptance of model license.

    Args:
        audio_path: Path to audio file (16kHz mono WAV recommended)
        huggingface_token: HuggingFace API token for gated model access
        model_manager: Optional ModelManager for sequential loading (created if None)
        min_speakers: Minimum expected speakers (default: 2)
        max_speakers: Maximum expected speakers (default: 5)

    Returns:
        DiarizationResult: Speaker segments with labels and timestamps

    Example:
        >>> result = diarize_audio("audio.wav", token="hf_...")
        >>> print(f"Detected {result.num_speakers} speakers")
        >>> for turn in result.turns:
        ...     print(f"{turn.speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    Note:
        - HuggingFace token required: https://huggingface.co/pyannote/speaker-diarization-community-1
        - User must accept model license on HuggingFace Hub before first use
        - Model requires ~2-4GB VRAM
    """
    # Create model manager if not provided (allows reuse across pipeline)
    if model_manager is None:
        model_manager = ModelManager(verbose=True)

    # Load pyannote speaker-diarization-community-1 via ModelManager
    print("Loading pyannote diarization...")
    diarization_pipeline = model_manager.load_model(
        "pyannote-diarization",
        lambda: Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=huggingface_token
        ).to(torch.device("cuda"))
    )

    # Run speaker diarization
    print("Detecting speakers...")
    diarization = diarization_pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    # Extract speaker turns from pyannote output
    turns = []
    for segment, track_id, speaker_label in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            speaker=speaker_label,
            start=round(segment.start, 2),
            end=round(segment.end, 2)
        ))

    # Count unique speakers
    unique_speakers = set(turn.speaker for turn in turns)
    num_speakers = len(unique_speakers)

    # Calculate total duration from last turn
    duration = turns[-1].end if turns else 0.0

    print(f"Diarization complete: {num_speakers} speakers detected")

    # DO NOT unload model - leave that to caller or next model load
    # This allows chaining with alignment in the same pipeline

    return DiarizationResult(
        num_speakers=num_speakers,
        turns=turns,
        duration=duration
    )
