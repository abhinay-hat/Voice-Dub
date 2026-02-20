"""Lip synchronization module for voice dubbing pipeline.

Provides LatentSync 1.6 (primary) and Wav2Lip (fallback) inference runners,
plus audio preparation utilities and chunking for long-video processing.
"""
from .audio_prep import prepare_audio_for_lipsync
from .latentsync_runner import run_latentsync_inference, LATENTSYNC_PYTHON, LATENTSYNC_REPO
from .wav2lip_runner import run_wav2lip_inference, WAV2LIP_REPO, WAV2LIP_CHECKPOINT
from .chunker import (
    split_video_into_chunks,
    concatenate_video_chunks,
    VideoChunk,
    get_video_duration,
)

__all__ = [
    # Audio preparation
    "prepare_audio_for_lipsync",
    # LatentSync primary runner
    "run_latentsync_inference",
    "LATENTSYNC_PYTHON",
    "LATENTSYNC_REPO",
    # Wav2Lip fallback runner
    "run_wav2lip_inference",
    "WAV2LIP_REPO",
    "WAV2LIP_CHECKPOINT",
    # Chunker
    "split_video_into_chunks",
    "concatenate_video_chunks",
    "VideoChunk",
    "get_video_duration",
]
