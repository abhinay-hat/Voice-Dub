"""Lip synchronization module.

Provides LatentSync and Wav2Lip GAN runners for lip sync,
plus chunking utilities for long-video processing.
"""
from src.lip_sync.wav2lip_runner import run_wav2lip_inference, WAV2LIP_REPO, WAV2LIP_CHECKPOINT
from src.lip_sync.chunker import (
    split_video_into_chunks,
    concatenate_video_chunks,
    VideoChunk,
    get_video_duration,
)

__all__ = [
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
