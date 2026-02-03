"""
Audio-video assembly infrastructure for drift-free synchronization.

This module provides core components for timestamp validation, audio normalization,
and segment concatenation to enable precise audio-video alignment over long videos
(20+ minutes) without drift.
"""

from . import timestamp_validator
from . import audio_normalizer
from . import audio_concatenator

__all__ = [
    'timestamp_validator',
    'audio_normalizer',
    'audio_concatenator',
]
