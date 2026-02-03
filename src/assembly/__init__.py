"""
Audio-Video Assembly Module (Phase 6)

Provides frame-perfect synchronization infrastructure for dubbing pipeline:
- Timestamp validation with float64 precision
- Sample rate normalization to 48kHz
- Audio segment concatenation
- Drift detection at checkpoint intervals
- FFmpeg merge with explicit sync flags

Usage:
    from src.assembly import (
        # Timestamp validation
        TimedSegment,
        validate_timestamps_precision,
        ensure_float64,

        # Audio normalization
        normalize_sample_rate,
        validate_sample_rate,
        batch_normalize,

        # Audio concatenation
        concatenate_audio_segments,
        get_total_duration,

        # Drift detection
        SyncCheckpoint,
        DriftValidationResult,
        validate_sync_at_intervals,
        check_segment_continuity,

        # Video merging
        MergeResult,
        merge_with_sync_validation,
        get_merge_config,
        validate_audio_video_compatibility,
    )
"""

from .timestamp_validator import (
    TimedSegment,
    validate_timestamps_precision,
    ensure_float64,
)

from .audio_normalizer import (
    normalize_sample_rate,
    validate_sample_rate,
    batch_normalize,
)

from .audio_concatenator import (
    concatenate_audio_segments,
    get_total_duration,
)

from .drift_detector import (
    SyncCheckpoint,
    DriftValidationResult,
    validate_sync_at_intervals,
    check_segment_continuity,
)

from .video_merger import (
    MergeResult,
    merge_with_sync_validation,
    get_merge_config,
    validate_audio_video_compatibility,
)

__all__ = [
    # Timestamp validation
    "TimedSegment",
    "validate_timestamps_precision",
    "ensure_float64",
    # Audio normalization
    "normalize_sample_rate",
    "validate_sample_rate",
    "batch_normalize",
    # Audio concatenation
    "concatenate_audio_segments",
    "get_total_duration",
    # Drift detection
    "SyncCheckpoint",
    "DriftValidationResult",
    "validate_sync_at_intervals",
    "check_segment_continuity",
    # Video merging
    "MergeResult",
    "merge_with_sync_validation",
    "get_merge_config",
    "validate_audio_video_compatibility",
]
