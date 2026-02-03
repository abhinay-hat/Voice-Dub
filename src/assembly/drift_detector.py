"""
Drift detection at checkpoint intervals for audio-video synchronization.

Validates audio-video sync at regular intervals (5 minutes by default) to catch
progressive drift before it becomes noticeable. Uses ATSC standard of 45ms tolerance.

Example:
    >>> from src.assembly.drift_detector import validate_sync_at_intervals
    >>> from src.assembly.timestamp_validator import TimedSegment
    >>> segments = [
    ...     TimedSegment(0.0, 300.0, 'seg1.wav', 'S1'),
    ...     TimedSegment(300.0, 600.0, 'seg2.wav', 'S2'),
    ... ]
    >>> result = validate_sync_at_intervals(segments, 30.0, 48000, 600.0)
    >>> print(f'Synced: {result.is_synced}, Max drift: {result.max_drift_ms:.2f}ms')
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.assembly.timestamp_validator import TimedSegment, ensure_float64
from src.config.settings import ASSEMBLY_DRIFT_TOLERANCE_MS, ASSEMBLY_CHECKPOINT_INTERVAL

logger = logging.getLogger(__name__)


@dataclass
class SyncCheckpoint:
    """Validation point for audio-video synchronization.

    Attributes:
        timestamp: Expected timestamp in seconds (float64)
        frame_number: Expected video frame at this checkpoint
        audio_sample: Expected audio sample index at this checkpoint
        expected_duration: Expected cumulative duration at checkpoint
        actual_duration: Actual cumulative duration from segments
        drift_ms: Measured drift (actual - expected) in milliseconds
        within_tolerance: True if abs(drift_ms) <= tolerance
    """
    timestamp: float
    frame_number: int
    audio_sample: int
    expected_duration: float
    actual_duration: float
    drift_ms: float
    within_tolerance: bool


@dataclass
class DriftValidationResult:
    """Complete drift validation result.

    Attributes:
        is_synced: True if all checkpoints pass tolerance check
        checkpoints: List of validation checkpoints
        max_drift_ms: Maximum absolute drift observed across all checkpoints
        total_duration: Total video duration validated
        checkpoint_count: Number of checkpoints checked
    """
    is_synced: bool
    checkpoints: List[SyncCheckpoint]
    max_drift_ms: float
    total_duration: float
    checkpoint_count: int


def validate_sync_at_intervals(
    segments: List[TimedSegment],
    video_fps: float,
    audio_sr: int,
    total_duration: float,
    interval_seconds: float = ASSEMBLY_CHECKPOINT_INTERVAL,
    tolerance_ms: float = ASSEMBLY_DRIFT_TOLERANCE_MS
) -> DriftValidationResult:
    """Validate audio-video sync at regular intervals.

    Checks that cumulative segment durations match expected timestamps at
    regular intervals. Detects progressive drift before it becomes noticeable.

    Args:
        segments: List of timed audio segments (assumed chronological)
        video_fps: Video frame rate (e.g., 30.0, 29.97, 24.0)
        audio_sr: Audio sample rate (should be 48000)
        total_duration: Expected total video duration in seconds
        interval_seconds: Checkpoint interval (default: 300.0 = 5 minutes)
        tolerance_ms: Drift tolerance in milliseconds (default: 45.0 ATSC)

    Returns:
        DriftValidationResult with sync status and checkpoint data

    Example:
        >>> segments = [
        ...     TimedSegment(0.0, 300.0, 'seg1.wav', 'S1'),
        ...     TimedSegment(300.0, 600.0, 'seg2.wav', 'S2'),
        ... ]
        >>> result = validate_sync_at_intervals(segments, 30.0, 48000, 600.0)
        >>> for cp in result.checkpoints:
        ...     print(f'{cp.timestamp}s: {cp.drift_ms:+.2f}ms')
    """
    if not segments:
        raise ValueError("Empty segments list provided")

    if total_duration <= 0:
        raise ValueError(f"Invalid total_duration: {total_duration}")

    if interval_seconds <= 0:
        raise ValueError(f"Invalid interval_seconds: {interval_seconds}")

    # Generate checkpoint timestamps
    checkpoint_times = []
    num_intervals = int(total_duration / interval_seconds)

    for i in range(num_intervals):
        checkpoint_times.append((i + 1) * interval_seconds)

    # Always add final checkpoint at video end
    if total_duration not in checkpoint_times:
        checkpoint_times.append(total_duration)

    logger.info(
        f"Validating sync at {len(checkpoint_times)} checkpoints "
        f"(interval: {interval_seconds}s, tolerance: {tolerance_ms}ms)"
    )

    checkpoints = []
    is_synced = True
    max_drift_ms = 0.0

    for expected_timestamp in checkpoint_times:
        expected_timestamp = ensure_float64(expected_timestamp)

        # Calculate actual duration from all segments up to this checkpoint
        # Find which segments fall within [0, expected_timestamp]
        actual_duration = np.float64(0.0)

        for seg in segments:
            seg_start = ensure_float64(seg.start)
            seg_end = ensure_float64(seg.end)

            # If segment starts after checkpoint, skip it
            if seg_start >= expected_timestamp:
                continue

            # If segment ends before or at checkpoint, count full duration
            if seg_end <= expected_timestamp:
                actual_duration += (seg_end - seg_start)
            else:
                # Segment spans across checkpoint, count partial duration
                actual_duration += (expected_timestamp - seg_start)
                break  # No more segments to count after this one

        # Calculate drift
        drift_seconds = actual_duration - expected_timestamp
        drift_ms = float(drift_seconds * 1000.0)

        # Frame and sample indices at this checkpoint
        frame_number = int(expected_timestamp * video_fps)
        audio_sample = int(expected_timestamp * audio_sr)

        # Check tolerance
        within_tolerance = abs(drift_ms) <= tolerance_ms
        if not within_tolerance:
            is_synced = False

        max_drift_ms = max(max_drift_ms, abs(drift_ms))

        checkpoint = SyncCheckpoint(
            timestamp=float(expected_timestamp),
            frame_number=frame_number,
            audio_sample=audio_sample,
            expected_duration=float(expected_timestamp),
            actual_duration=float(actual_duration),
            drift_ms=drift_ms,
            within_tolerance=within_tolerance
        )
        checkpoints.append(checkpoint)

        logger.debug(
            f"Checkpoint at {expected_timestamp:.1f}s: "
            f"drift={drift_ms:+.2f}ms "
            f"(frame {frame_number}, {'PASS' if within_tolerance else 'FAIL'})"
        )

    result = DriftValidationResult(
        is_synced=is_synced,
        checkpoints=checkpoints,
        max_drift_ms=max_drift_ms,
        total_duration=float(total_duration),
        checkpoint_count=len(checkpoints)
    )

    if is_synced:
        logger.info(
            f"Sync validation PASSED: max drift {max_drift_ms:.2f}ms "
            f"(tolerance: {tolerance_ms}ms)"
        )
    else:
        logger.warning(
            f"Sync validation FAILED: max drift {max_drift_ms:.2f}ms exceeds "
            f"tolerance of {tolerance_ms}ms"
        )

    return result


def check_segment_continuity(
    segments: List[TimedSegment],
    max_gap_seconds: float = 1.0
) -> Tuple[bool, List[str]]:
    """Validate segments are in chronological order with no overlaps.

    Checks that segments are ordered by start time, have no overlaps,
    and have gaps smaller than the threshold.

    Args:
        segments: List of TimedSegment objects
        max_gap_seconds: Maximum acceptable gap (default: 1.0 second)

    Returns:
        Tuple of (is_continuous: bool, warning_messages: List[str])

    Example:
        >>> segments = [
        ...     TimedSegment(0.0, 5.0, 'seg1.wav', 'S1'),
        ...     TimedSegment(5.0, 10.0, 'seg2.wav', 'S2'),
        ... ]
        >>> is_continuous, warnings = check_segment_continuity(segments)
        >>> print(f'Continuous: {is_continuous}')
    """
    if not segments:
        return True, []

    warnings = []
    is_continuous = True

    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]

        current_start = ensure_float64(current.start)
        current_end = ensure_float64(current.end)
        next_start = ensure_float64(next_seg.start)
        next_end = ensure_float64(next_seg.end)

        # Check chronological order
        if next_start < current_start:
            warnings.append(
                f"Segment {i+1} starts before segment {i}: "
                f"{next_start:.3f}s < {current_start:.3f}s"
            )
            is_continuous = False

        # Check for overlaps
        if next_start < current_end:
            overlap = current_end - next_start
            warnings.append(
                f"Segment {i+1} overlaps segment {i} by {overlap:.3f}s"
            )
            is_continuous = False

        # Check for large gaps
        gap = next_start - current_end
        if gap > max_gap_seconds:
            warnings.append(
                f"Gap of {gap:.3f}s between segment {i} and {i+1} "
                f"(exceeds {max_gap_seconds}s threshold)"
            )
            # Large gaps don't make it discontinuous, just warn

    if warnings:
        logger.warning(f"Segment continuity issues found: {len(warnings)} warnings")
        for warning in warnings:
            logger.warning(f"  {warning}")
    else:
        logger.debug(f"Segment continuity validated: {len(segments)} segments OK")

    return is_continuous, warnings
