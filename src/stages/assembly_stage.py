"""
Complete assembly stage orchestration module.
Orchestrates audio-video assembly: normalization, concatenation, drift validation, and merging.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Callable
import json
import time
import logging

from src.assembly import (
    TimedSegment,
    validate_timestamps_precision,
    batch_normalize,
    concatenate_audio_segments,
    validate_sync_at_intervals,
    merge_with_sync_validation,
    SyncCheckpoint,
    DriftValidationResult,
    MergeResult,
)
from src.config.settings import (
    TEMP_DATA_DIR,
    ASSEMBLY_CHECKPOINT_INTERVAL,
    ASSEMBLY_DRIFT_TOLERANCE_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class AssemblyResult:
    """Result of audio-video assembly stage."""
    output_path: Path
    total_duration: float
    segment_count: int
    sync_checkpoints: List[SyncCheckpoint]
    drift_detected: bool
    max_drift_ms: float
    sample_rate_normalized: bool
    original_sample_rates: List[int]  # Sample rates before normalization
    processing_time: float
    video_fps: float

    def to_dict(self) -> dict:
        """Serialize for JSON export."""
        return {
            'output_path': str(self.output_path),
            'total_duration': self.total_duration,
            'segment_count': self.segment_count,
            'checkpoints': [
                {
                    'timestamp': cp.timestamp,
                    'drift_ms': cp.drift_ms,
                    'frame_number': cp.frame_number,
                    'within_tolerance': cp.within_tolerance
                }
                for cp in self.sync_checkpoints
            ],
            'drift_detected': self.drift_detected,
            'max_drift_ms': self.max_drift_ms,
            'sample_rate_normalized': self.sample_rate_normalized,
            'original_sample_rates': self.original_sample_rates,
            'processing_time': self.processing_time,
            'video_fps': self.video_fps
        }


class AssemblyStageFailed(Exception):
    """Raised when assembly stage fails validation."""
    pass


def run_assembly_stage(
    video_path: Path,
    tts_result_path: Path,  # JSON from TTS stage with segment paths
    output_path: Path,
    video_fps: float = 30.0,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> AssemblyResult:
    """
    Complete assembly pipeline: normalization → concatenation → drift validation → merge.

    Orchestrates the entire assembly pipeline:
    1. Load TTS result JSON
    2. Create TimedSegment objects from TTS output
    3. Validate timestamp precision
    4. Normalize all audio segments to 48kHz
    5. Concatenate normalized segments
    6. Validate sync at 5-minute intervals
    7. Merge video with audio (async_correction=True)
    8. Export assembly result JSON
    9. Cleanup temporary files

    Args:
        video_path: Path to original video file
        tts_result_path: Path to TTS stage output JSON with segment paths
        output_path: Path for final dubbed video output
        video_fps: Video frame rate (default: 30.0)
        progress_callback: Optional callback(progress: float, status: str) for UI updates

    Returns:
        AssemblyResult: Complete result with sync checkpoints, drift detection, and metadata

    Raises:
        AssemblyStageFailed: If TTS JSON missing/invalid, no segments, or merge fails

    Example:
        >>> result = run_assembly_stage(
        ...     video_path=Path("input.mp4"),
        ...     tts_result_path=Path("tts_result.json"),
        ...     output_path=Path("dubbed.mp4"),
        ...     video_fps=30.0,
        ...     progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
        ... )
        >>> print(f"Drift detected: {result.drift_detected}")
        >>> print(f"Max drift: {result.max_drift_ms:.2f}ms")
    """
    # Start timer for performance tracking
    start_time = time.time()

    # Default progress callback to no-op if None
    if progress_callback is None:
        progress_callback = lambda progress, status: None

    temp_files = []  # Track temporary files for cleanup

    try:
        # Step 1: Load TTS result JSON (0.05)
        progress_callback(0.05, "Loading TTS result JSON...")
        logger.info(f"Loading TTS result from: {tts_result_path}")

        if not tts_result_path.exists():
            raise AssemblyStageFailed(f"TTS result JSON not found: {tts_result_path}")

        try:
            with open(tts_result_path, 'r', encoding='utf-8') as f:
                tts_data = json.load(f)
        except json.JSONDecodeError as e:
            raise AssemblyStageFailed(f"Invalid TTS result JSON: {e}")

        segments_data = tts_data.get("segments", [])
        if not segments_data:
            raise AssemblyStageFailed("TTS result contains no segments")

        logger.info(f"Loaded {len(segments_data)} segments from TTS result")

        # Step 2: Create TimedSegment objects (0.10)
        progress_callback(0.10, "Creating timed segments...")
        timed_segments = []
        original_sample_rates = []

        for seg in segments_data:
            timed_seg = TimedSegment(
                segment_id=seg.get("segment_id", 0),
                audio_path=Path(seg["audio_path"]),
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=seg.get("speaker", "UNKNOWN")
            )
            timed_segments.append(timed_seg)

        logger.info(f"Created {len(timed_segments)} timed segments")

        # Step 3: Validate timestamp precision (0.15)
        progress_callback(0.15, "Validating timestamp precision...")
        logger.info("Validating timestamp precision with float64...")

        try:
            validate_timestamps_precision(timed_segments)
        except ValueError as e:
            raise AssemblyStageFailed(f"Timestamp validation failed: {e}")

        logger.info("Timestamp validation passed")

        # Step 4: Normalize all audio segments to 48kHz (0.25)
        progress_callback(0.25, "Normalizing audio to 48kHz...")
        logger.info("Normalizing audio segments to 48kHz...")

        try:
            normalized_paths, sample_rates_before = batch_normalize(
                [seg.audio_path for seg in timed_segments]
            )
            original_sample_rates = sample_rates_before

            # Update timed segments with normalized paths
            for seg, norm_path in zip(timed_segments, normalized_paths):
                if norm_path != seg.audio_path:
                    temp_files.append(norm_path)  # Track for cleanup
                seg.audio_path = norm_path

        except Exception as e:
            raise AssemblyStageFailed(f"Audio normalization failed: {e}")

        logger.info(f"Normalized {len(normalized_paths)} audio segments")
        sample_rate_normalized = any(
            before != 48000 for before in sample_rates_before
        )

        # Step 5: Concatenate normalized segments (0.45)
        progress_callback(0.45, "Concatenating audio segments...")
        logger.info("Concatenating normalized audio segments...")

        # Generate temp path for concatenated audio
        concat_audio_path = TEMP_DATA_DIR / f"assembly_concat_{int(time.time())}.wav"
        temp_files.append(concat_audio_path)

        try:
            total_duration = concatenate_audio_segments(
                timed_segments,
                output_path=concat_audio_path
            )
        except Exception as e:
            raise AssemblyStageFailed(f"Audio concatenation failed: {e}")

        logger.info(f"Concatenated audio duration: {total_duration:.2f}s")

        # Step 6: Validate sync at 5-minute intervals (0.60)
        progress_callback(0.60, "Validating sync at checkpoints...")
        logger.info("Validating sync at 5-minute intervals...")

        drift_validation: DriftValidationResult = validate_sync_at_intervals(
            timed_segments=timed_segments,
            video_fps=video_fps,
            checkpoint_interval=ASSEMBLY_CHECKPOINT_INTERVAL,
            tolerance_ms=ASSEMBLY_DRIFT_TOLERANCE_MS
        )

        # Check for drift issues
        drift_detected = not drift_validation.is_synced
        max_drift_ms = max(
            (abs(cp.drift_ms) for cp in drift_validation.checkpoints),
            default=0.0
        )

        if drift_detected:
            logger.warning(
                f"Drift detected: max {max_drift_ms:.2f}ms "
                f"(tolerance: {ASSEMBLY_DRIFT_TOLERANCE_MS}ms)"
            )
            logger.warning("User should review sync quality")
        else:
            logger.info(f"Sync validation passed (max drift: {max_drift_ms:.2f}ms)")

        # Step 7: Merge video with audio (0.80)
        progress_callback(0.80, "Merging video and audio with FFmpeg...")
        logger.info("Merging video with audio using sync-aware FFmpeg flags...")

        try:
            merge_result: MergeResult = merge_with_sync_validation(
                video_path=video_path,
                audio_path=concat_audio_path,
                output_path=output_path,
                async_correction=True  # Enable clock drift correction
            )
        except Exception as e:
            raise AssemblyStageFailed(f"FFmpeg merge failed: {e}")

        if not merge_result.success:
            raise AssemblyStageFailed(
                f"Merge failed: {merge_result.error_message}"
            )

        logger.info(f"Video merge successful: {output_path}")

        # Step 8: Export assembly result JSON (0.95)
        progress_callback(0.95, "Exporting assembly result...")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create result object
        result = AssemblyResult(
            output_path=output_path,
            total_duration=total_duration,
            segment_count=len(timed_segments),
            sync_checkpoints=drift_validation.checkpoints,
            drift_detected=drift_detected,
            max_drift_ms=max_drift_ms,
            sample_rate_normalized=sample_rate_normalized,
            original_sample_rates=original_sample_rates,
            processing_time=processing_time,
            video_fps=video_fps
        )

        # Export result JSON to output directory
        result_json_path = output_path.parent / f"{output_path.stem}_assembly_result.json"
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Assembly result exported to: {result_json_path}")

        # Step 9: Cleanup temporary files (1.0)
        progress_callback(1.0, "Cleanup complete")
        logger.info("Cleaning up temporary files...")

        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.debug(f"Deleted temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")

        logger.info(
            f"Assembly stage complete in {processing_time:.2f}s: "
            f"{len(timed_segments)} segments, "
            f"drift {'detected' if drift_detected else 'within tolerance'}"
        )

        return result

    except AssemblyStageFailed:
        # Clean up temp files on failure
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
        raise
    except Exception as e:
        # Clean up temp files on unexpected error
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
        raise AssemblyStageFailed(f"Unexpected error during assembly: {e}")
