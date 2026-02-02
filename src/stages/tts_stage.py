"""
Complete TTS stage orchestration module.
Orchestrates reference extraction, speaker embedding, synthesis, and quality validation.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable, Dict
import json
import time

from src.models.model_manager import ModelManager
from src.tts.reference_extractor import extract_reference_samples
from src.tts.speaker_embeddings import SpeakerEmbeddingCache
from src.tts.xtts_generator import XTTSGenerator
from src.tts.quality_validator import QualityValidator
from src.config.settings import TEMP_DATA_DIR


@dataclass
class SynthesizedSegment:
    """Single synthesized audio segment with metadata."""
    segment_id: int
    speaker: str
    start: float
    end: float
    original_duration: float
    translated_text: str

    # Synthesis results
    audio_path: str
    actual_duration: float
    duration_error: float  # Percentage
    speed_used: float
    synthesis_attempts: int

    # Quality metrics
    quality_passed: bool
    flagged_for_review: bool
    rejection_reason: Optional[str]

    # Emotion preservation
    emotion_preserved: bool  # True if pitch variance ratio in acceptable range
    pitch_variance_ratio: Optional[float]


@dataclass
class TTSResult:
    """Complete TTS stage output."""
    video_id: str
    total_segments: int
    successful_segments: int
    failed_segments: int
    flagged_count: int
    flagged_segment_ids: List[int]
    emotion_flagged_count: int  # Segments with emotion preservation issues
    avg_duration_error: float
    segments: List[SynthesizedSegment]
    processing_time: float
    output_dir: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class TTSStageFailed(Exception):
    """Raised when TTS stage fails quality validation."""
    pass


def run_tts_stage(
    translation_json_path: str,
    audio_path: str,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> TTSResult:
    """
    Complete TTS pipeline: reference extraction → embedding generation → synthesis → validation.

    Orchestrates the entire TTS pipeline:
    1. Load translation JSON
    2. Create output directory
    3. Extract reference samples per speaker
    4. Generate speaker embeddings
    5. Synthesize all segments with duration matching
    6. Validate audio quality with emotion preservation check
    7. Build result structure
    8. Export result JSON
    9. Cleanup model and CUDA cache

    Args:
        translation_json_path: Path to translation stage output JSON
        audio_path: Path to extracted audio file (from video processing)
        output_dir: Optional directory for outputs (default: auto-generate in temp)
        progress_callback: Optional callback(progress: float, status: str) for UI updates

    Returns:
        TTSResult: Complete result with synthesized segments, quality flags, and metadata

    Raises:
        TTSStageFailed: If >50% segments fail quality validation

    Example:
        >>> result = run_tts_stage(
        ...     translation_json_path="data/temp/video_translation.json",
        ...     audio_path="data/temp/video_audio.wav",
        ...     output_dir="data/temp/tts_output",
        ...     progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
        ... )
        >>> print(f"Synthesized {result.successful_segments}/{result.total_segments} segments")
        >>> print(f"Flagged for review: {result.flagged_count}")
        >>> print(f"Emotion issues: {result.emotion_flagged_count}")
    """
    # Start timer for performance tracking
    start_time = time.time()

    # Create model manager for sequential loading
    model_manager = ModelManager(verbose=True)

    # Default progress callback to no-op if None
    if progress_callback is None:
        progress_callback = lambda progress, status: None

    try:
        # Step 1: Load translation JSON
        progress_callback(0.05, "Loading translation JSON...")
        print(f"Loading translation data from: {translation_json_path}")

        with open(translation_json_path, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)

        video_id = translation_data["video_id"]
        segments = translation_data["segments"]

        print(f"Loaded {len(segments)} segments for TTS")

        # Step 2: Create output directory if needed
        progress_callback(0.10, "Setting up output directory...")
        if output_dir is None:
            output_dir = str(TEMP_DATA_DIR / f"{video_id}_tts")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")

        # Step 3: Extract reference samples per speaker
        progress_callback(0.15, "Extracting reference samples...")
        print("\nExtracting reference audio samples per speaker...")

        reference_paths = extract_reference_samples(
            translation_json=Path(translation_json_path),
            audio_path=Path(audio_path),
            output_dir=output_path
        )

        if not reference_paths:
            raise TTSStageFailed("No reference samples extracted - cannot proceed with TTS")

        print(f"✓ Extracted references for {len(reference_paths)} speakers")

        # Step 4: Generate speaker embeddings
        progress_callback(0.25, "Generating speaker embeddings...")
        print("\nGenerating speaker embeddings...")

        embedding_cache = SpeakerEmbeddingCache(model_manager)

        for speaker_id, ref_path in reference_paths.items():
            print(f"  Processing speaker: {speaker_id}")
            embedding_cache.add_speaker(speaker_id, str(ref_path))

        print(f"✓ Generated embeddings for {len(reference_paths)} speakers")

        # Step 5: Synthesize all segments with duration matching
        progress_callback(0.35, "Initializing XTTS generator...")
        print("\nInitializing XTTS generator...")

        generator = XTTSGenerator(model_manager, embedding_cache)

        # Prepare segment data for synthesis
        synthesis_segments = []
        for seg in segments:
            synthesis_segments.append({
                'segment_id': seg['segment_id'],
                'speaker': seg['speaker'],
                'translated_text': seg['translated_text'],
                'duration': seg['duration'],
                'start': seg['start'],
                'end': seg['end']
            })

        # Synthesize with progress updates
        def synthesis_progress(progress: float, message: str):
            # Map synthesis progress (0.0-1.0) to overall progress (0.35-0.80)
            overall_progress = 0.35 + (progress * 0.45)
            progress_callback(overall_progress, message)

        print("\nSynthesizing segments...")
        synthesis_results = generator.synthesize_all_segments(
            segments=synthesis_segments,
            output_dir=output_path,
            progress_callback=synthesis_progress
        )

        # Step 6: Validate audio quality with emotion preservation check
        progress_callback(0.85, "Validating audio quality...")
        print("\nValidating audio quality...")

        validator = QualityValidator()

        # Prepare validation data - need to add speaker_id for reference lookup
        validation_data = []
        for result in synthesis_results:
            validation_data.append({
                'segment_id': result['segment_id'],
                'audio_path': result['audio_path'],
                'target_duration': result['target_duration'],
                'speaker_id': result['speaker']  # Use 'speaker' as 'speaker_id' for reference lookup
            })

        quality_results, quality_summary = validator.validate_batch(
            synthesis_results=validation_data,
            reference_dir=output_path  # Use speaker references for emotion comparison
        )

        # Process validation results - update segment metadata
        rejected_count = 0
        emotion_flagged_count = 0
        for synth_result, quality in zip(synthesis_results, quality_results):
            # Check emotion preservation
            emotion_preserved = (
                quality.pitch_variance_ratio is not None and
                0.6 <= quality.pitch_variance_ratio <= 1.5
            )

            # Count emotion flags (marginal or poor emotion preservation)
            if quality.pitch_variance_ratio is not None:
                if quality.pitch_variance_ratio < 0.6 or quality.pitch_variance_ratio > 1.5:
                    emotion_flagged_count += 1

            # Update synthesis result with quality data
            synth_result['quality_passed'] = quality.passes_quality
            synth_result['flagged_for_review'] = quality.flagged_for_review
            synth_result['rejection_reason'] = quality.rejection_reason
            synth_result['emotion_preserved'] = emotion_preserved
            synth_result['pitch_variance_ratio'] = quality.pitch_variance_ratio

            if not quality.passes_quality:
                rejected_count += 1

        # Fail stage if too many segments rejected
        if rejected_count > len(segments) * 0.5:
            raise TTSStageFailed(
                f"Quality validation failed: {rejected_count}/{len(segments)} segments rejected. "
                f"Check audio quality and reference samples."
            )

        # Log quality summary
        print(f"\nQuality validation: {quality_summary['passed']}/{quality_summary['total']} passed, "
              f"{quality_summary['flagged']} flagged, {quality_summary['rejected']} rejected")
        print(f"Emotion preservation: {emotion_flagged_count} segments flagged for emotion issues")

        # Step 7: Build result structure
        progress_callback(0.90, "Building TTS result...")
        print("\nBuilding TTS result structure...")

        synthesized_segment_objects = []
        flagged_segment_ids = []
        successful_count = 0
        failed_count = 0
        total_duration_error = 0.0

        for synth_result in synthesis_results:
            # Find corresponding original segment data
            original_seg = next(
                (s for s in segments if s['segment_id'] == synth_result['segment_id']),
                None
            )

            if original_seg is None:
                print(f"WARNING: Could not find original segment for ID {synth_result['segment_id']}")
                continue

            # Determine if segment failed
            is_failed = synth_result.get('failed', False)
            if is_failed:
                failed_count += 1
            else:
                successful_count += 1

            # Check if flagged for review
            is_flagged = synth_result.get('flagged_for_review', False) or synth_result.get('flagged', False)
            if is_flagged:
                flagged_segment_ids.append(synth_result['segment_id'])

            # Calculate duration error as percentage
            duration_error_pct = (
                abs(synth_result['actual_duration'] - synth_result['target_duration'])
                / synth_result['target_duration'] * 100
                if synth_result['target_duration'] > 0 else 0.0
            )

            total_duration_error += duration_error_pct

            # Create SynthesizedSegment object
            synth_obj = SynthesizedSegment(
                segment_id=synth_result['segment_id'],
                speaker=synth_result['speaker'],
                start=original_seg['start'],
                end=original_seg['end'],
                original_duration=original_seg['duration'],
                translated_text=original_seg['translated_text'],
                audio_path=synth_result['audio_path'] if synth_result['audio_path'] else "",
                actual_duration=synth_result['actual_duration'],
                duration_error=duration_error_pct,
                speed_used=synth_result['speed_used'],
                synthesis_attempts=synth_result['attempts'],
                quality_passed=synth_result.get('quality_passed', False),
                flagged_for_review=is_flagged,
                rejection_reason=synth_result.get('rejection_reason'),
                emotion_preserved=synth_result.get('emotion_preserved', False),
                pitch_variance_ratio=synth_result.get('pitch_variance_ratio')
            )
            synthesized_segment_objects.append(synth_obj)

        # Calculate aggregate metrics
        avg_duration_error = total_duration_error / len(synthesized_segment_objects) if synthesized_segment_objects else 0.0

        processing_time = time.time() - start_time

        result = TTSResult(
            video_id=video_id,
            total_segments=len(synthesized_segment_objects),
            successful_segments=successful_count,
            failed_segments=failed_count,
            flagged_count=len(flagged_segment_ids),
            flagged_segment_ids=flagged_segment_ids,
            emotion_flagged_count=emotion_flagged_count,
            avg_duration_error=avg_duration_error,
            segments=synthesized_segment_objects,
            processing_time=processing_time,
            output_dir=str(output_path)
        )

        # Step 8: Export result JSON (segment manifest)
        progress_callback(0.95, "Exporting result JSON...")
        print("\nExporting TTS result JSON...")

        result_json_path = output_path / "tts_result.json"
        result_dict = result.to_dict()

        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        print(f"✓ Result JSON saved to: {result_json_path}")

        # Summary message
        print(f"\n✓ TTS complete: {successful_count}/{result.total_segments} successful, "
              f"{failed_count} failed, {len(flagged_segment_ids)} flagged")
        print(f"  Average duration error: {avg_duration_error:.2f}%")
        print(f"  Emotion preservation issues: {emotion_flagged_count} segments")
        print(f"  Processing time: {processing_time:.1f}s")

        progress_callback(1.0, "TTS stage complete")

        return result

    finally:
        # Step 9: Cleanup - unload models and clear CUDA cache
        print("\nCleaning up TTS models...")
        model_manager.unload_current_model()
        print("✓ Cleanup complete")
