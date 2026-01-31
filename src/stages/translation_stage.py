"""
Complete translation stage orchestration module.
Orchestrates translation from ASR JSON to translated JSON output with multi-candidate selection.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Callable
import json
import time

from src.models.model_manager import ModelManager
from src.stages.translation import (
    Translator,
    CandidateRanker,
    chunk_transcript_with_overlap,
    merge_translated_chunks,
    rank_candidates
)
from src.config.settings import (
    TEMP_DATA_DIR,
    TRANSLATION_NUM_CANDIDATES,
    TRANSLATION_CONFIDENCE_THRESHOLD,
    TRANSLATION_DURATION_TOLERANCE,
    TRANSLATION_MAX_CHUNK_TOKENS,
    TRANSLATION_TARGET_LANGUAGE,
    TRANSLATION_APPROX_CHARS_PER_TOKEN
)


@dataclass
class TranslatedSegment:
    """Single translated segment with metadata."""
    segment_id: int
    speaker: str
    start: float
    end: float
    duration: float
    original_text: str
    source_language: str
    translated_text: str
    translation_confidence: float
    duration_ratio: float
    is_valid_duration: bool
    all_candidates: List[str]  # All candidate translations (for review if needed)
    flagged: bool  # True if confidence below threshold


@dataclass
class TranslationResult:
    """Complete translation stage output."""
    video_id: str
    source_language: str
    target_language: str  # Always "eng" for Phase 4
    total_segments: int
    flagged_count: int
    flagged_segment_ids: List[int]
    avg_confidence: float
    avg_duration_ratio: float
    segments: List[TranslatedSegment]
    processing_time: float
    output_path: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def run_translation_stage(
    asr_json_path: str,
    output_json_path: Optional[str] = None,
    num_candidates: int = None,
    confidence_threshold: float = None,
    duration_tolerance: float = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> TranslationResult:
    """
    Complete translation pipeline: ASR JSON → translated JSON with multi-candidate selection.

    Orchestrates the entire translation pipeline:
    1. Load ASR results from JSON (speaker-labeled transcript with timestamps)
    2. Determine chunking strategy (single batch vs overlapping chunks)
    3. Load SeamlessM4T v2 via ModelManager
    4. Translate segments with multi-candidate beam search
    5. Rank candidates by confidence (60%) + duration fit (40%)
    6. Select best translation per segment
    7. Validate duration constraints (flag if outside ±10%)
    8. Flag low-confidence segments (<70% threshold)
    9. Export translated JSON with all metadata
    10. Cleanup model and CUDA cache

    Args:
        asr_json_path: Path to ASR output JSON file (from run_asr_stage)
        output_json_path: Optional path for translation JSON output (default: auto-generate)
        num_candidates: Beam width for candidate generation (default: TRANSLATION_NUM_CANDIDATES)
        confidence_threshold: Flag segments below this (default: TRANSLATION_CONFIDENCE_THRESHOLD)
        duration_tolerance: Acceptable duration deviation ±% (default: TRANSLATION_DURATION_TOLERANCE)
        progress_callback: Optional callback(progress: float, status: str) for UI updates

    Returns:
        TranslationResult: Complete result with translated segments, confidence flags, and metadata

    Example:
        >>> result = run_translation_stage(
        ...     asr_json_path="data/temp/video123_transcript.json",
        ...     output_json_path="data/temp/video123_translation.json",
        ...     progress_callback=lambda p, s: print(f"[{p*100:.0f}%] {s}")
        ... )
        >>> print(f"Translated {result.total_segments} segments")
        >>> print(f"Average confidence: {result.avg_confidence:.2f}")
        >>> print(f"Flagged for review: {result.flagged_count} segments")
    """
    # Start timer for performance tracking
    start_time = time.time()

    # Use default values if not provided
    if num_candidates is None:
        num_candidates = TRANSLATION_NUM_CANDIDATES
    if confidence_threshold is None:
        confidence_threshold = TRANSLATION_CONFIDENCE_THRESHOLD
    if duration_tolerance is None:
        duration_tolerance = TRANSLATION_DURATION_TOLERANCE

    # Create model manager for sequential loading
    model_manager = ModelManager(verbose=True)

    # Default progress callback to no-op if None
    if progress_callback is None:
        progress_callback = lambda progress, status: None

    try:
        # Step 1: Load ASR results from JSON
        progress_callback(0.05, "Loading ASR results from JSON...")
        print(f"Loading ASR results from: {asr_json_path}")

        with open(asr_json_path, 'r', encoding='utf-8') as f:
            asr_data = json.load(f)

        video_id = asr_data["video_id"]
        source_language = asr_data["detected_language"]
        asr_segments = asr_data["segments"]

        print(f"Loaded {len(asr_segments)} segments in {source_language}")

        # Step 2: Determine chunking strategy
        progress_callback(0.10, "Analyzing transcript length...")
        print("Analyzing transcript length for chunking strategy...")

        # Estimate total tokens
        total_tokens = sum(
            len(seg["text"]) // TRANSLATION_APPROX_CHARS_PER_TOKEN
            for seg in asr_segments
        )

        needs_chunking = total_tokens > TRANSLATION_MAX_CHUNK_TOKENS

        if needs_chunking:
            print(f"Transcript has ~{total_tokens} tokens, will use chunking with overlap")
        else:
            print(f"Transcript has ~{total_tokens} tokens, single batch translation")

        # Step 3: Initialize components
        progress_callback(0.15, "Loading translation model...")
        print("Initializing translation components...")

        translator = Translator(model_manager)
        ranker = CandidateRanker(
            confidence_weight=0.6,
            duration_weight=0.4
        )

        # Step 4: Translate all segments with context preservation
        all_translated_segments = []

        if needs_chunking:
            # Step 4a: Chunk transcript with overlap
            progress_callback(0.20, "Chunking transcript for context preservation...")
            print("Chunking transcript with overlap...")

            chunks = chunk_transcript_with_overlap(
                segments=asr_segments,
                max_tokens=TRANSLATION_MAX_CHUNK_TOKENS,
                overlap_tokens=128
            )

            print(f"Split into {len(chunks)} chunks")

            # Step 4b: Translate each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_progress = 0.20 + (chunk_idx / len(chunks)) * 0.60
                progress_callback(
                    chunk_progress,
                    f"Translating chunk {chunk_idx + 1}/{len(chunks)}..."
                )
                print(f"\nTranslating chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk['segments'])} segments)...")

                # Extract texts from chunk segments
                chunk_texts = [seg["text"] for seg in chunk["segments"]]

                # Batch translate this chunk's segments
                batch_results = translator.translate_batch(
                    segments=chunk_texts,
                    source_lang=source_language,
                    num_candidates=num_candidates
                )

                # Step 4c: Rank candidates for each segment in chunk
                for seg_idx, (segment, result) in enumerate(zip(chunk["segments"], batch_results)):
                    best, all_ranked = rank_candidates(
                        candidates=result["candidates"],
                        scores=result["scores"],
                        original_duration=segment["duration"]
                    )

                    translated_segment = {
                        "original_idx": chunk["start_idx"] + seg_idx,
                        "segment": segment,
                        "text": best["candidate"],
                        "confidence": best["model_confidence"],
                        "duration_ratio": best["duration_ratio"],
                        "is_valid_duration": best["is_valid_duration"],
                        "all_candidates": result["candidates"]
                    }
                    all_translated_segments.append(translated_segment)

            # Step 4d: Merge overlapping chunks (deduplicate)
            progress_callback(0.85, "Merging translated chunks...")
            print("\nMerging translated chunks (deduplicating overlaps)...")

            final_segments = merge_translated_chunks(all_translated_segments, chunks)
            print(f"Merged to {len(final_segments)} unique segments")

        else:
            # No chunking: single batch translation
            progress_callback(0.25, "Translating all segments...")
            print("Translating all segments in single batch...")

            all_texts = [seg["text"] for seg in asr_segments]

            batch_results = translator.translate_batch(
                segments=all_texts,
                source_lang=source_language,
                num_candidates=num_candidates
            )

            # Rank candidates for each segment
            progress_callback(0.70, "Ranking translation candidates...")
            print("Ranking translation candidates...")

            final_segments = []
            for seg_idx, (segment, result) in enumerate(zip(asr_segments, batch_results)):
                best, all_ranked = rank_candidates(
                    candidates=result["candidates"],
                    scores=result["scores"],
                    original_duration=segment["duration"]
                )

                translated_segment = {
                    "original_idx": seg_idx,
                    "segment": segment,
                    "text": best["candidate"],
                    "confidence": best["model_confidence"],
                    "duration_ratio": best["duration_ratio"],
                    "is_valid_duration": best["is_valid_duration"],
                    "all_candidates": result["candidates"]
                }
                final_segments.append(translated_segment)

        # Step 5: Validate duration constraints and build result
        progress_callback(0.90, "Building translation result...")
        print("\nValidating duration constraints...")

        translated_segment_objects = []
        flagged_segment_ids = []
        total_confidence = 0.0
        total_duration_ratio = 0.0
        invalid_duration_count = 0

        for trans_seg in final_segments:
            seg = trans_seg["segment"]

            # Check if flagged (low confidence)
            is_flagged = trans_seg["confidence"] < confidence_threshold
            if is_flagged:
                flagged_segment_ids.append(seg["id"])

            # Track invalid durations
            if not trans_seg["is_valid_duration"]:
                invalid_duration_count += 1

            # Create TranslatedSegment object
            translated_obj = TranslatedSegment(
                segment_id=seg["id"],
                speaker=seg["speaker"],
                start=seg["start"],
                end=seg["end"],
                duration=seg["duration"],
                original_text=seg["text"],
                source_language=source_language,
                translated_text=trans_seg["text"],
                translation_confidence=trans_seg["confidence"],
                duration_ratio=trans_seg["duration_ratio"],
                is_valid_duration=trans_seg["is_valid_duration"],
                all_candidates=trans_seg["all_candidates"],
                flagged=is_flagged
            )
            translated_segment_objects.append(translated_obj)

            total_confidence += trans_seg["confidence"]
            total_duration_ratio += trans_seg["duration_ratio"]

        # Calculate aggregate metrics
        num_segments = len(translated_segment_objects)
        avg_confidence = total_confidence / num_segments if num_segments > 0 else 0.0
        avg_duration_ratio = total_duration_ratio / num_segments if num_segments > 0 else 0.0

        # Log warnings
        if invalid_duration_count > num_segments * 0.2:
            print(f"WARNING: {invalid_duration_count}/{num_segments} segments ({invalid_duration_count/num_segments*100:.1f}%) have invalid duration (outside ±{duration_tolerance*100:.0f}%)")

        if len(flagged_segment_ids) > 0:
            print(f"Flagged {len(flagged_segment_ids)} segments with confidence < {confidence_threshold:.2f}")

        # Step 6: Build result structure
        processing_time = time.time() - start_time

        result = TranslationResult(
            video_id=video_id,
            source_language=source_language,
            target_language=TRANSLATION_TARGET_LANGUAGE,
            total_segments=num_segments,
            flagged_count=len(flagged_segment_ids),
            flagged_segment_ids=flagged_segment_ids,
            avg_confidence=avg_confidence,
            avg_duration_ratio=avg_duration_ratio,
            segments=translated_segment_objects,
            processing_time=processing_time
        )

        # Step 7: Export JSON (if output_json_path provided)
        if output_json_path:
            progress_callback(0.95, "Saving JSON output...")
            print(f"\nSaving translation result to: {output_json_path}")

            # Ensure output directory exists
            Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert result to dict and write JSON
            result_dict = result.to_dict()

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

            result.output_path = str(output_json_path)
            print(f"Translation saved to: {output_json_path}")

        # Summary message
        print(f"\nTranslation complete: {num_segments} segments, avg confidence {avg_confidence:.2f}, {len(flagged_segment_ids)} flagged")
        print(f"Processing time: {processing_time:.1f}s")

        progress_callback(1.0, "Translation complete")

        return result

    finally:
        # Step 8: Cleanup - unload models and clear CUDA cache
        print("\nCleaning up translation model...")
        model_manager.unload_current_model()
