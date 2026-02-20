"""
Lip sync stage orchestration module.

Connects audio_prep, latentsync_runner, wav2lip_runner, chunker, and validator
into a single pipeline-compatible run_lip_sync_stage() function.

Mirrors the pattern established by run_assembly_stage() in Phase 6.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.lip_sync.audio_prep import prepare_audio_for_lipsync
from src.lip_sync.latentsync_runner import run_latentsync_inference
from src.lip_sync.wav2lip_runner import run_wav2lip_inference
from src.lip_sync.chunker import (
    split_video_into_chunks,
    concatenate_video_chunks,
    get_video_duration,
)
from src.lip_sync.validator import validate_lip_sync_output, SyncValidation

logger = logging.getLogger(__name__)


class LipSyncStageFailed(Exception):
    """Raised when the lip sync stage cannot complete."""
    pass


@dataclass
class LipSyncResult:
    """Result of the lip sync stage."""

    output_path: Path
    model_used: str           # "latentsync" or "wav2lip"
    model_version: str        # "1.6" for LatentSync, "gan" for Wav2Lip GAN
    inference_steps: int      # LatentSync denoising steps (0 for Wav2Lip)
    guidance_scale: float     # LatentSync CFG scale (0.0 for Wav2Lip)
    processing_time: float    # Total wall-clock seconds
    input_video_path: Path
    input_audio_path: Path    # Path to the 16kHz resampled audio (not 48kHz original)
    chunks_processed: int     # 1 if no chunking, N if long video
    fallback_used: bool       # True when Wav2Lip was used instead of LatentSync
    sync_validation: Optional[SyncValidation]  # None if validation was skipped
    multi_speaker_mode: bool  # True when speakers_detected > 1

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "output_path": str(self.output_path),
            "model_used": self.model_used,
            "model_version": self.model_version,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "processing_time": self.processing_time,
            "input_video_path": str(self.input_video_path),
            "input_audio_path": str(self.input_audio_path),
            "chunks_processed": self.chunks_processed,
            "fallback_used": self.fallback_used,
            "multi_speaker_mode": self.multi_speaker_mode,
            "sync_validation": (
                self.sync_validation.to_dict() if self.sync_validation else None
            ),
        }


def run_lip_sync_stage(
    assembled_video_path: Path,
    output_dir: Path,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    enable_deepcache: bool = True,
    chunk_duration: int = 300,
    long_video_threshold: int = 300,
    speakers_detected: int = 1,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> LipSyncResult:
    """
    Run the complete lip sync stage: audio prep -> (chunking) -> inference -> validation.

    Uses LatentSync 1.6 as the primary model. Falls back to Wav2Lip GAN on RuntimeError
    (e.g. OOM, face detection failure, InsightFace Windows issue).

    For videos longer than long_video_threshold seconds the video is split into
    chunk_duration-second segments, processed individually, then concatenated.

    Args:
        assembled_video_path: Path to the assembled video from Phase 6 (any sample rate).
        output_dir: Directory for all outputs including the result JSON.
        inference_steps: LatentSync denoising steps. 20 = fast, 40-50 = higher quality.
        guidance_scale: LatentSync classifier-free guidance scale (default 1.5).
        enable_deepcache: Enable ~2x speedup via DeepCache (LatentSync commit f5040cf+).
        chunk_duration: Duration of each chunk when splitting long videos (seconds).
        long_video_threshold: Videos longer than this are chunked. Default 300 (5 min).
        speakers_detected: Number of speakers found in Phase 3. Logged as a warning
                           when > 1 because LatentSync targets a single face.
        progress_callback: Optional callback(progress: float, status: str) for UI updates.

    Returns:
        LipSyncResult with full metadata and a path to the lip-synced video.

    Raises:
        LipSyncStageFailed: If the input video is missing or inference fails for both models.

    Progress checkpoints:
        0.05 - Input validation complete
        0.10 - Audio resampled to 16kHz
        0.15 - Video duration checked, chunking decision made
        0.25 - Inference starting (or chunks split)
        0.45 - Inference 50% complete (or first chunk done)
        0.70 - Inference complete (or all chunks done)
        0.85 - Sync validation complete
        0.95 - Result JSON written
        1.00 - Stage complete
    """
    start_time = time.time()

    # Default progress callback to no-op if None
    if progress_callback is None:
        progress_callback = lambda p, s: None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_files: list[Path] = []
    temp_dirs: list[Path] = []

    try:
        # --- Step 1: Validate inputs (0.05) ---
        progress_callback(0.05, "Validating inputs...")

        assembled_video_path = Path(assembled_video_path)
        if not assembled_video_path.exists():
            raise LipSyncStageFailed(
                f"Assembled video not found: {assembled_video_path}"
            )

        logger.info(f"Lip sync stage starting: {assembled_video_path}")

        # Multi-speaker awareness
        multi_speaker_mode = speakers_detected > 1
        if multi_speaker_mode:
            logger.warning(
                f"speakers_detected={speakers_detected}: LatentSync targets a single face. "
                "Lip sync quality may be reduced for non-primary speakers."
            )

        # --- Step 2: Resample audio to 16kHz (0.10) ---
        progress_callback(0.10, "Resampling audio to 16kHz...")

        audio_work_dir = output_dir / "audio_prep"
        audio_16k_path = prepare_audio_for_lipsync(
            source_media_path=assembled_video_path,
            output_dir=audio_work_dir,
        )
        temp_files.append(audio_16k_path)
        logger.info(f"16kHz audio ready: {audio_16k_path}")

        # --- Step 3: Check duration and decide chunking (0.15) ---
        progress_callback(0.15, "Checking video duration...")

        video_duration = get_video_duration(assembled_video_path)
        use_chunking = video_duration > long_video_threshold
        logger.info(
            f"Video duration: {video_duration:.1f}s — "
            f"{'chunking required' if use_chunking else 'no chunking needed'}"
        )

        # Final output path for the lip-synced video
        output_video_path = output_dir / "lip_synced.mp4"

        fallback_used = False
        model_used = "latentsync"
        model_version = "1.6"

        if not use_chunking:
            # --- Single-pass inference (short video) ---
            progress_callback(0.25, "Starting LatentSync inference...")

            try:
                run_latentsync_inference(
                    video_path=assembled_video_path,
                    audio_path=audio_16k_path,
                    output_path=output_video_path,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    enable_deepcache=enable_deepcache,
                )
                progress_callback(0.70, "LatentSync inference complete.")
                logger.info("LatentSync inference succeeded.")

            except RuntimeError as latentsync_err:
                logger.warning(
                    f"LatentSync failed ({latentsync_err}), falling back to Wav2Lip..."
                )
                fallback_used = True
                model_used = "wav2lip"
                model_version = "gan"

                progress_callback(0.45, "LatentSync failed — running Wav2Lip fallback...")
                try:
                    run_wav2lip_inference(
                        video_path=assembled_video_path,
                        audio_path=audio_16k_path,
                        output_path=output_video_path,
                    )
                    progress_callback(0.70, "Wav2Lip fallback inference complete.")
                    logger.info("Wav2Lip fallback inference succeeded.")
                except Exception as wav2lip_err:
                    raise LipSyncStageFailed(
                        f"Both LatentSync and Wav2Lip failed.\n"
                        f"LatentSync error: {latentsync_err}\n"
                        f"Wav2Lip error: {wav2lip_err}"
                    )

            chunks_processed = 1

        else:
            # --- Chunked inference (long video) ---
            progress_callback(0.25, f"Splitting video into {chunk_duration}s chunks...")

            chunk_work_dir = output_dir / "chunks"
            temp_dirs.append(chunk_work_dir)

            chunks = split_video_into_chunks(
                video_path=assembled_video_path,
                audio_path=audio_16k_path,
                work_dir=chunk_work_dir,
                chunk_duration=chunk_duration,
            )
            logger.info(f"Split into {len(chunks)} chunks.")

            processed_chunk_paths: list[Path] = []

            for i, chunk in enumerate(chunks):
                # Report progress: 0.25 to 0.70 spread across chunks
                chunk_progress = 0.25 + (0.45 * (i / len(chunks)))
                progress_callback(
                    chunk_progress,
                    f"Processing chunk {i + 1}/{len(chunks)}..."
                )

                chunk_output = chunk_work_dir / f"chunk_{chunk.index:03d}_lipsync.mp4"

                try:
                    run_latentsync_inference(
                        video_path=chunk.video_path,
                        audio_path=chunk.audio_path,
                        output_path=chunk_output,
                        inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        enable_deepcache=enable_deepcache,
                    )
                    logger.info(f"LatentSync chunk {i + 1}/{len(chunks)} complete.")

                except RuntimeError as latentsync_err:
                    logger.warning(
                        f"LatentSync failed on chunk {i + 1} ({latentsync_err}), "
                        "falling back to Wav2Lip for this chunk..."
                    )
                    fallback_used = True
                    model_used = "wav2lip"
                    model_version = "gan"

                    try:
                        run_wav2lip_inference(
                            video_path=chunk.video_path,
                            audio_path=chunk.audio_path,
                            output_path=chunk_output,
                        )
                        logger.info(f"Wav2Lip fallback chunk {i + 1}/{len(chunks)} complete.")
                    except Exception as wav2lip_err:
                        raise LipSyncStageFailed(
                            f"Both LatentSync and Wav2Lip failed on chunk {i + 1}.\n"
                            f"LatentSync error: {latentsync_err}\n"
                            f"Wav2Lip error: {wav2lip_err}"
                        )

                processed_chunk_paths.append(chunk_output)

            # Concatenate processed chunks
            progress_callback(0.70, "Concatenating processed chunks...")
            concatenate_video_chunks(
                processed_chunk_paths=processed_chunk_paths,
                output_path=output_video_path,
                work_dir=chunk_work_dir,
            )
            logger.info(f"Chunk concatenation complete: {output_video_path}")
            chunks_processed = len(chunks)

        # --- Step 4: Sync validation (advisory only — never fails stage) (0.85) ---
        progress_callback(0.85, "Running sync validation...")

        sync_validation: Optional[SyncValidation] = None
        try:
            sync_validation = validate_lip_sync_output(output_video_path)
            if sync_validation.passed:
                logger.info(
                    f"Sync validation passed: {sync_validation.pass_rate:.1%} valid frames "
                    f"({sync_validation.valid_frames}/{sync_validation.sampled_frames} sampled)"
                )
            else:
                logger.warning(
                    f"Sync validation flagged low quality: "
                    f"{sync_validation.pass_rate:.1%} valid frames "
                    f"({sync_validation.valid_frames}/{sync_validation.sampled_frames} sampled). "
                    "Output may have black frames — review manually."
                )
        except Exception as val_err:
            logger.warning(f"Sync validation skipped (non-fatal): {val_err}")

        # --- Step 5: Write result JSON (0.95) ---
        progress_callback(0.95, "Writing result JSON...")

        processing_time = time.time() - start_time

        result = LipSyncResult(
            output_path=output_video_path,
            model_used=model_used,
            model_version=model_version,
            inference_steps=inference_steps if not fallback_used else 0,
            guidance_scale=guidance_scale if not fallback_used else 0.0,
            processing_time=processing_time,
            input_video_path=assembled_video_path,
            input_audio_path=audio_16k_path,
            chunks_processed=chunks_processed,
            fallback_used=fallback_used,
            sync_validation=sync_validation,
            multi_speaker_mode=multi_speaker_mode,
        )

        result_json_path = output_dir / "lip_sync_result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Lip sync result exported: {result_json_path}")

        # --- Complete (1.0) ---
        progress_callback(1.0, "Lip sync stage complete.")
        logger.info(
            f"Lip sync stage finished in {processing_time:.1f}s: "
            f"model={model_used}, chunks={chunks_processed}, "
            f"fallback={fallback_used}"
        )

        return result

    except LipSyncStageFailed:
        raise

    except Exception as exc:
        raise LipSyncStageFailed(
            f"Unexpected error during lip sync stage: {exc}"
        ) from exc
