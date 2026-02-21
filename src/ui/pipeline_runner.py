"""
Pipeline runner for Voice Dub UI.
Generator functions that wrap pipeline stages with progress tracking,
stage validation, and cancellation support.

Import strategy: Stage modules (asr_stage, translation_stage, tts_stage,
assembly_stage, lip_sync_stage) are imported lazily inside each generator
to avoid cascading ML library imports (pyannote, torch, TTS, etc.) at
module load time. The TYPE_CHECKING block provides type hints for editors.
"""
from __future__ import annotations

import gc
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gradio as gr
import pandas as pd
import torch

from src.config.settings import TEMP_DATA_DIR, OUTPUT_DATA_DIR
from src.ui.validators import (
    validate_asr_output,
    validate_translation_output,
    validate_tts_output,
    validate_lip_sync_output,
)

if TYPE_CHECKING:
    # Only imported during static analysis — never at runtime.
    # Prevents cascading pyannote/TTS/assembly imports when this module is loaded.
    from src.stages.asr_stage import ASRResult
    from src.stages.translation_stage import TranslationResult
    from src.stages.tts_stage import TTSResult
    from src.stages.assembly_stage import AssemblyResult
    from src.stages.lip_sync_stage import LipSyncResult

# Module-level cancellation event — set by cancel_pipeline(), checked between stages.
_cancel_event = threading.Event()


def run_asr_ui(video_path: str, hf_token: str, app_state: dict):
    """
    Generator for Step 1: validate video, extract audio, run ASR, return transcript DataFrame.

    Yields 6-tuple:
        (upload_col_visible, review_col_visible, transcript_df, app_state,
         status_text, asr_status_visible)
    """
    if not video_path:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            app_state,
            "No video uploaded.",
            gr.update(visible=True),
        )
        return

    if not hf_token or not hf_token.strip():
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            app_state,
            "HuggingFace token is required for speaker diarization.",
            gr.update(visible=True),
        )
        return

    try:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            app_state,
            "Validating video...",
            gr.update(visible=True),
        )

        # Lazy imports — keep module-level load free of heavy ML dependencies
        from src.video_processing import validate_video_file
        from src.video_processing.extractor import extract_audio
        from src.stages.asr_stage import run_asr_stage

        # Validate the video file
        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            yield (
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                app_state,
                f"Invalid video: {error_msg}",
                gr.update(visible=True),
            )
            return

        # Generate unique video ID and extract audio
        video_id = str(uuid.uuid4())[:8]
        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = str(TEMP_DATA_DIR / f"{video_id}_audio.wav")

        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            app_state,
            "Extracting audio...",
            gr.update(visible=True),
        )

        extract_audio(video_path, audio_path)

        # Run ASR stage
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            app_state,
            "Transcribing speech (this may take several minutes)...",
            gr.update(visible=True),
        )

        asr_result = run_asr_stage(
            audio_path=audio_path,
            video_id=video_id,
            huggingface_token=hf_token.strip(),
            progress_callback=lambda p, s: None,
            save_json=True,
        )

        # Validate ASR output
        is_valid, msg = validate_asr_output(asr_result)
        if not is_valid:
            yield (
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                app_state,
                f"ASR failed: {msg}",
                gr.update(visible=True),
            )
            return

        # Convert ASR segments to DataFrame for transcript editing
        rows = []
        for seg in asr_result.segments:
            rows.append([
                seg.id,
                seg.speaker,
                round(seg.start, 2),
                round(seg.end, 2),
                seg.text,
                round(getattr(seg, "confidence", 1.0), 3),
            ])
        df = pd.DataFrame(rows, columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])

        # Update app state with video and ASR paths for downstream stages
        new_state = {
            **app_state,
            "video_path": video_path,
            "video_id": video_id,
            "audio_path": audio_path,
            "asr_output_path": asr_result.output_path,
            "step": "review",
        }

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            df,
            new_state,
            msg,
            gr.update(visible=False),
        )

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_full_pipeline(transcript_df: pd.DataFrame, app_state: dict):
    """
    Generator for the full dubbing pipeline: Translation -> TTS -> Assembly -> Lip Sync.
    Called after the user reviews and edits the transcript.

    Yields 6-tuple:
        (review_col_visible, processing_col_visible, status_text, progress_value,
         output_col_visible, output_video_path)
    """
    global _cancel_event
    _cancel_event.clear()

    video_path = app_state.get("video_path")
    video_id = app_state.get("video_id")
    audio_path = app_state.get("audio_path")

    if not video_path or not video_id:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            "Error: no video loaded. Please complete Step 1 first.",
            0,
            gr.update(visible=False),
            None,
        )
        return

    # Switch UI to processing view
    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        "Starting pipeline...",
        0,
        gr.update(visible=False),
        None,
    )

    try:
        # Lazy imports — deferred until pipeline actually runs
        from src.stages.translation_stage import run_translation_stage
        from src.stages.tts_stage import run_tts_stage
        from src.stages.assembly_stage import run_assembly_stage
        from src.stages.lip_sync_stage import run_lip_sync_stage

        # ------------------------------------------------------------------ #
        # Stage 1: Serialize edited transcript to JSON                         #
        # ------------------------------------------------------------------ #
        asr_json_path = str(TEMP_DATA_DIR / f"{video_id}_transcript_edited.json")
        _write_edited_transcript(transcript_df, asr_json_path, video_id)

        # ------------------------------------------------------------------ #
        # Stage 2: Translation                                                 #
        # ------------------------------------------------------------------ #
        if _cancel_event.is_set():
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                "Cancelled.",
                0,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            "Translating segments...",
            10,
            gr.update(visible=False),
            None,
        )

        translation_json_path = str(TEMP_DATA_DIR / f"{video_id}_translation.json")
        translation_result = run_translation_stage(
            asr_json_path=asr_json_path,
            output_json_path=translation_json_path,
        )

        is_valid, msg = validate_translation_output(translation_result)
        if not is_valid:
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                f"Translation stopped: {msg}",
                20,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            f"Translation: {msg}",
            25,
            gr.update(visible=False),
            None,
        )

        # ------------------------------------------------------------------ #
        # Stage 3: Voice Cloning (TTS)                                         #
        # ------------------------------------------------------------------ #
        if _cancel_event.is_set():
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                "Cancelled.",
                25,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            "Cloning voices and synthesizing audio...",
            30,
            gr.update(visible=False),
            None,
        )

        tts_result = run_tts_stage(
            translation_json_path=translation_json_path,
            audio_path=audio_path,
        )

        is_valid, msg = validate_tts_output(tts_result)
        if not is_valid:
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                f"Voice cloning stopped: {msg}",
                50,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            f"TTS: {msg}",
            55,
            gr.update(visible=False),
            None,
        )

        # ------------------------------------------------------------------ #
        # Stage 4: Audio-Video Assembly                                        #
        # ------------------------------------------------------------------ #
        if _cancel_event.is_set():
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                "Cancelled.",
                55,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            "Assembling audio and video...",
            60,
            gr.update(visible=False),
            None,
        )

        # run_assembly_stage reads segment audio paths from the TTS result JSON
        tts_result_json = Path(tts_result.output_dir) / "tts_result.json"
        assembled_output = TEMP_DATA_DIR / f"{video_id}_assembled.mp4"

        assembly_result = run_assembly_stage(
            video_path=Path(video_path),
            tts_result_path=tts_result_json,
            output_path=assembled_output,
        )

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            "Assembly complete.",
            75,
            gr.update(visible=False),
            None,
        )

        # ------------------------------------------------------------------ #
        # Stage 5: Lip Sync                                                    #
        # ------------------------------------------------------------------ #
        if _cancel_event.is_set():
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                "Cancelled.",
                75,
                gr.update(visible=False),
                None,
            )
            return

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            "Synchronizing lip movements (this takes the longest)...",
            80,
            gr.update(visible=False),
            None,
        )

        lip_sync_output_dir = TEMP_DATA_DIR / f"{video_id}_lipsync"

        lip_sync_result = run_lip_sync_stage(
            assembled_video_path=assembly_result.output_path,
            output_dir=lip_sync_output_dir,
        )

        is_valid, msg = validate_lip_sync_output(lip_sync_result)
        if not is_valid:
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                f"Lip sync stopped: {msg}",
                90,
                gr.update(visible=False),
                None,
            )
            return

        # Copy final output to outputs directory
        OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_output = str(OUTPUT_DATA_DIR / f"{video_id}_dubbed.mp4")
        shutil.copy2(str(lip_sync_result.output_path), final_output)

        # ------------------------------------------------------------------ #
        # Complete                                                             #
        # ------------------------------------------------------------------ #
        yield (
            gr.update(visible=False),
            gr.update(visible=False),
            f"Complete! {msg}",
            100,
            gr.update(visible=True),
            final_output,
        )

    except Exception as e:
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            f"Pipeline error: {str(e)}",
            0,
            gr.update(visible=False),
            None,
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cancel_pipeline() -> str:
    """
    Signal the pipeline to stop at the next stage boundary.

    Sets the module-level _cancel_event. run_full_pipeline() checks this
    event between every stage and yields a 'Cancelled.' status if set.

    Returns:
        str: User-facing status message confirming cancellation was requested.
    """
    _cancel_event.set()
    return "Cancellation requested. Pipeline will stop at next stage boundary..."


def _write_edited_transcript(
    transcript_df: pd.DataFrame,
    output_path: str,
    video_id: str,
) -> None:
    """
    Serialize the user-edited transcript DataFrame back to the ASR JSON format
    that run_translation_stage() expects.

    The translation stage reads ``detected_language`` from the JSON. We default
    to "und" (undetermined) because the original language is not stored in the
    DataFrame — the translation stage will re-detect if needed.

    Args:
        transcript_df: DataFrame with columns [ID, Speaker, Start, End, Text, Confidence].
        output_path: Destination JSON file path.
        video_id: Video identifier written into the JSON envelope.
    """
    segments = []
    for _, row in transcript_df.iterrows():
        confidence = float(row["Confidence"])
        segments.append({
            "id": int(row["ID"]),
            "speaker": str(row["Speaker"]),
            "start": float(row["Start"]),
            "end": float(row["End"]),
            "duration": round(float(row["End"]) - float(row["Start"]), 6),
            "text": str(row["Text"]),
            "confidence": confidence,
            "needs_review": confidence < 0.7,
            "words": [],
        })

    output = {
        "video_id": video_id,
        "detected_language": "und",   # undetermined; translation stage will use SeamlessM4T
        "segments": segments,
        "total_segments": len(segments),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
