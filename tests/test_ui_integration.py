"""
Integration tests for Voice Dub UI wiring and pipeline runner.
No GPU required — ML stage calls are mocked or avoided via input validation.

Tests cover:
  - App structure (imports, event handler registration)
  - run_asr_ui generator input validation
  - cancel_pipeline event signalling
  - run_full_pipeline generator error/cancel handling
  - _write_edited_transcript serialisation
  - validators importability smoke test
"""
import json
import os
import tempfile

import gradio as gr
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# App structure
# ---------------------------------------------------------------------------

class TestAppStructure:
    def test_app_imports_without_error(self):
        """demo must be a gr.Blocks instance."""
        from src.ui.app import demo
        assert isinstance(demo, gr.Blocks)

    def test_demo_has_event_handlers(self):
        """At least one event handler must be registered."""
        from src.ui.app import demo
        # Gradio stores event handler functions in demo.fns
        assert hasattr(demo, "fns")
        assert len(demo.fns) > 0

    def test_app_has_all_seven_handlers(self):
        """Seven UI interactions must be wired (includes queue registration)."""
        from src.ui.app import demo
        # We wired 7 user interactions (change + 6 clicks). Gradio may add
        # internal callbacks; assert we have at least 7 registered functions.
        assert len(demo.fns) >= 7

    def test_pipeline_runner_importable(self):
        """Core pipeline runner functions must be importable."""
        from src.ui.pipeline_runner import (
            run_asr_ui, run_full_pipeline, cancel_pipeline, _cancel_event,
        )
        import inspect
        assert inspect.isgeneratorfunction(run_asr_ui)
        assert inspect.isgeneratorfunction(run_full_pipeline)
        assert callable(cancel_pipeline)
        assert hasattr(_cancel_event, "set")
        assert hasattr(_cancel_event, "is_set")

    def test_validators_importable_from_ui(self):
        """All four validators must be importable and callable."""
        from src.ui.validators import (
            validate_asr_output,
            validate_translation_output,
            validate_tts_output,
            validate_lip_sync_output,
        )
        assert all(callable(f) for f in [
            validate_asr_output,
            validate_translation_output,
            validate_tts_output,
            validate_lip_sync_output,
        ])


# ---------------------------------------------------------------------------
# run_asr_ui generator — input validation (no real ASR)
# ---------------------------------------------------------------------------

class TestRunASRUIGenerator:
    def test_rejects_empty_video_path(self):
        """Generator must yield error state immediately for empty video path."""
        from src.ui.pipeline_runner import run_asr_ui
        gen = run_asr_ui("", "hf_token", {})
        result = next(gen)
        # upload_section stays visible, review_section stays hidden
        assert result[0] == gr.update(visible=True)
        assert result[1] == gr.update(visible=False)

    def test_rejects_missing_hf_token(self):
        """Generator must yield error state for empty HuggingFace token."""
        from src.ui.pipeline_runner import run_asr_ui
        gen = run_asr_ui("/some/video.mp4", "", {})
        result = next(gen)
        assert result[0] == gr.update(visible=True)
        assert result[1] == gr.update(visible=False)

    def test_rejects_none_hf_token(self):
        """Generator must reject None HuggingFace token."""
        from src.ui.pipeline_runner import run_asr_ui
        gen = run_asr_ui("/some/video.mp4", None, {})
        result = next(gen)
        assert result[0] == gr.update(visible=True)

    def test_yields_six_tuple_on_error(self):
        """Error yield must be a 6-element tuple (matches outputs list in app.py)."""
        from src.ui.pipeline_runner import run_asr_ui
        gen = run_asr_ui("", "hf_token", {})
        result = next(gen)
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_error_status_text_is_nonempty_string(self):
        """5th element (status_text) must be a non-empty string on error."""
        from src.ui.pipeline_runner import run_asr_ui
        gen = run_asr_ui("", "hf_token", {})
        result = next(gen)
        status_text = result[4]
        assert isinstance(status_text, str)
        assert len(status_text) > 0


# ---------------------------------------------------------------------------
# cancel_pipeline
# ---------------------------------------------------------------------------

class TestCancelPipeline:
    def test_cancel_sets_event(self):
        """cancel_pipeline() must set the threading.Event."""
        from src.ui.pipeline_runner import cancel_pipeline, _cancel_event
        _cancel_event.clear()
        assert not _cancel_event.is_set()
        cancel_pipeline()
        assert _cancel_event.is_set()
        _cancel_event.clear()  # cleanup

    def test_cancel_returns_string(self):
        """cancel_pipeline() must return a non-empty string status message."""
        from src.ui.pipeline_runner import cancel_pipeline, _cancel_event
        _cancel_event.clear()
        result = cancel_pipeline()
        assert isinstance(result, str)
        assert len(result) > 0
        _cancel_event.clear()

    def test_cancel_message_contains_cancel_word(self):
        """Status message must mention cancellation."""
        from src.ui.pipeline_runner import cancel_pipeline, _cancel_event
        _cancel_event.clear()
        result = cancel_pipeline()
        assert "cancel" in result.lower() or "stop" in result.lower()
        _cancel_event.clear()


# ---------------------------------------------------------------------------
# run_full_pipeline generator — error paths (no real ML)
# ---------------------------------------------------------------------------

class TestRunFullPipelineGenerator:
    def test_rejects_missing_video_in_state(self):
        """Generator must yield error when state has no video_path."""
        from src.ui.pipeline_runner import run_full_pipeline, _cancel_event
        _cancel_event.clear()
        df = pd.DataFrame(
            columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"]
        )
        gen = run_full_pipeline(df, {})
        result = next(gen)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_clears_cancel_event_on_start(self):
        """Generator must clear _cancel_event before starting, even if pre-set."""
        from src.ui.pipeline_runner import run_full_pipeline, _cancel_event
        _cancel_event.set()  # simulate a leftover cancel from a previous run
        df = pd.DataFrame(
            columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"]
        )
        # Generator with no video in state will error early
        gen = run_full_pipeline(df, {})
        next(gen)  # trigger execution
        # Cancel event must be cleared at generator start
        assert not _cancel_event.is_set()

    def test_error_yield_is_six_tuple(self):
        """Error yield from run_full_pipeline must be 6-tuple."""
        from src.ui.pipeline_runner import run_full_pipeline, _cancel_event
        _cancel_event.clear()
        df = pd.DataFrame(
            columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"]
        )
        gen = run_full_pipeline(df, {})
        result = next(gen)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# _write_edited_transcript
# ---------------------------------------------------------------------------

class TestWriteEditedTranscript:
    def test_serializes_dataframe_to_json(self):
        """Edited transcript DataFrame must round-trip through JSON correctly."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "SPEAKER_00", 0.0, 2.5, "Hello world", 0.95],
            [1, "SPEAKER_01", 2.5, 5.0, "How are you", 0.88],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "test_vid")
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["video_id"] == "test_vid"
            assert len(data["segments"]) == 2
            assert data["segments"][0]["text"] == "Hello world"
            assert data["segments"][1]["speaker"] == "SPEAKER_01"
        finally:
            os.unlink(tmp_path)

    def test_low_confidence_flagged_needs_review(self):
        """Segments with confidence < 0.7 must have needs_review=True."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "S1", 0.0, 1.0, "low confidence text", 0.3],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "vid")
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["segments"][0]["needs_review"] is True
        finally:
            os.unlink(tmp_path)

    def test_high_confidence_not_flagged(self):
        """Segments with confidence >= 0.7 must have needs_review=False."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "S1", 0.0, 1.0, "clear speech", 0.95],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "vid")
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["segments"][0]["needs_review"] is False
        finally:
            os.unlink(tmp_path)

    def test_json_envelope_has_total_segments(self):
        """JSON envelope must include total_segments count."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "S1", 0.0, 1.0, "first", 0.9],
            [1, "S1", 1.0, 2.0, "second", 0.9],
            [2, "S2", 2.0, 3.0, "third", 0.9],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "v1")
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["total_segments"] == 3
        finally:
            os.unlink(tmp_path)

    def test_duration_computed_from_start_end(self):
        """Segment duration must equal end - start."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "S1", 1.5, 4.0, "test", 0.8],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "v1")
            with open(tmp_path) as f:
                data = json.load(f)
            seg = data["segments"][0]
            assert abs(seg["duration"] - (4.0 - 1.5)) < 1e-4
        finally:
            os.unlink(tmp_path)

    def test_detected_language_is_und(self):
        """JSON envelope must set detected_language to 'und'."""
        from src.ui.pipeline_runner import _write_edited_transcript
        df = pd.DataFrame([
            [0, "S1", 0.0, 1.0, "text", 0.9],
        ], columns=["ID", "Speaker", "Start", "End", "Text", "Confidence"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            _write_edited_transcript(df, tmp_path, "v1")
            with open(tmp_path) as f:
                data = json.load(f)
            assert data["detected_language"] == "und"
        finally:
            os.unlink(tmp_path)
