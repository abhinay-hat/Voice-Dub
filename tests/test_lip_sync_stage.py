"""
Integration tests for lip sync stage orchestration.

Tests verify module structure, imports, dataclasses, audio preparation,
chunking, fallback logic, and progress callbacks without requiring actual
video files, GPU, or the LatentSync conda environment.

All subprocess calls are mocked so LatentSync/Wav2Lip binaries are not needed.
"""
import sys
import os
import json
import tempfile
import dataclasses
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Test 1: All imports
# ---------------------------------------------------------------------------

def test_lip_sync_imports():
    """Verify all lip sync modules import correctly."""
    from src.lip_sync import (
        prepare_audio_for_lipsync,
        run_latentsync_inference,
        LATENTSYNC_PYTHON,
        LATENTSYNC_REPO,
        run_wav2lip_inference,
        WAV2LIP_REPO,
        WAV2LIP_CHECKPOINT,
        split_video_into_chunks,
        concatenate_video_chunks,
        VideoChunk,
        get_video_duration,
        validate_lip_sync_output,
        SyncValidation,
    )
    from src.stages.lip_sync_stage import (
        run_lip_sync_stage,
        LipSyncResult,
        LipSyncStageFailed,
    )
    print("OK: All lip sync stage imports successful")


# ---------------------------------------------------------------------------
# Test 2: LipSyncResult dataclass fields and to_dict()
# ---------------------------------------------------------------------------

def test_lip_sync_result_dataclass():
    """Test LipSyncResult fields and to_dict() serialization."""
    from src.stages.lip_sync_stage import LipSyncResult
    from src.lip_sync.validator import SyncValidation

    sync_val = SyncValidation(
        total_frames=900,
        sampled_frames=30,
        valid_frames=29,
        pass_rate=29 / 30,
        passed=True,
    )

    result = LipSyncResult(
        output_path=Path("data/outputs/lip_synced.mp4"),
        model_used="latentsync",
        model_version="1.6",
        inference_steps=20,
        guidance_scale=1.5,
        processing_time=42.7,
        input_video_path=Path("data/outputs/assembled_video.mp4"),
        input_audio_path=Path("data/outputs/audio_prep/lipsync_audio_16k.wav"),
        chunks_processed=1,
        fallback_used=False,
        sync_validation=sync_val,
        multi_speaker_mode=False,
    )

    # Verify all 12 fields present via dataclasses.fields
    field_names = {f.name for f in dataclasses.fields(result)}
    expected_fields = {
        "output_path", "model_used", "model_version",
        "inference_steps", "guidance_scale", "processing_time",
        "input_video_path", "input_audio_path", "chunks_processed",
        "fallback_used", "sync_validation", "multi_speaker_mode",
    }
    assert expected_fields == field_names, f"Field mismatch: {field_names ^ expected_fields}"
    print("OK: LipSyncResult has all 12 fields")

    # Verify to_dict()
    d = result.to_dict()
    assert d["model_used"] == "latentsync"
    assert d["model_version"] == "1.6"
    assert d["inference_steps"] == 20
    assert d["guidance_scale"] == 1.5
    assert d["fallback_used"] is False
    assert d["multi_speaker_mode"] is False
    assert d["chunks_processed"] == 1
    assert isinstance(d["output_path"], str)
    assert isinstance(d["input_video_path"], str)
    assert isinstance(d["input_audio_path"], str)

    # sync_validation should be a dict (not None)
    assert isinstance(d["sync_validation"], dict)
    assert "pass_rate" in d["sync_validation"]
    print("OK: LipSyncResult.to_dict() serializes all fields correctly")


# ---------------------------------------------------------------------------
# Test 3: SyncValidation dataclass fields and to_dict()
# ---------------------------------------------------------------------------

def test_sync_validation_dataclass():
    """Test SyncValidation fields and to_dict() serialization."""
    from src.lip_sync.validator import SyncValidation

    sv = SyncValidation(
        total_frames=1800,
        sampled_frames=60,
        valid_frames=59,
        pass_rate=59 / 60,
        passed=True,
    )

    field_names = {f.name for f in dataclasses.fields(sv)}
    expected = {"total_frames", "sampled_frames", "valid_frames", "pass_rate", "passed"}
    assert expected == field_names, f"Unexpected fields: {field_names ^ expected}"
    print("OK: SyncValidation has correct 5 fields")

    d = sv.to_dict()
    assert d["total_frames"] == 1800
    assert d["sampled_frames"] == 60
    assert d["valid_frames"] == 59
    assert abs(d["pass_rate"] - (59 / 60)) < 1e-9
    assert d["passed"] is True
    print("OK: SyncValidation.to_dict() serializes correctly")


# ---------------------------------------------------------------------------
# Test 4: SyncValidation fail case (pass_rate < 0.95)
# ---------------------------------------------------------------------------

def test_sync_validation_fail_case():
    """Test SyncValidation reflects passed=False when pass_rate < 0.95."""
    from src.lip_sync.validator import SyncValidation, DEFAULT_PASS_THRESHOLD

    # Exactly at threshold: 94 out of 100 = 0.94 < 0.95 -> failed
    sv_fail = SyncValidation(
        total_frames=3000,
        sampled_frames=100,
        valid_frames=94,
        pass_rate=0.94,
        passed=0.94 >= DEFAULT_PASS_THRESHOLD,
    )
    assert sv_fail.passed is False, "pass_rate=0.94 should be below threshold 0.95"
    print("OK: SyncValidation correctly reflects pass_rate=0.94 -> passed=False")

    # At threshold: 95 out of 100 = 0.95 -> passed
    sv_pass = SyncValidation(
        total_frames=3000,
        sampled_frames=100,
        valid_frames=95,
        pass_rate=0.95,
        passed=0.95 >= DEFAULT_PASS_THRESHOLD,
    )
    assert sv_pass.passed is True, "pass_rate=0.95 should pass threshold"
    print("OK: SyncValidation correctly reflects pass_rate=0.95 -> passed=True")

    # Verify DEFAULT_PASS_THRESHOLD is 0.95
    assert DEFAULT_PASS_THRESHOLD == 0.95, f"Expected 0.95, got {DEFAULT_PASS_THRESHOLD}"
    print("OK: DEFAULT_PASS_THRESHOLD is 0.95")


# ---------------------------------------------------------------------------
# Test 5: validate_lip_sync_output raises FileNotFoundError for missing video
# ---------------------------------------------------------------------------

def test_validate_lip_sync_output_missing_file():
    """Test validate_lip_sync_output raises FileNotFoundError for missing video."""
    from src.lip_sync.validator import validate_lip_sync_output

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "nonexistent.mp4"
        try:
            validate_lip_sync_output(missing)
            raise AssertionError("Should raise FileNotFoundError for missing video")
        except FileNotFoundError as e:
            assert str(missing) in str(e) or "not found" in str(e).lower()
            print("OK: validate_lip_sync_output raises FileNotFoundError for missing video")


# ---------------------------------------------------------------------------
# Test 6: LipSyncStageFailed exception
# ---------------------------------------------------------------------------

def test_lip_sync_stage_failed_exception():
    """Test LipSyncStageFailed is a proper exception subclass."""
    from src.stages.lip_sync_stage import LipSyncStageFailed

    assert issubclass(LipSyncStageFailed, Exception)

    # Can be raised and caught
    try:
        raise LipSyncStageFailed("Test error message")
    except LipSyncStageFailed as e:
        assert "Test error message" in str(e)
        print("OK: LipSyncStageFailed is a proper exception with message")

    # Also caught as generic Exception
    try:
        raise LipSyncStageFailed("Another error")
    except Exception as e:
        assert isinstance(e, LipSyncStageFailed)
        print("OK: LipSyncStageFailed is caught as Exception")


# ---------------------------------------------------------------------------
# Test 7: audio_prep raises FileNotFoundError for missing source
# ---------------------------------------------------------------------------

def test_audio_prep_missing_source():
    """Test prepare_audio_for_lipsync raises FileNotFoundError for missing source."""
    from src.lip_sync.audio_prep import prepare_audio_for_lipsync

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "no_such_file.mp4"
        output_dir = Path(tmpdir) / "audio_out"

        try:
            prepare_audio_for_lipsync(missing, output_dir)
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError as e:
            assert str(missing) in str(e) or "not found" in str(e).lower()
            print("OK: prepare_audio_for_lipsync raises FileNotFoundError for missing source")


# ---------------------------------------------------------------------------
# Test 8: audio_prep subprocess call has correct -ar 16000 and -ac 1 flags
# ---------------------------------------------------------------------------

def test_audio_prep_subprocess_flags():
    """Test prepare_audio_for_lipsync calls FFmpeg with -ar 16000 -ac 1."""
    from src.lip_sync.audio_prep import prepare_audio_for_lipsync

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a real source file so the FileNotFoundError check passes
        source = Path(tmpdir) / "input.mp4"
        source.touch()
        output_dir = Path(tmpdir) / "audio_out"

        # Mock subprocess.run to avoid actual FFmpeg invocation
        with patch("src.lip_sync.audio_prep.subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            # Also mock stat() so logger.info works (output_path.stat().st_size)
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 12345
                prepare_audio_for_lipsync(source, output_dir)

        # Verify subprocess.run was called with the correct FFmpeg flags
        assert mock_run.called, "subprocess.run should have been called"
        cmd = mock_run.call_args[0][0]  # First positional arg = command list
        assert "-ar" in cmd, "FFmpeg command should include -ar flag"
        assert "16000" in cmd, "FFmpeg command should set -ar to 16000"
        assert "-ac" in cmd, "FFmpeg command should include -ac flag"
        assert "1" in cmd, "FFmpeg command should set -ac to 1 (mono)"
        print("OK: audio_prep calls FFmpeg with -ar 16000 -ac 1")
        print(f"  FFmpeg cmd: {' '.join(str(c) for c in cmd)}")


# ---------------------------------------------------------------------------
# Test 9: VideoChunk dataclass fields
# ---------------------------------------------------------------------------

def test_video_chunk_dataclass():
    """Test VideoChunk dataclass has correct fields."""
    from src.lip_sync.chunker import VideoChunk

    field_names = {f.name for f in dataclasses.fields(VideoChunk)}
    expected = {"index", "start_seconds", "end_seconds", "duration_seconds",
                "video_path", "audio_path"}
    assert expected == field_names, f"Field mismatch: {field_names ^ expected}"
    print("OK: VideoChunk has all 6 expected fields")

    chunk = VideoChunk(
        index=0,
        start_seconds=0.0,
        end_seconds=300.0,
        duration_seconds=300.0,
        video_path=Path("chunk_000_video.mp4"),
        audio_path=Path("chunk_000_audio.wav"),
    )
    assert chunk.index == 0
    assert chunk.start_seconds == 0.0
    assert chunk.end_seconds == 300.0
    assert chunk.duration_seconds == 300.0
    assert isinstance(chunk.video_path, Path)
    assert isinstance(chunk.audio_path, Path)
    print("OK: VideoChunk instantiates and retains all field values")


# ---------------------------------------------------------------------------
# Test 10: wav2lip_runner raises ValueError for s3fd.pth checkpoint
# ---------------------------------------------------------------------------

def test_wav2lip_s3fd_guard():
    """Test run_wav2lip_inference raises ValueError when s3fd.pth is given as checkpoint."""
    from src.lip_sync.wav2lip_runner import run_wav2lip_inference

    with tempfile.TemporaryDirectory() as tmpdir:
        # s3fd.pth path triggers the safety guard before any existence checks
        s3fd_path = Path(tmpdir) / "s3fd.pth"
        video = Path(tmpdir) / "video.mp4"
        audio = Path(tmpdir) / "audio.wav"

        try:
            run_wav2lip_inference(
                video_path=video,
                audio_path=audio,
                output_path=Path(tmpdir) / "out.mp4",
                checkpoint_path=s3fd_path,
            )
            raise AssertionError("Should raise ValueError for s3fd.pth")
        except ValueError as e:
            assert "s3fd.pth" in str(e).lower() or "face detector" in str(e).lower()
            print("OK: run_wav2lip_inference raises ValueError for s3fd.pth checkpoint")


# ---------------------------------------------------------------------------
# Test 11: wav2lip_runner raises FileNotFoundError for missing repo
# ---------------------------------------------------------------------------

def test_wav2lip_missing_repo():
    """Test run_wav2lip_inference raises FileNotFoundError when Wav2Lip repo is missing."""
    from src.lip_sync.wav2lip_runner import run_wav2lip_inference

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "video.mp4"
        audio = Path(tmpdir) / "audio.wav"
        out = Path(tmpdir) / "out.mp4"
        # Use a checkpoint that doesn't trigger the s3fd guard
        checkpoint = Path(tmpdir) / "wav2lip_gan.pth"

        # Patch WAV2LIP_REPO to a nonexistent path so FileNotFoundError is raised
        nonexistent_repo = Path(tmpdir) / "Wav2Lip_nonexistent"
        with patch("src.lip_sync.wav2lip_runner.WAV2LIP_REPO", nonexistent_repo):
            try:
                run_wav2lip_inference(
                    video_path=video,
                    audio_path=audio,
                    output_path=out,
                    checkpoint_path=checkpoint,
                )
                raise AssertionError("Should raise FileNotFoundError for missing repo")
            except FileNotFoundError as e:
                assert "wav2lip" in str(e).lower() or "not found" in str(e).lower()
                print("OK: run_wav2lip_inference raises FileNotFoundError for missing Wav2Lip repo")


# ---------------------------------------------------------------------------
# Test 12: latentsync_runner raises FileNotFoundError when conda env missing
# ---------------------------------------------------------------------------

def test_latentsync_missing_conda_env():
    """Test run_latentsync_inference raises FileNotFoundError when conda env Python is missing."""
    from src.lip_sync.latentsync_runner import run_latentsync_inference

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "video.mp4"
        audio = Path(tmpdir) / "audio.wav"
        out = Path(tmpdir) / "out.mp4"

        # Override LATENTSYNC_PYTHON to a nonexistent path
        missing_python = Path(tmpdir) / "latentsync" / "python.exe"
        with patch("src.lip_sync.latentsync_runner.LATENTSYNC_PYTHON", missing_python):
            try:
                run_latentsync_inference(
                    video_path=video,
                    audio_path=audio,
                    output_path=out,
                )
                raise AssertionError("Should raise FileNotFoundError for missing conda env")
            except FileNotFoundError as e:
                msg = str(e).lower()
                assert "latentsync" in msg or "not found" in msg or "python" in msg
                print("OK: run_latentsync_inference raises FileNotFoundError when conda env missing")


# ---------------------------------------------------------------------------
# Test 13: run_lip_sync_stage raises LipSyncStageFailed for missing input
# ---------------------------------------------------------------------------

def test_run_lip_sync_stage_missing_input():
    """Test run_lip_sync_stage raises LipSyncStageFailed when assembled video is missing."""
    from src.stages.lip_sync_stage import run_lip_sync_stage, LipSyncStageFailed

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_video = Path(tmpdir) / "nonexistent_assembled.mp4"
        output_dir = Path(tmpdir) / "output"

        try:
            run_lip_sync_stage(
                assembled_video_path=missing_video,
                output_dir=output_dir,
            )
            raise AssertionError("Should raise LipSyncStageFailed for missing input video")
        except LipSyncStageFailed as e:
            msg = str(e).lower()
            assert "not found" in msg or "missing" in msg or "assembled" in msg
            print("OK: run_lip_sync_stage raises LipSyncStageFailed for missing assembled video")


# ---------------------------------------------------------------------------
# Test 14: Progress callback receives values 0.0–1.0
# ---------------------------------------------------------------------------

def test_progress_callback_values():
    """Test progress callback is called with values in [0.0, 1.0] range."""
    from src.stages.lip_sync_stage import run_lip_sync_stage

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a real input file so file-existence check passes
        assembled_video = Path(tmpdir) / "assembled.mp4"
        assembled_video.touch()
        output_dir = Path(tmpdir) / "output"

        progress_values = []

        def capture_callback(progress: float, status: str):
            progress_values.append(progress)

        # Patch all subprocess/inference calls so nothing real runs
        with patch("src.stages.lip_sync_stage.prepare_audio_for_lipsync") as mock_prep, \
             patch("src.stages.lip_sync_stage.get_video_duration") as mock_dur, \
             patch("src.stages.lip_sync_stage.run_latentsync_inference") as mock_ls, \
             patch("src.stages.lip_sync_stage.validate_lip_sync_output") as mock_val:

            # Set return values
            audio_path = Path(tmpdir) / "audio_prep" / "lipsync_audio_16k.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.touch()
            mock_prep.return_value = audio_path

            # Short video: no chunking (< 300s threshold)
            mock_dur.return_value = 60.0

            # LatentSync "succeeds" (no-op)
            mock_ls.return_value = None

            # Create the output file that run_lip_sync_stage will try to reference
            output_video = output_dir / "lip_synced.mp4"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video.touch()

            from src.lip_sync.validator import SyncValidation
            mock_val.return_value = SyncValidation(
                total_frames=1800,
                sampled_frames=60,
                valid_frames=60,
                pass_rate=1.0,
                passed=True,
            )

            run_lip_sync_stage(
                assembled_video_path=assembled_video,
                output_dir=output_dir,
                progress_callback=capture_callback,
            )

        assert len(progress_values) > 0, "Progress callback should have been called"
        for val in progress_values:
            assert 0.0 <= val <= 1.0, f"Progress value {val} is outside [0.0, 1.0]"

        # First value should be at or near 0, last should be 1.0
        assert progress_values[-1] == 1.0, "Last progress value should be 1.0"
        print(f"OK: Progress callback received {len(progress_values)} values, all in [0.0, 1.0]")
        print(f"  Values: {progress_values}")


# ---------------------------------------------------------------------------
# Test 15: fallback_used=True when LatentSync raises RuntimeError
# ---------------------------------------------------------------------------

def test_fallback_used_on_latentsync_runtimeerror():
    """Test fallback_used=True in result when LatentSync raises RuntimeError."""
    from src.stages.lip_sync_stage import run_lip_sync_stage

    with tempfile.TemporaryDirectory() as tmpdir:
        assembled_video = Path(tmpdir) / "assembled.mp4"
        assembled_video.touch()
        output_dir = Path(tmpdir) / "output"

        with patch("src.stages.lip_sync_stage.prepare_audio_for_lipsync") as mock_prep, \
             patch("src.stages.lip_sync_stage.get_video_duration") as mock_dur, \
             patch("src.stages.lip_sync_stage.run_latentsync_inference") as mock_ls, \
             patch("src.stages.lip_sync_stage.run_wav2lip_inference") as mock_wl, \
             patch("src.stages.lip_sync_stage.validate_lip_sync_output") as mock_val:

            audio_path = Path(tmpdir) / "audio_prep" / "lipsync_audio_16k.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.touch()
            mock_prep.return_value = audio_path
            mock_dur.return_value = 60.0  # Short video, no chunking

            # LatentSync fails with RuntimeError
            mock_ls.side_effect = RuntimeError("OOM: CUDA out of memory")

            # Wav2Lip succeeds (no-op)
            mock_wl.return_value = None

            # Create expected output file
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "lip_synced.mp4").touch()

            from src.lip_sync.validator import SyncValidation
            mock_val.return_value = SyncValidation(
                total_frames=1800,
                sampled_frames=60,
                valid_frames=60,
                pass_rate=1.0,
                passed=True,
            )

            result = run_lip_sync_stage(
                assembled_video_path=assembled_video,
                output_dir=output_dir,
            )

        assert result.fallback_used is True, "fallback_used should be True"
        assert result.model_used == "wav2lip", f"model_used should be 'wav2lip', got {result.model_used}"
        assert result.model_version == "gan", f"model_version should be 'gan', got {result.model_version}"
        # When fallback: inference_steps=0, guidance_scale=0.0
        assert result.inference_steps == 0, "inference_steps should be 0 for fallback"
        assert result.guidance_scale == 0.0, "guidance_scale should be 0.0 for fallback"
        print("OK: fallback_used=True, model_used='wav2lip' when LatentSync raises RuntimeError")
        print(f"  inference_steps={result.inference_steps}, guidance_scale={result.guidance_scale}")


# ---------------------------------------------------------------------------
# Test 16: multi_speaker_mode set correctly
# ---------------------------------------------------------------------------

def test_multi_speaker_mode_field():
    """Test multi_speaker_mode=True when speakers_detected > 1."""
    from src.stages.lip_sync_stage import run_lip_sync_stage

    with tempfile.TemporaryDirectory() as tmpdir:
        assembled_video = Path(tmpdir) / "assembled.mp4"
        assembled_video.touch()
        output_dir = Path(tmpdir) / "output"

        with patch("src.stages.lip_sync_stage.prepare_audio_for_lipsync") as mock_prep, \
             patch("src.stages.lip_sync_stage.get_video_duration") as mock_dur, \
             patch("src.stages.lip_sync_stage.run_latentsync_inference") as mock_ls, \
             patch("src.stages.lip_sync_stage.validate_lip_sync_output") as mock_val:

            audio_path = Path(tmpdir) / "audio_prep" / "lipsync_audio_16k.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.touch()
            mock_prep.return_value = audio_path
            mock_dur.return_value = 60.0
            mock_ls.return_value = None
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "lip_synced.mp4").touch()

            from src.lip_sync.validator import SyncValidation
            mock_val.return_value = SyncValidation(
                total_frames=1800, sampled_frames=60,
                valid_frames=60, pass_rate=1.0, passed=True,
            )

            # speakers_detected=1 -> multi_speaker_mode=False
            result_single = run_lip_sync_stage(
                assembled_video_path=assembled_video,
                output_dir=output_dir,
                speakers_detected=1,
            )
            assert result_single.multi_speaker_mode is False, \
                "multi_speaker_mode should be False when speakers_detected=1"
            print("OK: multi_speaker_mode=False when speakers_detected=1")

            # speakers_detected=3 -> multi_speaker_mode=True
            result_multi = run_lip_sync_stage(
                assembled_video_path=assembled_video,
                output_dir=output_dir,
                speakers_detected=3,
            )
            assert result_multi.multi_speaker_mode is True, \
                "multi_speaker_mode should be True when speakers_detected=3"
            print("OK: multi_speaker_mode=True when speakers_detected=3")


# ---------------------------------------------------------------------------
# Test 17: SyncValidation None when validation raises exception
# ---------------------------------------------------------------------------

def test_sync_validation_none_on_exception():
    """Test sync_validation=None in result when validation raises an exception."""
    from src.stages.lip_sync_stage import run_lip_sync_stage

    with tempfile.TemporaryDirectory() as tmpdir:
        assembled_video = Path(tmpdir) / "assembled.mp4"
        assembled_video.touch()
        output_dir = Path(tmpdir) / "output"

        with patch("src.stages.lip_sync_stage.prepare_audio_for_lipsync") as mock_prep, \
             patch("src.stages.lip_sync_stage.get_video_duration") as mock_dur, \
             patch("src.stages.lip_sync_stage.run_latentsync_inference") as mock_ls, \
             patch("src.stages.lip_sync_stage.validate_lip_sync_output") as mock_val:

            audio_path = Path(tmpdir) / "audio_prep" / "lipsync_audio_16k.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.touch()
            mock_prep.return_value = audio_path
            mock_dur.return_value = 60.0
            mock_ls.return_value = None
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "lip_synced.mp4").touch()

            # Validation raises exception (e.g. ffprobe unavailable)
            mock_val.side_effect = RuntimeError("ffprobe signalstats unavailable")

            result = run_lip_sync_stage(
                assembled_video_path=assembled_video,
                output_dir=output_dir,
            )

        assert result.sync_validation is None, \
            "sync_validation should be None when validation raises exception"
        print("OK: sync_validation=None when validate_lip_sync_output raises exception")

        # to_dict() should still work with sync_validation=None
        d = result.to_dict()
        assert d["sync_validation"] is None
        print("OK: to_dict() with sync_validation=None serializes to null")


# ---------------------------------------------------------------------------
# Test 18: SyncValidation embedded in LipSyncResult serializes correctly
# ---------------------------------------------------------------------------

def test_sync_validation_in_lip_sync_result_serialization():
    """Test that SyncValidation embedded in LipSyncResult round-trips through JSON."""
    from src.stages.lip_sync_stage import LipSyncResult
    from src.lip_sync.validator import SyncValidation

    sync_val = SyncValidation(
        total_frames=2700,
        sampled_frames=90,
        valid_frames=87,
        pass_rate=87 / 90,
        passed=(87 / 90) >= 0.95,
    )

    result = LipSyncResult(
        output_path=Path("out.mp4"),
        model_used="latentsync",
        model_version="1.6",
        inference_steps=40,
        guidance_scale=1.5,
        processing_time=120.5,
        input_video_path=Path("in.mp4"),
        input_audio_path=Path("audio_16k.wav"),
        chunks_processed=2,
        fallback_used=False,
        sync_validation=sync_val,
        multi_speaker_mode=True,
    )

    d = result.to_dict()

    # Round-trip through JSON
    json_str = json.dumps(d)
    d2 = json.loads(json_str)

    sv_dict = d2["sync_validation"]
    assert sv_dict is not None
    assert sv_dict["total_frames"] == 2700
    assert sv_dict["sampled_frames"] == 90
    assert sv_dict["valid_frames"] == 87
    assert abs(sv_dict["pass_rate"] - (87 / 90)) < 1e-9
    assert sv_dict["passed"] is True  # 87/90 = 0.9667 >= 0.95
    assert d2["multi_speaker_mode"] is True
    assert d2["chunks_processed"] == 2
    print("OK: SyncValidation embedded in LipSyncResult round-trips through JSON correctly")


# ---------------------------------------------------------------------------
# Test 19: LATENTSYNC_PYTHON constant is a Path
# ---------------------------------------------------------------------------

def test_latentsync_python_constant():
    """Test LATENTSYNC_PYTHON is a Path and can be overridden via env var."""
    from src.lip_sync.latentsync_runner import LATENTSYNC_PYTHON, LATENTSYNC_REPO

    assert isinstance(LATENTSYNC_PYTHON, Path), \
        f"LATENTSYNC_PYTHON should be a Path, got {type(LATENTSYNC_PYTHON)}"
    print(f"OK: LATENTSYNC_PYTHON is a Path: {LATENTSYNC_PYTHON}")

    assert isinstance(LATENTSYNC_REPO, Path), \
        f"LATENTSYNC_REPO should be a Path, got {type(LATENTSYNC_REPO)}"
    # LATENTSYNC_REPO should point to models/LatentSync under project root
    assert "LatentSync" in str(LATENTSYNC_REPO), \
        f"LATENTSYNC_REPO should contain 'LatentSync': {LATENTSYNC_REPO}"
    print(f"OK: LATENTSYNC_REPO is a Path pointing to: {LATENTSYNC_REPO}")


# ---------------------------------------------------------------------------
# Test 20: WAV2LIP_REPO and WAV2LIP_CHECKPOINT constants
# ---------------------------------------------------------------------------

def test_wav2lip_constants():
    """Test WAV2LIP_REPO and WAV2LIP_CHECKPOINT are correctly set Path constants."""
    from src.lip_sync.wav2lip_runner import WAV2LIP_REPO, WAV2LIP_CHECKPOINT

    assert isinstance(WAV2LIP_REPO, Path), \
        f"WAV2LIP_REPO should be a Path, got {type(WAV2LIP_REPO)}"
    assert "Wav2Lip" in str(WAV2LIP_REPO), \
        f"WAV2LIP_REPO should contain 'Wav2Lip': {WAV2LIP_REPO}"
    print(f"OK: WAV2LIP_REPO is a Path: {WAV2LIP_REPO}")

    assert isinstance(WAV2LIP_CHECKPOINT, Path), \
        f"WAV2LIP_CHECKPOINT should be a Path, got {type(WAV2LIP_CHECKPOINT)}"
    assert WAV2LIP_CHECKPOINT.name == "wav2lip_gan.pth", \
        f"WAV2LIP_CHECKPOINT should be wav2lip_gan.pth, got {WAV2LIP_CHECKPOINT.name}"
    print(f"OK: WAV2LIP_CHECKPOINT points to wav2lip_gan.pth: {WAV2LIP_CHECKPOINT}")


# ---------------------------------------------------------------------------
# Test 21: run_lip_sync_stage progress_callback signature accepted
# ---------------------------------------------------------------------------

def test_progress_callback_signature():
    """Test run_lip_sync_stage accepts progress_callback with correct signature."""
    import inspect
    from src.stages.lip_sync_stage import run_lip_sync_stage

    sig = inspect.signature(run_lip_sync_stage)
    assert "progress_callback" in sig.parameters, \
        "run_lip_sync_stage should accept progress_callback"
    param = sig.parameters["progress_callback"]
    assert param.default is None, "progress_callback should default to None"
    print("OK: run_lip_sync_stage accepts progress_callback parameter (default None)")

    assert "speakers_detected" in sig.parameters, \
        "run_lip_sync_stage should accept speakers_detected"
    assert "inference_steps" in sig.parameters, \
        "run_lip_sync_stage should accept inference_steps"
    assert "guidance_scale" in sig.parameters, \
        "run_lip_sync_stage should accept guidance_scale"
    assert "enable_deepcache" in sig.parameters, \
        "run_lip_sync_stage should accept enable_deepcache"
    print("OK: run_lip_sync_stage has all expected parameters")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Lip Sync Stage Integration Tests ===\n")

    tests = [
        test_lip_sync_imports,
        test_lip_sync_result_dataclass,
        test_sync_validation_dataclass,
        test_sync_validation_fail_case,
        test_validate_lip_sync_output_missing_file,
        test_lip_sync_stage_failed_exception,
        test_audio_prep_missing_source,
        test_audio_prep_subprocess_flags,
        test_video_chunk_dataclass,
        test_wav2lip_s3fd_guard,
        test_wav2lip_missing_repo,
        test_latentsync_missing_conda_env,
        test_run_lip_sync_stage_missing_input,
        test_progress_callback_values,
        test_fallback_used_on_latentsync_runtimeerror,
        test_multi_speaker_mode_field,
        test_sync_validation_none_on_exception,
        test_sync_validation_in_lip_sync_result_serialization,
        test_latentsync_python_constant,
        test_wav2lip_constants,
        test_progress_callback_signature,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
