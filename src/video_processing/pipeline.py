"""
Video processing pipeline orchestration.
Provides end-to-end video processing workflow with extraction, processing, and merging.
"""
from pathlib import Path
from typing import Union, Optional, Callable
from dataclasses import dataclass
import time
import shutil

from .video_utils import validate_video_file, detect_container_format, get_video_info, VideoInfo
from .extractor import extract_streams
from .merger import merge_audio_video, get_optimal_merge_config
from ..storage.temp_manager import TempFileManager


@dataclass
class ProcessingResult:
    """Result of video processing pipeline."""
    input_path: Path
    output_path: Path
    duration: float  # seconds (video duration)
    video_info: VideoInfo
    temp_dir_used: str  # for debugging
    processing_time: float  # seconds (processing elapsed time)


def process_video(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> ProcessingResult:
    """
    Process video through extraction, processing, and merging pipeline.

    Currently performs a "round trip" (extract -> pass through -> merge) to validate
    the toolchain. Future phases will insert ML processing (transcription, translation,
    voice cloning, lip sync) between extraction and merging.

    Args:
        input_path: Path to input video file
        output_path: Path for output video (if None, uses input name + "_processed")
        progress_callback: Optional callback(progress: float, description: str)
                          where progress is 0.0-1.0

    Returns:
        ProcessingResult: Contains output path, metadata, and processing metrics

    Raises:
        FileNotFoundError: If input video doesn't exist
        ValueError: If video validation fails (unsupported format, no video stream)
        ffmpeg.Error: If FFmpeg operations fail

    Example:
        >>> def progress(value, desc):
        ...     print(f"{int(value*100)}% - {desc}")
        >>> result = process_video("input.mp4", progress_callback=progress)
        >>> print(f"Processed in {result.processing_time:.1f}s")
    """
    start_time = time.time()
    input_path = Path(input_path)

    # Default progress callback (no-op)
    if progress_callback is None:
        progress_callback = lambda p, d: None

    # 1. Validate input video
    progress_callback(0.0, "Validating input video")
    is_valid, error_msg = validate_video_file(input_path)
    if not is_valid:
        raise ValueError(f"Invalid input video: {error_msg}")

    # 2. Detect input format
    input_format = detect_container_format(input_path)

    # 3. Determine output path
    if output_path is None:
        # Use input name + "_processed" + same extension
        output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
    else:
        output_path = Path(output_path)

    output_format = detect_container_format(output_path) if output_path.suffix else input_format

    # 4. Report progress: starting
    progress_callback(0.05, "Starting processing")

    # 5. Process video with temporary file management
    temp_dir_used = None
    with TempFileManager() as temp:
        temp_dir_used = str(temp.temp_dir)

        # Extract streams
        progress_callback(0.2, "Extracting audio and video streams")
        extraction = extract_streams(input_path, temp.temp_dir)

        # Phase 2: Audio is NOT processed - just passed through
        # Future phases will insert ML processing here:
        # - Phase 3: Speech recognition (Whisper)
        # - Phase 4: Translation (SeamlessM4T)
        # - Phase 5: Voice cloning (XTTS-v2)
        # - Phase 6: Lip sync (Wav2Lip/LatentSync)
        processed_audio = extraction.audio_path

        progress_callback(0.6, "Merging streams")

        # Get optimal merge configuration for container formats
        merge_config = get_optimal_merge_config(input_format, output_format)

        # Merge video and processed audio back together
        merge_audio_video(
            extraction.video_path,
            processed_audio,
            output_path,
            merge_config
        )

    # Temp files automatically cleaned up here

    # 6. Report progress: complete
    progress_callback(1.0, "Complete")

    # 7. Calculate processing time
    processing_time = time.time() - start_time

    # 8. Return result
    return ProcessingResult(
        input_path=input_path,
        output_path=output_path,
        duration=extraction.duration,
        video_info=extraction.video_info,
        temp_dir_used=temp_dir_used,
        processing_time=processing_time
    )


def validate_processing_environment() -> tuple[bool, str]:
    """
    Validate video processing environment is ready.

    Checks:
    - FFmpeg is available on system PATH
    - Temporary directory is writable

    Returns:
        tuple[bool, str]: (is_valid, error_message)
                         Returns (True, "") if valid
                         Returns (False, "error description") if invalid

    Example:
        >>> valid, error = validate_processing_environment()
        >>> if not valid:
        ...     print(f"Environment check failed: {error}")
    """
    import subprocess
    import tempfile

    # Check FFmpeg is available
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "FFmpeg is not available (command failed)"
    except FileNotFoundError:
        return False, "FFmpeg is not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg command timed out"
    except Exception as e:
        return False, f"Failed to check FFmpeg: {str(e)}"

    # Verify temp directory is writable
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
    except Exception as e:
        return False, f"Temporary directory is not writable: {str(e)}"

    # All checks passed
    return True, ""
