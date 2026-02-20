"""Audio preparation utilities for lip synchronization models.

Phase 6 outputs 48kHz audio (broadcast standard).
LatentSync's Whisper tiny encoder requires 16kHz mono WAV.
Wav2Lip also expects 16kHz mel spectrograms.
This module handles the conversion using FFmpeg.
"""
from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


def prepare_audio_for_lipsync(source_media_path: Path, output_dir: Path) -> Path:
    """
    Resample audio from any sample rate to 16kHz mono WAV for lip sync models.

    Args:
        source_media_path: Input media file (video or audio). FFmpeg extracts audio track if video.
        output_dir: Directory to write resampled audio. Created if it doesn't exist.

    Returns:
        Path to lipsync_audio_16k.wav in output_dir.

    Raises:
        RuntimeError: If FFmpeg resampling fails.
        FileNotFoundError: If source_media_path does not exist.
    """
    if not source_media_path.exists():
        raise FileNotFoundError(f"Source media not found: {source_media_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lipsync_audio_16k.wav"

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(source_media_path),
            "-vn",                    # No video stream
            "-acodec", "pcm_s16le",  # 16-bit PCM WAV (universally compatible)
            "-ar", "16000",           # 16kHz - required by Whisper tiny and Wav2Lip
            "-ac", "1",               # Mono - both models expect mono
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio resampling failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info(f"Resampled audio to 16kHz: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path
