"""Video chunking for long-video lip synchronization.

Both LatentSync and Wav2Lip work best on shorter video segments:
- 5-minute chunks prevent memory spikes during face detection
- Scene changes at chunk boundaries are handled gracefully
- Chunk boundaries align to 5-minute marks regardless of scene content

Pattern: split_video_into_chunks -> [process each chunk] -> concatenate_video_chunks
"""
import subprocess
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DURATION = 300  # 5 minutes in seconds


@dataclass
class VideoChunk:
    """Metadata for a single video chunk."""
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    video_path: Path
    audio_path: Path  # Pre-extracted 16kHz audio for this chunk


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def split_video_into_chunks(
    video_path: Path,
    audio_path: Path,
    work_dir: Path,
    chunk_duration: int = DEFAULT_CHUNK_DURATION,
) -> list[VideoChunk]:
    """
    Split a long video and its audio into fixed-duration chunks.

    Uses FFmpeg stream copy for video (no quality loss) and PCM copy for audio.
    Each chunk gets its own video file and pre-extracted 16kHz audio file.

    Args:
        video_path: Input MP4 video (any duration).
        audio_path: Input 16kHz mono WAV audio matching video_path.
        work_dir: Directory for chunk files. Created if missing.
        chunk_duration: Duration of each chunk in seconds. Default 300 (5 minutes).

    Returns:
        List of VideoChunk in order (index 0, 1, 2, ...).

    Raises:
        FileNotFoundError: If video_path or audio_path don't exist.
        RuntimeError: If FFmpeg splitting fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    total_duration = get_video_duration(video_path)
    logger.info(f"Video duration: {total_duration:.1f}s, splitting into {chunk_duration}s chunks")

    chunks = []
    start = 0.0
    index = 0

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        duration = end - start

        chunk_video = work_dir / f"chunk_{index:03d}_video.mp4"
        chunk_audio = work_dir / f"chunk_{index:03d}_audio.wav"

        # Extract video chunk with re-encode for clean keyframe alignment.
        # Using -ss -t AFTER -i (accurate seek) prevents GOP misalignment that
        # causes duration inflation when chunks are later concatenated.
        # Stream copy with -ss before -i causes PTS discontinuities that inflate
        # reported chunk duration (e.g. 10s chunk reports 10.08s, 17s total for 12s).
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", str(start),
                "-t", str(duration),
                "-c:v", "libx264",   # Re-encode for clean keyframes at chunk boundaries
                "-c:a", "pcm_s16le", # Keep audio as PCM (wav2lip/latsync expect this)
                str(chunk_video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Extract audio chunk (16kHz WAV already, just trim)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", str(audio_path),
                "-t", str(duration),
                "-acodec", "pcm_s16le",
                str(chunk_audio),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        chunks.append(VideoChunk(
            index=index,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration,
            video_path=chunk_video,
            audio_path=chunk_audio,
        ))
        logger.debug(f"Chunk {index}: {start:.1f}s - {end:.1f}s -> {chunk_video.name}")

        start = end
        index += 1

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def concatenate_video_chunks(
    processed_chunk_paths: list[Path],
    output_path: Path,
    work_dir: Path,
) -> Path:
    """
    Concatenate processed lip-synced chunk videos into final output.

    Uses FFmpeg concat demuxer with stream copy (frame-accurate, no re-encoding).
    The concat list file is written to work_dir/concat_list.txt.

    Args:
        processed_chunk_paths: List of chunk MP4 paths in order (chunk 0, 1, 2, ...).
        output_path: Path for the concatenated output MP4.
        work_dir: Working directory for concat list file.

    Returns:
        output_path on success.

    Raises:
        ValueError: If processed_chunk_paths is empty.
        FileNotFoundError: If any chunk path doesn't exist.
        RuntimeError: If FFmpeg concat fails.
    """
    if not processed_chunk_paths:
        raise ValueError("No chunk paths provided for concatenation")

    for p in processed_chunk_paths:
        if not p.exists():
            raise FileNotFoundError(f"Processed chunk not found: {p}")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write FFmpeg concat demuxer file list
    concat_list = work_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for chunk_path in processed_chunk_paths:
            # Use forward slashes for FFmpeg on Windows
            f.write(f"file '{chunk_path.as_posix()}'\n")

    logger.info(f"Concatenating {len(processed_chunk_paths)} chunks -> {output_path}")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",            # Allow absolute paths in concat list
            "-i", str(concat_list),
            "-c", "copy",            # Stream copy: no re-encoding
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    logger.info(f"Concatenation complete: {output_path}")
    return output_path
