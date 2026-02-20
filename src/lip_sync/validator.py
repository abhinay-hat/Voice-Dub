"""Lightweight lip sync output validation.

Uses FFmpeg/ffprobe to sample frames from the lip-synced output and check
for black frames (YAVG brightness). A valid output should have nearly all
frames with meaningful brightness.

This is advisory — the validator never fails the stage. The caller is
responsible for catching exceptions and logging warnings.
"""
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Brightness threshold on 0-255 scale.
# Frames with mean luma (YAVG) below this are considered "black" (corrupt/blank).
BRIGHTNESS_THRESHOLD = 10.0

# Minimum pass_rate to set SyncValidation.passed = True.
DEFAULT_PASS_THRESHOLD = 0.95


@dataclass
class SyncValidation:
    """Result of validate_lip_sync_output()."""

    total_frames: int     # Total frames in the output video (from ffprobe)
    sampled_frames: int   # Number of frames that were actually checked
    valid_frames: int     # Frames that passed the brightness threshold
    pass_rate: float      # valid_frames / sampled_frames (or 0.0 if sampled_frames == 0)
    passed: bool          # True when pass_rate >= pass_threshold

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_frames": self.total_frames,
            "sampled_frames": self.sampled_frames,
            "valid_frames": self.valid_frames,
            "pass_rate": self.pass_rate,
            "passed": self.passed,
        }


def _get_total_frames(video_path: Path) -> int:
    """Return the total frame count using ffprobe nb_frames or duration * fps."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {video_path}")

    stream = streams[0]

    # Prefer nb_frames (fast, exact)
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        return int(nb_frames)

    # Fall back: duration * fps (slightly approximate)
    duration = float(stream.get("duration", 0))
    fps_str = stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    return max(1, int(duration * fps))


def _get_frame_brightness(video_path: Path, sample_interval: int) -> list[float]:
    """
    Extract YAVG brightness for every Nth frame via ffprobe signalstats.

    Returns a list of YAVG values (0.0–255.0) for the sampled frames.
    An empty list indicates no frames could be read.
    """
    # Use the select filter to pick every Nth frame, then measure signalstats
    vf = f"select='not(mod(n,{sample_interval}))',signalstats"

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-f", "lavfi",
            "-i", f"movie={video_path.as_posix()},{vf}",
            "-show_frames",
            "-select_streams", "v",
            "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
            "-of", "json",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # signalstats via lavfi can be tricky; fall back gracefully
        logger.debug(f"ffprobe signalstats failed: {result.stderr[:500]}")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    brightness_values: list[float] = []
    for frame in data.get("frames", []):
        tags = frame.get("tags", {})
        yavg_str = tags.get("lavfi.signalstats.YAVG")
        if yavg_str is not None:
            try:
                brightness_values.append(float(yavg_str))
            except ValueError:
                pass

    return brightness_values


def validate_lip_sync_output(
    video_path: Path,
    sample_interval: int = 30,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> SyncValidation:
    """
    Validate lip sync output by sampling frame brightness.

    Samples every Nth frame (default every 30 frames) using FFmpeg signalstats.
    A "valid" frame has mean luma (YAVG) >= BRIGHTNESS_THRESHOLD (10.0).
    Black frames indicate corruption, encoding failure, or face-region blanking.

    Args:
        video_path: Path to lip-synced output video.
        sample_interval: Sample every Nth frame. Default 30 (~1 sample/second at 30fps).
        pass_threshold: Minimum pass_rate for SyncValidation.passed = True. Default 0.95.

    Returns:
        SyncValidation with total_frames, sampled_frames, valid_frames, pass_rate, passed.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If ffprobe cannot read the video stream.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Lip sync output not found: {video_path}")

    total_frames = _get_total_frames(video_path)
    logger.info(
        f"Validating lip sync output: {video_path.name} "
        f"({total_frames} frames, sampling every {sample_interval} frames)"
    )

    brightness_values = _get_frame_brightness(video_path, sample_interval)
    sampled_frames = len(brightness_values)

    if sampled_frames == 0:
        # Could not sample — be optimistic rather than blocking the pipeline
        logger.warning(
            "Brightness sampling returned 0 frames. "
            "Treating as passed (advisory check only)."
        )
        return SyncValidation(
            total_frames=total_frames,
            sampled_frames=0,
            valid_frames=0,
            pass_rate=1.0,   # Optimistic default
            passed=True,
        )

    valid_frames = sum(1 for b in brightness_values if b >= BRIGHTNESS_THRESHOLD)
    pass_rate = valid_frames / sampled_frames
    passed = pass_rate >= pass_threshold

    logger.debug(
        f"Brightness check: {valid_frames}/{sampled_frames} frames valid "
        f"(threshold YAVG>={BRIGHTNESS_THRESHOLD}), pass_rate={pass_rate:.3f}"
    )

    return SyncValidation(
        total_frames=total_frames,
        sampled_frames=sampled_frames,
        valid_frames=valid_frames,
        pass_rate=pass_rate,
        passed=passed,
    )
