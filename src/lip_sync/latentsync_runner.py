"""LatentSync 1.6 inference runner via subprocess isolation.

LatentSync pins torch==2.5.1 (CUDA 12.1) which conflicts with project's
PyTorch nightly (CUDA 12.8, sm_120 for RTX 5090). Running LatentSync in
an isolated conda environment called via subprocess is the required pattern.

Environment: conda env 'latentsync' at C:/Users/ASBL/miniconda3/envs/latentsync
"""
import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path to isolated conda env Python interpreter (Windows)
# Override via LATENTSYNC_PYTHON_PATH env var for portability
_default_python = "C:/Users/ASBL/miniconda3/envs/latentsync/python.exe"
LATENTSYNC_PYTHON = Path(os.environ.get("LATENTSYNC_PYTHON_PATH", _default_python))

# Path to cloned LatentSync repo
LATENTSYNC_REPO = Path(__file__).parent.parent.parent / "models" / "LatentSync"


def run_latentsync_inference(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    enable_deepcache: bool = True,
    timeout: int = 3600,
) -> None:
    """
    Run LatentSync 1.6 inference via subprocess in isolated conda environment.

    Args:
        video_path: Input MP4 video (any FPS; LatentSync normalizes to 25FPS internally).
        audio_path: Input WAV audio at 16kHz mono (use prepare_audio_for_lipsync first).
        output_path: Output path for lip-synced MP4 video.
        inference_steps: Denoising steps. 20 = fast, 40-50 = higher quality.
        guidance_scale: Classifier-free guidance scale. Default 1.5 per LatentSync docs.
        enable_deepcache: Enable ~2x speedup. Requires latest LatentSync main (commit f5040cf+).
        timeout: Subprocess timeout in seconds. Default 3600 (1 hour) for long videos.

    Raises:
        FileNotFoundError: If LATENTSYNC_PYTHON or required checkpoints don't exist.
        RuntimeError: If LatentSync subprocess exits non-zero.
        subprocess.TimeoutExpired: If inference exceeds timeout.
    """
    # Validate environment before invoking subprocess
    if not LATENTSYNC_PYTHON.exists():
        raise FileNotFoundError(
            f"LatentSync Python interpreter not found: {LATENTSYNC_PYTHON}\n"
            "Set LATENTSYNC_PYTHON_PATH env var or run Plan 07-01 to set up conda env."
        )

    unet_config = LATENTSYNC_REPO / "configs" / "unet" / "stage2_512.yaml"
    unet_checkpoint = LATENTSYNC_REPO / "checkpoints" / "latentsync_unet.pt"
    for path in [unet_config, unet_checkpoint, video_path, audio_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(LATENTSYNC_PYTHON), "-m", "scripts.inference",
        "--unet_config_path", str(unet_config),
        "--inference_ckpt_path", str(unet_checkpoint),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
    ]
    if enable_deepcache:
        cmd.append("--enable_deepcache")

    logger.info(f"Running LatentSync inference: steps={inference_steps}, guidance={guidance_scale}")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(LATENTSYNC_REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.error(f"LatentSync stderr:\n{result.stderr}")
        raise RuntimeError(
            f"LatentSync inference failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"  # Last 2000 chars to avoid log flood
        )

    logger.info(f"LatentSync inference complete -> {output_path}")
    if result.stdout:
        logger.debug(f"LatentSync stdout: {result.stdout[-500:]}")
