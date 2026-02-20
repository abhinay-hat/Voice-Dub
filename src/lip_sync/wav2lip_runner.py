"""Wav2Lip GAN fallback inference runner.

Used when LatentSync fails due to:
- OOM (LatentSync 1.6 needs ~18GB; Wav2Lip needs ~2-4GB)
- Face detection failures (LatentSync InsightFace issue)
- InsightFace Windows installation failure

Wav2Lip tradeoffs vs LatentSync:
- ~10x faster inference
- Blurrier output (GAN not diffusion)
- No temporal consistency smoothing
- Simpler Windows installation

Checkpoint download (manual - Google Drive):
  wav2lip_gan.pth: https://iiitaphyd-my.sharepoint.com/personal/...
  s3fd.pth (face detector): stored at face_detection/detection/sfd/s3fd.pth
  See: github.com/Rudrabha/Wav2Lip README for current download links.
"""
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WAV2LIP_REPO = Path(__file__).parent.parent.parent / "models" / "Wav2Lip"
# GAN model: better visual quality than wav2lip.pth for fallback use
WAV2LIP_CHECKPOINT = WAV2LIP_REPO / "checkpoints" / "wav2lip_gan.pth"


def run_wav2lip_inference(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 64,
    timeout: int = 7200,
) -> None:
    """
    Run Wav2Lip GAN inference as fallback when LatentSync is unavailable.

    Args:
        video_path: Input MP4 video with faces.
        audio_path: Input WAV audio at 16kHz mono (use prepare_audio_for_lipsync first).
        output_path: Output path for lip-synced MP4.
        checkpoint_path: Path to wav2lip_gan.pth. Defaults to WAV2LIP_CHECKPOINT.
                         MUST be wav2lip_gan.pth or wav2lip.pth - NOT s3fd.pth.
        face_det_batch_size: Batch size for SFD face detector. Reduce if VRAM OOM.
        wav2lip_batch_size: Batch size for Wav2Lip model. Reduce from 128 if VRAM OOM.
        timeout: Subprocess timeout in seconds. Default 7200 (2 hours) for long videos.

    Raises:
        FileNotFoundError: If Wav2Lip repo, checkpoint, or input files are missing.
        ValueError: If checkpoint_path looks like a face detector (s3fd.pth) - wrong file.
        RuntimeError: If Wav2Lip subprocess exits non-zero.
    """
    if checkpoint_path is None:
        checkpoint_path = WAV2LIP_CHECKPOINT

    # Safety guard: prevent the most common mistake (passing face detector as model)
    if checkpoint_path.name == "s3fd.pth":
        raise ValueError(
            f"checkpoint_path points to the face detector (s3fd.pth), not the Wav2Lip model.\n"
            f"Use 'wav2lip_gan.pth' or 'wav2lip.pth' instead.\n"
            f"The face detector is loaded automatically from face_detection/detection/sfd/s3fd.pth"
        )

    # Validate required files exist
    if not WAV2LIP_REPO.exists():
        raise FileNotFoundError(
            f"Wav2Lip repo not found at {WAV2LIP_REPO}.\n"
            "Clone with: git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {checkpoint_path}\n"
            "Download wav2lip_gan.pth from the Wav2Lip GitHub README (Google Drive link)."
        )
    for path in [video_path, audio_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference.py",
        "--checkpoint_path", str(checkpoint_path),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        # NOTE: do not add --nosmooth; temporal smoothing helps visual quality
    ]

    logger.info(f"Running Wav2Lip fallback: {checkpoint_path.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(WAV2LIP_REPO),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.error(f"Wav2Lip stderr:\n{result.stderr}")
        raise RuntimeError(
            f"Wav2Lip inference failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    logger.info(f"Wav2Lip inference complete -> {output_path}")
