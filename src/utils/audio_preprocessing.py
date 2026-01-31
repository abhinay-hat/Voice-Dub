"""
Audio preprocessing utilities for ASR (Automatic Speech Recognition).
Converts audio to format required by Whisper and pyannote.
"""
from pathlib import Path
from typing import Union
import ffmpeg


def preprocess_audio_for_asr(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None
) -> str:
    """
    Convert audio to 16kHz mono PCM WAV required by Whisper and pyannote.

    Both Whisper Large V3 and pyannote.audio expect 16kHz mono audio.
    Preprocessing once upfront is more efficient than each model resampling separately.

    Args:
        input_path: Path to input audio (any format supported by FFmpeg)
        output_path: Path for output WAV file (default: same dir with _16khz_mono.wav suffix)

    Returns:
        str: Path to preprocessed 16kHz mono WAV file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If FFmpeg preprocessing fails

    Example:
        >>> preprocessed = preprocess_audio_for_asr("audio.mp3")
        >>> # Creates audio_16khz_mono.wav in same directory
        >>> preprocessed
        'audio_16khz_mono.wav'
    """
    input_path = Path(input_path)

    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_16khz_mono.wav"
    else:
        output_path = Path(output_path)

    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to 16kHz mono PCM WAV
    try:
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',  # 16-bit signed PCM
                ar=16000,            # 16kHz sample rate (REQUIRED by Whisper and pyannote)
                ac=1,                # Mono channel (REQUIRED by both Whisper and pyannote)
                f='wav'              # WAV container
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg preprocessing failed: {stderr}") from e

    return str(output_path)
