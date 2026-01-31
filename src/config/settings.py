"""
Project configuration settings.
Centralized constants for paths, model configurations, and pipeline parameters.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TEMP_DATA_DIR = DATA_DIR / "temp"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"

# Model cache directory
MODELS_DIR = PROJECT_ROOT / "models"

# GPU Configuration
GPU_DEVICE = 0  # Primary GPU index
EXPECTED_COMPUTE_CAPABILITY = (12, 0)  # RTX 5090 sm_120
EXPECTED_VRAM_GB = 32

# Memory Management
PYTORCH_CUDA_ALLOC_CONF = (
    "expandable_segments:True,"           # Prevent fragmentation
    "max_split_size_mb:128,"              # Don't split blocks >128MB
    "garbage_collection_threshold:0.8"    # Reclaim at 80% usage
)

# Model-specific settings (to be expanded in later phases)
WHISPER_MODEL = "large-v3"
SEAMLESS_MODEL = "seamlessM4T_v2_large"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
WAV2LIP_MODEL = "Wav2Lip"

# Pipeline settings (to be expanded in later phases)
MAX_VIDEO_DURATION_MINUTES = 30
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mkv", ".avi", ".mov"]
SUPPORTED_AUDIO_SAMPLE_RATE = 48000

# ASR (Automatic Speech Recognition) settings
ASR_SAMPLE_RATE = 16000  # Required by Whisper and pyannote
ASR_CONFIDENCE_THRESHOLD = 0.7  # Flag segments below this confidence for review

# Translation settings (Phase 4)
SEAMLESS_MODEL_ID = "facebook/seamless-m4t-v2-large"  # 2.3B parameters, ~6GB VRAM
TRANSLATION_TARGET_LANGUAGE = "eng"  # English output
TRANSLATION_CONFIDENCE_THRESHOLD = 0.7  # Flag segments below this
TRANSLATION_DURATION_TOLERANCE = 0.1  # ±10% duration fit tolerance
TRANSLATION_CHARS_PER_SECOND = 15  # English conversational speech rate

# Supported source languages (96 languages supported by SeamlessM4T v2)
# Reference: https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages
# Priority languages: jpn (Japanese), kor (Korean), cmn (Mandarin), spa (Spanish),
# fra (French), deu (German), hin (Hindi), ara (Arabic)

# Logging
LOG_LEVEL = "INFO"
VERBOSE_MEMORY_LOGGING = True
