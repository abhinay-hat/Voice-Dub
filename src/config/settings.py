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

# Multi-candidate generation (Phase 4, Plan 3)
TRANSLATION_NUM_CANDIDATES = 3  # Default beam width for multi-candidate generation
TRANSLATION_BATCH_SIZE = 8  # Process 8 segments at a time
TRANSLATION_MAX_TOKENS_PER_SEGMENT = 512  # Max output length

# Translation chunking for long videos (Phase 4, Plan 3)
TRANSLATION_MAX_CHUNK_TOKENS = 1024  # SeamlessM4T safe upper limit
TRANSLATION_OVERLAP_TOKENS = 128  # Context preservation overlap (128-256 recommended)
TRANSLATION_APPROX_CHARS_PER_TOKEN = 4  # Rough estimate for token counting

# Supported source languages (96 languages supported by SeamlessM4T v2)
# Reference: https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages
# Priority languages: jpn (Japanese), kor (Korean), cmn (Mandarin), spa (Spanish),
# fra (French), deu (German), hin (Hindi), ara (Arabic)

# TTS (Text-to-Speech) settings (Phase 5)
TTS_SAMPLE_RATE = 24000  # XTTS native sample rate
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_REFERENCE_MIN_DURATION = 6.0  # Minimum reference sample duration (seconds)
TTS_REFERENCE_MAX_DURATION = 10.0  # Maximum reference sample duration (seconds)
TTS_CONCATENATION_GAP_THRESHOLD = 0.5  # Max gap between segments for concatenation (seconds)

# XTTS synthesis parameters
TTS_TEMPERATURE = 0.65  # Creativity control (0.1-1.0, lower = more consistent)
TTS_LENGTH_PENALTY = 1.0  # >1.0 encourages longer output
TTS_REPETITION_PENALTY = 2.0  # Reduces repeated phrases
TTS_TOP_K = 50  # Decoder sampling parameter
TTS_TOP_P = 0.85  # Nucleus sampling

# Duration matching
TTS_DURATION_TOLERANCE = 0.05  # ±5% duration match tolerance
TTS_SPEED_MIN = 0.8  # Minimum speed adjustment
TTS_SPEED_MAX = 1.2  # Maximum speed adjustment
TTS_MAX_DURATION_RETRIES = 3  # Max retries for duration matching

# Quality validation
TTS_MIN_PESQ_SCORE = 2.5  # Minimum acceptable PESQ score
TTS_PESQ_REVIEW_THRESHOLD = 3.0  # Flag for review if below this
TTS_SHORT_TEXT_THRESHOLD = 3.0  # Minimum target duration (seconds) before short text handling

# Assembly settings (Phase 6)
ASSEMBLY_TARGET_SAMPLE_RATE = 48000  # Video production standard (48kHz)
ASSEMBLY_DRIFT_TOLERANCE_MS = 45.0  # ATSC recommendation: max 45ms audio-video offset
ASSEMBLY_CHECKPOINT_INTERVAL = 300.0  # 5-minute validation intervals (seconds)
ASSEMBLY_RESAMPLING_QUALITY = 'kaiser_best'  # High-quality sinc interpolation

# Logging
LOG_LEVEL = "INFO"
VERBOSE_MEMORY_LOGGING = True
