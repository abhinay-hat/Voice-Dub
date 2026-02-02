"""
TTS (Text-to-Speech) voice cloning components.
Phase 5: Voice cloning reference extraction, speaker embeddings, and XTTS synthesis.
"""
from .reference_extractor import (
    extract_reference_samples,
    select_best_segment,
    ReferenceExtractor
)
from .speaker_embeddings import (
    SpeakerEmbeddingCache,
    generate_speaker_embeddings,
    generate_single_embedding
)
from .xtts_generator import (
    XTTSGenerator,
    BatchSynthesisError
)
from .quality_validator import (
    AudioQualityResult,
    QualityValidator,
    validate_audio_quality
)

__all__ = [
    'extract_reference_samples',
    'select_best_segment',
    'ReferenceExtractor',
    'SpeakerEmbeddingCache',
    'generate_speaker_embeddings',
    'generate_single_embedding',
    'XTTSGenerator',
    'BatchSynthesisError',
    'AudioQualityResult',
    'QualityValidator',
    'validate_audio_quality'
]
