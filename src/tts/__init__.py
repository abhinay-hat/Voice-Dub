"""
TTS (Text-to-Speech) voice cloning components.
Phase 5: Voice cloning reference extraction and speaker embeddings.
"""
from .reference_extractor import (
    extract_reference_samples,
    select_best_segment,
    ReferenceExtractor
)

__all__ = [
    'extract_reference_samples',
    'select_best_segment',
    'ReferenceExtractor'
]
