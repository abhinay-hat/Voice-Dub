"""
Context-preserving chunking for long video transcripts.
Splits transcripts into overlapping chunks to maintain conversational coherence.
"""
from typing import Optional

from src.config.settings import (
    TRANSLATION_MAX_CHUNK_TOKENS,
    TRANSLATION_OVERLAP_TOKENS,
    TRANSLATION_APPROX_CHARS_PER_TOKEN
)


class ContextChunker:
    """
    Chunks long transcripts into overlapping segments for translation.

    Maintains conversational context by including overlap between chunks,
    preventing jarring transitions in long videos that exceed model token limits.

    Usage:
        >>> chunker = ContextChunker(max_tokens=1024, overlap_tokens=128)
        >>> segments = [{"text": "...", "duration": 2.0}, ...]
        >>> chunks = chunker.chunk_transcript_with_overlap(segments)
        >>> # Translate each chunk...
        >>> merged = chunker.merge_translated_chunks(translated_segments, chunks)
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None
    ):
        """
        Initialize context chunker with configuration.

        Args:
            max_tokens: Maximum tokens per chunk (default: TRANSLATION_MAX_CHUNK_TOKENS)
            overlap_tokens: Overlap size for context (default: TRANSLATION_OVERLAP_TOKENS)
        """
        self.max_tokens = max_tokens if max_tokens is not None else TRANSLATION_MAX_CHUNK_TOKENS
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else TRANSLATION_OVERLAP_TOKENS

        # Validate configuration
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < max_tokens ({self.max_tokens})"
            )

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from character length.

        Args:
            text: Input text

        Returns:
            Approximate token count (chars / TRANSLATION_APPROX_CHARS_PER_TOKEN)

        Note: Conservative estimate for SeamlessM4T tokenizer. Actual token count
              may vary, so we use conservative chunking to stay within limits.
        """
        return len(text) // TRANSLATION_APPROX_CHARS_PER_TOKEN

    def chunk_transcript_with_overlap(self, segments: list[dict]) -> list[dict]:
        """
        Split transcript segments into overlapping chunks.

        Args:
            segments: List of segment dicts with "text" key (and optional metadata)

        Returns:
            List of chunk dicts:
                {
                    "segments": list[dict],  # Segment dicts from input
                    "start_idx": int,        # Index of first segment in original list
                    "end_idx": int,          # Index of last segment in original list
                    "total_tokens": int,     # Approximate token count
                    "has_overlap": bool      # True if this chunk includes overlap
                }

        Example:
            >>> segments = [
            ...     {"text": "First sentence.", "duration": 1.5},
            ...     {"text": "Second sentence.", "duration": 2.0},
            ...     ...
            ... ]
            >>> chunks = chunker.chunk_transcript_with_overlap(segments)
            >>> print(f"Split into {len(chunks)} chunks")
            Split into 3 chunks

        Note: If transcript fits in single chunk, returns single-element list.
              Adjacent chunks will have overlapping segments for context preservation.
        """
        # Handle edge cases
        if not segments:
            return []

        if len(segments) == 1:
            # Single segment, no chunking needed
            single_tokens = self._estimate_tokens(segments[0].get("text", ""))
            return [{
                "segments": segments,
                "start_idx": 0,
                "end_idx": 0,
                "total_tokens": single_tokens,
                "has_overlap": False
            }]

        # Validate segments have "text" key
        for i, seg in enumerate(segments):
            if "text" not in seg:
                raise ValueError(f"Segment at index {i} missing 'text' key")

        # Build chunks with overlap
        chunks = []
        current_chunk_segments = []
        current_chunk_tokens = 0
        current_start_idx = 0
        overlap_start_idx = None  # Track where overlap begins in current chunk

        for i, segment in enumerate(segments):
            segment_tokens = self._estimate_tokens(segment["text"])

            # Check if adding this segment exceeds max_tokens
            if current_chunk_tokens + segment_tokens > self.max_tokens and current_chunk_segments:
                # Save current chunk
                chunks.append({
                    "segments": current_chunk_segments,
                    "start_idx": current_start_idx,
                    "end_idx": current_start_idx + len(current_chunk_segments) - 1,
                    "total_tokens": current_chunk_tokens,
                    "has_overlap": overlap_start_idx is not None
                })

                # Start new chunk with overlap from previous chunk
                overlap_segments = []
                overlap_tokens = 0

                # Take segments from end of previous chunk as overlap
                for prev_seg in reversed(current_chunk_segments):
                    prev_tokens = self._estimate_tokens(prev_seg["text"])
                    if overlap_tokens + prev_tokens <= self.overlap_tokens:
                        overlap_segments.insert(0, prev_seg)
                        overlap_tokens += prev_tokens
                    else:
                        break

                # Initialize new chunk
                if overlap_segments:
                    current_chunk_segments = overlap_segments
                    current_chunk_tokens = overlap_tokens
                    # Track that this chunk starts with overlap
                    current_start_idx = current_start_idx + len(chunks[0]["segments"]) - len(overlap_segments)
                    overlap_start_idx = 0  # Overlap starts at beginning of new chunk
                else:
                    current_chunk_segments = []
                    current_chunk_tokens = 0
                    current_start_idx = i
                    overlap_start_idx = None

            # Add current segment to chunk
            current_chunk_segments.append(segment)
            current_chunk_tokens += segment_tokens

        # Add final chunk
        if current_chunk_segments:
            chunks.append({
                "segments": current_chunk_segments,
                "start_idx": current_start_idx,
                "end_idx": current_start_idx + len(current_chunk_segments) - 1,
                "total_tokens": current_chunk_tokens,
                "has_overlap": overlap_start_idx is not None
            })

        return chunks

    def merge_translated_chunks(
        self,
        translated_segments: list[dict],
        chunks: list[dict]
    ) -> list[dict]:
        """
        Merge translated chunks back to original segment list, deduplicating overlaps.

        Args:
            translated_segments: Flat list of all translated segments from all chunks
            chunks: Original chunk metadata from chunk_transcript_with_overlap()

        Returns:
            List of translated segments in original order, duplicates removed

        Example:
            >>> # After translating each chunk
            >>> translated = []
            >>> for chunk in chunks:
            ...     # Translate chunk["segments"]...
            ...     translated.extend(chunk_translated_results)
            >>> merged = chunker.merge_translated_chunks(translated, chunks)

        Note: For overlapped segments (appear in multiple chunks), keeps translation
              from later chunk (has more context from subsequent segments).
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            # No overlap to merge, return as-is
            return translated_segments

        # Build map of segment index to translation
        # For overlaps, later chunks override earlier ones (more context)
        segment_translations = {}

        translated_idx = 0
        for chunk in chunks:
            chunk_start = chunk["start_idx"]
            chunk_segment_count = len(chunk["segments"])

            for i in range(chunk_segment_count):
                original_idx = chunk_start + i
                segment_translations[original_idx] = translated_segments[translated_idx]
                translated_idx += 1

        # Rebuild in original order
        max_idx = max(segment_translations.keys())
        merged = []
        for i in range(max_idx + 1):
            if i in segment_translations:
                merged.append(segment_translations[i])

        return merged


def chunk_transcript_with_overlap(
    segments: list[dict],
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None
) -> list[dict]:
    """
    Convenience function for chunking transcript with overlap.

    Creates a ContextChunker instance and chunks the transcript in one call.

    Args:
        segments: List of segment dicts with "text" key
        max_tokens: Maximum tokens per chunk (default: TRANSLATION_MAX_CHUNK_TOKENS)
        overlap_tokens: Overlap size for context (default: TRANSLATION_OVERLAP_TOKENS)

    Returns:
        List of chunk dicts (see ContextChunker.chunk_transcript_with_overlap)

    Example:
        >>> from src.stages.translation import chunk_transcript_with_overlap
        >>> segments = [{"text": "...", "duration": 2.0}, ...]
        >>> chunks = chunk_transcript_with_overlap(segments)
    """
    chunker = ContextChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    return chunker.chunk_transcript_with_overlap(segments)


def merge_translated_chunks(
    translated_segments: list[dict],
    chunks: list[dict]
) -> list[dict]:
    """
    Convenience function for merging translated chunks.

    Creates a ContextChunker instance and merges chunks in one call.

    Args:
        translated_segments: Flat list of all translated segments from all chunks
        chunks: Original chunk metadata from chunk_transcript_with_overlap()

    Returns:
        List of translated segments in original order, duplicates removed

    Example:
        >>> from src.stages.translation import chunk_transcript_with_overlap, merge_translated_chunks
        >>> chunks = chunk_transcript_with_overlap(segments)
        >>> # ... translate each chunk ...
        >>> merged = merge_translated_chunks(all_translated, chunks)
    """
    chunker = ContextChunker()
    return chunker.merge_translated_chunks(translated_segments, chunks)
