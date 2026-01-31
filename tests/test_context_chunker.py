"""
Test suite for context-preserving chunking logic.

Tests:
1. No chunking needed for short transcript
2. Chunking with overlap for long transcript
3. Chunk merging after translation
4. Edge cases (empty, single segment, very long segment)
"""

# Add project root to path
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.translation import ContextChunker, chunk_transcript_with_overlap, merge_translated_chunks
from src.config.settings import TRANSLATION_MAX_CHUNK_TOKENS, TRANSLATION_OVERLAP_TOKENS


def test_no_chunking_short_transcript():
    """Test 1: Short transcript fits in single chunk."""
    print("\n" + "="*70)
    print("Test 1: No Chunking Needed for Short Transcript")
    print("="*70)

    # Create 10 short segments (~50 chars each = ~500 chars total = ~125 tokens)
    segments = [
        {"text": f"This is segment {i} with some short text.", "duration": 2.0}
        for i in range(10)
    ]

    total_chars = sum(len(seg["text"]) for seg in segments)
    estimated_tokens = total_chars // 4  # Rough estimate
    print(f"\nTotal segments: {len(segments)}")
    print(f"Total characters: {total_chars}")
    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Max chunk tokens: {TRANSLATION_MAX_CHUNK_TOKENS}")

    # Chunk transcript
    chunker = ContextChunker()
    chunks = chunker.chunk_transcript_with_overlap(segments)

    # Verify single chunk
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    print(f"\n✓ Single chunk created (no splitting needed)")

    chunk = chunks[0]
    assert len(chunk["segments"]) == len(segments), (
        f"Expected {len(segments)} segments in chunk, got {len(chunk['segments'])}"
    )
    assert chunk["has_overlap"] == False, "First chunk should not have overlap"
    assert chunk["start_idx"] == 0, "Chunk should start at index 0"
    assert chunk["end_idx"] == len(segments) - 1, (
        f"Chunk should end at index {len(segments)-1}, got {chunk['end_idx']}"
    )

    print(f"✓ Chunk contains all {len(segments)} segments")
    print(f"✓ has_overlap = False (expected)")
    print(f"✓ Token count: {chunk['total_tokens']}")

    print("\n✓ Test 1 PASSED: Short transcript handled correctly")
    return True


def test_chunking_with_overlap():
    """Test 2: Long transcript splits into overlapping chunks."""
    print("\n" + "="*70)
    print("Test 2: Chunking with Overlap for Long Transcript")
    print("="*70)

    # Create 100 segments (~200 chars each = ~20,000 chars = ~5,000 tokens)
    segments = [
        {
            "text": f"Segment {i} contains a moderately long sentence with multiple words "
                    f"to ensure we exceed the token limit and trigger chunking behavior. "
                    f"This helps test the overlap functionality properly.",
            "duration": 3.0
        }
        for i in range(100)
    ]

    total_chars = sum(len(seg["text"]) for seg in segments)
    estimated_tokens = total_chars // 4
    print(f"\nTotal segments: {len(segments)}")
    print(f"Total characters: {total_chars}")
    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Max chunk tokens: {TRANSLATION_MAX_CHUNK_TOKENS}")
    print(f"Overlap tokens: {TRANSLATION_OVERLAP_TOKENS}")

    # Chunk transcript
    chunker = ContextChunker(max_tokens=1024, overlap_tokens=128)
    chunks = chunker.chunk_transcript_with_overlap(segments)

    # Verify multiple chunks created
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    print(f"\n✓ Split into {len(chunks)} chunks")

    # Verify chunk properties
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Segments: {len(chunk['segments'])} (idx {chunk['start_idx']}-{chunk['end_idx']})")
        print(f"  Tokens: {chunk['total_tokens']}")
        print(f"  Has overlap: {chunk['has_overlap']}")

        # Verify token limit
        assert chunk["total_tokens"] <= 1024, (
            f"Chunk {i} exceeds token limit: {chunk['total_tokens']} > 1024"
        )

        # Verify overlap property
        if i == 0:
            assert chunk["has_overlap"] == False, "First chunk should not have overlap"
        else:
            assert chunk["has_overlap"] == True, f"Chunk {i} should have overlap"

    # Verify adjacent chunks have overlap (segments appear in both)
    for i in range(len(chunks) - 1):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]

        # Check if there's segment overlap
        # chunk_b should start before chunk_a ends if there's overlap
        if chunk_b["has_overlap"]:
            print(f"\n✓ Overlap detected between chunk {i+1} and {i+2}")
            print(f"  Chunk {i+1} ends at segment {chunk_a['end_idx']}")
            print(f"  Chunk {i+2} starts at segment {chunk_b['start_idx']}")
            assert chunk_b["start_idx"] <= chunk_a["end_idx"], (
                f"Expected overlap: chunk {i+2} should start before chunk {i+1} ends"
            )

    print("\n✓ Test 2 PASSED: Long transcript chunked with overlap")
    return True


def test_chunk_merging():
    """Test 3: Merge translated chunks, removing duplicates."""
    print("\n" + "="*70)
    print("Test 3: Chunk Merging After Translation")
    print("="*70)

    # Create mock overlapping chunks
    # Original transcript: 10 segments (indices 0-9)
    # Chunk 1: segments 0-5 (6 segments)
    # Chunk 2: segments 4-9 (6 segments, overlaps with 4-5 from chunk 1)

    original_segments = [
        {"text": f"Original segment {i}", "duration": 2.0}
        for i in range(10)
    ]

    chunks = [
        {
            "segments": original_segments[0:6],
            "start_idx": 0,
            "end_idx": 5,
            "total_tokens": 300,
            "has_overlap": False
        },
        {
            "segments": original_segments[4:10],
            "start_idx": 4,
            "end_idx": 9,
            "total_tokens": 300,
            "has_overlap": True
        }
    ]

    print(f"\nOriginal segments: {len(original_segments)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk 1: segments 0-5 (6 segments)")
    print(f"Chunk 2: segments 4-9 (6 segments, overlap on 4-5)")

    # Mock translated segments (all segments from all chunks, with duplicates)
    translated_segments = []

    # Chunk 1 translations
    for i in range(6):
        translated_segments.append({
            "text": f"Translated segment {i} from chunk 1",
            "duration": 2.0
        })

    # Chunk 2 translations (includes re-translation of segments 4-5)
    for i in range(4, 10):
        translated_segments.append({
            "text": f"Translated segment {i} from chunk 2",  # Better context
            "duration": 2.0
        })

    print(f"\nTotal translated segments (with duplicates): {len(translated_segments)}")
    assert len(translated_segments) == 12, "Expected 12 segments (6 + 6)"

    # Merge translated chunks
    chunker = ContextChunker()
    merged = chunker.merge_translated_chunks(translated_segments, chunks)

    print(f"Merged segments (duplicates removed): {len(merged)}")

    # Verify correct count (should match original)
    assert len(merged) == len(original_segments), (
        f"Expected {len(original_segments)} merged segments, got {len(merged)}"
    )

    # Verify overlapped segments use translation from later chunk (better context)
    # Segments 4-5 should come from chunk 2 (later chunk)
    for i in [4, 5]:
        assert "chunk 2" in merged[i]["text"], (
            f"Segment {i} should use translation from chunk 2 (later chunk), "
            f"got: {merged[i]['text']}"
        )
        print(f"✓ Segment {i}: {merged[i]['text']}")

    # Verify segments 0-3 come from chunk 1
    for i in range(4):
        assert "chunk 1" in merged[i]["text"], (
            f"Segment {i} should use translation from chunk 1, got: {merged[i]['text']}"
        )

    # Verify segments 6-9 come from chunk 2
    for i in range(6, 10):
        assert "chunk 2" in merged[i]["text"], (
            f"Segment {i} should use translation from chunk 2, got: {merged[i]['text']}"
        )

    print("\n✓ Test 3 PASSED: Chunks merged correctly, overlaps resolved")
    return True


def test_edge_cases():
    """Test 4: Edge cases (empty, single segment, very long segment)."""
    print("\n" + "="*70)
    print("Test 4: Edge Cases")
    print("="*70)

    chunker = ContextChunker()

    # Edge case 1: Empty segments list
    print("\nEdge case 1: Empty segments list")
    empty_chunks = chunker.chunk_transcript_with_overlap([])
    assert empty_chunks == [], "Empty segments should return empty chunks"
    print("✓ Empty list handled correctly")

    # Edge case 2: Single segment
    print("\nEdge case 2: Single segment")
    single_segment = [{"text": "Single segment text", "duration": 2.0}]
    single_chunks = chunker.chunk_transcript_with_overlap(single_segment)
    assert len(single_chunks) == 1, "Single segment should return single chunk"
    assert len(single_chunks[0]["segments"]) == 1, "Single chunk should have 1 segment"
    assert single_chunks[0]["has_overlap"] == False, "Single chunk has no overlap"
    print("✓ Single segment handled correctly")

    # Edge case 3: Very long single segment (still returns single chunk, no mid-segment split)
    print("\nEdge case 3: Very long single segment")
    very_long_text = "This is a very long segment. " * 500  # ~15,000 chars = ~3,750 tokens
    long_segment = [{"text": very_long_text, "duration": 60.0}]
    long_chunks = chunker.chunk_transcript_with_overlap(long_segment)
    assert len(long_chunks) == 1, "Very long segment should still return single chunk (no mid-segment split)"
    print(f"✓ Very long segment (~{len(very_long_text)} chars) kept as single chunk")
    print(f"  Estimated tokens: {long_chunks[0]['total_tokens']}")

    # Edge case 4: Segments missing "text" key
    print("\nEdge case 4: Invalid segment (missing 'text' key)")
    invalid_segments = [{"duration": 2.0}]  # Missing "text"
    try:
        chunker.chunk_transcript_with_overlap(invalid_segments)
        assert False, "Should raise ValueError for missing 'text' key"
    except ValueError as e:
        assert "missing 'text' key" in str(e), f"Unexpected error message: {e}"
        print(f"✓ ValueError raised correctly: {e}")

    print("\n✓ Test 4 PASSED: Edge cases handled correctly")
    return True


def test_convenience_functions():
    """Test 5: Module-level convenience functions."""
    print("\n" + "="*70)
    print("Test 5: Convenience Functions")
    print("="*70)

    # Test chunk_transcript_with_overlap convenience function
    print("\nTesting chunk_transcript_with_overlap() convenience function...")
    segments = [
        {"text": f"Segment {i} text content here.", "duration": 2.0}
        for i in range(20)
    ]

    chunks = chunk_transcript_with_overlap(segments, max_tokens=500, overlap_tokens=50)
    assert isinstance(chunks, list), "Should return list of chunks"
    print(f"✓ Created {len(chunks)} chunks using convenience function")

    # Test merge_translated_chunks convenience function
    print("\nTesting merge_translated_chunks() convenience function...")
    translated = []
    for chunk in chunks:
        for seg in chunk["segments"]:
            translated.append({"text": f"Translated: {seg['text']}", "duration": seg["duration"]})

    merged = merge_translated_chunks(translated, chunks)
    assert isinstance(merged, list), "Should return list of segments"
    print(f"✓ Merged into {len(merged)} segments using convenience function")

    print("\n✓ Test 5 PASSED: Convenience functions work correctly")
    return True


def run_all_tests():
    """Run all context chunker tests."""
    print("\n" + "="*70)
    print("CONTEXT CHUNKER TEST SUITE")
    print("="*70)

    tests = [
        ("No Chunking for Short Transcript", test_no_chunking_short_transcript),
        ("Chunking with Overlap", test_chunking_with_overlap),
        ("Chunk Merging", test_chunk_merging),
        ("Edge Cases", test_edge_cases),
        ("Convenience Functions", test_convenience_functions),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED ✓")
        return True
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
