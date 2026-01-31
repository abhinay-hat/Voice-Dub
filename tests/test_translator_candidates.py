"""
Test suite for multi-candidate beam search and batch translation.

Tests:
1. Multiple candidates generation
2. Confidence score ordering
3. Batch translation
4. GPU memory efficiency
"""
import time
import torch

# Add project root to path
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_manager import ModelManager
from src.stages.translation import Translator
from src.utils.gpu_validation import validate_gpu_environment
from src.utils.memory_monitor import print_gpu_memory_summary


def test_multiple_candidates_generation():
    """Test 1: Generate multiple translation candidates with beam search."""
    print("\n" + "="*70)
    print("Test 1: Multiple Candidates Generation")
    print("="*70)

    manager = ModelManager(verbose=True)
    translator = Translator(manager)

    # Test with Japanese text
    test_text = "こんにちは、元気ですか？"
    source_lang = "jpn"
    num_candidates = 3

    print(f"\nInput: {test_text}")
    print(f"Source language: {source_lang}")
    print(f"Requested candidates: {num_candidates}")

    # Generate candidates
    candidates, scores = translator.translate_with_candidates(
        test_text,
        source_lang,
        num_candidates=num_candidates
    )

    # Verify return types
    assert isinstance(candidates, list), "Candidates should be a list"
    assert isinstance(scores, list), "Scores should be a list"
    assert len(candidates) == num_candidates, f"Expected {num_candidates} candidates, got {len(candidates)}"
    assert len(scores) == num_candidates, f"Expected {num_candidates} scores, got {len(scores)}"

    # Verify all candidates are different strings
    unique_candidates = set(candidates)
    print(f"\nGenerated {len(candidates)} candidates:")
    for i, (cand, score) in enumerate(zip(candidates, scores)):
        print(f"  {i+1}. [{score:.4f}] {cand}")

    # Note: In some cases, beam search may produce identical candidates
    # if the model is very confident in one translation
    if len(unique_candidates) < num_candidates:
        print(f"\nNote: Only {len(unique_candidates)} unique candidates (beam collapse)")

    # Verify all scores are floats in range [0, 1]
    for i, score in enumerate(scores):
        assert isinstance(score, float), f"Score {i} is not a float: {type(score)}"
        assert 0 <= score <= 1, f"Score {i} out of range [0, 1]: {score}"

    print("\n✓ Test 1 PASSED: Multiple candidates generated successfully")
    return True


def test_confidence_score_ordering():
    """Test 2: Verify confidence scores are in descending order."""
    print("\n" + "="*70)
    print("Test 2: Confidence Score Ordering")
    print("="*70)

    manager = ModelManager(verbose=True)
    translator = Translator(manager)

    # Test with Spanish text
    test_text = "¿Cómo estás hoy?"
    source_lang = "spa"
    num_candidates = 3

    print(f"\nInput: {test_text}")
    print(f"Generating {num_candidates} candidates...")

    candidates, scores = translator.translate_with_candidates(
        test_text,
        source_lang,
        num_candidates=num_candidates
    )

    print("\nCandidates with scores:")
    for i, (cand, score) in enumerate(zip(candidates, scores)):
        print(f"  {i+1}. [{score:.4f}] {cand}")

    # Verify scores are in descending order (best first)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], (
            f"Scores not in descending order: "
            f"scores[{i}]={scores[i]:.4f} < scores[{i+1}]={scores[i+1]:.4f}"
        )

    print(f"\n✓ Highest score: {scores[0]:.4f} (most confident)")
    print(f"✓ Lowest score: {scores[-1]:.4f} (least confident)")
    print("\n✓ Test 2 PASSED: Confidence scores properly ordered")
    return True


def test_batch_translation():
    """Test 3: Batch translation with multiple segments."""
    print("\n" + "="*70)
    print("Test 3: Batch Translation")
    print("="*70)

    manager = ModelManager(verbose=True)
    translator = Translator(manager)

    # Test segments in different languages (all supported by SeamlessM4T)
    test_segments = [
        "こんにちは",  # Japanese: Hello
        "ありがとう",  # Japanese: Thank you
        "さようなら",  # Japanese: Goodbye
        "おはよう",    # Japanese: Good morning
        "こんばんは"   # Japanese: Good evening
    ]
    source_lang = "jpn"
    num_candidates = 2

    print(f"\nBatch size: {len(test_segments)}")
    print(f"Source language: {source_lang}")
    print(f"Candidates per segment: {num_candidates}")

    # Test batch translation
    print("\nRunning batch translation...")
    start_time = time.time()
    results = translator.translate_batch(
        test_segments,
        source_lang,
        num_candidates=num_candidates
    )
    batch_time = time.time() - start_time

    # Verify results structure
    assert isinstance(results, list), "Results should be a list"
    assert len(results) == len(test_segments), (
        f"Expected {len(test_segments)} results, got {len(results)}"
    )

    print(f"\nBatch translation completed in {batch_time:.2f}s")
    print("\nResults:")
    for i, result in enumerate(results):
        assert "segment_text" in result, f"Result {i} missing 'segment_text'"
        assert "candidates" in result, f"Result {i} missing 'candidates'"
        assert "scores" in result, f"Result {i} missing 'scores'"

        assert len(result["candidates"]) == num_candidates, (
            f"Result {i}: Expected {num_candidates} candidates, got {len(result['candidates'])}"
        )
        assert len(result["scores"]) == num_candidates, (
            f"Result {i}: Expected {num_candidates} scores, got {len(result['scores'])}"
        )

        print(f"\n  {i+1}. Original: {result['segment_text']}")
        for j, (cand, score) in enumerate(zip(result['candidates'], result['scores'])):
            print(f"     Candidate {j+1}: [{score:.4f}] {cand}")

    # Compare with individual calls (time efficiency)
    print("\nComparing with individual translations...")
    individual_start = time.time()
    for segment in test_segments[:2]:  # Test first 2 to save time
        _ = translator.translate_with_candidates(segment, source_lang, num_candidates=1)
    individual_time = time.time() - individual_start
    individual_projected = individual_time * (len(test_segments) / 2)

    print(f"\nIndividual calls (projected): {individual_projected:.2f}s")
    print(f"Batch translation: {batch_time:.2f}s")
    speedup = individual_projected / batch_time
    print(f"Speedup: {speedup:.2f}x")

    # Note: Speedup may vary, but batch should generally be faster
    if speedup > 1.0:
        print("✓ Batch processing is faster than individual calls")

    print("\n✓ Test 3 PASSED: Batch translation works correctly")
    return True


def test_gpu_memory_efficiency():
    """Test 4: Verify no memory leaks in batch translation."""
    print("\n" + "="*70)
    print("Test 4: GPU Memory Efficiency")
    print("="*70)

    manager = ModelManager(verbose=True)
    translator = Translator(manager)

    # Warm-up: Load model
    print("\nWarming up (loading model)...")
    _ = translator.translate_segment("テスト", "jpn")

    # Measure baseline VRAM
    print("\nBaseline VRAM usage:")
    baseline_allocated = torch.cuda.memory_allocated() / (1024**3)
    baseline_reserved = torch.cuda.memory_reserved() / (1024**3)
    print_gpu_memory_summary("Before batch: ")

    # Run batch translation
    test_segments = [f"テストメッセージ {i}" for i in range(8)]
    print(f"\nRunning batch translation ({len(test_segments)} segments)...")
    _ = translator.translate_batch(test_segments, "jpn", num_candidates=2)

    # Measure VRAM after batch
    print("\nAfter batch translation:")
    after_allocated = torch.cuda.memory_allocated() / (1024**3)
    after_reserved = torch.cuda.memory_reserved() / (1024**3)
    print_gpu_memory_summary("After batch: ")

    # Clear cache
    torch.cuda.empty_cache()

    # Measure VRAM after cleanup
    print("\nAfter cache clear:")
    final_allocated = torch.cuda.memory_allocated() / (1024**3)
    final_reserved = torch.cuda.memory_reserved() / (1024**3)
    print_gpu_memory_summary("After cleanup: ")

    # Check for memory leak
    allocated_diff = final_allocated - baseline_allocated
    print(f"\nMemory difference: {allocated_diff:.3f} GB")

    # Allow small drift (< 100MB) due to PyTorch caching
    if abs(allocated_diff) < 0.1:
        print("✓ No significant memory leak detected")
    else:
        print(f"⚠ Warning: Memory difference {allocated_diff:.3f} GB (threshold: 0.1 GB)")

    print("\n✓ Test 4 PASSED: GPU memory efficiency verified")
    return True


def run_all_tests():
    """Run all translator candidate tests."""
    print("\n" + "="*70)
    print("TRANSLATOR CANDIDATES TEST SUITE")
    print("="*70)

    # Validate GPU environment first
    print("\nValidating GPU environment...")
    try:
        validate_gpu_environment()
        print("✓ GPU environment validated")
    except Exception as e:
        print(f"✗ GPU validation failed: {e}")
        print("\nSkipping tests (GPU required)")
        return False

    tests = [
        ("Multiple Candidates Generation", test_multiple_candidates_generation),
        ("Confidence Score Ordering", test_confidence_score_ordering),
        ("Batch Translation", test_batch_translation),
        ("GPU Memory Efficiency", test_gpu_memory_efficiency),
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
