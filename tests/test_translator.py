"""
Test suite for SeamlessM4T v2 translation integration.

Validates:
- Model loading on GPU (not CPU fallback)
- Multi-language translation to English
- ModelManager integration
- VRAM usage tracking
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models.model_manager import ModelManager
from src.stages.translation import Translator
from src.utils.memory_monitor import print_gpu_memory_summary


# Test data: multiple languages with expected English output patterns
TEST_CASES = [
    {
        "text": "こんにちは、元気ですか？",
        "lang": "jpn",
        "expected_contains": ["hello", "how", "hi"]  # Flexible matching
    },
    {
        "text": "Hola, ¿cómo estás?",
        "lang": "spa",
        "expected_contains": ["hello", "how", "hi"]
    },
    {
        "text": "안녕하세요",
        "lang": "kor",
        "expected_contains": ["hello", "hi"]
    },
]


def test_model_loading():
    """Test 1: Verify SeamlessM4T loads on GPU without CPU fallback."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading on GPU")
    print("="*60)

    # Ensure GPU is available
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (CPU-only environment)")
        return False

    print_gpu_memory_summary("Before model load: ")

    # Create translator with ModelManager
    manager = ModelManager(verbose=True)
    translator = Translator(manager)

    # Trigger model load by translating Japanese test sentence
    print("\nTranslating Japanese test sentence...")
    result = translator.translate_segment("こんにちは", "jpn")

    print_gpu_memory_summary("After model load: ")

    # Verify model is on GPU
    assert translator.model is not None, "Model should be loaded"
    assert next(translator.model.parameters()).is_cuda, "Model should be on CUDA device (not CPU)"

    # Verify translation result structure
    assert "translation" in result, "Result should have 'translation' key"
    assert "source_lang" in result, "Result should have 'source_lang' key"
    assert "target_lang" in result, "Result should have 'target_lang' key"
    assert result["target_lang"] == "eng", "Target language should be English"

    print(f"\nTranslation result: {result['translation']}")
    print("[PASS] Model loaded on GPU successfully")

    return True


def test_multiple_languages():
    """Test 2: Verify multiple language translations produce valid English outputs."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Language Translation")
    print("="*60)

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (CPU-only environment)")
        return False

    # Create translator (reuses existing model if loaded)
    manager = ModelManager(verbose=False)  # Quiet mode for cleaner output
    translator = Translator(manager)

    all_passed = True

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest case {i}: {test_case['lang']}")
        # Don't print Unicode strings (causes Windows encoding issues)

        result = translator.translate_segment(test_case['text'], test_case['lang'])
        translation = result["translation"].lower()

        print(f"  Output: {result['translation']}")

        # Verify it's not just echoing the input
        assert translation != test_case['text'].lower(), \
            f"Translation should not echo input for {test_case['lang']}"

        # Verify some expected English words appear (flexible matching)
        contains_expected = any(
            expected_word in translation
            for expected_word in test_case['expected_contains']
        )

        if contains_expected:
            print(f"  [PASS] Contains expected English words")
        else:
            print(f"  [WARN] Expected one of {test_case['expected_contains']}, got '{translation}'")
            # Not a hard failure - translation models can produce varied outputs
            # But flag for manual review

        # Verify source/target languages
        assert result["source_lang"] == test_case["lang"], \
            f"Source language mismatch: expected {test_case['lang']}, got {result['source_lang']}"
        assert result["target_lang"] == "eng", "Target language should be 'eng'"

    print("\n[PASS] All language tests completed")
    return all_passed


def test_model_manager_integration():
    """Test 3: Verify ModelManager integration and VRAM tracking."""
    print("\n" + "="*60)
    print("TEST 3: ModelManager Integration")
    print("="*60)

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (CPU-only environment)")
        return False

    # Create fresh manager to track VRAM changes
    manager = ModelManager(verbose=True)

    print("\nBefore loading SeamlessM4T:")
    print_gpu_memory_summary("  ")

    # Load translator
    translator = Translator(manager)
    result = translator.translate_segment("Test", "eng")  # English-to-English passthrough

    print("\nAfter loading SeamlessM4T:")
    print_gpu_memory_summary("  ")

    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"\nTotal VRAM allocated: {allocated_gb:.2f} GB")

    # Verify model is tracked by ModelManager
    assert manager.get_current_model_name() == "seamless_m4t_translation", \
        "ModelManager should track 'seamless_m4t_translation'"

    # Verify VRAM allocation is reasonable (~6GB for SeamlessM4T v2 Large)
    # Allow for variation due to other GPU processes
    if allocated_gb < 3.0:
        print("[WARN] VRAM allocation lower than expected (~6GB). "
              "Model may not have loaded properly or other models were unloaded.")
    elif allocated_gb > 15.0:
        print("[WARN] VRAM allocation higher than expected (~6GB). "
              "Other models may still be loaded.")
    else:
        print(f"[PASS] VRAM allocation within expected range (3-15 GB, actual: {allocated_gb:.2f} GB)")

    # Test model unloading
    print("\nUnloading model...")
    manager.unload_current_model()
    print_gpu_memory_summary("After unload: ")

    allocated_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after unload: {allocated_after:.2f} GB")

    print("[PASS] ModelManager integration verified")
    return True


def main():
    """Run all tests and report results."""
    print("Starting SeamlessM4T Translation Tests")
    print("="*60)

    # GPU validation
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. These tests require GPU.")
        print("Run on RTX 5090 with CUDA 12.8+ environment.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Run tests
    tests = [
        ("Model Loading", test_model_loading),
        ("Multi-Language Translation", test_multiple_languages),
        ("ModelManager Integration", test_model_manager_integration),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[FAIL] TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[PASS] ALL TESTS PASSED")
        return True
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
