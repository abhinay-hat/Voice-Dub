"""
Tests for duration validation logic in translation pipeline.

Validates duration estimation accuracy, tolerance checking, and edge case handling.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from file to avoid __init__.py's translator import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "duration_validator",
    project_root / "src" / "stages" / "translation" / "duration_validator.py"
)
duration_validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duration_validator)

estimate_duration = duration_validator.estimate_duration
validate_duration = duration_validator.validate_duration
normalize_text_for_duration = duration_validator.normalize_text_for_duration


def test_normalize_text():
    """Test text normalization for duration estimation."""
    print("\n=== Test 1: Text Normalization ===")

    # Test case 1: Multiple spaces
    text = "  Hello   World  "
    normalized = normalize_text_for_duration(text)
    assert normalized == "hello world", f"Expected 'hello world', got '{normalized}'"
    print(f"[OK] Multiple spaces: '{text}' -> '{normalized}'")

    # Test case 2: Mixed case
    text = "Hello WORLD"
    normalized = normalize_text_for_duration(text)
    assert normalized == "hello world", f"Expected 'hello world', got '{normalized}'"
    print(f"[OK] Mixed case: '{text}' -> '{normalized}'")

    # Test case 3: Empty text
    text = ""
    normalized = normalize_text_for_duration(text)
    assert normalized == "", f"Expected '', got '{normalized}'"
    print(f"[OK] Empty text: '{text}' -> '{normalized}'")

    print("[OK] Text normalization tests passed\n")


def test_duration_estimation():
    """Test duration estimation accuracy."""
    print("=== Test 2: Duration Estimation ===")

    # Test case 1: Known text sample
    # "Hello,howareyoutoday?" = 21 chars (no spaces) at 15 chars/sec = 1.40 seconds
    text = "Hello, how are you today?"
    expected = 21 / 15  # 1.40
    duration = estimate_duration(text)
    assert abs(duration - expected) < 0.01, f"Expected {expected:.2f}, got {duration:.2f}"
    print(f"[OK] '{text}' -> {duration:.2f}s (expected {expected:.2f}s)")

    # Test case 2: Short text
    text = "Hi"
    expected = 2 / 15  # 0.133...
    duration = estimate_duration(text)
    assert abs(duration - expected) < 0.01, f"Expected {expected:.2f}, got {duration:.2f}"
    print(f"[OK] '{text}' -> {duration:.2f}s (expected {expected:.2f}s)")

    # Test case 3: Long text
    text = "This is a longer sentence that should take more time to speak out loud."
    # Remove spaces and count: 61 chars / 15 = 4.07 seconds
    char_count = len(text.replace(" ", "").lower())
    expected = char_count / 15
    duration = estimate_duration(text)
    assert abs(duration - expected) < 0.01, f"Expected {expected:.2f}, got {duration:.2f}"
    print(f"[OK] Long text ({char_count} chars) -> {duration:.2f}s (expected {expected:.2f}s)")

    # Test case 4: Empty text
    text = ""
    expected = 0.0
    duration = estimate_duration(text)
    assert duration == expected, f"Expected {expected}, got {duration}"
    print(f"[OK] Empty text -> {duration}s")

    # Test case 5: Custom chars_per_second
    text = "Hello"
    chars_per_second = 10  # Slower speech
    expected = 5 / 10  # 0.5 seconds
    duration = estimate_duration(text, chars_per_second=chars_per_second)
    assert abs(duration - expected) < 0.01, f"Expected {expected:.2f}, got {duration:.2f}"
    print(f"[OK] Custom rate (10 cps): '{text}' -> {duration:.2f}s")

    print("[OK] Duration estimation tests passed\n")


def test_duration_validation():
    """Test duration validation with tolerance."""
    print("=== Test 3: Duration Validation with Tolerance ===")

    # Test case 1: Perfect fit (ratio = 1.0)
    # "Hello" = 5 chars / 15 cps = 0.333s
    original_duration = 0.333
    translated_text = "Hello"
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert result["is_valid"], "Perfect fit should be valid"
    assert abs(result["ratio"] - 1.0) < 0.01, f"Ratio should be ~1.0, got {result['ratio']}"
    print(f"[OK] Perfect fit: ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    # Test case 2: Within tolerance (5% over)
    # Original: 5.0s, tolerance ±10% -> valid range 4.5-5.5s
    # "HelloHowareyou?" = 15 chars / 15 cps = 1.0s (fits in 5.0s baseline for testing)
    # Let's use a longer text that's 5% over
    original_duration = 5.0
    # Need 5.25s duration -> 5.25 * 15 = 78.75 chars
    # "HelloHowareyoutodayI'mdoingwellthanksforthisconversationandI'mhappytobehere"
    # Simpler: just verify tolerance math works
    translated_text = "A" * int(5.25 * 15)  # 78 chars = 5.2s
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert result["is_valid"], f"5% over should be valid (ratio={result['ratio']:.3f})"
    print(f"[OK] Within tolerance (5% over): ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    # Test case 3: Outside tolerance (15% over)
    translated_text = "A" * int(5.75 * 15)  # 86 chars = 5.73s (15% over)
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert not result["is_valid"], f"15% over should be invalid (ratio={result['ratio']:.3f})"
    print(f"[OK] Outside tolerance (15% over): ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    # Test case 4: Within tolerance (5% under)
    translated_text = "A" * int(4.75 * 15)  # 71 chars = 4.73s (5% under)
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert result["is_valid"], f"5% under should be valid (ratio={result['ratio']:.3f})"
    print(f"[OK] Within tolerance (5% under): ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    # Test case 5: Outside tolerance (15% under)
    translated_text = "A" * int(4.25 * 15)  # 63 chars = 4.2s (15% under)
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert not result["is_valid"], f"15% under should be invalid (ratio={result['ratio']:.3f})"
    print(f"[OK] Outside tolerance (15% under): ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    print("[OK] Duration validation tests passed\n")


def test_edge_cases():
    """Test edge cases in duration validation."""
    print("=== Test 4: Edge Cases ===")

    # Test case 1: Empty text
    result = validate_duration(5.0, "", tolerance=0.1)
    assert not result["is_valid"], "Empty text should be invalid"
    assert result["estimated_duration"] == 0.0, "Empty text duration should be 0.0"
    print(f"[OK] Empty text: duration={result['estimated_duration']}, valid={result['is_valid']}")

    # Test case 2: Zero original duration
    result = validate_duration(0.0, "Hello", tolerance=0.1)
    assert not result["is_valid"], "Zero original duration should be invalid"
    assert result["ratio"] == 0.0, "Ratio should be 0.0 for zero original"
    print(f"[OK] Zero original duration: ratio={result['ratio']}, valid={result['is_valid']}")

    # Test case 3: Negative original duration
    result = validate_duration(-1.0, "Hello", tolerance=0.1)
    assert not result["is_valid"], "Negative original duration should be invalid"
    print(f"[OK] Negative original duration: valid={result['is_valid']}")

    # Test case 4: Very short text (< 1 second)
    original_duration = 0.5
    translated_text = "Hi"  # 2 chars / 15 cps = 0.133s
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    # This should be invalid (0.133 / 0.5 = 0.266, which is < 0.9)
    assert not result["is_valid"], f"Very short text should be invalid (ratio={result['ratio']:.3f})"
    print(f"[OK] Very short text: ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    # Test case 5: Very long text (> 30 seconds)
    original_duration = 30.0
    # 30 * 15 = 450 chars for perfect fit
    translated_text = "A" * 450
    result = validate_duration(original_duration, translated_text, tolerance=0.1)
    assert result["is_valid"], "Perfect fit should be valid regardless of duration"
    print(f"[OK] Very long text (30s): ratio={result['ratio']:.3f}, valid={result['is_valid']}")

    print("[OK] Edge case tests passed\n")


def run_all_tests():
    """Run all duration validation tests."""
    print("\n" + "=" * 60)
    print("DURATION VALIDATION TEST SUITE")
    print("=" * 60)

    try:
        test_normalize_text()
        test_duration_estimation()
        test_duration_validation()
        test_edge_cases()

        print("=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60 + "\n")
        return True

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        return False
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
