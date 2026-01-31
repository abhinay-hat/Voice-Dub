"""
Tests for candidate ranking logic in translation pipeline.

Validates multi-candidate ranking with weighted scoring of confidence and duration fit.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from files to avoid __init__.py's translator import
import importlib.util

# Load settings first
spec_settings = importlib.util.spec_from_file_location(
    "src.config.settings",
    project_root / "src" / "config" / "settings.py"
)
settings = importlib.util.module_from_spec(spec_settings)
sys.modules["src.config.settings"] = settings
spec_settings.loader.exec_module(settings)

# Load duration_validator
spec_dv = importlib.util.spec_from_file_location(
    "src.stages.translation.duration_validator",
    project_root / "src" / "stages" / "translation" / "duration_validator.py"
)
duration_validator = importlib.util.module_from_spec(spec_dv)
sys.modules["src.stages.translation.duration_validator"] = duration_validator
spec_dv.loader.exec_module(duration_validator)

# Create a mock translation package module
import types
translation_pkg = types.ModuleType("src.stages.translation")
translation_pkg.duration_validator = duration_validator
sys.modules["src.stages.translation"] = translation_pkg

# Load candidate_ranker (with relative imports working now)
with open(project_root / "src" / "stages" / "translation" / "candidate_ranker.py") as f:
    code = f.read()
    # Replace relative import with absolute import
    code = code.replace("from .duration_validator import estimate_duration",
                       "from src.stages.translation.duration_validator import estimate_duration")
    code = code.replace("from src.config.settings import",
                       "from src.config.settings import")

# Execute the modified code
spec_cr = importlib.util.spec_from_file_location(
    "candidate_ranker",
    project_root / "src" / "stages" / "translation" / "candidate_ranker.py"
)
candidate_ranker = importlib.util.module_from_spec(spec_cr)
sys.modules["candidate_ranker"] = candidate_ranker

# Execute with modified code
exec(code, candidate_ranker.__dict__)

CandidateRanker = candidate_ranker.CandidateRanker
rank_candidates = candidate_ranker.rank_candidates


def test_single_best_candidate():
    """Test selection of best candidate from multiple options."""
    print("\n=== Test 1: Single Best Candidate Selection ===")

    # Create mock candidates with different confidence + duration profiles
    # Candidate A: high confidence (0.9), poor duration (1.5x = ratio 1.5, outside tolerance)
    # Candidate B: medium confidence (0.75), perfect duration (ratio 1.0)
    # Candidate C: low confidence (0.6), good duration (ratio 1.05)

    original_duration = 2.0  # 2 seconds original
    # At 15 chars/sec:
    # 2.0s = 30 chars (perfect fit)
    # 3.0s = 45 chars (1.5x, way over)
    # 2.1s = 31-32 chars (1.05x, within tolerance)

    candidates = [
        "A" * 45,  # Candidate A: 45 chars = 3.0s (ratio 1.5, outside tolerance)
        "B" * 30,  # Candidate B: 30 chars = 2.0s (ratio 1.0, perfect)
        "C" * 31,  # Candidate C: 31 chars = 2.07s (ratio 1.03, good fit)
    ]
    scores = [0.9, 0.75, 0.6]  # Confidence scores

    ranker = CandidateRanker(confidence_weight=0.6, duration_weight=0.4)
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)

    # Candidate B should win (balanced score)
    # A: 0.9 * 0.6 + 0.0 * 0.4 = 0.54 (duration score = 0.0, outside tolerance)
    # B: 0.75 * 0.6 + 1.0 * 0.4 = 0.45 + 0.4 = 0.85 (perfect duration)
    # C: 0.6 * 0.6 + ~0.7 * 0.4 = 0.36 + 0.28 = 0.64 (good duration)

    assert best is not None, "Best candidate should not be None"
    assert best["candidate"] == candidates[1], f"Expected candidate B, got {best['candidate'][:10]}..."
    print(f"[OK] Best candidate: B (confidence={best['model_confidence']}, duration_score={best['duration_score']:.3f}, combined={best['combined_score']:.3f})")

    # Verify all candidates are ranked
    assert len(all_ranked) == 3, f"Expected 3 ranked candidates, got {len(all_ranked)}"
    print(f"[OK] All candidates ranked: {len(all_ranked)} candidates")

    # Verify ranking order (B > C > A)
    assert all_ranked[0]["candidate"] == candidates[1], "Candidate B should be ranked 1st"
    assert all_ranked[1]["candidate"] == candidates[2], "Candidate C should be ranked 2nd"
    assert all_ranked[2]["candidate"] == candidates[0], "Candidate A should be ranked 3rd"
    print(f"[OK] Ranking order correct: B > C > A")

    print("[OK] Single best candidate tests passed\n")


def test_duration_fit_scoring():
    """Test duration fit scoring logic."""
    print("=== Test 2: Duration Fit Scoring ===")

    ranker = CandidateRanker()
    original_duration = 2.0

    # Test case 1: Perfect fit (ratio 1.0) -> duration_score = 1.0
    text = "A" * 30  # 30 chars / 15 cps = 2.0s (ratio 1.0)
    score = ranker._calculate_duration_score(text, original_duration)
    assert abs(score - 1.0) < 0.01, f"Perfect fit should score 1.0, got {score:.3f}"
    print(f"[OK] Perfect fit (ratio 1.0): score={score:.3f}")

    # Test case 2: 5% over (ratio 1.05) -> duration_score = 0.5
    # Formula: 1.0 - abs(1.0 - 1.05) / 0.1 = 1.0 - 0.5 = 0.5
    text = "A" * 31  # 31 chars / 15 cps = 2.07s (ratio 1.03, ~5% over)
    score = ranker._calculate_duration_score(text, original_duration)
    expected = 1.0 - abs(1.0 - 2.07 / 2.0) / 0.1
    assert abs(score - expected) < 0.1, f"5% over should score ~{expected:.3f}, got {score:.3f}"
    print(f"[OK] 5% over (ratio 1.03): score={score:.3f}")

    # Test case 3: 10% over (ratio 1.1, at tolerance boundary) -> duration_score = 0.0
    text = "A" * 33  # 33 chars / 15 cps = 2.2s (ratio 1.1)
    score = ranker._calculate_duration_score(text, original_duration)
    assert abs(score - 0.0) < 0.01, f"10% over should score 0.0, got {score:.3f}"
    print(f"[OK] 10% over (ratio 1.1): score={score:.3f}")

    # Test case 4: 15% over (ratio 1.15, outside tolerance) -> duration_score = 0.0
    text = "A" * 34  # 34 chars / 15 cps = 2.27s (ratio 1.13)
    score = ranker._calculate_duration_score(text, original_duration)
    assert abs(score - 0.0) < 0.01, f"15% over should score 0.0, got {score:.3f}"
    print(f"[OK] 15% over (ratio 1.13): score={score:.3f}")

    # Test case 5: 5% under (ratio 0.95) -> similar to 5% over
    text = "A" * 28  # 28 chars / 15 cps = 1.87s (ratio 0.93)
    score = ranker._calculate_duration_score(text, original_duration)
    expected = 1.0 - abs(1.0 - 1.87 / 2.0) / 0.1
    assert abs(score - expected) < 0.1, f"5% under should score ~{expected:.3f}, got {score:.3f}"
    print(f"[OK] 5% under (ratio 0.93): score={score:.3f}")

    print("[OK] Duration fit scoring tests passed\n")


def test_weighted_scoring():
    """Test weighted scoring with different weight configurations."""
    print("=== Test 3: Weighted Scoring ===")

    original_duration = 2.0

    # Candidates:
    # High confidence, poor duration (A)
    # Low confidence, perfect duration (B)
    candidates = [
        "A" * 45,  # 45 chars = 3.0s (ratio 1.5, outside tolerance)
        "B" * 30,  # 30 chars = 2.0s (ratio 1.0, perfect)
    ]

    # Test case 1: Default weights (60% confidence, 40% duration)
    scores = [0.9, 0.5]  # A has high confidence, B has low
    ranker = CandidateRanker(confidence_weight=0.6, duration_weight=0.4)
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)

    # A: 0.9 * 0.6 + 0.0 * 0.4 = 0.54
    # B: 0.5 * 0.6 + 1.0 * 0.4 = 0.3 + 0.4 = 0.7
    # B should win
    assert best["candidate"] == candidates[1], "With default weights, B should win"
    print(f"[OK] Default weights (0.6, 0.4): B wins (score={best['combined_score']:.3f})")

    # Test case 2: Favor confidence more (80% confidence, 20% duration)
    ranker = CandidateRanker(confidence_weight=0.8, duration_weight=0.2)
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)

    # A: 0.9 * 0.8 + 0.0 * 0.2 = 0.72
    # B: 0.5 * 0.8 + 1.0 * 0.2 = 0.4 + 0.2 = 0.6
    # A should win now
    assert best["candidate"] == candidates[0], "With 0.8 confidence weight, A should win"
    print(f"[OK] Favor confidence (0.8, 0.2): A wins (score={best['combined_score']:.3f})")

    # Test case 3: Favor duration more (40% confidence, 60% duration)
    ranker = CandidateRanker(confidence_weight=0.4, duration_weight=0.6)
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)

    # A: 0.9 * 0.4 + 0.0 * 0.6 = 0.36
    # B: 0.5 * 0.4 + 1.0 * 0.6 = 0.2 + 0.6 = 0.8
    # B should win strongly
    assert best["candidate"] == candidates[1], "With 0.6 duration weight, B should win"
    print(f"[OK] Favor duration (0.4, 0.6): B wins (score={best['combined_score']:.3f})")

    print("[OK] Weighted scoring tests passed\n")


def test_edge_cases():
    """Test edge cases in candidate ranking."""
    print("=== Test 4: Edge Cases ===")

    ranker = CandidateRanker()
    original_duration = 2.0

    # Test case 1: Empty candidates list
    best, all_ranked = ranker.rank_candidates([], [], original_duration)
    assert best is None, "Empty candidates should return None"
    assert all_ranked == [], "Empty candidates should return empty list"
    print(f"[OK] Empty candidates: best={best}, ranked={all_ranked}")

    # Test case 2: Single candidate (should return as best)
    candidates = ["Hello"]
    scores = [0.8]
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)
    assert best is not None, "Single candidate should not be None"
    assert best["candidate"] == "Hello", "Single candidate should be returned as best"
    assert len(all_ranked) == 1, "Single candidate should have 1 ranked item"
    print(f"[OK] Single candidate: returned as best")

    # Test case 3: All candidates outside tolerance (best is least-bad)
    candidates = [
        "A" * 50,  # 50 chars = 3.33s (ratio 1.67, way over)
        "B" * 45,  # 45 chars = 3.0s (ratio 1.5, way over)
        "C" * 40,  # 40 chars = 2.67s (ratio 1.33, way over)
    ]
    scores = [0.9, 0.85, 0.8]
    best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)

    # All have duration_score = 0.0, so ranking is purely by confidence
    # A should win (highest confidence)
    assert best["candidate"] == candidates[0], "Highest confidence should win when all invalid"
    assert best["duration_score"] == 0.0, "Duration score should be 0.0 for invalid"
    print(f"[OK] All candidates invalid: highest confidence wins")

    # Test case 4: Mismatched candidates and scores length
    try:
        candidates = ["A", "B", "C"]
        scores = [0.9, 0.8]  # Only 2 scores for 3 candidates
        best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration)
        assert False, "Should raise ValueError for length mismatch"
    except ValueError as e:
        print(f"[OK] Length mismatch raises ValueError: {e}")

    # Test case 5: Invalid weights (don't sum to 1.0)
    try:
        ranker = CandidateRanker(confidence_weight=0.7, duration_weight=0.4)
        assert False, "Should raise ValueError for invalid weights"
    except ValueError as e:
        print(f"[OK] Invalid weights raise ValueError: {e}")

    print("[OK] Edge case tests passed\n")


def test_convenience_function():
    """Test convenience rank_candidates() function."""
    print("=== Test 5: Convenience Function ===")

    candidates = ["Hello", "Hi there", "Greetings"]
    scores = [0.9, 0.85, 0.75]
    original_duration = 1.5

    # Test default weights
    best, all_ranked = rank_candidates(candidates, scores, original_duration)
    assert best is not None, "Convenience function should return best candidate"
    assert len(all_ranked) == 3, "Convenience function should return all ranked"
    print(f"[OK] Convenience function with default weights: best='{best['candidate']}'")

    # Test custom weights
    best, all_ranked = rank_candidates(
        candidates, scores, original_duration,
        confidence_weight=0.8, duration_weight=0.2
    )
    assert best is not None, "Convenience function should work with custom weights"
    print(f"[OK] Convenience function with custom weights: best='{best['candidate']}'")

    print("[OK] Convenience function tests passed\n")


def run_all_tests():
    """Run all candidate ranking tests."""
    print("\n" + "=" * 60)
    print("CANDIDATE RANKING TEST SUITE")
    print("=" * 60)

    try:
        test_single_best_candidate()
        test_duration_fit_scoring()
        test_weighted_scoring()
        test_edge_cases()
        test_convenience_function()

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
