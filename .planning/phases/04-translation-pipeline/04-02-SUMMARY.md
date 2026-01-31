---
phase: 04-translation-pipeline
plan: 02
subsystem: translation
tags: [translation, duration-validation, candidate-ranking, seamless-m4t, timing-constraints]

# Dependency graph
requires:
  - phase: 04-01
    provides: Translation infrastructure with SeamlessM4T v2 integration
provides:
  - Duration validation using character-count heuristics (±10% tolerance)
  - Multi-candidate ranking with weighted scoring (confidence + duration fit)
  - Reusable modules for translation quality optimization
affects: [04-03-translation-stage-integration, 05-voice-cloning, 06-lip-sync]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Character-count based duration estimation (15 chars/sec for English)
    - Linear penalty scoring within tolerance (0.0 outside tolerance)
    - Weighted candidate ranking (60% confidence, 40% duration)

key-files:
  created:
    - src/stages/translation/duration_validator.py
    - src/stages/translation/candidate_ranker.py
    - tests/test_duration_validation.py
    - tests/test_candidate_ranking.py
  modified:
    - src/stages/translation/__init__.py

key-decisions:
  - "Character-count heuristic for duration estimation (15 chars/sec) instead of phoneme analysis or TTS preview"
  - "Linear penalty within tolerance (1.0 - deviation/tolerance) with hard 0.0 penalty outside"
  - "60% confidence / 40% duration default weights (speed-first priority)"
  - "Normalize text to lowercase and collapse whitespace for consistent character counting"

patterns-established:
  - "Duration validation returns detailed dict with is_valid, ratio, estimated_duration, tolerance bounds"
  - "Candidate ranking returns tuple of (best_candidate_dict, all_ranked_list) for flexibility"
  - "Module-level convenience functions alongside classes for one-off usage"

# Metrics
duration: 10min
completed: 2026-01-31
---

# Phase 4 Plan 2: Duration Validation and Candidate Ranking Summary

**Character-count based duration validation (±10% tolerance) and multi-candidate ranking with weighted scoring (60% confidence, 40% duration fit) for translation quality optimization**

## Performance

- **Duration:** 10 min
- **Started:** 2026-01-31T18:07:43Z
- **Completed:** 2026-01-31T18:17:30Z
- **Tasks:** 2 (Task 1 already completed in 04-01)
- **Files modified:** 5

## Accomplishments
- Duration validation module estimates translated text duration using character count heuristics
- Candidate ranking module scores translations by weighted combination of model confidence and duration fit
- Comprehensive test suites validate all functionality (text normalization, estimation accuracy, tolerance checking, weighted scoring, edge cases)
- All tests pass successfully with correct duration calculations and ranking behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement duration validation module** - Already completed in 04-01 (b3ce335)
2. **Task 2: Implement candidate ranking module** - `dc4d344` (feat)
3. **Task 3: Test duration validation and candidate ranking** - `71af3dd` (test)

## Files Created/Modified
- `src/stages/translation/duration_validator.py` - Duration estimation and validation (created in 04-01)
  - `estimate_duration()`: Character-count based duration estimation (15 chars/sec)
  - `validate_duration()`: Check if translation fits within ±10% tolerance
  - `normalize_text_for_duration()`: Text normalization for consistent counting
- `src/stages/translation/candidate_ranker.py` - Multi-candidate ranking with weighted scoring
  - `CandidateRanker` class: Configurable weights for confidence and duration fit
  - `rank_candidates()` method: Returns best candidate and full ranking
  - `_calculate_duration_score()`: Linear penalty within tolerance, 0.0 outside
  - Module-level `rank_candidates()` convenience function
- `src/stages/translation/__init__.py` - Export duration validation and ranking functions
- `tests/test_duration_validation.py` - 4 test suites for duration validation
  - Text normalization, duration estimation, tolerance validation, edge cases
- `tests/test_candidate_ranking.py` - 5 test suites for candidate ranking
  - Best candidate selection, duration fit scoring, weighted scoring, edge cases, convenience function

## Decisions Made

**Duration estimation method:**
- Chose character-count heuristic (15 chars/sec for English) over phoneme analysis or TTS preview
- Rationale: Speed-first priority - character count is instant vs linguistic analysis overhead
- Normalized text to lowercase with collapsed whitespace for consistent counting
- Excludes spaces from character count (spaces don't contribute to speech duration)

**Duration scoring:**
- Linear penalty within tolerance: `score = 1.0 - abs(1.0 - ratio) / tolerance`
  - Perfect fit (ratio 1.0) → score 1.0
  - 5% deviation → score 0.5
  - 10% deviation (at boundary) → score 0.0
- Hard penalty outside tolerance: score = 0.0
- Rationale: Gradual degradation within acceptable range, clear cutoff for unacceptable translations

**Default weights:**
- 60% confidence, 40% duration fit
- Rationale: Speed-first priority favors model confidence, but duration fit ensures downstream stages (voice cloning, lip sync) can process the translation
- Weights configurable for different quality/speed tradeoffs

**API design:**
- Return detailed dict from `validate_duration()` with all metadata (ratio, tolerance bounds, estimated duration)
- Return tuple (best_candidate, all_ranked) from `rank_candidates()` for flexibility
- Provide module-level convenience functions alongside classes for one-off usage

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Transformers library not installed:**
- Plan 04-01 created translator.py with transformers import, but library not installed yet
- Tests couldn't import translation modules due to __init__.py triggering translator import
- Solution: Used importlib to directly load modules and modified test imports to avoid __init__.py
- Tests now use direct module loading with sys.modules setup for relative imports
- No changes needed to source code - issue was test-only

**Unicode output in tests:**
- Initial tests used checkmark/X unicode symbols that failed on Windows cp1252 encoding
- Solution: Replaced with [OK]/[FAIL] ASCII markers
- All tests now pass successfully on Windows

## Next Phase Readiness

**Ready for Phase 4 Plan 3 (Translation Stage Integration):**
- Duration validation can verify translations fit timing constraints
- Candidate ranking can select best translation from multiple model outputs
- Test infrastructure validates correctness

**Ready for downstream phases:**
- Phase 5 (Voice Cloning): Validated translations will fit original segment duration (±10%)
- Phase 6 (Lip Sync): Duration-constrained translations enable accurate lip synchronization
- Quality optimization: Multi-candidate approach ensures better translations than single-pass

**No blockers or concerns.**

---
*Phase: 04-translation-pipeline*
*Completed: 2026-01-31*
