---
plan: 08-02
phase: 08-quality-controls
status: complete
subsystem: ui-quality-controls
tags: [tdd, validators, clip-preview, ffmpeg, quality-controls]

dependency-graph:
  requires:
    - "07-04: LipSyncResult and SyncValidation dataclasses (lip sync stage)"
    - "05-04: TTSResult with failed_segments int and emotion_flagged_count fields"
    - "04-04: TranslationResult with is_valid_duration on TranslatedSegment"
    - "03-03: ASRResult with confidence field on AlignedSegment"
  provides:
    - "validate_asr_output(): (bool, str) for ASR stage QC gate"
    - "validate_translation_output(): (bool, str) for translation stage QC gate"
    - "validate_tts_output(): (bool, str) for TTS stage QC gate"
    - "validate_lip_sync_output(): (bool, str) for lip sync stage QC gate"
    - "extract_preview_clip(): FFmpeg 30s clip extractor for UI preview"
  affects:
    - "08-03: pipeline_runner.py will call all four validators after each stage"
    - "08-04: final QC and export stage uses validators to gate pipeline flow"

tech-stack:
  added:
    - "pytest (test runner, installed in venv)"
  patterns:
    - "TYPE_CHECKING imports: avoids ML library cascade at import time"
    - "Duck typing with getattr() defaults: validators accept any object with matching attrs"
    - "TDD RED-GREEN: 30 failing tests committed before implementation"

file-tracking:
  created:
    - "src/ui/validators.py (165 lines, 4 exported functions)"
    - "src/ui/clip_preview.py (65 lines, 1 exported function)"
    - "tests/test_ui_validators.py (430 lines, 30 tests)"
  modified: []

decisions:
  - id: type-checking-imports
    title: "Use TYPE_CHECKING for stage result type imports in validators.py"
    impact: "Validators importable in test/UI contexts without pyannote, torch, TTS installed"
    status: implemented
  - id: duck-typing-validators
    title: "Validators use getattr() with defaults instead of isinstance() checks"
    impact: "Any object with matching attributes works (SimpleNamespace, real dataclasses)"
    status: implemented
  - id: tts-failed-segments-int
    title: "TTSResult.failed_segments is an int count (not a list)"
    impact: "validate_tts_output() uses getattr(result, 'failed_segments', 0) directly"
    status: confirmed-from-source

metrics:
  duration: "3 minutes"
  completed: "2026-02-21"
  tests-written: 30
  tests-passing: 30
  lines-validators: 165
  lines-clip-preview: 65
  lines-tests: 430
---

# Phase 8 Plan 02: Stage Validators + Clip Preview — Summary

**One-liner:** TYPE_CHECKING-guarded stage validators with duck-typed attribute inspection and FFmpeg subprocess clip extractor, fully TDD-verified with 30 tests and no GPU required.

## What Was Built

### src/ui/validators.py

Four pure functions implementing QC-04 stage gates:

- **`validate_asr_output(result)`**: Checks segment count and confidence ratio. Returns warning when >50% of segments fall below ASR_CONFIDENCE_THRESHOLD (0.7). Returns error when no segments detected.

- **`validate_translation_output(result)`**: Checks `is_valid_duration` on each TranslatedSegment. Fails when >20% are invalid (hard stop), warns when 0–20% are invalid (continue with caution).

- **`validate_tts_output(result)`**: Uses `failed_segments` (int count) and `emotion_flagged_count` from TTSResult. Fails when failed/total > 50%, warns when emotion issues detected.

- **`validate_lip_sync_output(result)`**: Inspects `sync_validation.passed` and `sync_validation.pass_rate`. Detects Wav2Lip fallback separately. Treats `sync_validation=None` as success (advisory design from Phase 7).

### src/ui/clip_preview.py

`extract_preview_clip(video_path, start_seconds, output_path, duration=30.0)`:
- Validates video_path exists (ValueError if not)
- Clamps negative start_seconds to 0
- Runs `ffmpeg -y -ss {start} -i {input} -t {duration} -c copy {output}` (fast stream copy)
- Raises RuntimeError on non-zero ffmpeg exit code
- Returns output_path on success

### tests/test_ui_validators.py

30 tests across 5 test classes. All tests run without GPU hardware using SimpleNamespace mocks:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestValidateASROutput | 6 | none, empty, high-conf, low-conf >50%, exactly-50%, speaker-count |
| TestValidateTranslationOutput | 6 | none, empty, >20% invalid, all valid, partial warning, exactly-20% |
| TestValidateTTSOutput | 6 | none, majority-failed, exactly-50%, all-passed, emotion-flagged, zero-total |
| TestValidateLipSyncOutput | 6 | none, failed-sync, valid, fallback, no-validation, message-content |
| TestExtractPreviewClip | 6 | missing-file, ffmpeg-failure, negative-clamp, success, default-30s, custom-duration |

## TDD Protocol Followed

**RED phase** (commit `9c5d11d`): 30 tests written first, all failing with `ModuleNotFoundError: No module named 'src.ui.validators'`.

**GREEN phase** (commit `545a429`): validators.py and clip_preview.py implemented. All 30 tests pass in 0.04s.

**REFACTOR phase**: No structural changes needed. Logic is already minimal and readable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] TYPE_CHECKING import pattern to avoid ML library cascade**

- **Found during:** Task 2 (GREEN phase) — first test run showed `ModuleNotFoundError: No module named 'pyannote'`
- **Issue:** `from src.stages.asr_stage import ASRResult` triggered `asr_stage.py` → `diarization.py` → `from pyannote.audio import Pipeline`, which is not installed in the test environment. This cascaded through all 24 validator tests.
- **Fix:** Moved stage result type imports inside `if TYPE_CHECKING:` block. Validators use `getattr()` duck typing instead of type-based dispatch — no runtime type checking needed.
- **Files modified:** `src/ui/validators.py`
- **Impact:** Validators now importable in any Python environment without ML stack. Full type safety still available to static analysis tools (mypy, pyright).

## Next Phase Readiness

Plan 08-03 (pipeline_runner.py) can now call all four validators:

```python
from src.ui.validators import (
    validate_asr_output,
    validate_translation_output,
    validate_tts_output,
    validate_lip_sync_output,
)
from src.ui.clip_preview import extract_preview_clip

# After each stage:
ok, msg = validate_asr_output(asr_result)
if not ok:
    raise PipelineStageError(msg)
```

All four validators follow the same `(bool, str)` contract, enabling uniform error handling in the pipeline runner.

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| `9c5d11d` | test | RED: 30 failing tests for validators + clip preview |
| `545a429` | feat | GREEN: validators.py and clip_preview.py implemented, all 30 pass |
