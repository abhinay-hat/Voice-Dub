---
phase: 05-voice-cloning-and-tts
plan: 04
subsystem: tts-stage-integration
tags: [tts, stage-orchestration, xtts, voice-cloning, quality-validation, emotion-preservation]

requires:
  phases: [05-01, 05-02, 05-03]
  components: [reference-extractor, speaker-embeddings, xtts-generator, quality-validator]
provides:
  - Complete TTS stage orchestration with run_tts_stage() function
  - TTSResult and SynthesizedSegment dataclasses for pipeline output
  - Integration of all TTS components into single stage module
  - Quality validation with explicit failure handling (>50% rejection)
  - Emotion preservation tracking via pitch_variance_ratio
  - JSON segment manifest export
  - Try-finally cleanup for model unloading
affects:
  phases: [06-lip-sync, 10-pipeline-integration]
  downstream: [assembly-export, quality-review-ui]

tech-stack:
  added: []
  patterns:
    - "Stage orchestration pattern (matching translation_stage.py)"
    - "Progress callbacks at 10 checkpoints (0.05 to 1.0)"
    - "Quality validation with explicit failure handling"
    - "Emotion preservation tracking separate from quality flags"
    - "Speaker-grouped batch processing for VRAM efficiency"
    - "Try-finally cleanup pattern for GPU resource release"

key-files:
  created:
    - src/stages/tts_stage.py: "Complete TTS stage orchestration with quality validation"
    - tests/test_tts_stage.py: "13 integration tests for TTS pipeline"
  modified:
    - README.md: "Phase 5 documentation with emotion preservation details"

decisions:
  - id: stage-orchestration-pattern
    title: Mirror translation_stage.py orchestration pattern
    impact: Consistent API and flow across pipeline stages for maintainability
    status: implemented

  - id: explicit-quality-failure-handling
    title: Explicit quality validation with TTSStageFailed exception
    impact: Stage fails if >50% segments rejected, prevents bad audio from proceeding to lip sync
    status: implemented

  - id: emotion-preservation-tracking
    title: Track emotion preservation separately from quality_passed
    impact: Enables UI to show emotion flags without failing stage, gives user visibility
    status: implemented

  - id: generate-speaker-embeddings-helper
    title: Use generate_speaker_embeddings() helper instead of manual cache population
    impact: Cleaner API, handles XTTS model loading correctly
    status: implemented

metrics:
  duration: 7.5 minutes
  completed: 2026-02-02
---

# Phase 5 Plan 4: TTS Stage Integration Summary

Complete TTS stage orchestration with reference extraction, speaker embedding generation, XTTS synthesis, quality validation, emotion preservation tracking, and JSON export.

## Objective Achieved

Implemented complete TTS stage orchestration (`run_tts_stage()`) that integrates all Phase 5 components into a single pipeline function. The stage transforms translation JSON + original audio into cloned English speech with duration matching, quality validation, and emotion preservation checks. Follows the established stage pattern from ASR and translation for consistency.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement TTS stage orchestration | 77db54d + 442f87f | src/stages/tts_stage.py |
| 2 | Create integration test suite | 8e171e2 | tests/test_tts_stage.py |
| 3 | Update README with Phase 5 documentation | 1918f43 | README.md |

## What Was Built

### 1. TTS Stage Orchestration Module (Task 1)

Created `src/stages/tts_stage.py` with complete pipeline orchestration:

**Dataclasses:**
- `SynthesizedSegment`: Single synthesized segment with metadata
  - Original timing: segment_id, speaker, start, end, original_duration, translated_text
  - Synthesis results: audio_path, actual_duration, duration_error, speed_used, synthesis_attempts
  - Quality metrics: quality_passed, flagged_for_review, rejection_reason
  - Emotion preservation: emotion_preserved (bool), pitch_variance_ratio (float)

- `TTSResult`: Complete stage output
  - Counts: total_segments, successful_segments, failed_segments, flagged_count, emotion_flagged_count
  - Segment list: segments (List[SynthesizedSegment])
  - Metadata: video_id, avg_duration_error, processing_time, output_dir
  - Serialization: to_dict() for JSON export

**run_tts_stage() Pipeline:**
1. (0.05) Load translation JSON
2. (0.10) Create output directory
3. (0.15) Extract reference samples per speaker (extract_reference_samples)
4. (0.25) Generate speaker embeddings (generate_speaker_embeddings via XTTS model)
5. (0.35-0.80) Synthesize all segments with duration matching
   - Initialize XTTSGenerator with model_manager and embedding_cache
   - Group segments by speaker for efficiency
   - Report per-segment progress via callback
6. (0.85) Validate audio quality with emotion preservation check:
   ```python
   validator = QualityValidator()
   quality_results, quality_summary = validator.validate_batch(
       synthesis_results=validation_data,
       reference_dir=output_path  # Use speaker references for emotion comparison
   )

   # Process validation results - update segment metadata
   for synth_result, quality in zip(synthesis_results, quality_results):
       synth_result['quality_passed'] = quality.passes_quality
       synth_result['flagged_for_review'] = quality.flagged_for_review
       synth_result['rejection_reason'] = quality.rejection_reason
       synth_result['emotion_preserved'] = (
           quality.pitch_variance_ratio is not None and
           0.6 <= quality.pitch_variance_ratio <= 1.5
       )
       synth_result['pitch_variance_ratio'] = quality.pitch_variance_ratio

   # Fail stage if >50% segments rejected
   if rejected_count > len(segments) * 0.5:
       raise TTSStageFailed(f"Quality validation failed: {rejected_count}/{len(segments)} segments rejected")
   ```
7. (0.90) Build result structure (aggregate metrics, flagged segments)
8. (0.95) Export result JSON (tts_result.json in output_dir)
9. (1.0) Cleanup (model_manager.unload_current_model() in finally block)

**Error Handling:**
- If reference extraction fails for speaker: log warning, raise TTSStageFailed
- If synthesis fails for segment: log error, mark as failed, continue
- If >50% segments fail quality validation: raise TTSStageFailed exception
- If quality validation flags segment: mark flagged_for_review=True, continue processing

**Output Directory Structure:**
```
output_dir/
  reference_SPEAKER_00.wav  # Reference samples
  reference_SPEAKER_01.wav
  segment_0.wav             # Synthesized segments
  segment_1.wav
  ...
  tts_result.json           # Result manifest with quality flags
```

**Bug Fix (442f87f):**
- Corrected speaker embedding generation to use `generate_speaker_embeddings()` helper
- Loads XTTS model once via ModelManager, reused for embeddings and synthesis
- SpeakerEmbeddingCache created empty, populated via helper function (not non-existent add_speaker method)

### 2. Integration Test Suite (Task 2)

Created `tests/test_tts_stage.py` with 13 comprehensive tests:

1. **test_tts_stage_imports**: Verify all TTS modules import correctly
2. **test_dataclass_structure**: Verify SynthesizedSegment and TTSResult fields (including emotion fields)
3. **test_reference_extractor_logic**: Test segment selection with mock data
   - Single long segment
   - Multiple segments (select longest)
   - Segment too long (needs centering)
   - Concatenation fallback
   - No viable segment
4. **test_speaker_embedding_cache**: Test cache operations (get/put/has)
5. **test_duration_matching_logic**: Test speed adjustment binary search
6. **test_quality_validator_thresholds**: Test PESQ tier classification
7. **test_emotion_preservation_validation**: Test pitch variance ratio calculation and thresholds
   - Ratio 0.6-1.5: Emotion preserved
   - Ratio <0.6 or >1.5: Emotion lost/exaggerated (flagged)
8. **test_batch_synthesis_grouping**: Test speaker grouping optimization
9. **test_short_text_handling**: Test < 3s target flagging
10. **test_json_output_format**: Test result JSON structure matches schema with emotion fields
11. **test_progress_callback_points**: Test progress callback fired at expected points
12. **test_quality_failure_handling**: Test that >50% rejection raises TTSStageFailed
13. **test_emotion_flag_count**: Test emotion_flagged_count is correctly calculated

**Graceful Dependency Handling:**
```python
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

@pytest.mark.skipif(not TTS_AVAILABLE, reason="TTS library not installed")
def test_xtts_model_loading():
    # Tests that require actual XTTS model
    pass
```

**All 13 tests pass** without requiring GPU or actual TTS models.

### 3. README Documentation (Task 3)

Updated `README.md` with comprehensive Phase 5 documentation:

**Phase 5 Section:**
- Overview: Voice cloning with XTTS-v2, reference extraction, emotion preservation
- Components: Reference Extractor, Speaker Embeddings, XTTS Generator, Quality Validator
- Emotion Preservation explanation:
  - Pitch variance ratio 0.6-1.5: Emotion preserved (acceptable range)
  - Ratio <0.6 or >1.5: Emotion lost/exaggerated - flagged for review
- Usage example with progress callbacks
- Pipeline flow (9 steps with progress percentages)
- Configuration settings from src/config/settings.py
- Testing instructions
- Key decisions and requirements (including non-commercial license note)

**Updated AI Models Table:**
| Model | Purpose | VRAM | Speed | Status |
|-------|---------|------|-------|--------|
| **XTTS-v2** | Voice cloning + TTS | ~2GB | ~0.5-1s/segment | ✅ Integrated |
| - Model: coqui/XTTS-v2 | Emotion preservation, duration matching | Non-commercial | Speed adjust 0.8-1.2x | Phase 5 |

**Updated Development Progress Table:**
| Phase | Plans | Status | Completed |
|-------|-------|--------|-----------|
| 5. Voice Cloning & TTS | 4/4 | Complete | 2026-02-02 |

**Current Status**: Phase 5 Complete - Voice Cloning & TTS with emotion preservation ready
**Next**: Phase 6 - Lip Synchronization

## Accomplishments

1. **Complete TTS stage orchestration** following established stage pattern from ASR and translation
2. **Quality validation with explicit failure handling** - stage fails if >50% segments rejected
3. **Emotion preservation tracking** separate from quality flags for user visibility
4. **Comprehensive test suite** validating all logic without requiring GPU/models
5. **Phase 5 documentation complete** in README with usage examples and emotion explanation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected speaker embedding generation API**
- **Found during:** Task 1 (TTS stage implementation)
- **Issue:** Attempted to use non-existent `SpeakerEmbeddingCache.add_speaker()` method
- **Fix:** Use `generate_speaker_embeddings()` helper function from src/tts/speaker_embeddings.py
  - Load XTTS model once via ModelManager
  - Pass model and reference_paths to helper
  - Returns populated SpeakerEmbeddingCache
- **Files modified:** src/stages/tts_stage.py
- **Verification:** Test imports pass, API matches actual implementation
- **Committed in:** 442f87f (bug fix commit)

---

**Total deviations:** 1 auto-fixed (1 bug - incorrect API usage)
**Impact on plan:** Necessary correction to match actual speaker_embeddings.py API. No scope change.

## Issues Encountered

None - plan executed smoothly after API correction.

## Next Phase Readiness

**Phase 5 complete** - all TTS components integrated and tested:
- ✅ Reference extraction with RMS-based selection
- ✅ Speaker embedding generation and caching
- ✅ XTTS synthesis with duration matching (±5% tolerance)
- ✅ Quality validation with PESQ, STOI, emotion preservation
- ✅ Stage orchestration with progress callbacks
- ✅ Comprehensive test coverage (13 integration tests)
- ✅ README documentation with usage examples

**Ready for Phase 6: Lip Synchronization**
- TTS stage produces individual segment WAV files with timing metadata
- Quality validation ensures audio meets minimum standards
- Emotion preservation tracking helps identify segments needing review
- Duration matching (±5% tolerance) provides lip sync compatibility

**No blockers for Phase 6.**

---
*Phase: 05-voice-cloning-and-tts*
*Completed: 2026-02-02*
