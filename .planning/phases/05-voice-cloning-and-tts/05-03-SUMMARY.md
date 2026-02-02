---
phase: 05-voice-cloning-and-tts
plan: 03
subsystem: tts-quality-validation
tags: [tts, audio-quality, pesq, stoi, emotion-preservation]

requires:
  phases: [05-01, 05-02]
  components: [reference-extractor, speaker-embeddings, xtts-generator]
provides:
  - Audio quality validation with PESQ and STOI metrics
  - Emotion preservation validation via pitch variance ratio
  - Duration accuracy checking for lip sync compatibility
affects:
  phases: [05-04]
  downstream: [tts-stage-integration]

tech-stack:
  added:
    - pesq: "ITU-T P.862 perceptual speech quality (MOS 1-5 scale)"
    - pystoi: "Short-Time Objective Intelligibility (0-1 scale)"
  patterns:
    - "Emotion preservation proxy via pitch variance ratio"
    - "PESQ quality tiers: excellent (4.0+), good (3.0-4.0), fair (2.5-3.0), poor (<2.5)"
    - "Duration-only validation fallback when reference unavailable"

key-files:
  created:
    - src/tts/quality_validator.py: "Audio quality validation with PESQ/STOI/emotion metrics"
  modified:
    - src/tts/__init__.py: "Export AudioQualityResult, QualityValidator, validate_audio_quality"
    - requirements.txt: "Add pesq, pystoi dependencies"

decisions:
  - id: pesq-stoi-combination
    title: Use PESQ + STOI for comprehensive quality assessment
    impact: Combines perceptual quality (PESQ) with intelligibility (STOI)
    status: implemented

  - id: pitch-variance-emotion-proxy
    title: Use pitch variance ratio as emotion preservation proxy
    impact: Detects flat/monotone (ratio < 0.6) or exaggerated (ratio > 1.5) output
    status: implemented

  - id: duration-only-fallback
    title: Support duration-only validation without reference
    impact: Enables quality checking when reference audio unavailable
    status: implemented

  - id: pesq-quality-tiers
    title: Four-tier PESQ classification
    impact: Maps MOS scores to human-readable categories for UI/logging
    status: implemented

metrics:
  duration: 4 minutes
  completed: 2026-02-02
---

# Phase 5 Plan 3: Audio Quality Validation Summary

Audio quality validator with PESQ scoring, STOI intelligibility, emotion preservation checking, and duration accuracy validation.

## Objective Achieved

Implemented comprehensive audio quality validation module using industry-standard PESQ (ITU-T P.862) and STOI metrics to automatically flag low-quality TTS segments before lip sync. Includes emotion preservation validation via pitch variance ratio comparison.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add quality validation dependencies | f150a1d | requirements.txt |
| 2 | Create quality validator module | 43415e8 | quality_validator.py, __init__.py |
| 3 | Add audio file utilities | 43415e8 | quality_validator.py (integrated) |

## What Was Built

### 1. Quality Validation Dependencies (Task 1)

Added audio quality validation libraries to `requirements.txt`:

- **pesq>=0.0.4**: ITU-T P.862 PESQ algorithm for perceptual speech quality (MOS 1-5 scale)
- **pystoi>=0.3.3**: Short-Time Objective Intelligibility for speech clarity (0-1 scale)
- **soundfile>=0.12.1**: Already present from Phase 5 Plan 1

### 2. Quality Validator Module (Task 2)

Created `src/tts/quality_validator.py` with:

**AudioQualityResult dataclass:**
- Duration metrics: target, actual, error percentage, validation status
- PESQ score (1.0-5.0): Perceptual speech quality with human-readable tier
- STOI score (0.0-1.0): Speech intelligibility measurement
- Pitch variance ratio: Emotion preservation proxy (generated/reference)
- Overall assessment: passes_quality, flagged_for_review, rejection_reason

**QualityValidator class:**
- `validate_single()`: Full validation with reference (PESQ, STOI, emotion preservation)
- `validate_duration_only()`: Simplified validation without reference
- `validate_batch()`: Batch processing with summary statistics
- Configurable thresholds from settings.py (min_pesq, review_pesq, duration_tolerance)

**Quality tier classification:**
- **Excellent** (PESQ > 4.0): Auto-accept
- **Good** (PESQ 3.0-4.0): Accept
- **Fair** (PESQ 2.5-3.0): Flag for review
- **Poor** (PESQ < 2.5): Reject

**Emotion preservation classification:**
- Ratio 0.8-1.2: Preserved (good)
- Ratio 0.6-0.8 or 1.2-1.5: Marginal (flag for review)
- Ratio <0.6 or >1.5: Lost/exaggerated (flag for review)

### 3. Audio File Utilities (Task 3)

Integrated into quality_validator.py:

- `get_audio_duration()`: Duration in seconds using librosa
- `load_audio_for_validation()`: Load at 16kHz, mono, normalized [-1, 1]
- `calculate_rms_energy()`: RMS energy as quality proxy
- `flag_silent_audio()`: Detect silent/quiet audio (RMS < 0.01)
- `extract_pitch_contour()`: F0 extraction using librosa.pyin()
- `validate_emotion_preservation()`: Compare pitch dynamics (variance ratio)

**Error handling:**
- Returns 0.0 duration for unreadable files
- Graceful fallbacks for PESQ/STOI calculation failures
- Warnings for insufficient voiced frames in emotion checks

### 4. Module Exports

Updated `src/tts/__init__.py` to export:
- `AudioQualityResult`: Quality validation result dataclass
- `QualityValidator`: Main validation class
- `validate_audio_quality`: Convenience function for single validation

## Technical Implementation

### PESQ Validation

Uses ITU-T P.862 wideband PESQ at 16kHz:
```python
pesq_score = pesq(16000, reference_audio, generated_audio, 'wb')
```

Maps MOS scores (1-5) to quality tiers for human-readable feedback.

### STOI Intelligibility

Short-Time Objective Intelligibility at 16kHz:
```python
stoi_score = stoi(reference_audio, generated_audio, 16000)
```

Provides 0-1 scale for speech clarity measurement.

### Emotion Preservation Proxy

Compares pitch variance between generated and reference audio:
1. Extract F0 contours using librosa.pyin()
2. Calculate variance for voiced regions (exclude NaN)
3. Compute ratio: `generated_variance / reference_variance`

Ratio close to 1.0 indicates preserved pitch dynamics (emotion maintained). Ratio < 0.5 suggests flat/monotone output (emotion lost). Ratio > 2.0 suggests excessive variation (emotion exaggerated).

### Duration Validation

Calculates percentage error:
```python
duration_error = abs(actual - target) / target
duration_valid = duration_error <= tolerance  # Default 5%
```

Critical for lip sync compatibility - excessive duration mismatch causes visual desync.

### Reference Audio Strategy

Two validation modes:
1. **Full validation**: Uses speaker reference sample for PESQ/STOI/emotion checks
2. **Duration-only**: Fallback when no reference available (TTS has no "ground truth")

Default to option 1 for emotion checking, with duration-only as fallback.

## Decisions Made

### 1. PESQ + STOI Combination
**Decision:** Use both PESQ and STOI for comprehensive quality assessment
**Rationale:** PESQ measures perceptual quality (how natural it sounds), STOI measures intelligibility (how understandable it is). Both needed for good TTS.
**Impact:** More robust quality detection than single metric alone
**Status:** ✅ Implemented

### 2. Pitch Variance as Emotion Proxy
**Decision:** Use pitch variance ratio instead of prosody/spectral analysis
**Rationale:** Fast calculation, directly correlated with emotional expression (flat pitch = monotone, varied pitch = expressive). More complex methods (prosody trees, spectral features) add overhead for marginal gains.
**Impact:** Instant emotion preservation check without linguistic analysis
**Trade-offs:** Less precise than full prosody analysis, but 80/20 rule applies
**Status:** ✅ Implemented

### 3. Duration-Only Fallback
**Decision:** Support validation without reference audio
**Rationale:** TTS doesn't have "ground truth" reference - generated audio IS the output. PESQ/STOI need reference-to-degraded comparison, so only useful when using speaker sample as reference.
**Impact:** Enables quality checking in all scenarios (with or without reference)
**Status:** ✅ Implemented

### 4. Four-Tier PESQ Classification
**Decision:** Map PESQ scores to excellent/good/fair/poor tiers
**Rationale:** MOS scores (1-5) are not intuitive for non-experts. Human-readable tiers make quality feedback understandable in UI/logs.
**Impact:** Better UX for quality assessment and debugging
**Status:** ✅ Implemented

## Verification Results

All verification criteria met:

1. ✅ pesq and pystoi libraries install and import correctly
2. ✅ AudioQualityResult has all required fields including pitch_variance_ratio
3. ✅ QualityValidator uses configurable thresholds from settings.py
4. ✅ Duration-only validation works without reference audio
5. ✅ Audio utilities handle edge cases (missing files return 0.0 duration)
6. ✅ Emotion preservation check uses pitch variance ratio

## Integration Points

### Upstream Dependencies
- **Phase 05-01**: Reference extractor provides speaker reference samples
- **Phase 05-02**: XTTS generator produces audio files to validate

### Downstream Impact
- **Phase 05-04**: TTS stage integration will use QualityValidator for post-synthesis checks
- **Phase 07**: Lip sync stage needs duration-validated audio (timing critical)
- **Phase 08**: Review UI will surface flagged segments for manual inspection

### Usage Pattern
```python
from src.tts import QualityValidator

validator = QualityValidator()

# Full validation with reference
result = validator.validate_single(
    audio_path=Path("output/segment_0.wav"),
    reference_path=Path("references/speaker_0.wav"),
    segment_id=0,
    target_duration=3.5
)

if not result.passes_quality:
    print(f"Rejected: {result.rejection_reason}")
elif result.flagged_for_review:
    print(f"Review needed: PESQ={result.pesq_score}, emotion={result.pitch_variance_ratio}")
```

## Deviations from Plan

None - plan executed exactly as written. All three tasks completed successfully with full integration.

## Next Phase Readiness

### Blockers
None.

### Prerequisites for Phase 05-04 (TTS Stage Integration)
- ✅ Quality validation module complete
- ✅ PESQ/STOI dependencies installed
- ✅ Emotion preservation checking implemented
- ✅ Batch validation support ready

### Known Limitations

1. **PESQ requires reference audio**: Can't compute PESQ/STOI without reference sample. Duration-only fallback available but less comprehensive.

2. **16kHz requirement**: PESQ wideband mode requires 16kHz audio. XTTS outputs 24kHz, so resampling needed for validation.

3. **Pitch extraction for unvoiced audio**: librosa.pyin() returns NaN for unvoiced regions (silence, consonants). Need 10+ voiced frames for meaningful emotion check.

4. **Emotion proxy simplicity**: Pitch variance ratio is a proxy, not full prosody analysis. May miss subtle emotional nuances (stress patterns, rhythm, timbre).

5. **PESQ/STOI library warnings**: pesq library may warn about audio length mismatches. Handled by padding/trimming to same length.

### Recommendations for Next Plan

1. **Test with real XTTS output**: Validate quality checker against actual XTTS-generated audio
2. **Calibrate thresholds**: Current thresholds (min_pesq=2.5, review_pesq=3.0) are reasonable defaults but may need tuning based on XTTS output characteristics
3. **Add batch validation examples**: Document batch validation usage for TTS stage integration
4. **Consider GPU acceleration**: PESQ/STOI are CPU-bound. May add latency for large batches.

## Performance Metrics

- **Duration**: 4 minutes (faster than average 31.5m Phase 5 estimate)
- **LOC added**: ~500 lines (quality_validator.py)
- **Dependencies added**: 2 (pesq, pystoi)
- **Commits**: 2 (dependencies + implementation)

## Key Learnings

1. **Emotion preservation is challenging**: No ground truth for "correct" emotion in TTS. Pitch variance ratio is best fast proxy available without linguistic analysis.

2. **PESQ needs careful length matching**: PESQ requires reference and degraded audio to have same length. Padding/trimming to min_length prevents errors.

3. **TTS validation differs from ASR**: ASR has ground truth transcripts (can use WER). TTS has no ground truth audio, so validation compares to reference sample (voice similarity) rather than "correctness".

4. **Silence detection is critical**: Silent or very quiet audio indicates synthesis failure. Must check before PESQ (PESQ fails on silent audio).

5. **Configurable thresholds matter**: Different use cases may want different quality bars. Centralized settings.py constants enable tuning without code changes.
