# Phase 5: Voice Cloning & TTS - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate English audio that clones each speaker's voice and emotional tone from 6-10 second reference samples. The cloned voices must be recognizably similar to the original speakers, preserve emotional expression, and match translated text timing. This delivers the core "voice preservation" value that distinguishes Voice Dub from generic TTS.

Input: JSON transcript with translated English text and speaker IDs (from Phase 4)
Output: English audio files per segment with cloned voices matching original speakers

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

User has delegated all implementation decisions to Claude. The following areas are open to technical judgment during research and planning:

**Reference Sample Strategy:**
- Selection algorithm (cleanest audio automatic, first appearance, or hybrid)
- Minimum duration handling (strict 6-10s vs fallback to 3-6s vs segment concatenation)
- Quality validation approach (audio metrics, SNR checks, user preview)
- Edge case handling (speakers without clean samples, fallback to generic TTS or error)

**Voice Similarity Priority:**
- Balance between recognizability, pitch accuracy, and speech pattern matching
- Gender/age mismatch handling (strict matching vs approximation vs pitch shifting)
- Uncanny valley detection and retry logic (automatic parameter adjustment or user approval)
- Speed vs quality optimization (XTTS quality settings, inference speed trade-offs)

**Emotional Preservation:**
- Emotion capture method (analyze original audio, infer from text, or hybrid)
- Intensity matching (exact match vs normalization vs cultural adaptation)
- Conflict resolution (prioritize audio emotion vs text emotion vs user review)
- Granularity (segment-level, speaker-turn-level, or scene-level emotion tracking)

**Quality Validation Approach:**
- Automatic validation metrics (duration matching ±5%, voice similarity scoring, audio quality checks)
- Validation timing (per-segment, per-speaker, or full generation batch)
- Error handling strategy (automatic retry, user notification, or fail-fast)
- Preview workflow (reference sample playback, first segment per speaker, or no previews)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches and technical best practices.

User expects:
- Recognizable voice similarity (user can identify who's speaking)
- Emotional tone preservation (excited, calm, angry preserved in English)
- Duration accuracy (within 5% of target per success criteria)
- Quality validation before proceeding to lip sync

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-voice-cloning-and-tts*
*Context gathered: 2026-02-02*
