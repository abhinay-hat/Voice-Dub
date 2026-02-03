# Phase 6: Audio-Video Assembly - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Merges dubbed audio with video streams maintaining frame-perfect synchronization over full 20-minute duration without gradual drift. This phase establishes the infrastructure for precise timestamp handling, sample rate consistency, and sync validation that enables Phase 7 (Lip Synchronization). Does not include lip movement generation or quality controls/preview UI.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

**All implementation areas delegated to Claude:**

The user has delegated all technical decisions for this phase to Claude, trusting the builder to make optimal choices based on requirements. Claude has full discretion over:

#### Timestamp Precision Strategy
- Numeric type for storing timestamps (float64 vs int64 milliseconds)
- Boundary handling when segments don't align to frame boundaries
- Validation points throughout pipeline (after ASR, translation, TTS, or only at assembly)
- Acceptable precision loss tolerance over 20-minute duration

**Constraints to consider:**
- Phase 7 (Lip Sync) depends on precise timestamps
- Success criteria requires "no noticeable drift at 10-minute and 20-minute marks"
- Must maintain "high-precision timestamps (float64)" per roadmap
- Validation should catch issues before expensive lip sync processing

#### Sample Rate Normalization
- When to enforce target sample rate (48kHz) in pipeline flow
- Handling multi-speaker scenarios where TTS may generate different rates
- Resampling quality vs speed tradeoff (sinc vs polyphase vs linear)
- Whether to validate rate consistency before merging or trust upstream stages

**Constraints to consider:**
- Success criteria requires "consistent sample rate (48kHz) enforced throughout"
- Whisper requires 16kHz preprocessing (Phase 3)
- XTTS outputs at specific rate (Phase 5)
- Quality must support voice cloning fidelity

#### Drift Detection Approach
- Checkpoint interval frequency (every 5min, 10min, or start/middle/end)
- Response to drift exceeding tolerance (fail, auto-correct, warn)
- Tolerance threshold for triggering warnings/errors
- Measurement method (expected vs actual timestamp, frame vs sample, boundary alignment)

**Constraints to consider:**
- Success criteria requires "audio and video stay synchronized at 5-minute intervals"
- Must validate "at 10-minute and 20-minute marks"
- 20-minute test videos are the target duration
- Should prevent wasted computation if drift makes lip sync impossible

#### FFmpeg Sync Flags
- Primary sync parameter (-async, -vsync, -af aresample)
- Stream mapping order (video-first vs audio-first vs explicit indices)
- Format-specific handling (MP4 vs MKV differences)
- Original audio track handling (replace vs keep both vs configurable)

**Constraints to consider:**
- Success criteria requires "FFmpeg merge completes with explicit sync flags (-async 1)"
- Must preserve video quality (no re-encoding unless necessary)
- Phase 2 established MP4/MKV/AVI support
- Should support container format flexibility

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

The user has expressed trust in Claude's technical judgment for this infrastructure phase. Implementation should prioritize:
1. Preventing drift accumulation (primary user concern)
2. Maintaining precision for downstream lip sync
3. Following established patterns from Phases 2-5 (FFmpeg usage, progress callbacks, JSON I/O)
4. Meeting success criteria from roadmap

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-audio-video-assembly*
*Context gathered: 2026-02-02*
