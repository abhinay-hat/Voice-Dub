# Phase 4: Translation Pipeline - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Translates transcribed speech from any language to English while preserving context, meaning, and timing constraints. System must handle 20-30 source languages, maintain conversational coherence across speakers, and ensure translated text fits within original segment durations (within 10% tolerance) for downstream voice cloning and lip sync stages.

</domain>

<decisions>
## Implementation Decisions

### Context handling
- Full transcript context for translation (not sliding window or isolated segments)
- Long videos: Claude's discretion on chunking strategy (scene detection vs single batch based on model limits)
- Speaker labels: Claude's discretion on whether to include in translation input
- Pronoun/reference handling: Claude's discretion on when to clarify vs preserve ambiguous pronouns

### Duration constraints
- Moderate tolerance: Translated segments must fit within 10% of original duration
- Too-long translations: Claude's discretion (compress, split, or flag based on specific case)
- Too-short translations: Claude's discretion (keep as-is, pad naturally, or merge based on context)
- Duration estimation: Claude's discretion on method (character count, phoneme analysis, or TTS preview)

### Translation quality vs speed
- **Priority: Speed first** - optimize for fast processing of 20-minute videos
- **Generate multiple candidates** (2-3 alternatives per segment) and rank them
- Ranking criterion: Claude's discretion (balance model confidence and duration fit)
- **Auto-flag low confidence segments** (e.g., <70% threshold) for later review in Phase 8

### Multi-speaker handling
- Cross-speaker approach: User wants "what is best" - Claude should choose optimal strategy (likely hybrid with cross-speaker context + speaker consistency)
- Conversational references: Claude's discretion on when to clarify pronouns vs keep as-is
- Formality preservation: Claude's discretion on maintaining speaker-specific language styles
- Interruptions/overlaps: Claude's discretion on how to handle overlapping speech naturally

### Claude's Discretion
- Long video chunking strategy (scene detection, context window sizing)
- Whether to include speaker labels in translation input
- Pronoun clarification logic (when to replace 'he/she/it' with specific nouns)
- Duration adjustment approach (compress/split/flag for over-long, pad/merge for short)
- Duration estimation method (heuristic vs linguistic analysis vs TTS preview)
- Translation candidate ranking formula (confidence + duration weighting)
- Multi-speaker translation strategy (independent vs cross-context vs hybrid)
- Formality level handling across speakers
- Interruption and speech overlap translation

</decisions>

<specifics>
## Specific Ideas

- User wants the system to prioritize speed over perfection - this is for friends to watch dubbed videos, not professional translation service
- Multiple translation candidates should help catch better duration fits even though it takes more processing
- Low confidence threshold (~70%) flagging ensures quality control without manual review of every segment
- The "what is best" response for multi-speaker handling suggests user trusts Claude to figure out the right technical approach

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 04-translation-pipeline*
*Context gathered: 2026-01-31*
