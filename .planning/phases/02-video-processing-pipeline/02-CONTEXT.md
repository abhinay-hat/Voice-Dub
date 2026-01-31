# Phase 2: Video Processing Pipeline - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can upload videos and the system extracts audio/video streams using FFmpeg, then merges them back. This validates the FFmpeg toolchain before ML complexity enters the pipeline. No ML models involved yet - pure video I/O infrastructure.

</domain>

<decisions>
## Implementation Decisions

### Output Format & Codec
- **Container format:** Match input format (MKV in → MKV out, MP4 in → MP4 out)
- **Video codec:** Claude's choice - select best codec for dubbing use case
- **Quality settings:** Match input quality (preserve original bitrate and resolution)
- **Metadata handling:** Claude's discretion - decide what's practical to preserve

### Claude's Discretion
- Video upload handling (file picker, drag-and-drop, size limits, format validation)
- Extraction strategy (frame extraction approach, audio format/sample rate)
- Temporary file management (storage location, cleanup timing, naming, disk space)
- Video codec selection (H.264 vs H.265 vs copy)
- Metadata preservation (chapters, subtitles, titles - what to keep/drop)

</decisions>

<specifics>
## Specific Ideas

No specific requirements - open to standard approaches for video processing infrastructure.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 02-video-processing-pipeline*
*Context gathered: 2026-01-31*
