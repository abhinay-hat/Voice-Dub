# Roadmap: Voice Dub

## Overview

Voice Dub transforms videos in any language into English-dubbed versions with cloned voices, preserved emotion, and lip synchronization. The roadmap builds a sequential AI pipeline from foundation to full dubbing: starting with environment setup and basic video processing, adding each ML model incrementally (speech recognition, translation, voice cloning, lip sync), then enhancing with quality controls, performance optimization, batch processing, and user experience polish. The journey validates core value early (watchable dubbed videos with recognizable voices) before adding production features.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Environment & Foundation** - CUDA setup and project scaffolding
- [x] **Phase 2: Video Processing Pipeline** - Video upload, extraction, and merging
- [x] **Phase 3: Speech Recognition** - Transcription with speaker diarization
- [x] **Phase 4: Translation Pipeline** - Context-aware English translation
- [x] **Phase 5: Voice Cloning & TTS** - Speaker voice cloning with emotion preservation
- [ ] **Phase 6: Audio-Video Assembly** - Sync infrastructure and duration validation
- [x] **Phase 7: Lip Synchronization** - Lip movement matching to English audio
- [ ] **Phase 8: Quality Controls** - User review, editing, and validation
- [ ] **Phase 9: Performance Optimization** - GPU utilization and processing speed
- [ ] **Phase 10: Batch Processing** - Queue management for multiple videos
- [ ] **Phase 11: User Experience Polish** - Web UI refinement and error handling

## Phase Details

### Phase 1: Environment & Foundation
**Goal**: Establish verified CUDA environment and project architecture that prevents silent CPU fallback and enables efficient GPU memory management for ML pipeline.
**Depends on**: Nothing (first phase)
**Requirements**: PERF-01, PERF-05
**Success Criteria** (what must be TRUE):
  1. PyTorch with CUDA 12.x installed and verified (torch.cuda.is_available() returns True)
  2. GPU detection utility confirms RTX 5090 is active with 32GB VRAM accessible
  3. Project structure exists with directories for models, temp files, pipeline stages
  4. GPU memory monitoring shows CUDA allocation during test inference (no CPU fallback)
  5. Environment variables configured (PYTORCH_CUDA_ALLOC_CONF for memory management)
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Install PyTorch nightly with CUDA 12.8 and create GPU validation utility
- [x] 01-02-PLAN.md — Create project structure and implement model manager with sequential loading
- [x] 01-03-PLAN.md — Run validation tests and verify environment on RTX 5090 hardware

### Phase 2: Video Processing Pipeline
**Goal**: Users can upload videos and system extracts audio/video streams, then merges them back without ML complexity, validating FFmpeg toolchain.
**Depends on**: Phase 1
**Requirements**: VID-01, VID-02, VID-03, VID-04
**Success Criteria** (what must be TRUE):
  1. User can upload video files (MP4, MKV, AVI) through web interface
  2. System extracts audio stream to WAV and video stream to frames using FFmpeg
  3. System merges processed audio and video back into original format
  4. Output video preserves input quality and resolution (up to 1080p tested)
  5. Temporary files are automatically cleaned up after processing completes
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — FFmpeg setup, video probing utilities, and temp file manager
- [x] 02-02-PLAN.md — Audio/video extraction, stream merging, and pipeline orchestration
- [x] 02-03-PLAN.md — Gradio web interface and end-to-end verification

### Phase 3: Speech Recognition
**Goal**: System transcribes speech from any language with timestamps and speaker labels, enabling translation and voice cloning downstream.
**Depends on**: Phase 2
**Requirements**: ASR-01, ASR-02, ASR-03, ASR-04, ASR-05
**Success Criteria** (what must be TRUE):
  1. System transcribes audio to text from 20-30 input languages (Japanese, Korean, Chinese, Spanish, French, German tested)
  2. Each transcribed segment has start/end timestamps accurate to 0.1 seconds
  3. System detects 2-5 speakers and labels each segment by speaker ID
  4. Low-confidence segments (below 70% confidence) are flagged for user review
  5. Whisper Large V3 loads on GPU and processes 1-minute audio in under 10 seconds
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Audio preprocessing and Whisper transcription with word-level timestamps
- [x] 03-02-PLAN.md — Speaker diarization with pyannote and temporal alignment
- [x] 03-03-PLAN.md — ASR stage integration, JSON output, and testing

### Phase 4: Translation Pipeline
**Goal**: System translates transcribed text to English preserving context and meaning, with duration validation ensuring translated speech fits original timing.
**Depends on**: Phase 3
**Requirements**: TRAN-01, TRAN-02, TRAN-03, TRAN-04
**Success Criteria** (what must be TRUE):
  1. System translates transcript segments to English maintaining original meaning
  2. Translation preserves context across segment boundaries (no isolated sentence translation)
  3. Translated text duration matches original segment timing (within 10% tolerance)
  4. System supports 20-30 source languages (same as ASR coverage)
  5. SeamlessM4T v2 loads on GPU without evicting Whisper (32GB VRAM sufficient)
**Plans**: 4 plans

Plans:
- [x] 04-01-PLAN.md — SeamlessM4T model setup and single-segment translation
- [x] 04-02-PLAN.md — Duration validation and candidate ranking modules
- [x] 04-03-PLAN.md — Multi-candidate generation and context chunking
- [x] 04-04-PLAN.md — Translation stage orchestration and integration testing

### Phase 5: Voice Cloning & TTS
**Goal**: System generates English audio that clones each speaker's voice and emotional tone from 6-10 second reference samples, delivering recognizable similarity.
**Depends on**: Phase 4
**Requirements**: TTS-01, TTS-02, TTS-03, TTS-04, TTS-05
**Success Criteria** (what must be TRUE):
  1. System extracts 6-10 second reference audio sample per speaker from original video
  2. Generated English voice is recognizably similar to original speaker (user can identify who's speaking)
  3. Emotional tone (excited, calm, angry) is preserved in English audio
  4. Generated audio matches translated text timing (within 5% of target duration)
  5. Audio quality validation (MOS scoring) rejects low-quality samples before proceeding
**Plans**: 4 plans

Plans:
- [x] 05-01-PLAN.md — Reference sample extraction and speaker embedding generation
- [x] 05-02-PLAN.md — XTTS-v2 synthesis wrapper with duration matching
- [x] 05-03-PLAN.md — Audio quality validation with PESQ scoring
- [x] 05-04-PLAN.md — TTS stage orchestration and integration testing

### Phase 6: Audio-Video Assembly
**Goal**: System merges dubbed audio with video maintaining frame-perfect synchronization over full 20-minute duration, preventing gradual drift.
**Depends on**: Phase 5
**Requirements**: None (infrastructure for Phase 7)
**Success Criteria** (what must be TRUE):
  1. Audio and video stay synchronized at 5-minute intervals throughout 20-minute test video
  2. No noticeable audio drift at 10-minute and 20-minute marks
  3. High-precision timestamps (float64) maintain accuracy through pipeline
  4. Consistent sample rate (48kHz) enforced throughout audio processing
  5. FFmpeg merge completes with explicit sync flags (-async 1) preventing drift
**Plans**: 3 plans

Plans:
- [ ] 06-01-PLAN.md — Core assembly infrastructure (timestamp validator, audio normalizer, concatenator)
- [ ] 06-02-PLAN.md — Drift detection and enhanced video merger with sync flags
- [ ] 06-03-PLAN.md — Assembly stage orchestration and integration testing

### Phase 7: Lip Synchronization
**Goal**: System synchronizes lip movements to English audio with frame-perfect accuracy while maintaining facial stability, completing core dubbing pipeline.
**Depends on**: Phase 6
**Requirements**: SYNC-01, SYNC-02, SYNC-03, SYNC-04
**Success Criteria** (what must be TRUE):
  1. Lip movements match English audio phonemes (p, b, m, w tested specifically)
  2. Facial expressions remain stable (no flickering or uncanny valley dead-face)
  3. Lip sync accuracy validated frame-by-frame (95%+ frames pass threshold)
  4. Multi-speaker videos maintain correct sync per speaker
  5. Full 20-minute dubbed video is watchable without distracting lip sync errors
**Plans**: 4 plans

Plans:
- [x] 07-01-PLAN.md — LatentSync conda env setup, checkpoint download, audio prep, and inference runner
- [x] 07-02-PLAN.md — Wav2Lip fallback runner and video chunker for long videos
- [x] 07-03-PLAN.md — Lip sync stage orchestration with run_lip_sync_stage() and LipSyncResult
- [x] 07-04-PLAN.md — Integration test suite and Phase 7 README documentation

### Phase 8: Quality Controls
**Goal**: Users can review and correct pipeline outputs before full processing, preventing wasted computation on ASR errors or bad voice cloning.
**Depends on**: Phase 7
**Requirements**: QC-01, QC-02, QC-03, QC-04
**Success Criteria** (what must be TRUE):
  1. User can edit transcript text in web UI before dubbing begins
  2. User can preview 10-30 second clips before committing to full 20-minute render
  3. Progress display shows real-time status (Transcribing 45% / Translating / Cloning voice / Syncing lips)
  4. System validates each pipeline stage (ASR confidence, translation duration, audio quality, sync accuracy) before proceeding
  5. User can cancel processing at any stage and return to editing
**Plans**: TBD

Plans:
- [ ] TBD during phase planning

### Phase 9: Performance Optimization
**Goal**: System processes 20-minute videos in 10-20 minutes leveraging full RTX 5090 capabilities, with memory management preventing OOM errors.
**Depends on**: Phase 8
**Requirements**: PERF-02, PERF-03, PERF-04
**Success Criteria** (what must be TRUE):
  1. 20-minute video completes full pipeline in 10-20 minutes (tested end-to-end)
  2. GPU memory usage optimized (multiple models loaded simultaneously in 32GB VRAM)
  3. No GPU OOM errors during 20-minute video processing
  4. GPU utilization above 80% during inference stages (Whisper, SeamlessM4T, XTTS, Wav2Lip)
  5. Model loading uses lazy initialization and LRU unloading for memory efficiency
**Plans**: TBD

Plans:
- [ ] TBD during phase planning

### Phase 10: Batch Processing
**Goal**: Users can queue multiple videos and system processes them sequentially with status tracking and cancellation support.
**Depends on**: Phase 9
**Requirements**: BATCH-01, BATCH-02, BATCH-03, BATCH-04
**Success Criteria** (what must be TRUE):
  1. User can add 5-10 videos to processing queue
  2. System processes queued videos one at a time (sequential execution)
  3. User can view status of all queued jobs (Pending, In Progress, Complete, Failed)
  4. User can cancel pending jobs from queue before processing starts
  5. Completed jobs remain accessible for download until manually deleted
**Plans**: TBD

Plans:
- [ ] TBD during phase planning

### Phase 11: User Experience Polish
**Goal**: Web interface is intuitive and accessible for friends, with clear error messages and automatic cleanup preventing disk space bloat.
**Depends on**: Phase 10
**Requirements**: UX-01, UX-02, UX-03, UX-04, UX-05
**Success Criteria** (what must be TRUE):
  1. Web interface accessible at localhost:7860 (Gradio default) on network
  2. User can upload videos via drag-and-drop (not just file picker)
  3. User can download dubbed video with single click when processing completes
  4. Error messages are clear and actionable (not stack traces or "Unknown error")
  5. Temporary files deleted automatically after 24 hours or manual cleanup button works
**Plans**: TBD

Plans:
- [ ] TBD during phase planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment & Foundation | 3/3 | Complete | 2026-01-31 |
| 2. Video Processing Pipeline | 3/3 | Complete | 2026-01-31 |
| 3. Speech Recognition | 3/3 | Complete | 2026-01-31 |
| 4. Translation Pipeline | 4/4 | Complete | 2026-01-31 |
| 5. Voice Cloning & TTS | 4/4 | Complete | 2026-02-02 |
| 6. Audio-Video Assembly | 3/3 | Complete | 2026-02-03 |
| 7. Lip Synchronization | 4/4 | Complete | 2026-02-21 |
| 8. Quality Controls | 0/TBD | Not started | - |
| 9. Performance Optimization | 0/TBD | Not started | - |
| 10. Batch Processing | 0/TBD | Not started | - |
| 11. User Experience Polish | 0/TBD | Not started | - |

---
*Created: 2026-01-31*
*Last updated: 2026-02-21 (Phase 7 execution complete)*
