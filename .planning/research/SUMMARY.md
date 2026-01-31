# Project Research Summary

**Project:** Voice Dub
**Domain:** AI-Powered Video Dubbing with Voice Cloning
**Researched:** 2026-01-31
**Confidence:** HIGH

## Executive Summary

Voice Dub is an AI-powered video dubbing tool that leverages voice cloning to translate videos while preserving the original speaker's voice, emotions, and lip synchronization. The expert approach involves a sequential pipeline combining state-of-the-art open-source models: Whisper Large V3 for transcription, SeamlessM4T v2 for translation with prosody preservation, XTTS-v2 for voice cloning with emotion transfer, and Wav2Lip for lip synchronization. This pipeline must run on AMD GPUs using ROCm 7.2, which adds complexity but enables local, privacy-focused processing as the core differentiator against cloud-only competitors.

The recommended approach prioritizes a sequential pipeline with lazy model loading and explicit GPU memory management. Start with a single-language MVP (English target) to validate the core value proposition - watch any video in your language while preserving original speaker voices and emotional expression. Build the pipeline incrementally: video extraction, then ASR, translation, TTS, and finally lip sync. Each stage produces checkpointed intermediate files to enable debugging and restart on failure. The architecture uses Gradio for the web interface, async queue management for non-blocking processing, and context managers for automatic cleanup of temporary files.

Critical risks include silent CPU fallback on AMD GPUs (causing 10-100x slowdown), ASR hallucinations cascading through the pipeline, audio-video synchronization drift in long videos, and GPU memory fragmentation causing OOM errors despite available VRAM. Mitigation requires explicit GPU verification utilities, confidence scoring for transcriptions, high-precision timestamp management, and proper PyTorch memory allocator configuration. The lip sync component is highest risk due to "uncanny valley" quality issues - mouths move correctly but faces look dead. Test with full 20-minute videos, not just 10-second clips, to catch these issues early.

## Key Findings

### Recommended Stack

The technology stack is built around PyTorch 2.9.1 with ROCm 7.2.0, which provides official AMD GPU support for Python 3.12. The ML pipeline uses Whisper Large V3 (1.55B params) for ASR with 10-20% error reduction over V2, SeamlessM4T v2 (2.3B params) for translation with prosody preservation, XTTS-v2 for zero-shot voice cloning from 6-second samples supporting emotion transfer, and Wav2Lip for battle-tested lip synchronization. Video processing leverages FFmpeg with AMD AMF hardware acceleration on Windows or VAAPI on Linux, with OpenCV for frame extraction and MoviePy for high-level editing.

**Core technologies:**
- **PyTorch 2.9.1+rocm7.2**: Primary ML framework - Official ROCm 7.2 support, install from AMD's repo.radeon.com (NOT PyPI which has CPU-only variant for ROCm 7.x)
- **Whisper Large V3 via faster-whisper**: State-of-the-art ASR with 99 languages, 4x faster than standard implementation, requires custom ROCm build via community forks
- **SeamlessM4T v2**: Meta's multilingual translation with speech-to-speech and prosody preservation, available via Hugging Face Transformers with native ROCm support
- **XTTS-v2**: Zero-shot voice cloning from 6-second samples with emotional prosody support for 17 languages, ROCm support via YellowRoseCx/XTTS-WebUI-ROCm fork
- **Wav2Lip**: Battle-tested lip sync model with proven reliability and PyTorch/ROCm compatibility
- **Gradio 4.x**: Purpose-built ML web UI framework with one-line hosting, simpler than Streamlit for AI model interfaces
- **RQ + Redis**: Background job processing with simpler architecture than Celery, sufficient for single-GPU local tool
- **FFmpeg with AMF/VAAPI**: Hardware-accelerated video encoding/decoding on AMD GPUs

**Critical version constraint:** PyTorch, torchvision, and torchaudio must all match the same ROCm version (7.2.0) to avoid runtime errors. Install from AMD's repo.radeon.com, NOT PyPI.

**ROCm-specific requirements:** Consumer GPUs like RX 7900 XT (gfx1100) need `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable. Some community tools (faster-whisper ROCm builds) may be Linux-only.

### Expected Features

The 2026 AI dubbing landscape shows convergence on core features: multi-language support (100+ languages standard), voice cloning from 2-3 second samples with emotion preservation, automatic speaker diarization for 2-10 speakers, frame-perfect lip synchronization, HD/4K output quality, and processing times under 3-10 minutes for shorts (cloud services). For a local, privacy-focused tool, the key differentiator is eliminating cloud upload while accepting longer processing times (under 1 hour for 20-minute videos).

**Must have (table stakes):**
- Multi-language support (20-30 languages for MVP vs. 175+ industry leader) - users expect this
- Voice cloning with emotional preservation from 2-3 second samples - core value proposition
- Automatic transcription with 90%+ accuracy (ASR foundation) - users expect this
- Speaker diarization for 2-5 speakers with per-speaker voice cloning - breaks without it
- Lip synchronization with frame-perfect accuracy - without lip-sync, dubbed video looks broken
- HD output preserving input resolution up to 1080p - users expect this
- Progress tracking showing pipeline stages - UX baseline requirement
- Transcript editing to fix ASR errors before dubbing - quality control necessity
- Preview clips (10-30 sec) before full render to avoid wasting processing time - users expect this

**Should have (competitive):**
- Local processing (privacy-first) - YOUR core differentiator vs. cloud services
- Subtitle generation with sync (SRT export) - accessibility and user value
- Custom voice library (200+ voices as fallback when cloning fails) - flexibility
- Batch processing with queue management for 2-5 concurrent jobs - efficiency improvement
- Custom terminology glossary (CSV upload for brand terms) - quality enhancement
- Processing time under 1 hour for 20-minute videos - acceptable tradeoff for privacy

**Defer (v2+):**
- API integration with webhooks for automation - power users only
- GPU utilization optimization for 5-10x speedup - performance enhancement
- Voice emotion control (manual tuning beyond preservation) - niche, high complexity
- Cultural intelligence engine for joke timing and references - extremely complex, overkill for personal use
- Real-time dubbing for live streaming - very high complexity, not needed for personal video files
- Support for 175+ languages - 80% of use is 5-10 languages, maintain quality over completeness

**Anti-features to avoid:**
- "Instant" processing under 1 minute - local GPU can't match cloud, sets unrealistic expectations
- Automatic quality without review options - dubbing always needs human review for ASR errors, cultural context
- Real-time lip-sync preview during editing - technically impossible without cloud infrastructure
- Zero-click auto-dubbing on upload - users need to choose target language and review settings

### Architecture Approach

The standard architecture for AI video dubbing is a sequential pipeline with discrete, independent stages: video extraction, ASR transcription, translation, TTS voice synthesis, lip synchronization, and final video merge. Each stage reads from previous output and writes to next input, enabling isolation for testing/debugging and restart from any stage on failure. The pipeline uses lazy model loading with a singleton ModelManager to prevent loading the same model multiple times and handle GPU OOM by unloading least-recently-used models. An async producer-consumer queue pattern enables non-blocking UI and parallel processing of multiple videos, while context managers guarantee cleanup of temporary files even on errors.

**Major components:**
1. **Web UI Layer (Gradio)** - File upload/download, progress display, parameter configuration with File components and status updates
2. **Task Queue Manager (RQ + Redis)** - Job management, worker distribution, status tracking for async processing
3. **Processing Pipeline** - Sequential stages: Video Extract → ASR (Whisper) → Translation (Seamless) → TTS (XTTS) → Lip Sync (Wav2Lip) → Video Merge (FFmpeg)
4. **Model Manager (Singleton)** - Lazy loading, GPU memory management with LRU unloading on OOM
5. **GPU Scheduler (ROCm/PyTorch)** - Device allocation, memory monitoring, OOM recovery with `torch.cuda.empty_cache()`
6. **Storage Manager (tempfile)** - Per-job temporary workspaces with automatic cleanup via context managers

**Key architectural patterns:**
- **Pipeline as Sequential Stages:** Each stage isolated with intermediate file checkpoints, enables debugging and restart on failure
- **Lazy Model Loading with LRU:** Load models on-demand, keep in memory across requests, unload LRU on OOM to prevent manual intervention
- **Producer-Consumer Queue:** Gradio UI submits jobs to queue, async workers process in background, enables non-blocking UI and concurrent users
- **Context Managers for Cleanup:** Temporary workspaces automatically cleaned up even on errors, critical for video processing generating GBs of files

**Build order recommendation:** Start with foundation (project structure, FFmpeg video extraction/merge, storage management, basic Gradio UI) to validate toolchain without ML complexity. Then single model integration (Whisper ASR + model manager base class) to establish pattern. Expand pipeline with translation and TTS for complete audio pipeline. Finally integrate lip sync (highest risk component) after audio quality is proven.

### Critical Pitfalls

Research identified 8 critical pitfalls that can derail the project. The top risks are AMD ROCm-specific (silent CPU fallback, GPU architecture mismatch), pipeline quality issues (ASR hallucinations, audio-video sync drift), and GPU resource management (memory fragmentation causing OOM despite available VRAM). Each pitfall has specific warning signs and prevention strategies that must be built into the architecture from day one.

1. **Silent CPU Fallback on AMD ROCm** - PyTorch silently falls back to CPU execution when ROCm kernels unavailable, causing 10-100x slowdown with no warning. Prevention: Explicit GPU verification utilities testing each component, assertions checking `tensor.device`, monitor GPU utilization (should be >80% during inference). Address in Phase 1 (Setup & Validation).

2. **AMD GPU Architecture Mismatch (gfx Targeting)** - PyTorch compiled for wrong gfx target causes failures or 20-30% performance degradation. Consumer GPUs (RX 7900 = gfx1100) differ from professional GPUs (MI200 = CDNA). Prevention: Verify GPU's gfx code with `rocminfo | grep gfx`, ensure PyTorch ROCm build includes your gfx target, set `HSA_OVERRIDE_GFX_VERSION` environment variable. Address in Phase 0 (Environment Setup).

3. **ASR Hallucination Cascade** - Whisper invents content not actually spoken (1 in 5 transcripts have 10%+ errors), errors cascade through translation and TTS, final video says things not in original. Prevention: Confidence scoring for transcription segments, audio preprocessing (noise reduction, music removal), set `--condition_on_previous_text False`, add human validation checkpoint for low-confidence segments. Address in Phase 2 (ASR Pipeline).

4. **Audio-Video Synchronization Drift** - Audio and video gradually lose sync over 20+ minute videos despite correct initial timestamps. Small rounding errors accumulate, TTS audio duration differs from original. Prevention: Consistent sample rates (48kHz), high-precision timestamps (float64), validate audio duration matches video duration, use FFmpeg `-async 1` flag, test with full 20-minute videos not just 30-second clips. Address in Phase 5 (Audio/Video Assembly).

5. **GPU Memory Fragmentation** - "CUDA out of memory" errors despite GB of free VRAM due to PyTorch memory allocator fragmentation. Prevention: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable, use gradient checkpointing, enable mixed precision with `torch.cuda.amp.autocast()`, explicitly call `.to("cpu")` before deleting models, call `torch.cuda.empty_cache()` after deletion. Address in Phase 1 (Setup & Validation).

6. **TTS Audio Quality Degradation** - Voice cloning produces robotic/distorted voices despite quality models, emotional nuance lost. Extremely sensitive to reference audio quality (background noise, MP3 compression, incorrect sample rate). Prevention: Use uncompressed WAV for reference audio (not MP3/AAC), ensure correct sample rate (22050Hz or 24000Hz model-specific), apply noise reduction, use 10-30 second reference clips, adjust temperature (0.65-0.75 sweet spot). Address in Phase 3 (Voice Cloning TTS).

7. **Uncanny Valley Lip Sync** - Technically correct mouth movements but dead-eyed, robotic, unnatural. Mouth moves but rest of face frozen, micro-expressions don't match speech. Models designed for avatars not dynamic real-world content. Prevention: Test with difficult sounds (p, b, m, w), verify facial stability over time, test multi-speaker scenarios, use 1080p+ high-quality source video, evaluate full 20-minute outputs not just 10-second clips. Address in Phase 4 (Lip Sync).

8. **"Open Source" Model Licensing** - Models claim Apache 2.0 but inherit LLaMA commercial restrictions, "personal use" build can't be shared without violating licenses. Prevention: Check LICENSE file in repo not just model card, verify licenses of base models if fine-tuned/derived, document license for every model component, prefer MIT/Apache 2.0 for personal/research use. Address in Phase 0 (Planning).

## Implications for Roadmap

Based on research, the roadmap should follow a 7-phase structure that builds foundation first, adds ML models incrementally, and defers lip sync (highest risk) until audio pipeline is proven. Each phase delivers working functionality and validates critical assumptions before adding complexity.

### Phase 0: Environment Setup & License Audit
**Rationale:** AMD ROCm environment is complex and must be correct before any model work. Wrong gfx target or PyTorch build causes subtle failures or massive performance degradation. License violations discovered late require rebuilding with different models.

**Delivers:** Verified ROCm 7.2 + PyTorch 2.9.1 installation, GPU detection utilities, license audit for all planned models, environment variable configuration (HSA_OVERRIDE_GFX_VERSION, PYTORCH_ROCM_ARCH).

**Addresses:** Pitfall #2 (AMD GPU architecture mismatch), Pitfall #8 (licensing violations).

**Avoids:** Silent CPU fallback by establishing GPU verification utilities early.

**Research needed:** NO - Well-documented in official ROCm docs.

---

### Phase 1: Foundation Pipeline (No ML Models)
**Rationale:** Validate toolchain (FFmpeg, Gradio, ROCm) without ML complexity. Establish project structure, storage patterns, and GPU memory management infrastructure. Can test with simple passthrough (upload → extract → merge → download).

**Delivers:** Project structure, FFmpeg video extractor/merger, storage manager with context managers, basic Gradio UI (upload, download, progress display), GPU memory monitoring utilities.

**Addresses:** Foundation for all future phases, establishes patterns for temporary file cleanup.

**Avoids:** Pitfall #5 (GPU memory fragmentation) by configuring PyTorch memory allocator early.

**Uses:** FFmpeg with AMD AMF/VAAPI, Gradio, tempfile module, PyTorch device management.

**Research needed:** NO - Standard patterns, well-documented.

---

### Phase 2: ASR Pipeline (First ML Model)
**Rationale:** Establishes model loading pattern with simplest ML task. Whisper is well-documented and robust. Tests GPU memory management with real model. ASR is foundation - everything depends on accurate transcription.

**Delivers:** Model manager base class with lazy loading, Whisper Large V3 integration, transcription with timestamps, confidence scoring, transcript editing UI, ASR quality validation with WER measurement.

**Addresses:** Must-have feature (automatic transcription), establishes model loading pattern for all future ML components.

**Avoids:** Pitfall #3 (ASR hallucination cascade) by implementing confidence scoring and validation checkpoints.

**Uses:** Whisper Large V3 via faster-whisper or Transformers, Hugging Face Transformers library, ModelManager singleton pattern.

**Research needed:** YES - faster-whisper ROCm custom build may need research, fallback to Transformers implementation.

---

### Phase 3: Translation Pipeline
**Rationale:** Builds on proven ASR output. SeamlessM4T v2 has native ROCm support via Transformers. Completing translation enables testing full text pipeline before adding voice complexity.

**Delivers:** SeamlessM4T v2 integration, translation to English (single target language for MVP), duration validation (translated text vs. source timing), terminology glossary support (CSV upload).

**Addresses:** Must-have feature (translation with cultural adaptation).

**Avoids:** Integration gotcha (duration mismatch between source and translation) by implementing validation early.

**Uses:** SeamlessM4T v2 via Hugging Face Transformers, sentencepiece tokenization.

**Research needed:** NO - Well-documented official Meta model with Transformers integration.

---

### Phase 4: Voice Cloning TTS Pipeline
**Rationale:** Most complex audio component. XTTS-v2 requires careful audio preprocessing and quality validation. Building complete audio pipeline (ASR → translate → TTS) enables testing quality before adding video complexity.

**Delivers:** XTTS-v2 integration with ROCm support (YellowRoseCx fork), voice cloning from 6-second reference samples, emotion preservation, audio preprocessing (noise reduction, format conversion to WAV 22050Hz), temperature tuning, audio quality validation (MOS scoring).

**Addresses:** Must-have feature (voice cloning with emotional preservation), core value proposition differentiator.

**Avoids:** Pitfall #6 (TTS audio quality degradation) by implementing audio preprocessing and quality validation.

**Uses:** XTTS-v2 (Coqui TTS via community ROCm fork), eSpeak NG for phoneme synthesis, scipy for audio I/O.

**Research needed:** YES - XTTS-v2 ROCm fork compatibility needs validation, may need dependency version pinning.

---

### Phase 5: Audio/Video Assembly & Sync
**Rationale:** Critical infrastructure before lip sync. Audio-video sync drift is subtle and only appears in long videos. Must test with full 20-minute content, not clips. Validates timestamp precision through entire pipeline.

**Delivers:** FFmpeg merge with explicit sync flags, timestamp validation at 5-minute intervals, high-precision timestamp calculations (float64), consistent sample rate enforcement (48kHz), duration matching verification.

**Addresses:** Must-have feature (HD output quality, audio quality preservation).

**Avoids:** Pitfall #4 (audio-video synchronization drift) by implementing validation checkpoints.

**Uses:** FFmpeg with `-async 1` flag, torchaudio for audio processing, OpenCV for frame manipulation.

**Research needed:** NO - Standard FFmpeg patterns, well-documented.

---

### Phase 6: Lip Sync Integration (Highest Risk)
**Rationale:** Lip sync is most complex and least documented component. Uncanny valley quality issues require careful evaluation. Having working audio pipeline first enables testing audio quality independently - lip sync failures easier to debug when audio is proven correct.

**Delivers:** Wav2Lip integration, lip synchronization with frame-perfect accuracy, facial stability validation, multi-speaker testing, quality evaluation on full 20-minute outputs.

**Addresses:** Must-have feature (lip synchronization), completes core value proposition.

**Avoids:** Pitfall #7 (uncanny valley lip sync) by testing with realistic scenarios (difficult sounds, full-length videos, multi-speaker).

**Uses:** Wav2Lip via PyTorch/ROCm, Pillow for image processing.

**Research needed:** YES - Wav2Lip ROCm compatibility needs validation, may need to evaluate alternative models (Wav2Lip-HD) if quality insufficient.

---

### Phase 7: Queue Management & Deployment
**Rationale:** With complete pipeline proven, add production features: async processing, batch jobs, error recovery, resource limits. Completes MVP with non-blocking UI and graceful handling of long-running tasks.

**Delivers:** RQ + Redis async queue, job status tracking, batch processing (queue up to 5 videos), priority management, cancellation support, error recovery with checkpointing, Gradio resource cleanup configuration.

**Addresses:** Should-have feature (batch processing with queue management).

**Avoids:** UX pitfalls (no progress indication, processing failure with no context) by implementing granular status updates.

**Uses:** RQ (Redis Queue), Redis 7.x, SQLite for job metadata, Gradio delete_cache parameter.

**Research needed:** NO - Standard queue patterns, well-documented.

---

### Phase Ordering Rationale

**Why this order:**
- **Environment setup first (Phase 0):** ROCm complexity and licensing issues discovered late cause catastrophic rework
- **Foundation before ML (Phase 1):** Validates toolchain and establishes patterns without ML complexity
- **ASR before translation (Phase 2-3):** Sequential dependency - can't translate without transcription, establishes model loading pattern
- **Complete audio pipeline before video (Phase 2-4):** Audio quality can be tested independently, cheaper to debug than video
- **Assembly before lip sync (Phase 5-6):** Sync validation infrastructure required before adding lip sync complexity
- **Lip sync second-to-last (Phase 6):** Highest risk component, benefits from proven audio pipeline for debugging
- **Queue management last (Phase 7):** Production feature that depends on complete working pipeline

**Why this grouping:**
- Phases 0-1 establish foundation without ML models (can test passthrough pipeline)
- Phases 2-4 build complete audio pipeline (ASR → translate → TTS) as cohesive unit
- Phases 5-6 add video integration (assembly → lip sync) building on audio foundation
- Phase 7 adds production features (queue, batch, deployment) on proven pipeline

**How this avoids pitfalls:**
- GPU verification utilities in Phase 0 prevent silent CPU fallback (Pitfall #1)
- ASR confidence scoring in Phase 2 prevents hallucination cascade (Pitfall #3)
- Audio preprocessing in Phase 4 prevents TTS quality degradation (Pitfall #6)
- Sync validation in Phase 5 prevents drift issues (Pitfall #4)
- Memory allocator config in Phase 1 prevents fragmentation OOM (Pitfall #5)
- Incremental build order reduces risk of late-stage architectural changes

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 2 (ASR):** faster-whisper ROCm custom build - community fork may need compatibility verification, fallback to Transformers if issues
- **Phase 4 (TTS):** XTTS-v2 ROCm fork (YellowRoseCx) - deprecated Coqui TTS with limited ROCm testing, may need dependency version pinning
- **Phase 6 (Lip Sync):** Wav2Lip ROCm compatibility - no official ROCm support documented, may need to evaluate alternatives (Wav2Lip-HD, LatentSync) if quality issues

**Phases with standard patterns (skip research-phase):**
- **Phase 0 (Environment):** Well-documented official ROCm installation process, standard PyTorch setup
- **Phase 1 (Foundation):** Standard FFmpeg, Gradio, tempfile patterns - no novel integration
- **Phase 3 (Translation):** SeamlessM4T v2 has official Hugging Face Transformers integration with native ROCm support
- **Phase 5 (Assembly):** Standard FFmpeg merge patterns, well-documented sync techniques
- **Phase 7 (Queue):** RQ + Redis is standard Python task queue pattern, mature ecosystem

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official ROCm documentation verified for PyTorch 2.9.1+rocm7.2, Hugging Face Transformers, FFmpeg. Community forks for faster-whisper and XTTS-v2 ROCm support documented with specific versions. |
| Features | HIGH | Verified with multiple current sources including ElevenLabs, HeyGen, CAMB.AI, Synthesia official documentation and 2026 industry analysis. Competitor feature comparison shows clear table stakes vs. differentiators. |
| Architecture | HIGH | Comprehensive review of 2026 documentation for all major components (Whisper, XTTS-v2, Wav2Lip, SeamlessM4T, ROCm, PyTorch, Gradio, FFmpeg). Standard patterns validated across multiple production implementations. |
| Pitfalls | HIGH | ROCm-specific pitfalls verified via official AMD documentation and PyTorch forums. ASR hallucination rates and lip sync quality issues documented in peer-reviewed research and industry reports. Memory management patterns confirmed in PyTorch official docs. |

**Overall confidence:** HIGH

All core technologies have official documentation or well-established community support. ROCm 7.2 + PyTorch 2.9.1 compatibility verified in official AMD docs. ML models (Whisper Large V3, SeamlessM4T v2, XTTS-v2, Wav2Lip) have 2026-current documentation with production usage examples. Community ROCm forks for faster-whisper and XTTS-v2 identified with specific repositories and versions.

Medium-confidence areas limited to community-maintained ROCm builds (faster-whisper, XTTS-v2) which may require version pinning or fallback to official Transformers implementations. These are flagged for research during respective phases.

### Gaps to Address

**faster-whisper ROCm compatibility:** Community fork (davidguttman/whisper-rocm) tested on ROCm 6.4.3 but may need updates for 7.2. Fallback plan: use official Transformers Whisper implementation which has verified ROCm support. Validate during Phase 2.

**XTTS-v2 ROCm stability:** YellowRoseCx/XTTS-WebUI-ROCm fork confirmed working on ROCm 6.3, but Coqui TTS deprecated with limited ongoing support. May require dependency version pinning. Consider alternative TTS models if quality issues emerge. Validate during Phase 4.

**Wav2Lip uncanny valley quality:** Research confirms "dead face" issue is inherent to current lip sync models designed for avatars not real-world content. No perfect solution exists. Mitigation: test with high-quality source video (1080p+), evaluate alternatives (Wav2Lip-HD, LatentSync if 24GB+ VRAM available). Accept as known limitation and communicate to users. Validate during Phase 6.

**GPU memory capacity:** Pipeline stages require sequential loading (Whisper Large V3 ~6GB, SeamlessM4T ~4GB, XTTS-v2 ~4GB, Wav2Lip ~2GB). 16GB VRAM supports one model at a time with LRU unloading. 8-12GB VRAM may need model quantization (FP16/INT8 via Optimum-AMD) or smaller model variants (Whisper Medium). Validate during Phase 1 with target GPU.

**Windows vs. Linux differences:** Research focused on Linux (VAAPI hardware acceleration, ROCm stability). Windows uses AMD AMF for hardware acceleration and has different ROCm installation process (requires AMD graphics driver 26.1.1+). Some community tools may be Linux-only. Document Windows-specific instructions if targeting Windows deployment.

## Sources

### Primary (HIGH confidence)
- **AMD ROCm Official Documentation** - PyTorch compatibility matrix, installation guides, ROCm 7.2.0 release notes, consumer GPU support
- **PyTorch for AMD ROCm** - Official blog posts on PyTorch 2.9 ROCm 7.2 support, GPU optimization guides
- **Hugging Face Official Docs** - Transformers library, Optimum-AMD, model documentation (Whisper Large V3, SeamlessM4T v2, XTTS-v2)
- **FFmpeg Official Documentation** - Codecs, AMD AMF hardware acceleration guide
- **Gradio Official Documentation** - Backend architecture, file access, resource cleanup guides
- **Meta AI Research** - SeamlessM4T v2 paper, Seamless Communication GitHub repository
- **OpenAI Whisper** - Whisper Large V3 model card, faster-whisper documentation
- **Industry Reports 2026** - AI dubbing software comparisons (CAMB.AI, Synthesia, ElevenLabs, HeyGen official documentation)

### Secondary (MEDIUM confidence)
- **Community ROCm Forks** - davidguttman/whisper-rocm (faster-whisper ROCm build), YellowRoseCx/XTTS-WebUI-ROCm (XTTS-v2 ROCm support)
- **Wav2Lip GitHub** - Original implementation by Rudrabha, integration guides, lip sync quality analysis
- **Technical Blogs** - AI dubbing pipeline architecture (RWS 2026 guide), video processing with AI (Forasoft 2025)
- **Framework Comparisons** - Gradio vs Streamlit 2025, Celery vs RQ task queues, VidGear video processing
- **PyTorch Performance** - GPU optimization guides, memory management patterns, OOM error handling (PyTorch FAQ)
- **Speaker Diarization** - AssemblyAI technical guides, multi-speaker video processing best practices

### Tertiary (LOW confidence)
- **Tutorial Articles** - AI dubbing tool reviews, best practices for voice cloning (needs validation during implementation)
- **GitHub Issues** - Gradio memory leak discussions, MoviePy memory issues (anecdotal, needs testing)
- **Community Forums** - PyTorch ROCm user experiences, troubleshooting discussions (informative but needs verification)

---
*Research completed: 2026-01-31*
*Ready for roadmap: yes*
