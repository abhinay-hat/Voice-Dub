# Pitfalls Research

**Domain:** AI-Powered Video Dubbing with AMD GPU
**Researched:** 2026-01-31
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Silent CPU Fallback on AMD ROCm

**What goes wrong:**
PyTorch silently falls back to CPU execution when ROCm kernels are unavailable, causing massive performance degradation (10-100x slower) with no warning, logging, or error messages. Your 20-minute video dubbing pipeline that should take under 1 hour could take 10+ hours, and you won't know why.

**Why it happens:**
Many PyTorch operations lack full ROCm support. Functions like `torch.nn.functional.grid_sample`, sparse operations, bfloat16, xformers, and flash-attention are unavailable or partially supported under ROCm. When these operations are called, PyTorch quietly executes them on CPU instead of failing loudly.

**How to avoid:**
- Explicitly verify GPU execution for each pipeline component during setup phase
- Add assertions checking `tensor.device` for critical operations
- Monitor GPU utilization during pipeline runs (should be >80% during model inference)
- Test each model component individually with `torch.cuda.current_device()` checks
- Use ROCm-specific profiling tools to detect CPU fallback patterns

**Warning signs:**
- Pipeline runs much slower than benchmarks suggest
- GPU utilization (from `rocm-smi`) shows low percentage during inference
- CPU usage spikes during what should be GPU-bound operations
- Processing time doesn't scale linearly with video length

**Phase to address:**
Phase 1 (Setup & Validation) - Create GPU verification utilities that test each model component and fail loudly if CPU fallback is detected.

---

### Pitfall 2: AMD GPU Architecture Mismatch (gfx Targeting)

**What goes wrong:**
PyTorch compiled for wrong gfx target (AMD GPU architecture) causes either complete failure or severely degraded performance. Consumer GPUs like RX 7900 use RDNA architecture, but many ROCm builds target MI100/MI200 (CDNA) architectures, resulting in 20-30% of theoretical performance or runtime errors.

**Why it happens:**
ROCm was originally designed for AMD Instinct series (professional GPUs). Consumer GPU support is experimental. Pre-built PyTorch wheels may not include your specific gfx target (e.g., gfx1100 for RX 7900 XT). Libraries like hipBLAS and MIOpen aren't tuned for RDNA2/RDNA3 architectures.

**How to avoid:**
- Check your GPU's gfx code: `rocminfo | grep gfx`
- Verify PyTorch ROCm build includes your gfx target before installation
- Consider building PyTorch from source with correct `PYTORCH_ROCM_ARCH` flag
- Test hipBLAS operations for your specific GPU before building full pipeline
- Monitor for "unsupported architecture" warnings in logs

**Warning signs:**
- Installation succeeds but models fail to load
- Error messages mentioning "gfx" codes
- hipBLASLt falls back to slower hipBLAS
- Significantly slower performance than CUDA benchmarks suggest (>50% gap)

**Phase to address:**
Phase 0 (Environment Setup) - Verify GPU compatibility and compile/install correct ROCm+PyTorch stack before any model work.

---

### Pitfall 3: ASR Hallucination Cascade

**What goes wrong:**
Whisper and other ASR models "hallucinate" content not actually spoken in the audio (inventing phrases, repeating text, inserting random words). These errors cascade through translation and TTS, resulting in dubbed videos saying things that weren't in the original. Average Word Error Rate is 7.5%, but 1 in 5 transcripts have 10%+ errors.

**Why it happens:**
ASR models are trained on noisy web data with weak supervision. Background noise, music, overlapping speech, accents, and technical jargon trigger hallucinations. The models guess at unclear audio rather than admitting uncertainty.

**How to avoid:**
- Implement confidence scoring for transcription segments
- Use audio preprocessing: noise reduction, music removal, speaker diarization
- Set `--condition_on_previous_text False` for Whisper to reduce repetition hallucinations
- Compare multiple ASR models for critical segments
- Add human validation checkpoint for low-confidence segments
- Test with your target audio types (accented speech, noisy backgrounds) during setup

**Warning signs:**
- Transcription contains repeated phrases (sign of hallucination)
- Technical terms or names are wildly incorrect
- Translation seems nonsensical (may indicate source ASR error, not translation error)
- Silent/quiet sections have transcribed text
- Transcription length doesn't match video segment length

**Phase to address:**
Phase 2 (ASR Pipeline) - Build confidence scoring and validation checkpoints into ASR workflow before connecting to translation.

---

### Pitfall 4: Audio-Video Synchronization Drift

**What goes wrong:**
Audio and video gradually lose synchronization over the course of long videos (20+ minutes). Starts perfectly synced but by minute 15-20, lips are noticeably mismatched with audio. This happens even when initial segment timestamps are correct.

**Why it happens:**
Mismatch between video frame rate and audio sample rate causes cumulative drift. Video is typically 23.976/24/25/29.97/30 fps, audio is 44.1kHz or 48kHz. Small rounding errors in timestamp calculations accumulate. TTS-generated audio may have different duration than original speech. Timestamp correction applied after preprocessing doesn't fix lip sync errors created during format conversion.

**How to avoid:**
- Use consistent sample rates throughout pipeline (48kHz recommended)
- Calculate timestamps in high precision (float64, not float32)
- Validate audio duration matches video duration before lip sync
- Use FFmpeg with explicit `-async 1` flag for A/V sync correction
- Test with full-length videos (20+ min) during development, not just 30-second clips
- Implement periodic sync checkpoints that verify alignment every 5 minutes
- Consider frame-accurate timestamp preservation through entire pipeline

**Warning signs:**
- Sync is perfect at start but degrades over time
- Sync error proportional to video length (e.g., 0.5s at 10min, 1s at 20min)
- Different sample rates reported by `ffprobe` for audio streams
- Non-integer frame rates (23.976 instead of 24) combined with 44.1kHz audio

**Phase to address:**
Phase 5 (Audio/Video Assembly) - Implement timestamp validation and sync verification before and after video assembly.

---

### Pitfall 5: GPU Memory Fragmentation (OOM Despite Available VRAM)

**What goes wrong:**
"CUDA out of memory" errors occur even though `rocm-smi` shows GB of free VRAM. Pipeline that processes 5-minute videos successfully fails on 10-minute videos despite having enough total memory. Memory usage keeps increasing between batches even after deleting tensors.

**Why it happens:**
PyTorch memory allocator fragments GPU memory. Small allocations and deallocations create unusable gaps. Default allocator doesn't compact memory. Gradio interface holds references to previous inference outputs. Python garbage collector doesn't immediately free GPU tensors even after `del`.

**How to avoid:**
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable
- Use gradient checkpointing for memory-bound models (30-80% memory reduction)
- Enable mixed precision training/inference with `torch.cuda.amp.autocast()`
- Explicitly call `.to("cpu")` before deleting model objects
- Call `torch.cuda.empty_cache()` after `del` and `gc.collect()`
- Process videos in fixed-size chunks rather than variable-length segments
- Use Gradio's `delete_cache` parameter to periodically clean up

**Warning signs:**
- OOM errors with 2GB+ VRAM shown as free
- Memory usage increases over multiple pipeline runs
- First video processes successfully, second identical video fails
- `torch.cuda.memory_summary()` shows high fragmentation percentage

**Phase to address:**
Phase 1 (Setup & Validation) - Configure memory allocator settings and create memory management utilities before building pipeline.

---

### Pitfall 6: TTS Audio Quality Degrades Voice Cloning

**What goes wrong:**
Voice cloning produces robotic, unnatural, or distorted voices despite using quality models. Emotional nuance is lost. Output sounds flat and monotonic. Some phonemes sound garbled or wrong.

**Why it happens:**
Voice cloning quality is extremely sensitive to reference audio quality. Background noise, compression artifacts (MP3), incorrect sample rates, or clipped audio degrade results. Most models need "professional-quality" reference audio (clean, 22050Hz WAV, no background noise) but documentation doesn't specify this upfront. Audio tokens are lossy - can't perfectly reconstruct audio. Diffusion-based TTS requires many denoising steps, making it slow.

**How to avoid:**
- Use uncompressed WAV format for reference audio (not MP3/AAC)
- Ensure 22050Hz or 24000Hz sample rate (model-specific)
- Apply noise reduction to reference audio before voice cloning
- Use longer reference clips (10-30 seconds) for better quality
- Adjust temperature settings (0.65-0.75 typical sweet spot)
- Test voice cloning quality early with your actual audio types
- Consider using multiple reference clips and averaging embeddings

**Warning signs:**
- Cloned voice sounds robotic despite good model
- Quality varies significantly between runs with same reference
- Reference audio has background noise or echo
- Reference audio is compressed format (MP3, AAC)
- Output audio has artifacts or distortion

**Phase to address:**
Phase 3 (Voice Cloning TTS) - Build audio preprocessing and quality validation before implementing voice cloning pipeline.

---

### Pitfall 7: Lip Sync AI Treats Mouth Movement as Isolated Action

**What goes wrong:**
Lip-synced videos look "uncanny valley" - technically correct mouth movements but dead-eyed, robotic, unnatural. Mouth moves but rest of face is frozen. Micro-expressions don't match speech. Multiple speakers in frame causes breakdown (mouth sync only works for one person).

**Why it happens:**
Most lip sync models were designed for avatars (static, predictable shots), not dynamic real-world content. They treat speech as isolated mouth movement, ignoring that lip movements are connected to facial muscle shifts, eye blinks, head movement, and micro-expressions. Models prioritize speed over accuracy. Complex backgrounds or low-quality video reduce accuracy.

**How to avoid:**
- Test lip sync quality on difficult sounds (p, b, m, w sounds)
- Verify facial stability over time (not just per-frame accuracy)
- Test with multi-speaker scenarios if your content has them
- Use high-quality source video (1080p+, good lighting)
- Consider post-processing to blend lip sync with original facial movement
- Evaluate full 20-minute outputs, not just 10-second test clips
- Check if model supports your video characteristics (background complexity, camera movement)

**Warning signs:**
- Mouth moves correctly but face looks "dead"
- Lip sync accuracy degrades with multiple speakers in frame
- Sync is good for primary speaker but non-existent for background speakers
- Quality degrades during fast speech or emotional moments
- Model outputs require significant manual correction

**Phase to address:**
Phase 4 (Lip Sync) - Evaluate multiple lip sync models with realistic test cases before committing to architecture.

---

### Pitfall 8: "Open Source" Model Licensing Gotchas

**What goes wrong:**
You build entire pipeline using "open source" models, then discover commercial use restrictions, derivative work limitations, or mandatory branding requirements. LLaMA-based models claim Apache 2.0 license but inherit LLaMA's commercial restrictions. "Personal use" build suddenly can't be shared with friends or deployed online without violating licenses.

**Why it happens:**
Many popular models use "open weights" not truly open source. LLaMA license restricts commercial use if you have >700M monthly active users and requires "Built with Llama" branding. Derivative models inherit upstream license restrictions even if their model card shows permissive license. License applies to training data, weights, and output - not always clear which matters.

**How to avoid:**
- Check LICENSE file in model repository, not just model card
- Verify licenses of base models (if fine-tuned/derived)
- Understand MIT/Apache 2.0 come "as-is" with no warranties
- Check for geographic restrictions (some models blocked in EU)
- Document license for every model component in your pipeline
- Prefer models with clear MIT/Apache 2.0 for personal/research use
- Understand "non-commercial" vs "personal use" vs "research" distinctions

**Warning signs:**
- Model card says Apache 2.0 but README mentions LLaMA
- License includes user count thresholds or branding requirements
- Base model has different license than fine-tuned model
- "Non-commercial" license used but definition unclear
- No LICENSE file, only documentation claims

**Phase to address:**
Phase 0 (Planning) - Audit all model licenses before architecture decisions to avoid rebuilding pipeline with different models later.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using pre-installed PyTorch instead of ROCm-specific build | Faster setup, avoid compilation | Silent CPU fallback, terrible performance | Never for AMD GPUs - rebuild is inevitable |
| Skipping audio preprocessing (noise reduction) | Faster pipeline, less code | Poor ASR accuracy, hallucinations cascade through pipeline | Only for clean studio audio with controlled environment |
| Processing entire video in one pass | Simpler code, no chunking logic | OOM errors on long videos, can't resume partial failures | Only if videos guaranteed <5 minutes and memory validated |
| Using MP3/AAC for intermediate audio | Smaller file sizes | Cumulative quality degradation through pipeline stages | Never - storage is cheap, quality matters |
| Single ASR model without validation | Faster inference, simpler code | Hallucinations go undetected until final output | Only if manual review of all outputs |
| Fixed batch size for all video lengths | Simple code | Memory waste on short videos, OOM on long videos | MVP only - implement dynamic batching by Phase 3 |
| Gradio default settings without resource cleanup | Easy setup, works initially | Memory leaks over multiple runs, unstable server | Development only - production needs cleanup config |

## Integration Gotchas

Common mistakes when connecting pipeline components.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ASR → Translation | Passing raw transcript with timestamps | Strip timestamps, pass plain text to translation; reconstruct timing after |
| Translation → TTS | Assuming translation has same duration as source | Validate output duration, implement time-stretching if needed for lip sync |
| TTS → Lip Sync | Using different sample rates (TTS=22kHz, Video=48kHz) | Standardize on 48kHz throughout pipeline; resample TTS output before video merge |
| Video Decode → Processing | Loading entire video into RAM | Use frame-by-frame streaming with FFmpeg pipe |
| Model → Model | Keeping previous model on GPU when loading next | Move to CPU, delete, garbage collect before loading next model |
| Python → FFmpeg | Using shell=True with os.system() | Use subprocess with explicit args for security and error handling |
| ROCm → PyTorch | Assuming CUDA code works with `s/cuda/rocm/` | Test each CUDA-specific operation; many require ROCm-specific implementations |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading all models on GPU simultaneously | Fast model switching, no reload time | Each model ~2-4GB VRAM; 4 models = 8-16GB, exceeds consumer GPU | >2 models on 8GB GPU, >3 on 16GB |
| Synchronous pipeline (wait for each stage) | Simple sequential code | Each stage blocks next; 20min video * 4 stages = 80min total | Any production use; prevents parallelization |
| No progress tracking or resumability | Simpler code | 45min video processing crashes at minute 40, must restart from scratch | Videos >10 minutes or unreliable GPU stability |
| Video loaded entirely into numpy array | Simple indexing and manipulation | 1080p 20min video = 60GB+ RAM | Videos >5 minutes at 1080p |
| Batch size=1 for all operations | Safe, never OOM | 10-50x slower than optimal batching | Throughput matters (multiple videos) |
| CPU-GPU transfer for every frame | Conceptually clean separation | Transfer overhead dominates; 1080p frame = 6MB, 30fps = 180MB/sec | Real-time or near-real-time processing |
| No timeout on FFmpeg operations | Assume commands complete | Corrupted video causes FFmpeg hang, blocks entire pipeline indefinitely | Production with user-uploaded videos |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Executing FFmpeg with unsanitized file paths | Command injection via specially crafted filenames | Use subprocess with arg lists, never shell=True; validate filenames |
| Loading user-provided models without validation | Malicious pickle files execute arbitrary code | Only load models from trusted sources; use safe_load or convert to safetensors |
| No resource limits on Gradio interface | Single user submits 2-hour video, exhausts GPU/RAM for all users | Set max video length, timeout limits, concurrent request limits |
| Storing temporary files in predictable locations | Race conditions, data leakage between users | Use `tempfile.mkdtemp()` with random names, cleanup after processing |
| No validation on uploaded video codecs | Malicious video exploits FFmpeg vulnerabilities | Validate video format, re-encode all uploads with known-safe codec |
| Processing videos in-place without copies | Corrupted source file if crash during processing | Work on copies, only replace source on successful completion |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No progress indication for long operations | User sees "Processing..." for 45 minutes, assumes it crashed | Granular progress: "ASR: 45%, Translation: 0%, TTS: 0%, Lip Sync: 0%" |
| Processing failure after 40 minutes with no context | User has no idea what went wrong or how to fix | Checkpointing: save intermediate results, show specific failure point |
| No quality preview before full processing | User waits 1 hour only to find voice quality unacceptable | Quick 10-second preview of voice clone before processing full video |
| Accepting any video format without guidance | User uploads incompatible format, sees cryptic FFmpeg error | Pre-flight validation: show format details, convert if needed, clear error messages |
| No estimate of processing time | User doesn't know if 5 minutes or 5 hours | Estimate based on video length: "~45 minutes for 20-minute video" |
| Memory errors crash entire interface | Gradio server dies, user loses all state, must restart | Graceful degradation: catch OOM, suggest shorter video or lower quality |
| Silent failures in intermediate steps | Final output has mismatched audio, user doesn't know which stage failed | Validation checkpoints: verify ASR quality, translation, TTS before proceeding |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **ASR Pipeline:** Working transcription tested, but no validation for hallucinations - verify with known-content test video and measure WER
- [ ] **Translation:** Translates correctly, but duration mismatches original - verify translated audio duration within 10% of source
- [ ] **Voice Cloning:** Sounds like target voice in 10-second test, but quality degrades in full video - verify on full 20-minute content, not just clips
- [ ] **Lip Sync:** Mouths move, but facial expressions frozen - verify natural micro-expressions, not just mouth movement accuracy
- [ ] **ROCm Setup:** Models load and run, but CPU fallback happening silently - verify GPU utilization >80% during inference
- [ ] **Memory Management:** First video processes successfully, but second OOM - verify memory cleanup with 5+ consecutive runs
- [ ] **Audio/Video Sync:** Perfect sync in first minute, but drift develops - verify sync at 5min, 10min, 15min, 20min marks
- [ ] **Gradio Interface:** Works in development, but memory leaks in deployment - verify resource cleanup with delete_cache configured
- [ ] **Error Handling:** Happy path works, but crashes on corrupted videos - verify graceful handling of malformed inputs
- [ ] **License Compliance:** Models work, but license audit missing - verify each model's LICENSE file, not just model card claims

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Silent CPU fallback | LOW | Add device assertions, test single component, find which lacks ROCm support, swap model or wait for ROCm update |
| Wrong gfx target | MEDIUM | Rebuild PyTorch from source with correct PYTORCH_ROCM_ARCH (2-4 hours build time) |
| ASR hallucinations | LOW-MEDIUM | Implement post-processing filter (detect repeated phrases), or add human validation step |
| A/V sync drift | MEDIUM | Recalculate all timestamps with higher precision, re-encode with explicit sync flags |
| GPU OOM | LOW | Enable memory config flags (PYTORCH_CUDA_ALLOC_CONF), reduce batch size, use gradient checkpointing |
| Bad voice quality | LOW | Preprocess reference audio (noise reduction), adjust temperature, try longer reference clip |
| Uncanny lip sync | HIGH | Switch to different lip sync model (may require architecture changes), or accept limitation |
| License violation | HIGH | Replace violating model with compatible alternative (may require retraining/fine-tuning) |
| Memory leak in Gradio | LOW | Configure delete_cache, add explicit cleanup, restart server periodically |
| FFmpeg codec incompatibility | LOW | Add re-encoding step to normalize all inputs to H.264 MP4 |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent CPU fallback | Phase 1: Setup & Validation | GPU utilization >80% during test inference |
| AMD GPU architecture mismatch | Phase 0: Environment Setup | hipBLAS test passes for specific gfx code |
| ASR hallucination cascade | Phase 2: ASR Pipeline | WER <5% on test dataset with ground truth |
| Audio-video sync drift | Phase 5: Audio/Video Assembly | Sync validated at 5min intervals in 20min video |
| GPU memory fragmentation | Phase 1: Setup & Validation | 5 consecutive 20min video runs without OOM |
| TTS audio quality degradation | Phase 3: Voice Cloning TTS | MOS score >4.0 on sample outputs |
| Uncanny lip sync | Phase 4: Lip Sync | Manual review: natural facial movement rated acceptable |
| License violations | Phase 0: Planning | All model licenses documented, audit complete |
| Video processing memory issues | Phase 5: Audio/Video Assembly | 20min 1080p video processes with <8GB RAM |
| Model pipeline integration | Phase 6: Integration | Full pipeline runs end-to-end on 20min test video |
| Gradio memory leaks | Phase 7: Interface & Deployment | Interface stable after 10+ consecutive runs |

## Sources

**ROCm and AMD GPU Challenges:**
- [PyTorch on ROCm installation - ROCm Documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html)
- [Current state of PyTorch + ROCm - PyTorch Forums](https://discuss.pytorch.org/t/current-state-of-pytorch-rocm/223926)
- [GPU Software for AI: CUDA vs. ROCm in 2026](https://research.aimultiple.com/cuda-vs-rocm/)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [hipBLASLt falling back on gfx1151 - ROCm Issue #5643](https://github.com/ROCm/ROCm/issues/5643)

**AI Dubbing Pipeline Challenges:**
- [AI dubbing in 2026: complete guide - RWS](https://www.rws.com/blog/ai-dubbing-in-2026/)
- [AI Dubbing Limitations & Solutions - 3Play Media](https://www.3playmedia.com/blog/ai-dubbing-limitations-solutions/)
- [Safety Concerns of AI Dubbing - Sieve Blog](https://www.sieve.ai/blog/ai-dubbing-safety-concerns)

**Voice Cloning and TTS Issues:**
- [Qwen3-TTS: Complete 2026 Guide - DEV Community](https://dev.to/czmilo/qwen3-tts-the-complete-2026-guide-to-open-source-voice-cloning-and-ai-speech-generation-1in6)
- [Answers to TTS Problems - Murf.ai](https://murf.ai/blog/text-to-speech-voice-generation-common-issues-and-solutions)
- [Best Open Source AI Voice Cloning Tools 2026 - Resemble AI](https://www.resemble.ai/best-open-source-ai-voice-cloning-tools/)

**Lip Sync Accuracy Problems:**
- [AI Lip Sync: Why Most Tools Fall Short - LipDub AI](https://www.lipdub.ai/blogs/ai-lip-sync)
- [Best AI Lip Sync Tools for 2026 - Barchart](https://www.barchart.com/story/news/37179827/best-ai-lip-sync-tools-for-2026-how-to-choose-the-right-tool)

**ASR and Whisper Challenges:**
- [Whisper OpenAI challenges - Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/2202587/what-is-whisper-openai-challenges-and-security-cha)
- [How Big of a Deal Is 'Whisper' for ASR - Slator](https://slator.com/how-big-a-deal-is-whisper-for-asr-multilingual-transcription/)
- [Whisper-LM: Improving ASR for Low-Resource Languages](https://arxiv.org/html/2503.23542v1)

**Video Processing and Memory Management:**
- [MoviePy memory issues - GitHub Issue #1892](https://github.com/Zulko/moviepy/issues/1892)
- [Python mmap: Improved File I/O - Real Python](https://realpython.com/python-mmap/)
- [FFmpeg Codecs Documentation](https://ffmpeg.org/ffmpeg-codecs.html)

**PyTorch Performance and GPU Memory:**
- [7 PyTorch Memory Tricks That Stop GPU OOM - Medium](https://python.plainenglish.io/7-pytorch-memory-tricks-that-stop-gpu-oom-crashes-every-ml-engineer-needs-6873585cdd2c)
- [Solving Bottlenecks on Data Input Pipeline - Towards Data Science](https://towardsdatascience.com/solving-bottlenecks-on-the-data-input-pipeline-with-pytorch-profiler-and-tensorboard-5dced134dbe9/)
- [PyTorch GPU Optimization Guide - Medium](https://medium.com/@ishita.verma178/pytorch-gpu-optimization-step-by-step-guide-9dead5164ca2)

**Model Licensing:**
- [Licensing Machine Learning models - The Turing Way](https://book.the-turing-way.org/reproducible-research/licensing/licensing-ml/)
- [Open Source Licensing in LLMs - Medium](https://medium.com/@adnanmasood/open-source-licensing-modalities-in-large-language-models-insights-risks-and-opportunities-for-283416b2a40d)
- [Understanding AI Licenses - Viso.ai](https://viso.ai/deep-learning/ai-licenses/)

**Gradio Memory Leaks:**
- [Potential Memory Leak in Gradio - Issue #11602](https://github.com/gradio-app/gradio/issues/11602)
- [Gradio GPU memory release issues - Issue #6971](https://github.com/gradio-app/gradio/issues/6971)
- [Resource Cleanup - Gradio Guides](https://www.gradio.app/guides/resource-cleanup)

**Audio/Video Synchronization:**
- [Audio-to-video synchronization - Wikipedia](https://en.wikipedia.org/wiki/Audio-to-video_synchronization)
- [Timestamp Drifting in Live Capture - LEADTOOLS](https://www.leadtools.com/help/sdk/multimedia/filters/timestamp-drifting-in-live-capture-situations.html)
- [How to Match Video and Audio - LinkedIn](https://www.linkedin.com/advice/0/how-do-you-deal-audio-drift-long-video-projects)

---
*Pitfalls research for: AI-Powered Video Dubbing with AMD GPU*
*Researched: 2026-01-31*
