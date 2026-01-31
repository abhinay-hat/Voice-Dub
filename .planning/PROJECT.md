# Voice Dubbing Tool

## What This Is

A local video dubbing application that takes videos in any language and generates English-dubbed versions with voice cloning, emotion preservation, and lip synchronization. Users upload videos through a web interface, and the system processes them entirely on local hardware using open-source AI models, returning fully dubbed videos with the original speaker's voice characteristics preserved in English.

## Core Value

Watch any video content in English while preserving the original speaker's voice characteristics and emotional expression, without relying on cloud services or paying API fees.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] User can upload video files through web interface
- [ ] System transcribes speech from any language to text
- [ ] System translates transcribed text to English with context preservation
- [ ] System generates English audio that clones original speaker's voice and emotion
- [ ] System synchronizes video lip movements to English audio
- [ ] User can download complete dubbed video file
- [ ] Processing completes in under 1 hour for 20-minute videos
- [ ] Translation accuracy is high (preserves meaning and context)
- [ ] Voice cloning accuracy is high (recognizably similar to original speaker)
- [ ] System runs entirely on local AMD GPU hardware
- [ ] Multiple friends can access and use the web interface

### Out of Scope

- Public release or commercial deployment — Non-commercial use only (personal + friends)
- Real-time or streaming dubbing — Batch processing is acceptable
- Perfect Hollywood-quality results — "Watchable" quality with high accuracy for voice/translation is sufficient
- Multi-user authentication and accounts — Simple shared access for friends is fine
- Cloud deployment or API services — Local-only processing required
- NVIDIA-specific optimizations — Must work on AMD GPU with ROCm

## Context

**Motivation:** User wants to watch anime and other video content in English with better dubbing than currently available. Existing solutions either don't exist, require paid APIs, or fail to preserve emotion and voice characteristics.

**Use Case:** Personal entertainment (watching anime primarily, but supporting any video content) and sharing with friends who want the same capability.

**Technical Environment:** AMD GPU available (not NVIDIA), requiring ROCm support for all models. PyTorch-based ML pipeline running locally.

**Quality Expectations:** High accuracy for translation and voice cloning is critical. Lip sync and speech recognition can be "good enough" but translation must preserve meaning/context and voice cloning must sound recognizably similar to original speaker.

## Constraints

- **Hardware**: AMD GPU (ROCm 7.x required) — Limits some model choices optimized for NVIDIA CUDA
- **Processing Time**: Under 1 hour for 20-minute video — Constrains model size and complexity
- **Privacy**: Local-only processing, no external API calls — All models must run on-device
- **Licensing**: Non-commercial licenses acceptable — Personal + friends use only, not public release
- **Tech Stack**: Open source models only (Hugging Face, GitHub, etc.) — No proprietary or paid models
- **Languages**: Must support any input language — Universal language support required, not just Japanese

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| XTTS-v2 for voice cloning | Best open-source voice cloning with emotion preservation; non-commercial license acceptable for friends-only use | — Pending |
| Meta Seamless for translation | Preserves vocal expression during translation, critical for maintaining emotional context | — Pending |
| Whisper Large V3 for speech-to-text | 99+ language support, high accuracy, proven ROCm compatibility | — Pending |
| Gradio for web UI | Simplest framework for friends to use, good for rapid prototyping | — Pending |
| PyTorch + ROCm stack | Required for AMD GPU support; most ML models built on PyTorch | — Pending |
| Wav2Lip HD or LatentSync for lip sync | Good quality with acceptable processing speed; newer than original Wav2Lip | — Pending |

---
*Last updated: 2026-01-31 after initialization*
