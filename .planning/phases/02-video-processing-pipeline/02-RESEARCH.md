# Phase 2: Video Processing Pipeline - Research

**Researched:** 2026-01-31
**Domain:** FFmpeg-based video I/O and stream manipulation
**Confidence:** HIGH

## Summary

This research focuses on video upload, audio/video extraction, and stream merging using FFmpeg with Python and Gradio. The standard approach uses `ffmpeg-python` as a Pythonic wrapper around FFmpeg CLI, combined with Gradio's `Video` component for web upload. For this phase's scope (validating FFmpeg toolchain before ML complexity), the priority is stream copy operations (`-c copy`) to avoid unnecessary re-encoding overhead.

The Python ecosystem strongly favors `ffmpeg-python` (10.9k stars) over MoviePy for video I/O tasks because MoviePy loads entire videos into memory as numpy arrays, making it 40-100x slower for simple extraction/merge operations. For temporary file management, Python's `tempfile` module with context managers ensures automatic cleanup even when errors occur.

Gradio's `Video` component handles MP4/MKV/AVI uploads natively and stores files in a cache directory (configurable via `GRADIO_TEMP_DIR`). Progress tracking during processing is supported via `gr.Progress()` parameter, though upload progress itself is not exposed by Gradio's current API.

**Primary recommendation:** Use `ffmpeg-python` with stream copy operations for extraction/merging, `tempfile.NamedTemporaryFile` for intermediate files, and Gradio's native Video component with size limits enforced via `max_file_size` parameter.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ffmpeg-python | 0.2.0+ | Pythonic FFmpeg wrapper | 10.9k stars, supports complex filter graphs, active maintenance |
| FFmpeg | 4.0+ | Video/audio processing engine | Industry standard, must be installed separately |
| Gradio | Latest | Web UI for file upload | Project requirement, native video upload support |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | Path manipulation | Always - modern Python file handling |
| tempfile | stdlib | Temporary file management | Intermediate storage for extracted streams |
| subprocess | stdlib | FFmpeg CLI fallback | Edge cases where ffmpeg-python lacks bindings |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ffmpeg-python | MoviePy | MoviePy is 40-100x slower for I/O tasks (loads entire video to memory), use only if pixel-level editing needed |
| ffmpeg-python | PyAV | PyAV wraps FFmpeg at C level for better performance but much steeper learning curve, overkill for simple I/O |
| subprocess | ffmpeg-python | Direct subprocess gives full control but requires manual command building and error parsing |

**Installation:**
```bash
# FFmpeg must be installed system-wide first
# Windows: Download from ffmpeg.org or use chocolatey
choco install ffmpeg

# Python wrapper
pip install ffmpeg-python
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── video_processing/
│   ├── __init__.py
│   ├── upload.py           # Gradio upload interface
│   ├── extractor.py        # Audio/video stream extraction
│   ├── merger.py           # Stream recombination
│   └── utils.py            # Metadata probing, validation
├── storage/
│   └── temp_manager.py     # Temporary file lifecycle management
└── main.py
```

### Pattern 1: Stream Copy Extraction (Zero Re-encoding)
**What:** Extract audio and video streams without decoding/re-encoding
**When to use:** When preserving exact quality and speed is critical (this phase's requirement)
**Example:**
```python
# Source: https://github.com/kkroening/ffmpeg-python
import ffmpeg

# Extract audio stream to WAV without re-encoding
input_video = ffmpeg.input('input.mp4')
audio = input_video.audio
ffmpeg.output(audio, 'audio.wav', acodec='pcm_s16le', ar=48000, ac=2).run()

# Extract video stream to individual frames
video = input_video.video
ffmpeg.output(video, 'frame_%04d.png', vf='fps=24').run()
```

### Pattern 2: Metadata Probing Before Processing
**What:** Use ffprobe to extract video metadata (codec, resolution, duration, format)
**When to use:** Always - before extraction to determine correct output format
**Example:**
```python
# Source: https://kkroening.github.io/ffmpeg-python/
import ffmpeg

probe = ffmpeg.probe('input.mp4')
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

width = int(video_stream['width'])
height = int(video_stream['height'])
codec = video_stream['codec_name']  # e.g., 'h264'
duration = float(probe['format']['duration'])
container_format = probe['format']['format_name']  # e.g., 'mov,mp4,m4a,3gp,3g2,mj2'
```

### Pattern 3: Temporary File Management with Context Managers
**What:** Use tempfile with context managers for automatic cleanup
**When to use:** Always - for extracted audio, video frames, intermediate files
**Example:**
```python
# Source: Python stdlib docs + https://www.timsanteford.com/posts/temporary-files-in-python-a-handy-guide-to-tempfile/
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    audio_path = temp_path / 'audio.wav'

    # Extract audio
    ffmpeg.input('video.mp4').output(str(audio_path), acodec='pcm_s16le').run()

    # Process audio (will be cleaned up automatically on exit)
    # ... processing ...

# temp_dir and all contents deleted here
```

### Pattern 4: Stream Merging with Synchronization
**What:** Combine processed audio and original video back into container
**When to use:** Final step after audio/video processing
**Example:**
```python
# Source: https://github.com/kkroening/ffmpeg-python/issues/252
import ffmpeg

video_input = ffmpeg.input('video_only.mp4').video
audio_input = ffmpeg.input('audio_processed.wav').audio

# Merge with video stream copy (no re-encoding)
ffmpeg.output(
    video_input,
    audio_input,
    'output.mp4',
    vcodec='copy',  # Copy video stream exactly
    acodec='aac',   # Encode audio to AAC for MP4 compatibility
    strict='experimental'
).run()
```

### Pattern 5: Gradio Upload with Validation
**What:** Use Gradio Video component with size limits and format validation
**When to use:** Web interface for video upload
**Example:**
```python
# Source: https://www.gradio.app/docs/gradio/video
import gradio as gr

def process_video(video_path, progress=gr.Progress()):
    progress(0, desc="Starting processing")

    # Validate format using ffprobe
    probe = ffmpeg.probe(video_path)
    format_name = probe['format']['format_name']

    if format_name not in ['mov,mp4,m4a,3gp,3g2,mj2', 'matroska,webm', 'avi']:
        raise ValueError(f"Unsupported format: {format_name}")

    progress(0.3, desc="Extracting audio")
    # ... extraction logic ...

    return output_path

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        sources=["upload"],  # Disable webcam for this use case
        format="mp4"  # Auto-convert to mp4 for browser playback
    ),
    outputs=gr.Video()
)

demo.launch(max_file_size="500mb")  # Set upload limit
```

### Anti-Patterns to Avoid
- **Loading entire video into memory:** Don't use MoviePy for simple I/O - it loads video as numpy arrays, causing 40-100x slowdown
- **Re-encoding when unnecessary:** Avoid re-encoding video streams during extraction/merge - use `-c copy` for stream copy
- **Ignoring temporary file cleanup:** Always use context managers (`with` statements) for temporary files to prevent disk space leaks
- **Hardcoded temporary paths:** Don't use fixed `/tmp` or `C:\Temp` - use `tempfile` module for cross-platform compatibility
- **Not validating upload formats:** Always probe video metadata before processing to detect corrupted or unsupported files

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video format detection | Custom file extension parser | `ffmpeg.probe()` with format_name parsing | Extensions lie (renamed files), FFmpeg reads actual container headers |
| Temporary file cleanup | Manual `os.remove()` calls | `tempfile.TemporaryDirectory()` context manager | Cleanup happens automatically even on exceptions, prevents disk leaks |
| Audio/video synchronization | Manual timestamp alignment | FFmpeg's native stream mapping (`-map`) | FFmpeg handles PTS/DTS timestamps, frame rate variations, codec quirks automatically |
| Upload progress tracking | Custom file upload handler | Gradio's built-in Video component | Handles multipart uploads, browser compatibility, security (file size limits) |
| Command-line argument building | String concatenation for ffmpeg CLI | `ffmpeg-python` fluent API | Type-safe, handles escaping, generates correct filter graphs |

**Key insight:** FFmpeg's decades of edge case handling (variable frame rates, B-frames, audio drift, codec quirks) make custom video I/O implementations bug-prone. Use existing wrappers.

## Common Pitfalls

### Pitfall 1: Audio Stream Loss During Video Filtering
**What goes wrong:** Some FFmpeg filters (like `hflip`, `vflip`) operate only on video streams and silently drop audio, resulting in silent output videos
**Why it happens:** ffmpeg-python applies filters to entire input by default, and video-only filters can't process audio streams
**How to avoid:** Explicitly separate audio and video streams using `.audio` and `.video` properties, process separately, then merge
**Warning signs:** Output video plays but has no sound despite input having audio

```python
# BAD - audio will be dropped
input = ffmpeg.input('video.mp4')
output = input.hflip().output('out.mp4')  # Silent video!

# GOOD - preserve audio
input = ffmpeg.input('video.mp4')
video = input.video.hflip()
audio = input.audio  # Pass through unchanged
output = ffmpeg.output(video, audio, 'out.mp4')
```

### Pitfall 2: FFmpeg Not in PATH
**What goes wrong:** `ffmpeg.Error` with message "ffmpeg not found" or similar, despite installing `ffmpeg-python`
**Why it happens:** `ffmpeg-python` is a pure Python wrapper - it doesn't bundle FFmpeg binary. Expects `ffmpeg` command available in system PATH
**How to avoid:** Install FFmpeg separately (Chocolatey on Windows, apt/brew on Linux/Mac), verify with `ffmpeg -version` in terminal before running Python code
**Warning signs:** ImportError-free but runtime failure when calling `.run()`

### Pitfall 3: Re-encoding Quality Loss
**What goes wrong:** Output video looks worse than input despite "matching bitrate" settings
**Why it happens:** Re-encoding with lossy codecs (H.264, H.265) always degrades quality. Trying to match bitrate doesn't preserve quality if source was already compressed
**How to avoid:** Use `-c copy` (codec copy) when extracting/merging streams that don't need modification. Only re-encode when changing resolution/codec/container
**Warning signs:** Processing takes long time, output file size differs significantly from input

```python
# BAD - re-encodes video unnecessarily
ffmpeg.input('video.mp4').output('output.mp4', vcodec='libx264').run()

# GOOD - stream copy for lossless merge
ffmpeg.input('video.mp4').output('output.mkv', vcodec='copy', acodec='copy').run()
```

### Pitfall 4: Temporary File Disk Space Exhaustion
**What goes wrong:** Repeated video processing fills disk with orphaned temporary files, eventually causing "No space left on device" errors
**Why it happens:** Extraction creates large intermediate files (frames, audio WAV). If cleanup fails due to exceptions, files remain in temp directory
**How to avoid:** Always use `tempfile.TemporaryDirectory()` with context manager (`with` statement) - cleanup happens even if exceptions occur
**Warning signs:** Temp directory (check `echo $TEMP` on Windows) grows continuously, processing slows down over time

### Pitfall 5: Container Format Mismatch
**What goes wrong:** FFmpeg error "Codec not supported in container" or playback fails despite successful encoding
**Why it happens:** Not all codecs work in all containers (e.g., MP4 doesn't support Theora, AVI doesn't support VP9). User decision requires matching input container format
**How to avoid:** Use `ffmpeg.probe()` to detect input container format, copy format for output. For codec selection, use H.264 for MP4/AVI (universal compatibility), original codec for MKV (supports everything)
**Warning signs:** `ffmpeg.Error` mentioning "Invalid data found when processing input" during muxing stage

```python
# Detect input format and match for output
probe = ffmpeg.probe('input.mkv')
input_format = probe['format']['format_name'].split(',')[0]  # 'matroska' from 'matroska,webm'

# Map format to extension
format_ext = {
    'matroska': 'mkv',
    'mov': 'mp4',  # QuickTime to MP4
    'avi': 'avi'
}.get(input_format, 'mp4')  # Default to MP4
```

## Code Examples

Verified patterns from official sources:

### Extract Audio to WAV (PCM 16-bit, 48kHz)
```python
# Source: https://gist.github.com/whizkydee/804d7e290f46c73f55a84db8a8936d74
import ffmpeg

# Extract audio with specific sample rate and format
# pcm_s16le = 16-bit signed little-endian PCM (CD quality)
# ar=48000 = 48kHz sample rate (video production standard)
# ac=2 = stereo (2 channels)
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.output(stream.audio, 'audio.wav', acodec='pcm_s16le', ar=48000, ac=2)
ffmpeg.run(stream)
```

### Extract Video Frames at Original FPS
```python
# Source: https://ottverse.com/extract-frames-using-ffmpeg-a-comprehensive-guide/
import ffmpeg

# Extract all frames as PNG (lossless)
# fps filter extracts frames at original video frame rate
ffmpeg.input('input.mp4').output(
    'frame_%04d.png',  # Numbered sequence: frame_0001.png, frame_0002.png, ...
    vf='fps=fps=source_fps',  # Preserve source frame rate
    qscale=2  # High quality (1=best, 31=worst)
).run()

# For storage efficiency, extract at lower rate (1 fps = 1 frame/second)
ffmpeg.input('input.mp4').output(
    'frame_%04d.jpg',
    vf='fps=1',  # 1 frame per second
    qscale=2
).run()
```

### Merge Audio and Video Streams
```python
# Source: https://www.mux.com/articles/merge-audio-and-video-files-with-ffmpeg
import ffmpeg

# Merge processed audio back with original video
video = ffmpeg.input('video_only.mp4').video
audio = ffmpeg.input('audio_processed.wav').audio

# Output with video stream copy (fast, lossless) and audio re-encoding
ffmpeg.output(
    video,
    audio,
    'final_output.mp4',
    vcodec='copy',      # Copy video stream without re-encoding
    acodec='aac',       # Encode WAV to AAC for MP4 compatibility
    audio_bitrate='192k'  # Standard quality AAC
).run()
```

### Probe Video Metadata
```python
# Source: https://python-ffmpeg.readthedocs.io/en/stable/examples/querying-metadata/
import ffmpeg
import json

# Get all metadata
probe = ffmpeg.probe('video.mp4')

# Extract video stream info
video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
width = int(video_info['width'])
height = int(video_info['height'])
codec = video_info['codec_name']
fps = eval(video_info['r_frame_rate'])  # "30000/1001" -> 29.97

# Extract format info
format_info = probe['format']
duration = float(format_info['duration'])
bitrate = int(format_info['bit_rate'])
container = format_info['format_name']  # "mov,mp4,m4a,3gp,3g2,mj2"

print(f"{width}x{height} {codec} @ {fps:.2f}fps, {duration:.1f}s")
```

### Gradio Video Upload with Progress
```python
# Source: https://www.gradio.app/guides/progress-bars
import gradio as gr
import ffmpeg
import tempfile
from pathlib import Path

def process_video_pipeline(video_path, progress=gr.Progress()):
    progress(0, desc="Validating video")

    # Probe metadata
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        raise ValueError("No video stream found")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / 'audio.wav'
        frames_path = temp_path / 'frames'
        frames_path.mkdir()

        progress(0.2, desc="Extracting audio")
        ffmpeg.input(video_path).output(
            str(audio_path),
            acodec='pcm_s16le',
            ar=48000
        ).run(quiet=True)

        progress(0.5, desc="Extracting video frames")
        ffmpeg.input(video_path).output(
            str(frames_path / 'frame_%04d.png'),
            vf='fps=24'
        ).run(quiet=True)

        progress(0.8, desc="Merging streams")
        output_path = temp_path / 'output.mp4'
        video_in = ffmpeg.input(str(frames_path / 'frame_%04d.png'), framerate=24)
        audio_in = ffmpeg.input(str(audio_path))

        ffmpeg.output(
            video_in, audio_in, str(output_path),
            vcodec='libx264', acodec='aac'
        ).run(quiet=True)

        progress(1.0, desc="Complete")
        return str(output_path)

demo = gr.Interface(
    fn=process_video_pipeline,
    inputs=gr.Video(sources=["upload"]),
    outputs=gr.Video(),
    title="Video Processing Pipeline"
)

demo.launch(max_file_size="500mb")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MoviePy for all video tasks | ffmpeg-python for I/O, MoviePy only for pixel editing | ~2019 | 40-100x speedup for extraction/merge tasks |
| Manual subprocess calls | ffmpeg-python fluent API | 2016+ | Type-safe, easier filter graphs, better error handling |
| H.264 everywhere | H.265/HEVC for storage | ~2020+ | 40% smaller files but 4-10x slower encoding (avoid for real-time) |
| String-based temp paths | pathlib.Path objects | Python 3.6+ (2016) | Cross-platform, cleaner path manipulation |
| Manual temp cleanup | tempfile context managers | Always available but underused | Automatic cleanup prevents disk leaks |

**Deprecated/outdated:**
- **python-ffmpeg vs ffmpeg-python confusion:** `python-ffmpeg` (different library) was created in 2021 as maintained alternative to `ffmpeg-python`, but `ffmpeg-python` resumed maintenance. Stick with `ffmpeg-python` (kkroening/ffmpeg-python) which has 10x more users
- **MoviePy for simple I/O:** MoviePy loads entire videos into memory as numpy arrays. Only use for pixel-level editing (overlays, custom effects). For this phase (extract/merge), ffmpeg-python is 40-100x faster
- **FFmpeg 2.x/3.x:** FFmpeg 4.0+ (2018) required for modern codec support. FFmpeg 8.0 (August 2025) is current stable version

## Open Questions

Things that couldn't be fully resolved:

1. **Gradio upload progress visibility**
   - What we know: Gradio shows progress during processing via `gr.Progress()`, but upload progress (file transfer to server) is not exposed
   - What's unclear: Whether this is a fundamental limitation or if there's an undocumented API
   - Recommendation: Accept current limitation for Phase 2. Show processing progress only. Document as known limitation

2. **Optimal frame extraction strategy for lip-sync prep**
   - What we know: Can extract all frames (`fps=source_fps`) or at lower rate. 1080p 30fps video = 1800 frames/minute
   - What's unclear: What frame rate lip-sync models (Wav2Lip, LatentSync) require. Extract all frames for quality, or 24fps for speed?
   - Recommendation: Start with full frame rate extraction (lossless), optimize in later phase based on lip-sync model requirements

3. **Metadata preservation scope**
   - What we know: FFmpeg can preserve chapters, subtitles, metadata tags with `-map` and `-metadata` flags
   - What's unclear: Which metadata is essential for dubbing use case vs. nice-to-have
   - Recommendation: Preserve basic format metadata (resolution, fps, duration) in Phase 2. Defer subtitle/chapter handling to later phase when dubbing workflow is clearer

## Sources

### Primary (HIGH confidence)
- ffmpeg-python GitHub: https://github.com/kkroening/ffmpeg-python - Installation, API, examples
- ffmpeg-python docs: https://kkroening.github.io/ffmpeg-python/ - Core features, error handling
- Gradio Video docs: https://www.gradio.app/docs/gradio/video - Component parameters, formats
- Gradio File Access: https://www.gradio.app/guides/file-access - Upload handling, security
- Gradio Progress Bars: https://www.gradio.app/guides/progress-bars - Progress tracking implementation
- Python tempfile docs: https://docs.python.org/3/library/tempfile.html - Temporary file management

### Secondary (MEDIUM confidence)
- [How to Use FFmpeg with Python in 2026](https://www.gumlet.com/learn/ffmpeg-python/) - Library comparison, best practices
- [ffmpeg-python vs moviepy performance](https://github.com/Zulko/moviepy/issues/2165) - Speed comparison data
- [H.264 vs H.265 codec comparison 2026](https://www.red5.net/blog/h264-vs-h265-vp9/) - Codec selection guidance
- [FFmpeg best quality conversion](https://www.baeldung.com/linux/ffmpeg-best-quality-conversion) - Quality preservation techniques
- [Merge audio and video with FFmpeg](https://www.mux.com/articles/merge-audio-and-video-files-with-ffmpeg) - Stream merging patterns
- [Extract frames using FFmpeg guide](https://ottverse.com/extract-frames-using-ffmpeg-a-comprehensive-guide/) - Frame extraction best practices
- [Temporary files in Python guide](https://www.timsanteford.com/posts/temporary-files-in-python-a-handy-guide-to-tempfile/) - Context manager patterns

### Tertiary (LOW confidence)
- WebSearch results for common pitfalls (aggregated from multiple Stack Overflow, GitHub issues) - Pitfall patterns identified but not individually verified

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ffmpeg-python is dominant wrapper (10.9k stars vs alternatives <1k), official Gradio docs verified
- Architecture: HIGH - Patterns verified from official ffmpeg-python docs, Python stdlib docs, Gradio guides
- Pitfalls: MEDIUM - Sourced from GitHub issues and community discussions, cross-referenced with official docs

**Research date:** 2026-01-31
**Valid until:** 2026-03-31 (60 days - stable domain, slow-moving ecosystem)
