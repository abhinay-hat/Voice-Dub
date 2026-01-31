# Phase 4: Translation Pipeline - Research

**Researched:** 2026-01-31
**Domain:** Neural machine translation with speech synthesis constraints
**Confidence:** HIGH

## Summary

Phase 4 implements translation from any source language to English using Meta's SeamlessM4T v2 Large model (2.3B parameters), with critical requirements for context preservation, multi-speaker handling, and duration constraints (within 10% tolerance) for downstream voice cloning and lip sync. The research confirms SeamlessM4T v2 is well-suited for this task with 96 text languages, ~6GB VRAM footprint, and native support for beam search candidate generation.

The standard approach involves: (1) batching full transcript context through SeamlessM4T's text-to-text model, (2) generating 2-3 translation candidates per segment using beam search (num_beams=2-3), (3) ranking candidates by duration fit and model confidence, and (4) flagging low-confidence segments (<70% threshold) for Phase 8 review. For long videos (>1024 tokens), chunk with 128-256 token overlap to preserve cross-boundary context.

Duration estimation uses character-count heuristics (English: ~14-16 chars/second conversational speech) with explicit duration modeling inspired by recent TTS research. The system must validate translated segment duration against original ±10% tolerance before accepting translations.

**Primary recommendation:** Use SeamlessM4T v2 ForTextToText dedicated model with full-context batching, beam search candidate generation (num_beams=3), and character-count duration estimation with explicit validation.

## Standard Stack

The established libraries/tools for neural machine translation with duration awareness:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | ≥5.0.0 | SeamlessM4T v2 model access | Official Hugging Face implementation, actively maintained |
| torch | ≥2.0.0 (nightly with CUDA 12.8) | PyTorch backend | Required for RTX 5090 sm_120 compute capability |
| sentencepiece | latest | Tokenization for SeamlessM4T | Required dependency for SeamlessM4T processor |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | ≥1.24.0 | Array operations for duration calculations | Character counting, duration math |
| scipy | ≥1.10.0 | Statistical analysis for confidence scoring | Optional: advanced ranking algorithms |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SeamlessM4T v2 Large (2.3B) | SeamlessM4T v2 Medium (1.2B) | Faster inference but lower quality |
| Dedicated ForTextToText model | Unified SeamlessM4Tv2Model | More memory overhead, unnecessary speech components |
| Full-context batching | Sliding window | Loses cross-segment context, worse pronoun handling |

**Installation:**
```bash
# Already in requirements.txt, but explicit command:
pip install transformers>=5.0.0 sentencepiece
# PyTorch nightly already installed for GPU validation
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── stages/
│   └── translation/
│       ├── translator.py          # SeamlessM4T wrapper
│       ├── candidate_ranker.py    # Multi-candidate ranking logic
│       ├── duration_validator.py  # Duration constraint checking
│       └── context_chunker.py     # Long transcript chunking
├── models/
│   └── model_manager.py           # Existing ModelManager (reuse)
└── config/
    └── settings.py                # Add translation config constants
```

### Pattern 1: Full-Context Translation with Candidate Generation
**What:** Translate entire transcript at once (or in overlapping chunks) while generating multiple candidates per segment for duration optimization.

**When to use:** Standard workflow for videos under ~15 minutes (transcript <1024 tokens).

**Example:**
```python
# Source: Hugging Face SeamlessM4T v2 docs + beam search research
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

# Full transcript context (preserves cross-segment references)
full_transcript = "\n".join([seg["text"] for seg in transcript_segments])

# Process with source language
text_inputs = processor(text=full_transcript, src_lang="jpn", return_tensors="pt")

# Generate 3 candidates using beam search
output_tokens = model.generate(
    **text_inputs,
    tgt_lang="eng",
    num_beams=3,              # Generate 3 candidates
    num_return_sequences=3,   # Return all 3
    return_dict_in_generate=True,
    output_scores=True        # Get confidence scores
)

# Decode all candidates
candidates = [
    processor.decode(tokens, skip_special_tokens=True)
    for tokens in output_tokens.sequences
]
```

### Pattern 2: Duration-Aware Candidate Ranking
**What:** Rank multiple translation candidates based on weighted score of model confidence and duration fit.

**When to use:** After generating multiple candidates per segment.

**Example:**
```python
# Source: Machine translation quality estimation research + video dubbing paper
def rank_candidates(candidates, scores, original_duration, target_chars_per_sec=15):
    """
    Rank translation candidates by confidence and duration fit.

    Args:
        candidates: List of translation strings
        scores: Model confidence scores for each candidate
        original_duration: Original segment duration in seconds
        target_chars_per_sec: English speech rate (14-16 typical)

    Returns:
        Best candidate and metadata
    """
    ranked = []

    for candidate, score in zip(candidates, scores):
        # Estimate duration using character count heuristic
        estimated_duration = len(candidate) / target_chars_per_sec

        # Duration fit score (1.0 = perfect, 0.0 = outside ±10%)
        duration_ratio = estimated_duration / original_duration
        if 0.9 <= duration_ratio <= 1.1:
            duration_score = 1.0 - abs(1.0 - duration_ratio) * 10  # Linear penalty
        else:
            duration_score = 0.0  # Outside tolerance

        # Weighted ranking (60% confidence, 40% duration fit)
        combined_score = (score * 0.6) + (duration_score * 0.4)

        ranked.append({
            "candidate": candidate,
            "model_confidence": score,
            "duration_score": duration_score,
            "combined_score": combined_score,
            "estimated_duration": estimated_duration,
            "duration_ratio": duration_ratio
        })

    # Sort by combined score
    ranked.sort(key=lambda x: x["combined_score"], reverse=True)

    return ranked[0], ranked  # Best candidate + all rankings
```

### Pattern 3: Overlapping Chunk Translation for Long Videos
**What:** Split long transcripts into overlapping chunks to stay within model limits while preserving context.

**When to use:** Videos >15 minutes where full transcript exceeds ~1024 tokens.

**Example:**
```python
# Source: Document chunking research (late chunking, context preservation)
def chunk_transcript_with_overlap(segments, max_tokens=1024, overlap_tokens=128):
    """
    Chunk transcript segments with overlap to preserve context.

    Args:
        segments: List of transcript segments with text and timestamps
        max_tokens: Maximum tokens per chunk (~1024 for SeamlessM4T)
        overlap_tokens: Overlap size to preserve context (128-256 recommended)

    Returns:
        List of chunks with segment indices
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for idx, segment in enumerate(segments):
        # Approximate tokenization (1 token ≈ 4 characters)
        segment_tokens = len(segment["text"]) // 4

        if current_tokens + segment_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append({
                "segments": current_chunk.copy(),
                "start_idx": current_chunk[0]["idx"],
                "end_idx": current_chunk[-1]["idx"]
            })

            # Start new chunk with overlap
            overlap_segs = [s for s in current_chunk if current_tokens - s["approx_tokens"] < overlap_tokens]
            current_chunk = overlap_segs
            current_tokens = sum(s["approx_tokens"] for s in current_chunk)

        segment["idx"] = idx
        segment["approx_tokens"] = segment_tokens
        current_chunk.append(segment)
        current_tokens += segment_tokens

    # Add final chunk
    if current_chunk:
        chunks.append({
            "segments": current_chunk,
            "start_idx": current_chunk[0]["idx"],
            "end_idx": current_chunk[-1]["idx"]
        })

    return chunks
```

### Anti-Patterns to Avoid
- **Isolated segment translation:** Translating each segment independently loses conversational context, causes pronoun/reference errors, and misses formality consistency across speakers
- **Single-candidate translation:** Accepting first translation output misses opportunities to optimize for duration constraints
- **Character-only duration estimation:** Ignoring phoneme complexity (e.g., "three" vs "1") leads to poor duration predictions
- **Post-translation duration adjustment:** Trying to compress/expand already-translated text is less effective than duration-aware candidate selection

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Neural translation model | Custom transformer architecture | SeamlessM4T v2 ForTextToText | 2.3B parameters trained on massive multilingual corpus, proven quality |
| Tokenization for translation | Manual BPE/WordPiece implementation | AutoProcessor from transformers | SeamlessM4T requires specific sentencepiece tokenizer, processor handles it |
| Translation quality estimation | Custom confidence scoring | Model output_scores=True | SeamlessM4T provides logits/scores, established QE methods exist |
| Multi-language handling | Language detection + routing | SeamlessM4T unified model | Handles 96 languages in single model, no routing needed |
| Context-aware chunking | Fixed-size splitting | Overlap-based semantic chunking | Research shows 128-256 token overlap preserves context better than hard cuts |

**Key insight:** Neural machine translation is a solved problem with production-ready models. The complexity is in duration-aware candidate selection and context preservation, not the translation itself.

## Common Pitfalls

### Pitfall 1: Context Loss from Isolated Translation
**What goes wrong:** Translating each transcript segment independently causes pronouns to lose referents, formality levels to fluctuate inconsistently, and conversational flow to break.

**Why it happens:** Transformer models need surrounding context to resolve ambiguous references (e.g., "he said" → who is "he"?).

**How to avoid:** Always provide full preceding transcript as context. For long videos, use overlapping chunks (128-256 tokens) instead of isolated segments.

**Warning signs:** Translated output contains vague pronouns ("he/she/it/they") where original used names, or formality (casual/formal) shifts mid-conversation.

### Pitfall 2: Ignoring Model Token Limits
**What goes wrong:** SeamlessM4T has implicit sequence length limits (~1024-2048 tokens). Exceeding this causes truncation, crashes, or silent quality degradation.

**Why it happens:** Transformer models use quadratic attention (O(n²) complexity), requiring hard limits on input size.

**How to avoid:** Estimate token count (approx: chars/4 for English) before processing. Chunk transcripts >800 tokens proactively.

**Warning signs:** Model output is cut off mid-sentence, quality degrades for later segments, or inference raises shape mismatch errors.

### Pitfall 3: Naive Duration Estimation
**What goes wrong:** Simple character count (len(text)/15) fails for content with numbers ("2026" takes longer to say than "soon"), acronyms ("FBI" vs "bureau"), or punctuation-heavy text.

**Why it happens:** Not all characters take equal time to speak. Numbers require full pronunciation ("two thousand twenty-six" = 24 chars spoken vs 4 written).

**How to avoid:** Normalize text before counting: expand numbers to words, handle acronyms, remove punctuation. Use phoneme-based estimation for higher accuracy.

**Warning signs:** Duration validation frequently fails for segments with dates, numbers, or acronyms despite character count being close.

### Pitfall 4: Accepting First Translation Without Validation
**What goes wrong:** First beam search output may fit meaning but violate duration constraints, causing downstream lip sync failures.

**Why it happens:** Translation models optimize for semantic accuracy, not speech timing. Different phrasings have different durations.

**How to avoid:** Always generate multiple candidates (num_beams=3, num_return_sequences=3) and rank by duration fit before accepting.

**Warning signs:** Many segments fail duration validation (>10% mismatch) in post-processing, requiring manual re-translation.

### Pitfall 5: Losing Speaker Context in Multi-Speaker Videos
**What goes wrong:** Translations mix up who's speaking, lose speaker-specific formality levels, or mishandle conversational turn-taking.

**Why it happens:** Translation model doesn't see speaker labels unless explicitly provided in input.

**How to avoid:** Include speaker labels in input format: "Speaker A: <text>\nSpeaker B: <text>". This preserves conversational structure.

**Warning signs:** Translated conversations sound unnatural, responses don't match questions, or formality randomly shifts.

### Pitfall 6: Ignoring Low Confidence Segments
**What goes wrong:** Accepting all translations regardless of confidence leads to incorrect/nonsensical segments that ruin dubbed video quality.

**Why it happens:** Translation models struggle with rare words, technical terms, slang, or low-quality audio transcriptions.

**How to avoid:** Implement confidence threshold (e.g., <70% = flag for review). Store flagged segments for Phase 8 manual review interface.

**Warning signs:** Users report obviously wrong translations, technical terms mistranslated, or names corrupted.

## Code Examples

Verified patterns from official sources:

### Loading SeamlessM4T v2 for Text Translation
```python
# Source: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

# Use dedicated text-to-text model (smaller memory footprint)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

# Move to GPU
model = model.to('cuda')
```

### Batch Translation with Multiple Candidates
```python
# Source: https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2
# Process multiple segments in batch
batch_texts = [seg["text"] for seg in segments[:8]]  # Process 8 at a time

# Batch encoding with source language
inputs = processor(
    text=batch_texts,
    src_lang="jpn",  # Japanese source
    return_tensors="pt",
    padding=True
).to('cuda')

# Generate with beam search for multiple candidates
outputs = model.generate(
    **inputs,
    tgt_lang="eng",
    num_beams=3,              # Beam search with width 3
    num_return_sequences=3,   # Return all 3 beams per input
    return_dict_in_generate=True,
    output_scores=True,       # Get confidence scores
    max_new_tokens=512        # Limit output length
)

# Decode outputs (3 candidates per input)
all_translations = []
for i in range(len(batch_texts)):
    candidates = []
    for j in range(3):
        idx = i * 3 + j
        translation = processor.decode(
            outputs.sequences[idx],
            skip_special_tokens=True
        )
        candidates.append(translation)
    all_translations.append(candidates)
```

### Duration Validation with Character Count Heuristic
```python
# Source: TTS duration modeling research + video dubbing papers
def validate_duration(original_duration, translated_text, tolerance=0.1):
    """
    Validate translated text fits within duration constraint.

    Args:
        original_duration: Original segment duration (seconds)
        translated_text: Translated English text
        tolerance: Acceptable deviation (0.1 = ±10%)

    Returns:
        (is_valid, estimated_duration, ratio)
    """
    # English conversational speech: ~14-16 chars/second
    # Use 15 as middle estimate
    CHARS_PER_SECOND = 15

    # Normalize text for better estimation
    normalized = translated_text.lower()
    # Remove multiple spaces
    normalized = " ".join(normalized.split())
    # Count characters (excluding spaces for better accuracy)
    char_count = len(normalized.replace(" ", ""))

    # Estimate duration
    estimated_duration = char_count / CHARS_PER_SECOND

    # Calculate ratio
    ratio = estimated_duration / original_duration

    # Check tolerance (±10%)
    min_ratio = 1.0 - tolerance
    max_ratio = 1.0 + tolerance
    is_valid = min_ratio <= ratio <= max_ratio

    return is_valid, estimated_duration, ratio
```

### Extracting Model Confidence Scores
```python
# Source: Neural translation quality estimation research
import torch

def extract_confidence_scores(outputs):
    """
    Extract per-sequence confidence from model outputs.

    Args:
        outputs: GenerateOutput with scores

    Returns:
        List of confidence scores (0-1) per sequence
    """
    # outputs.scores is tuple of tensors (one per generation step)
    # Shape: (batch_size * num_return_sequences, vocab_size)

    confidences = []

    # Get transition scores (log probabilities of chosen tokens)
    # Average across sequence length for overall confidence
    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True
    )

    # Convert log probs to probabilities and average
    for i in range(len(outputs.sequences)):
        # Average probability across tokens (geometric mean in log space)
        avg_log_prob = transition_scores[i].mean().item()
        confidence = torch.exp(torch.tensor(avg_log_prob)).item()
        confidences.append(confidence)

    return confidences
```

### Integration with ModelManager
```python
# Source: Existing project architecture
from src.models.model_manager import ModelManager

manager = ModelManager()

# Load SeamlessM4T (will unload Whisper if still loaded)
translation_model = manager.load_model(
    "seamless_m4t_translation",
    lambda: SeamlessM4Tv2ForTextToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    ).to('cuda')
)

# Translation stage uses model
translated_segments = translate_with_candidates(
    transcript_segments,
    translation_model,
    processor
)

# Later: unload before loading XTTS
manager.unload_current_model()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Isolated segment translation | Full-context batching with overlap | 2024-2025 (late chunking research) | Better pronoun resolution, consistent formality |
| Single translation output | Beam search with multiple candidates | 2020-2023 (diverse beam search) | Enables duration-aware selection |
| Post-translation compression | Duration-aware candidate ranking | 2024 (video dubbing papers) | Higher quality, fewer artifacts |
| Character count only | Phoneme-based duration modeling | 2024-2025 (TTS duration research) | More accurate timing predictions |
| Manual language detection | Unified multilingual models | 2023 (SeamlessM4T release) | Simpler pipeline, fewer errors |

**Deprecated/outdated:**
- SeamlessM4T v1: Replaced by v2 in late 2023 with non-autoregressive T2U decoder (faster, better quality)
- NLLB-200: Standalone translation model superseded by SeamlessM4T's integrated approach
- MarianMT: Older NMT models with worse multilingual coverage and no speech integration

## Open Questions

Things that couldn't be fully resolved:

1. **Exact SeamlessM4T v2 sequence length limit**
   - What we know: Transformer models typically 512-2048 tokens, SeamlessM4T docs don't specify
   - What's unclear: Actual hard limit for this specific model
   - Recommendation: Test empirically with long inputs, use 1024 tokens as safe upper bound, implement chunking proactively

2. **Optimal confidence threshold for flagging**
   - What we know: Quality estimation research suggests 0.6-0.8 range, user wants ~70%
   - What's unclear: Whether SeamlessM4T confidence scores are calibrated (high correlation with actual quality)
   - Recommendation: Start with 0.7 threshold, validate against manual review in Phase 8, adjust based on false positive/negative rates

3. **Multi-candidate generation performance impact**
   - What we know: num_beams=3 generates 3x candidates, likely 2-3x slower than greedy
   - What's unclear: Actual wall-clock impact on 20-minute video processing
   - Recommendation: Implement both modes (fast=greedy, quality=beam3), test on real videos, let user choose speed vs quality

4. **Speaker label format for translation input**
   - What we know: Including speaker labels improves context preservation
   - What's unclear: Optimal format ("Speaker A: text" vs "[A] text" vs metadata)
   - Recommendation: Test multiple formats, measure translation quality with/without labels, pick best-performing format

5. **Batch size for GPU memory optimization**
   - What we know: RTX 5090 has 32GB VRAM, SeamlessM4T Large uses ~6GB
   - What's unclear: Optimal batch size for translation stage (memory vs throughput tradeoff)
   - Recommendation: Start batch_size=8, monitor VRAM, increase until 80% usage for max throughput

## Sources

### Primary (HIGH confidence)
- [SeamlessM4T v2 Documentation](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2) - Official API reference
- [SeamlessM4T v2 Large Model Card](https://huggingface.co/facebook/seamless-m4t-v2-large) - Language support, VRAM requirements
- [SeamlessM4T PyTorch Optimization](https://pytorch.org/blog/accelerating-generative-ai-4/) - Performance characteristics
- [Length-Aware Speech Translation Paper (arxiv 2506.00740)](https://arxiv.org/pdf/2506.00740) - Duration constraints for dubbing

### Secondary (MEDIUM confidence)
- [Beam Search Documentation (d2l.ai)](https://d2l.ai/chapter_recurrent-modern/beam-search.html) - Candidate generation patterns
- [Diverse Beam Search (arxiv 1610.02424)](https://arxiv.org/pdf/1610.02424) - Multiple output ranking
- [Total-Duration-Aware Duration Modeling (arxiv 2406.04281)](https://arxiv.org/html/2406.04281v1) - Duration estimation techniques
- [Context-Aware Translation Framework (arxiv 2412.04205)](https://arxiv.org/html/2412.04205v1) - Multi-turn context preservation
- [Late Chunking Research (arxiv 2409.04701)](https://arxiv.org/html/2409.04701v2) - Context-preserving chunking strategies

### Secondary (verified via web search)
- [Neural Translation Quality Estimation Survey (arxiv 2403.14118)](https://arxiv.org/html/2403.14118v1) - Confidence scoring methods
- [Seamless Communication GitHub](https://github.com/facebookresearch/seamless_communication) - Official implementation details
- [Machine Translation Pitfalls (Argo Translation)](https://www.argotrans.com/blog/9-common-pitfalls-machine-translation) - Common errors
- [Speech Synthesis Duration Modeling Research](https://link.springer.com/article/10.1007/s10772-010-9077-x) - Phoneme duration patterns

### Tertiary (LOW confidence - requires validation)
- WebSearch results on SeamlessM4T troubleshooting (GitHub issues) - Community-reported problems
- WebSearch results on document chunking for RAG (2024-2026) - Indirect relevance to translation chunking

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - SeamlessM4T v2 is documented, proven, and actively maintained by Meta
- Architecture: HIGH - Patterns verified from official docs and recent academic research
- Pitfalls: MEDIUM - Based on general NMT research and community reports, not SeamlessM4T-specific testing
- Duration estimation: MEDIUM - Character-count heuristic is common practice but accuracy needs validation
- Confidence scoring: MEDIUM - Quality estimation research is robust but SeamlessM4T score calibration unverified

**Research date:** 2026-01-31
**Valid until:** ~60 days (2026-04-01) - SeamlessM4T is stable, but transformers library updates frequently
