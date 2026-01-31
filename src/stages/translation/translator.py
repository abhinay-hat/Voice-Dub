"""
SeamlessM4T v2 translation wrapper for multi-language to English conversion.
Provides single-segment translation with lazy model loading.
"""
from typing import Optional, Dict
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

from src.models.model_manager import ModelManager
from src.config.settings import (
    SEAMLESS_MODEL_ID,
    TRANSLATION_TARGET_LANGUAGE,
    TRANSLATION_NUM_CANDIDATES,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_MAX_TOKENS_PER_SEGMENT
)


class Translator:
    """
    SeamlessM4T v2 Large translation wrapper.

    Translates text from any supported language to English using Meta's SeamlessM4T v2.
    Integrates with ModelManager for sequential loading and VRAM efficiency.

    Usage:
        >>> from src.models.model_manager import ModelManager
        >>> manager = ModelManager()
        >>> translator = Translator(manager)
        >>> result = translator.translate_segment("こんにちは", "jpn")
        >>> print(result["translation"])  # "Hello"
    """

    def __init__(self, model_manager: ModelManager):
        """
        Initialize translator with model manager.

        Args:
            model_manager: ModelManager instance for sequential model loading

        Note: Model is NOT loaded during initialization (lazy loading on first translate).
        """
        self.model_manager = model_manager
        self.model: Optional[SeamlessM4Tv2ForTextToText] = None
        self.processor: Optional[AutoProcessor] = None

    def _ensure_model_loaded(self) -> None:
        """
        Lazy load SeamlessM4T model and processor.

        Only loads on first translation call. Uses ModelManager to ensure
        sequential loading (unloads previous model if one exists).

        Memory footprint: ~6GB VRAM (fp16)
        """
        if self.model is not None:
            return  # Already loaded

        print("Loading SeamlessM4T v2 Large processor...")
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL_ID)

        print("Loading SeamlessM4T v2 Large model...")
        self.model = self.model_manager.load_model(
            "seamless_m4t_translation",
            lambda: SeamlessM4Tv2ForTextToText.from_pretrained(
                SEAMLESS_MODEL_ID,
                dtype=torch.float16  # Half precision for VRAM efficiency
            ).to('cuda')
        )

    def translate_segment(
        self,
        segment_text: str,
        source_lang: str
    ) -> Dict[str, str]:
        """
        Translate a single text segment to English.

        Args:
            segment_text: Text to translate (from transcription)
            source_lang: Source language code (SeamlessM4T 3-letter codes)
                        Examples: "jpn" (Japanese), "spa" (Spanish), "kor" (Korean),
                                 "cmn" (Mandarin), "fra" (French), "deu" (German)

        Returns:
            Dictionary with keys:
                - translation: English translation text
                - source_lang: Source language code (echoed back)
                - target_lang: Target language code (always "eng")

        Example:
            >>> result = translator.translate_segment("こんにちは", "jpn")
            >>> print(result["translation"])
            "Hello"
            >>> print(result["source_lang"])
            "jpn"

        Note: Uses greedy decoding (no beam search) for speed. Beam search will
              be added in Plan 2 for multi-candidate generation.
        """
        # Ensure model is loaded (lazy loading)
        self._ensure_model_loaded()

        # Validate inputs
        if not segment_text or not segment_text.strip():
            raise ValueError("segment_text cannot be empty")

        if not source_lang:
            raise ValueError("source_lang must be provided")

        # Log warning for potentially unsupported language codes
        # (SeamlessM4T supports 96 languages, but we don't validate all codes here)
        supported_priority = ["jpn", "kor", "cmn", "spa", "fra", "deu", "hin", "ara"]
        if source_lang not in supported_priority:
            print(f"Warning: '{source_lang}' not in priority language list. "
                  f"Attempting translation anyway (SeamlessM4T supports 96 languages).")

        # Prepare inputs for model
        inputs = self.processor(
            text=segment_text,
            src_lang=source_lang,
            return_tensors="pt"
        ).to('cuda')

        # Generate translation (greedy decoding for speed)
        # max_new_tokens=512 handles most reasonable segment lengths
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                tgt_lang=TRANSLATION_TARGET_LANGUAGE,
                max_new_tokens=512
            )

        # Decode translation
        translation = self.processor.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        return {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": TRANSLATION_TARGET_LANGUAGE
        }

    def translate_with_candidates(
        self,
        segment_text: str,
        source_lang: str,
        num_candidates: int = None
    ) -> tuple[list[str], list[float]]:
        """
        Generate multiple translation candidates using beam search.

        Args:
            segment_text: Text to translate (from transcription)
            source_lang: Source language code (SeamlessM4T 3-letter codes)
            num_candidates: Number of translation candidates to generate
                          (default: TRANSLATION_NUM_CANDIDATES from settings)

        Returns:
            Tuple of (candidates, confidences):
                - candidates: List of translation strings (length = num_candidates)
                - confidences: List of confidence scores [0, 1] (length = num_candidates)
                              Scores are in descending order (best first)

        Example:
            >>> candidates, scores = translator.translate_with_candidates(
            ...     "こんにちは、元気ですか？",
            ...     "jpn",
            ...     num_candidates=3
            ... )
            >>> for i, (cand, score) in enumerate(zip(candidates, scores)):
            ...     print(f"{i+1}. [{score:.3f}] {cand}")
            1. [0.892] Hello, how are you?
            2. [0.745] Hi, how are you doing?
            3. [0.631] Hello, are you well?

        Note: Uses beam search with num_beams = num_candidates. Higher num_candidates
              increases computation time but provides more diverse translation options.
        """
        # Ensure model is loaded
        self._ensure_model_loaded()

        # Default to config if not specified
        if num_candidates is None:
            num_candidates = TRANSLATION_NUM_CANDIDATES

        # Validate inputs
        if not segment_text or not segment_text.strip():
            raise ValueError("segment_text cannot be empty")

        if not source_lang:
            raise ValueError("source_lang must be provided")

        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        # Prepare inputs for model
        inputs = self.processor(
            text=segment_text,
            src_lang=source_lang,
            return_tensors="pt"
        ).to('cuda')

        # Generate with beam search
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                tgt_lang=TRANSLATION_TARGET_LANGUAGE,
                num_beams=num_candidates,           # Beam width
                num_return_sequences=num_candidates, # Return all beams
                return_dict_in_generate=True,        # Get metadata
                output_scores=True,                  # Get confidence scores
                max_new_tokens=TRANSLATION_MAX_TOKENS_PER_SEGMENT
            )

        # Extract confidence scores using compute_transition_scores()
        # This gives per-token log probabilities
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True
        )

        # Calculate average confidence per candidate
        confidences = []
        for i in range(len(outputs.sequences)):
            # Average log probability across tokens, convert to probability
            avg_log_prob = transition_scores[i].mean().item()
            confidence = torch.exp(torch.tensor(avg_log_prob)).item()
            confidences.append(confidence)

        # Decode all candidates
        candidates = [
            self.processor.decode(seq, skip_special_tokens=True)
            for seq in outputs.sequences
        ]

        return (candidates, confidences)

    def translate_batch(
        self,
        segments: list[str],
        source_lang: str,
        num_candidates: int = 1,
        batch_size: int = None
    ) -> list[dict]:
        """
        Batch translate multiple segments efficiently on GPU.

        Args:
            segments: List of text segments to translate
            source_lang: Source language code (same for all segments)
            num_candidates: Number of translation candidates per segment (default: 1)
            batch_size: Number of segments to process at once
                       (default: TRANSLATION_BATCH_SIZE from settings)

        Returns:
            List of result dictionaries (one per input segment):
                {
                    "segment_text": str,      # Original text
                    "candidates": list[str],  # Translation candidates
                    "scores": list[float]     # Confidence scores
                }

        Example:
            >>> segments = ["こんにちは", "ありがとう", "さようなら"]
            >>> results = translator.translate_batch(segments, "jpn", num_candidates=2)
            >>> for result in results:
            ...     print(f"Original: {result['segment_text']}")
            ...     print(f"Best: {result['candidates'][0]} (score: {result['scores'][0]:.3f})")

        Note: Batch processing is 2-4x faster than individual translate_with_candidates()
              calls due to GPU parallelization. Uses batches of size TRANSLATION_BATCH_SIZE
              to balance throughput and VRAM usage.
        """
        # Ensure model is loaded
        self._ensure_model_loaded()

        # Default to config if not specified
        if batch_size is None:
            batch_size = TRANSLATION_BATCH_SIZE

        # Validate inputs
        if not segments:
            return []

        if not source_lang:
            raise ValueError("source_lang must be provided")

        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        # Process in batches for memory efficiency
        all_results = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]

            # Batch encode with padding
            inputs = self.processor(
                text=batch,
                src_lang=source_lang,
                return_tensors="pt",
                padding=True
            ).to('cuda')

            # Generate for batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=TRANSLATION_TARGET_LANGUAGE,
                    num_beams=num_candidates,
                    num_return_sequences=num_candidates,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=TRANSLATION_MAX_TOKENS_PER_SEGMENT
                )

            # Compute confidence scores for all outputs
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )

            # Extract candidates and scores per input segment
            # (outputs has batch_size * num_candidates sequences)
            for j in range(len(batch)):
                candidates = []
                scores = []

                for k in range(num_candidates):
                    idx = j * num_candidates + k
                    translation = self.processor.decode(
                        outputs.sequences[idx],
                        skip_special_tokens=True
                    )
                    candidates.append(translation)

                    # Extract confidence score for this candidate
                    avg_log_prob = transition_scores[idx].mean().item()
                    confidence = torch.exp(torch.tensor(avg_log_prob)).item()
                    scores.append(confidence)

                all_results.append({
                    "segment_text": batch[j],
                    "candidates": candidates,
                    "scores": scores
                })

        return all_results


def translate_segment(
    segment_text: str,
    source_lang: str,
    model_manager: Optional[ModelManager] = None
) -> Dict[str, str]:
    """
    Convenience function for single-segment translation.

    Creates a Translator instance and translates a single segment.
    Useful for quick one-off translations without managing Translator lifecycle.

    Args:
        segment_text: Text to translate
        source_lang: Source language code (SeamlessM4T 3-letter codes)
        model_manager: Optional ModelManager (created if None)

    Returns:
        Dictionary with translation, source_lang, and target_lang

    Example:
        >>> result = translate_segment("Hola", "spa")
        >>> print(result["translation"])
        "Hello"
    """
    if model_manager is None:
        model_manager = ModelManager(verbose=True)

    translator = Translator(model_manager)
    return translator.translate_segment(segment_text, source_lang)
