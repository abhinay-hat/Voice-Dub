"""
SeamlessM4T v2 translation wrapper for multi-language to English conversion.
Provides single-segment translation with lazy model loading.
"""
from typing import Optional, Dict
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

from src.models.model_manager import ModelManager
from src.config.settings import SEAMLESS_MODEL_ID, TRANSLATION_TARGET_LANGUAGE


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
                torch_dtype=torch.float16  # Half precision for VRAM efficiency
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
