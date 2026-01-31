"""
Duration validation for translation pipeline.

Provides character-count based heuristics to estimate translated text duration
and validate translations fit within original segment timing constraints.
"""
from src.config.settings import TRANSLATION_CHARS_PER_SECOND, TRANSLATION_DURATION_TOLERANCE


def normalize_text_for_duration(text: str) -> str:
    """
    Normalize text for duration estimation.

    Converts to lowercase and collapses whitespace for consistent character counting.

    Args:
        text: Text to normalize

    Returns:
        Normalized text with collapsed whitespace

    Example:
        >>> normalize_text_for_duration("  Hello   World  ")
        'hello world'
    """
    # Convert to lowercase
    text = text.lower()

    # Collapse multiple spaces and strip leading/trailing whitespace
    text = " ".join(text.split())

    return text


def estimate_duration(text: str, chars_per_second: float = None) -> float:
    """
    Estimate speech duration for translated text using character count heuristic.

    Uses character count (excluding spaces) divided by speaking rate to estimate
    how long the text will take to speak. Default rate is 15 chars/second for
    English conversational speech.

    Args:
        text: Translated text to estimate duration for
        chars_per_second: Speaking rate in characters per second (default from settings)

    Returns:
        Estimated duration in seconds

    Example:
        >>> # "HelloHowareyoutoday?" = 20 chars (no spaces) at 15 chars/sec = 1.33 sec
        >>> estimate_duration("Hello, how are you today?")
        1.3333333333333333
    """
    if not text:
        return 0.0

    if chars_per_second is None:
        chars_per_second = TRANSLATION_CHARS_PER_SECOND

    # Normalize text for consistent counting
    normalized = normalize_text_for_duration(text)

    # Count characters excluding spaces (spaces don't contribute to speech duration)
    char_count = len(normalized.replace(" ", ""))

    # Calculate duration
    duration = char_count / chars_per_second

    return duration


def validate_duration(
    original_duration: float,
    translated_text: str,
    tolerance: float = None
) -> dict:
    """
    Validate that translated text duration fits within tolerance of original.

    Estimates translated text duration and checks if it falls within acceptable
    range (±10% by default) of the original segment duration. This ensures
    downstream voice cloning and lip sync stages can fit the translation.

    Args:
        original_duration: Duration of original audio segment in seconds
        translated_text: Translated text to validate
        tolerance: Acceptable deviation as fraction (0.1 = ±10%, default from settings)

    Returns:
        Dictionary with validation results:
        - is_valid: Whether translation fits within tolerance
        - estimated_duration: Estimated duration of translated text
        - original_duration: Original segment duration
        - ratio: estimated_duration / original_duration
        - tolerance: Tolerance used
        - min_ratio: Minimum acceptable ratio (1 - tolerance)
        - max_ratio: Maximum acceptable ratio (1 + tolerance)

    Example:
        >>> result = validate_duration(
        ...     original_duration=5.0,
        ...     translated_text="Hello, how are you today?",
        ...     tolerance=0.1
        ... )
        >>> # result["is_valid"] = True if estimated duration is 4.5-5.5 seconds
        >>> # result["ratio"] = estimated_duration / 5.0
    """
    if tolerance is None:
        tolerance = TRANSLATION_DURATION_TOLERANCE

    # Handle edge case: zero original duration
    if original_duration <= 0:
        return {
            "is_valid": False,
            "estimated_duration": 0.0,
            "original_duration": original_duration,
            "ratio": 0.0,
            "tolerance": tolerance,
            "min_ratio": 1 - tolerance,
            "max_ratio": 1 + tolerance,
        }

    # Estimate translated text duration
    estimated_duration = estimate_duration(translated_text)

    # Calculate ratio
    ratio = estimated_duration / original_duration

    # Check if within tolerance
    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance
    is_valid = min_ratio <= ratio <= max_ratio

    return {
        "is_valid": is_valid,
        "estimated_duration": estimated_duration,
        "original_duration": original_duration,
        "ratio": ratio,
        "tolerance": tolerance,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
    }
