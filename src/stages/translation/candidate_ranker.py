"""
Multi-candidate ranking for translation quality optimization.

Scores translation candidates using weighted combination of model confidence
and duration fit to select the best translation for downstream stages.
"""
from typing import List, Tuple, Dict, Optional

from .duration_validator import estimate_duration
from src.config.settings import TRANSLATION_CHARS_PER_SECOND, TRANSLATION_DURATION_TOLERANCE


class CandidateRanker:
    """
    Ranks multiple translation candidates by weighted score.

    Combines model confidence (how certain the translation model is) with
    duration fit (how well the translation matches timing constraints) to
    select the best candidate for voice cloning and lip sync stages.

    Default weights: 60% confidence, 40% duration fit (speed-first priority).

    Usage:
        >>> ranker = CandidateRanker(confidence_weight=0.6, duration_weight=0.4)
        >>> candidates = ["Hello there", "Hi", "Greetings"]
        >>> scores = [0.9, 0.75, 0.85]
        >>> best, all_ranked = ranker.rank_candidates(candidates, scores, original_duration=2.0)
        >>> print(best["candidate"], best["combined_score"])
    """

    def __init__(
        self,
        confidence_weight: float = 0.6,
        duration_weight: float = 0.4,
        chars_per_second: Optional[float] = None
    ):
        """
        Initialize candidate ranker with scoring weights.

        Args:
            confidence_weight: Weight for model confidence score (0-1, default 0.6)
            duration_weight: Weight for duration fit score (0-1, default 0.4)
            chars_per_second: Speaking rate for duration estimation (default from settings)

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        # Validate weights sum to 1.0 (allow small floating point error)
        if abs(confidence_weight + duration_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0 (got {confidence_weight + duration_weight:.6f})"
            )

        self.confidence_weight = confidence_weight
        self.duration_weight = duration_weight
        self.chars_per_second = chars_per_second or TRANSLATION_CHARS_PER_SECOND

    def _calculate_duration_score(
        self,
        translated_text: str,
        original_duration: float,
        tolerance: float = None
    ) -> float:
        """
        Calculate duration fit score for a translation candidate.

        Scores how well the translated text fits the original duration:
        - Perfect fit (ratio=1.0): score = 1.0
        - Within tolerance (0.9-1.1): linear penalty based on deviation
        - Outside tolerance: score = 0.0 (hard penalty)

        Args:
            translated_text: Translation candidate to score
            original_duration: Original segment duration in seconds
            tolerance: Acceptable deviation (default 0.1 = ±10%)

        Returns:
            Duration fit score from 0.0 (worst) to 1.0 (perfect)

        Example:
            >>> # Perfect fit
            >>> score = _calculate_duration_score("Hello", 1.0)  # ratio=1.0
            >>> # score = 1.0
            >>>
            >>> # 5% over (ratio=1.05)
            >>> score = _calculate_duration_score("Hello there", 1.0)  # ratio=1.05
            >>> # score = 1.0 - abs(1.0 - 1.05) * 10 = 0.5
            >>>
            >>> # 15% over (outside tolerance)
            >>> score = _calculate_duration_score("Hello there friend", 1.0)  # ratio=1.15
            >>> # score = 0.0
        """
        if tolerance is None:
            tolerance = TRANSLATION_DURATION_TOLERANCE

        # Estimate duration
        estimated_duration = estimate_duration(translated_text, self.chars_per_second)

        # Calculate ratio
        if original_duration <= 0:
            return 0.0

        ratio = estimated_duration / original_duration

        # Check if within tolerance
        min_ratio = 1 - tolerance
        max_ratio = 1 + tolerance

        if min_ratio <= ratio <= max_ratio:
            # Linear penalty within tolerance
            # ratio=1.0 (perfect) -> penalty=0 -> score=1.0
            # ratio=1.05 (5% over) -> penalty=0.05*10=0.5 -> score=0.5
            # ratio=1.1 (10% over, at boundary) -> penalty=1.0 -> score=0.0
            penalty = abs(1.0 - ratio) / tolerance
            score = 1.0 - penalty
            return max(0.0, score)  # Ensure non-negative
        else:
            # Outside tolerance: hard penalty
            return 0.0

    def rank_candidates(
        self,
        candidates: List[str],
        scores: List[float],
        original_duration: float
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Rank translation candidates by weighted score.

        Calculates combined score for each candidate using:
        combined_score = (confidence * confidence_weight) + (duration_fit * duration_weight)

        Args:
            candidates: List of translation candidates
            scores: Model confidence scores for each candidate (0-1)
            original_duration: Original segment duration in seconds

        Returns:
            Tuple of (best_candidate_dict, all_ranked_list):
            - best_candidate_dict: Top-ranked candidate or None if empty list
            - all_ranked_list: All candidates sorted by combined_score (descending)

            Each dict contains:
                - candidate: Translation text
                - model_confidence: Model confidence score
                - duration_score: Duration fit score
                - combined_score: Weighted combination
                - estimated_duration: Estimated speech duration
                - duration_ratio: estimated / original
                - is_valid_duration: Whether within tolerance

        Example:
            >>> candidates = ["Hello there", "Hi", "Greetings"]
            >>> scores = [0.9, 0.75, 0.85]  # Model confidence
            >>> best, all_ranked = ranker.rank_candidates(candidates, scores, 2.0)
            >>> print(f"Best: {best['candidate']}")
            >>> print(f"Score: {best['combined_score']:.3f}")
        """
        # Validate inputs
        if not candidates:
            return None, []

        if len(candidates) != len(scores):
            raise ValueError(
                f"Length mismatch: {len(candidates)} candidates, {len(scores)} scores"
            )

        # Score each candidate
        ranked = []
        for candidate, model_confidence in zip(candidates, scores):
            # Calculate duration fit score
            duration_score = self._calculate_duration_score(candidate, original_duration)

            # Calculate combined weighted score
            combined_score = (
                model_confidence * self.confidence_weight +
                duration_score * self.duration_weight
            )

            # Estimate duration for metadata
            estimated_duration = estimate_duration(candidate, self.chars_per_second)
            duration_ratio = estimated_duration / original_duration if original_duration > 0 else 0.0

            # Check if duration is valid (within tolerance)
            tolerance = TRANSLATION_DURATION_TOLERANCE
            is_valid_duration = (
                (1 - tolerance) <= duration_ratio <= (1 + tolerance)
            )

            # Create result dict
            result = {
                "candidate": candidate,
                "model_confidence": model_confidence,
                "duration_score": duration_score,
                "combined_score": combined_score,
                "estimated_duration": estimated_duration,
                "duration_ratio": duration_ratio,
                "is_valid_duration": is_valid_duration
            }

            ranked.append(result)

        # Sort by combined_score (descending)
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)

        # Return best candidate and full ranking
        best = ranked[0] if ranked else None
        return best, ranked


def rank_candidates(
    candidates: List[str],
    scores: List[float],
    original_duration: float,
    confidence_weight: float = 0.6,
    duration_weight: float = 0.4
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Convenience function for ranking translation candidates.

    Creates a CandidateRanker instance and ranks candidates in one call.
    Useful for quick one-off ranking without managing CandidateRanker lifecycle.

    Args:
        candidates: List of translation candidates
        scores: Model confidence scores for each candidate (0-1)
        original_duration: Original segment duration in seconds
        confidence_weight: Weight for confidence score (default 0.6)
        duration_weight: Weight for duration fit score (default 0.4)

    Returns:
        Tuple of (best_candidate_dict, all_ranked_list)

    Example:
        >>> candidates = ["Hello", "Hi there", "Greetings"]
        >>> scores = [0.9, 0.85, 0.75]
        >>> best, all_ranked = rank_candidates(candidates, scores, 1.5)
        >>> print(best["candidate"])
        'Hello'
    """
    ranker = CandidateRanker(
        confidence_weight=confidence_weight,
        duration_weight=duration_weight
    )
    return ranker.rank_candidates(candidates, scores, original_duration)
