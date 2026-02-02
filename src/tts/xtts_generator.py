"""
XTTS-v2 voice cloning synthesis wrapper with duration matching.

Provides voice cloning synthesis using XTTS-v2 with iterative duration matching
via speed parameter adjustment. Integrates with ModelManager for sequential loading
and SpeakerEmbeddingCache for voice characteristics.
"""
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import torch
import soundfile as sf

from src.models.model_manager import ModelManager
from src.tts.speaker_embeddings import SpeakerEmbeddingCache
from src.config.settings import (
    TTS_MODEL_ID,
    TTS_SAMPLE_RATE,
    TTS_TEMPERATURE,
    TTS_LENGTH_PENALTY,
    TTS_REPETITION_PENALTY,
    TTS_TOP_K,
    TTS_TOP_P,
    TTS_DURATION_TOLERANCE,
    TTS_SPEED_MIN,
    TTS_SPEED_MAX,
    TTS_MAX_DURATION_RETRIES,
    TTS_SHORT_TEXT_THRESHOLD
)


class XTTSGenerator:
    """XTTS-v2 voice cloning wrapper with duration matching."""

    def __init__(
        self,
        model_manager: ModelManager,
        embedding_cache: SpeakerEmbeddingCache,
        temperature: float = TTS_TEMPERATURE,
        length_penalty: float = TTS_LENGTH_PENALTY,
        repetition_penalty: float = TTS_REPETITION_PENALTY,
        top_k: int = TTS_TOP_K,
        top_p: float = TTS_TOP_P
    ):
        """
        Initialize generator with model and speaker embeddings.

        Args:
            model_manager: ModelManager for XTTS loading (loaded lazily)
            embedding_cache: Pre-generated speaker embeddings
            temperature: Creativity (0.1-1.0, lower = consistent)
            length_penalty: >1.0 encourages longer output
            repetition_penalty: Reduces phrase repetition
            top_k: Decoder sampling parameter
            top_p: Nucleus sampling threshold
        """
        self.model_manager = model_manager
        self.embedding_cache = embedding_cache
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self._model = None  # Lazy loaded

    def _ensure_model_loaded(self):
        """Load XTTS model via ModelManager if not already loaded."""
        if self._model is None:
            from TTS.api import TTS
            self._model = self.model_manager.load_model(
                "xtts",
                lambda: TTS(TTS_MODEL_ID, gpu=True)
            )
        return self._model

    def synthesize_segment(
        self,
        text: str,
        speaker_id: str,
        language: str = "en"
    ) -> np.ndarray:
        """
        Synthesize audio for text using cloned voice from speaker_id.

        Args:
            text: English text to synthesize
            speaker_id: Speaker ID to lookup in embedding cache
            language: Target language code (default "en")

        Returns:
            Audio as numpy array at 24kHz sample rate

        Raises:
            ValueError: If speaker_id not in embedding cache
        """
        model = self._ensure_model_loaded()

        # Get speaker embeddings from cache - CRITICAL for voice cloning
        if not self.embedding_cache.has(speaker_id):
            raise ValueError(f"Speaker {speaker_id} not in embedding cache")

        gpt_cond_latent, speaker_embedding = self.embedding_cache.get(speaker_id)

        # Call XTTS inference with explicit embedding parameters
        # These parameters are REQUIRED for voice cloning to work
        audio = model.tts(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,      # Voice conditioning
            speaker_embedding=speaker_embedding,   # Speaker characteristics
            temperature=self.temperature,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            enable_text_splitting=True  # Handle long text gracefully
        )

        # Convert to numpy array if needed (TTS.api returns list)
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return audio

    def synthesize_with_duration_matching(
        self,
        text: str,
        speaker_id: str,
        target_duration: float,
        tolerance: float = TTS_DURATION_TOLERANCE,
        max_retries: int = TTS_MAX_DURATION_RETRIES
    ) -> tuple[np.ndarray, dict]:
        """
        Synthesize audio with iterative duration matching.

        Uses binary search to find speed parameter that produces audio
        matching target duration within tolerance.

        Args:
            text: English text to synthesize
            speaker_id: Speaker ID for voice cloning
            target_duration: Target audio duration in seconds
            tolerance: Acceptable deviation as fraction (default 0.05 = ±5%)
            max_retries: Maximum duration matching attempts

        Returns:
            Tuple of (audio_array, metadata_dict) where metadata includes:
                - actual_duration: Final audio duration
                - target_duration: Target duration
                - duration_error: Absolute difference in seconds
                - speed_used: Speed adjustment factor applied
                - attempts_count: Number of synthesis attempts
                - tolerance_met: Whether duration is within tolerance
                - flagged: Whether segment needs manual review
        """
        # Handle short text warning
        if target_duration < TTS_SHORT_TEXT_THRESHOLD:
            print(f"WARNING: Short target duration ({target_duration:.2f}s < {TTS_SHORT_TEXT_THRESHOLD}s)")
            print("  XTTS may add artifacts or unnatural pacing for very short segments")

        # Track all attempts for best attempt selection
        attempts = []

        # First attempt with speed=1.0 (baseline)
        audio = self.synthesize_segment(text, speaker_id)
        actual_duration = len(audio) / TTS_SAMPLE_RATE
        duration_error = abs(actual_duration - target_duration)

        attempts.append({
            'speed': 1.0,
            'audio': audio,
            'actual_duration': actual_duration,
            'duration_error': duration_error
        })

        # Check if baseline meets tolerance
        tolerance_range = target_duration * tolerance
        if duration_error <= tolerance_range:
            return audio, {
                'actual_duration': actual_duration,
                'target_duration': target_duration,
                'duration_error': duration_error,
                'speed_used': 1.0,
                'attempts_count': 1,
                'tolerance_met': True,
                'flagged': False
            }

        # Binary search for speed adjustment
        # Speed > 1.0 = faster = shorter duration
        # Speed < 1.0 = slower = longer duration
        speed_min = TTS_SPEED_MIN
        speed_max = TTS_SPEED_MAX

        for attempt_num in range(2, max_retries + 1):
            # Determine search direction
            if actual_duration > target_duration:
                # Audio too long, need to speed up
                speed_min = 1.0
                speed = (1.0 + speed_max) / 2
            else:
                # Audio too short, need to slow down
                speed_max = 1.0
                speed = (speed_min + 1.0) / 2

            # Synthesize with adjusted speed
            # NOTE: TTS.api doesn't expose speed parameter directly
            # We'll need to use the low-level model.inference() method
            audio = self._synthesize_with_speed(text, speaker_id, speed)
            actual_duration = len(audio) / TTS_SAMPLE_RATE
            duration_error = abs(actual_duration - target_duration)

            attempts.append({
                'speed': speed,
                'audio': audio,
                'actual_duration': actual_duration,
                'duration_error': duration_error
            })

            # Check if tolerance met
            if duration_error <= tolerance_range:
                return audio, {
                    'actual_duration': actual_duration,
                    'target_duration': target_duration,
                    'duration_error': duration_error,
                    'speed_used': speed,
                    'attempts_count': attempt_num,
                    'tolerance_met': True,
                    'flagged': False
                }

        # Max retries exceeded - return best attempt (closest to target)
        best_attempt = min(attempts, key=lambda a: a['duration_error'])

        print(f"WARNING: Max retries ({max_retries}) reached for segment")
        print(f"  Target: {target_duration:.2f}s, Best: {best_attempt['actual_duration']:.2f}s")
        print(f"  Error: {best_attempt['duration_error']:.2f}s (tolerance: {tolerance_range:.2f}s)")

        return best_attempt['audio'], {
            'actual_duration': best_attempt['actual_duration'],
            'target_duration': target_duration,
            'duration_error': best_attempt['duration_error'],
            'speed_used': best_attempt['speed'],
            'attempts_count': len(attempts),
            'tolerance_met': False,
            'flagged': True  # Flag for manual review
        }

    def _synthesize_with_speed(
        self,
        text: str,
        speaker_id: str,
        speed: float
    ) -> np.ndarray:
        """
        Synthesize audio with speed adjustment using low-level API.

        XTTS TTS.api doesn't expose speed parameter, so we use the
        underlying model.inference() method directly.

        Args:
            text: Text to synthesize
            speaker_id: Speaker for voice cloning
            speed: Speed adjustment factor (0.8-1.2)

        Returns:
            Audio as numpy array at 24kHz
        """
        model = self._ensure_model_loaded()

        # Get speaker embeddings
        gpt_cond_latent, speaker_embedding = self.embedding_cache.get(speaker_id)

        # Use low-level inference API with speed parameter
        # This bypasses the high-level tts() wrapper
        audio = model.synthesizer.tts(
            text=text,
            language_name="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=self.temperature,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            speed=speed,  # Speed parameter for duration matching
            enable_text_splitting=True
        )

        # Convert to numpy array
        if isinstance(audio, dict) and 'wav' in audio:
            audio = audio['wav']

        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return audio

    def handle_short_text(self, text: str, target_duration: float) -> str:
        """
        Handle short text segments that may cause XTTS artifacts.

        For MVP: Log warning and return original text.
        Future: Could implement padding strategies or alternative synthesis.

        Args:
            text: Text to synthesize
            target_duration: Target duration in seconds

        Returns:
            Processed text (currently unchanged)
        """
        if target_duration < TTS_SHORT_TEXT_THRESHOLD:
            print(f"WARNING: Short text segment detected")
            print(f"  Text length: {len(text)} chars")
            print(f"  Target duration: {target_duration:.2f}s")
            print(f"  May produce artifacts or unnatural pacing")
            print(f"  Recommendation: Flag for manual review after synthesis")

        # For MVP, return original text
        # DO NOT add fake text to pad duration (causes lip sync issues)
        return text

    def synthesize_all_segments(
        self,
        segments: list[dict],
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> list[dict]:
        """
        Synthesize audio for all segments with speaker grouping.

        Groups segments by speaker for efficient embedding reuse.
        Synthesizes each segment with duration matching and saves to output_dir.

        Args:
            segments: List of segment dicts with:
                - segment_id: Unique segment identifier
                - speaker: Speaker ID (must be in embedding cache)
                - translated_text: English text to synthesize
                - duration: Target duration in seconds
            output_dir: Directory to save audio files
            progress_callback: Optional callback(progress, message) for UI updates

        Returns:
            List of result dicts with:
                - segment_id: Segment identifier
                - speaker: Speaker ID
                - audio_path: Path to saved WAV file
                - target_duration: Target duration
                - actual_duration: Actual synthesized duration
                - duration_error: Difference in seconds
                - speed_used: Speed adjustment factor
                - attempts: Number of synthesis attempts
                - tolerance_met: Whether duration is within tolerance
                - flagged: Whether needs manual review
                - failed: Whether synthesis failed (True if error occurred)

        Raises:
            BatchSynthesisError: If >20% of segments fail
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group segments by speaker for efficient processing
        speaker_groups = {}
        for segment in segments:
            speaker_id = segment['speaker']
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            speaker_groups[speaker_id].append(segment)

        results = []
        total_segments = len(segments)
        completed = 0
        failed_count = 0

        print(f"\nSynthesizing {total_segments} segments for {len(speaker_groups)} speakers...")

        # Process each speaker group
        for speaker_id, speaker_segments in speaker_groups.items():
            print(f"\n  Speaker {speaker_id}: {len(speaker_segments)} segments")

            # Verify speaker embeddings exist
            if not self.embedding_cache.has(speaker_id):
                print(f"    ERROR: Speaker {speaker_id} not in embedding cache")
                # Mark all segments for this speaker as failed
                for segment in speaker_segments:
                    results.append({
                        'segment_id': segment['segment_id'],
                        'speaker': speaker_id,
                        'audio_path': None,
                        'target_duration': segment.get('duration', 0),
                        'actual_duration': 0,
                        'duration_error': 0,
                        'speed_used': 0,
                        'attempts': 0,
                        'tolerance_met': False,
                        'flagged': True,
                        'failed': True
                    })
                    failed_count += 1
                    completed += 1
                continue

            # Synthesize each segment for this speaker
            for segment in speaker_segments:
                segment_id = segment['segment_id']
                text = segment['translated_text']
                target_duration = segment.get('duration', 0)

                try:
                    # Handle short text warning
                    text = self.handle_short_text(text, target_duration)

                    # Synthesize with duration matching
                    audio, metadata = self.synthesize_with_duration_matching(
                        text=text,
                        speaker_id=speaker_id,
                        target_duration=target_duration
                    )

                    # Save audio to file
                    audio_path = output_dir / f"segment_{segment_id}.wav"
                    sf.write(audio_path, audio, TTS_SAMPLE_RATE)

                    # Record results
                    results.append({
                        'segment_id': segment_id,
                        'speaker': speaker_id,
                        'audio_path': str(audio_path),
                        'target_duration': metadata['target_duration'],
                        'actual_duration': metadata['actual_duration'],
                        'duration_error': metadata['duration_error'],
                        'speed_used': metadata['speed_used'],
                        'attempts': metadata['attempts_count'],
                        'tolerance_met': metadata['tolerance_met'],
                        'flagged': metadata['flagged'],
                        'failed': False
                    })

                    completed += 1

                    # Report progress
                    if progress_callback:
                        progress = completed / total_segments
                        message = f"Synthesized segment {segment_id} ({completed}/{total_segments})"
                        progress_callback(progress, message)

                    print(f"    ✓ Segment {segment_id}: {metadata['actual_duration']:.2f}s "
                          f"(target: {target_duration:.2f}s, error: {metadata['duration_error']:.2f}s)")

                except Exception as e:
                    print(f"    ✗ Segment {segment_id} FAILED: {e}")

                    # Record failure
                    results.append({
                        'segment_id': segment_id,
                        'speaker': speaker_id,
                        'audio_path': None,
                        'target_duration': target_duration,
                        'actual_duration': 0,
                        'duration_error': 0,
                        'speed_used': 0,
                        'attempts': 0,
                        'tolerance_met': False,
                        'flagged': True,
                        'failed': True
                    })
                    failed_count += 1
                    completed += 1

        # Check failure rate
        failure_rate = failed_count / total_segments
        print(f"\n✓ Synthesis complete: {total_segments - failed_count}/{total_segments} successful")

        if failed_count > 0:
            print(f"  WARNING: {failed_count} segments failed ({failure_rate*100:.1f}%)")

        if failure_rate > 0.2:
            raise BatchSynthesisError(
                f"Batch synthesis failed: {failure_rate*100:.1f}% failure rate "
                f"({failed_count}/{total_segments} segments failed)"
            )

        return results


class BatchSynthesisError(Exception):
    """Raised when batch synthesis fails for >20% of segments."""
    pass
