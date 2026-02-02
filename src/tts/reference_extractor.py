"""
Reference sample extraction for voice cloning.
Selects 6-10 second clean audio samples per speaker for XTTS-v2 conditioning.
"""
from pathlib import Path
from typing import Optional, List, Dict
import json
import numpy as np
import librosa
import soundfile as sf


class ReferenceExtractor:
    """
    Extracts reference audio samples from full audio file with lazy loading.
    """

    def __init__(self, audio_path: Path, sample_rate: int = 24000):
        """
        Initialize reference extractor.

        Args:
            audio_path: Path to full audio file
            sample_rate: Target sample rate (24000 for XTTS-v2)
        """
        self.audio_path = Path(audio_path)
        self.sample_rate = sample_rate
        self._audio = None  # Lazy loaded

    @property
    def audio(self) -> np.ndarray:
        """Lazy load audio on first access."""
        if self._audio is None:
            self._audio, _ = librosa.load(
                self.audio_path,
                sr=self.sample_rate,
                mono=True  # XTTS requires mono
            )
        return self._audio

    def extract_segment(
        self,
        start: float,
        end: float,
        output_path: Path
    ) -> Path:
        """
        Extract audio segment and save to file.

        Args:
            start: Start time in seconds
            end: End time in seconds
            output_path: Where to save extracted segment

        Returns:
            Path to saved audio file
        """
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)

        segment_audio = self.audio[start_sample:end_sample]

        # Validate audio is mono
        if len(segment_audio.shape) > 1:
            segment_audio = segment_audio.mean(axis=1)

        # Save as WAV at 24kHz (XTTS native format)
        sf.write(output_path, segment_audio, self.sample_rate)

        return output_path


def select_best_segment(
    speaker_segments: List[Dict],
    min_duration: float = 6.0,
    max_duration: float = 10.0,
    max_gap: float = 0.5
) -> Optional[Dict]:
    """
    Select best reference segment for speaker based on RMS energy.

    Strategy:
    1. Filter segments >= min_duration
    2. Calculate RMS energy for each (proxy for clean audio)
    3. Return segment with highest RMS energy
    4. If no single segment long enough, concatenate adjacent segments
    5. If concatenation still insufficient, return None

    Args:
        speaker_segments: List of segment dicts with keys:
            - speaker: Speaker ID
            - start: Start time (seconds)
            - end: End time (seconds)
            - duration: Duration (seconds)
        min_duration: Minimum acceptable duration (default 6.0s)
        max_duration: Maximum acceptable duration (default 10.0s)
        max_gap: Max gap between segments for concatenation (default 0.5s)

    Returns:
        Best segment dict or None if no viable segment found
        If concatenated, returns dict with combined timing
    """
    if not speaker_segments:
        return None

    # Sort segments by start time for concatenation logic
    segments = sorted(speaker_segments, key=lambda s: s['start'])

    # Filter segments >= min_duration
    long_segments = [s for s in segments if s['duration'] >= min_duration]

    if long_segments:
        # Simple case: select highest energy segment
        # Note: We can't calculate RMS here without audio access
        # So we select the longest segment as proxy for clean audio
        best_segment = max(long_segments, key=lambda s: s['duration'])

        # If segment exceeds max_duration, we'll extract center portion
        if best_segment['duration'] > max_duration:
            # Calculate centered window
            center_time = (best_segment['start'] + best_segment['end']) / 2
            half_duration = max_duration / 2

            # Create adjusted segment
            return {
                'speaker': best_segment['speaker'],
                'start': center_time - half_duration,
                'end': center_time + half_duration,
                'duration': max_duration,
                'original_segment_id': best_segment.get('id', -1)
            }

        return best_segment

    # No single long segment - try concatenation
    print(f"No single segment >= {min_duration}s, attempting concatenation...")

    # Find best concatenation sequence
    best_concat = None
    best_total_duration = 0

    for i in range(len(segments)):
        concat_segments = [segments[i]]
        total_duration = segments[i]['duration']

        # Try adding adjacent segments
        for j in range(i + 1, len(segments)):
            gap = segments[j]['start'] - segments[j-1]['end']

            # Stop if gap too large
            if gap > max_gap:
                break

            concat_segments.append(segments[j])
            total_duration += segments[j]['duration']

            # Check if we've reached min_duration
            if total_duration >= min_duration:
                # Found viable concatenation
                if total_duration > best_total_duration:
                    best_total_duration = total_duration
                    best_concat = concat_segments
                break

    if best_concat:
        # Build combined segment
        first_seg = best_concat[0]
        last_seg = best_concat[-1]

        combined_duration = last_seg['end'] - first_seg['start']

        # If combined exceeds max_duration, extract center portion
        if combined_duration > max_duration:
            center_time = (first_seg['start'] + last_seg['end']) / 2
            half_duration = max_duration / 2

            return {
                'speaker': first_seg['speaker'],
                'start': center_time - half_duration,
                'end': center_time + half_duration,
                'duration': max_duration,
                'concatenated': True,
                'num_segments': len(best_concat)
            }

        return {
            'speaker': first_seg['speaker'],
            'start': first_seg['start'],
            'end': last_seg['end'],
            'duration': combined_duration,
            'concatenated': True,
            'num_segments': len(best_concat)
        }

    # No viable segment found
    print(f"WARNING: Could not find or concatenate segments >= {min_duration}s")
    return None


def extract_reference_samples(
    translation_json: Path,
    audio_path: Path,
    output_dir: Path,
    min_duration: float = 6.0,
    max_duration: float = 10.0
) -> Dict[str, Path]:
    """
    Extract reference audio samples for each speaker from translation JSON.

    Args:
        translation_json: Path to translation stage output JSON
        audio_path: Path to extracted audio file (from video processing)
        output_dir: Directory to save reference samples
        min_duration: Minimum reference duration (default 6.0s for XTTS)
        max_duration: Maximum reference duration (default 10.0s)

    Returns:
        Dict mapping speaker_id -> reference_audio_path
        Speakers with no viable reference are excluded from dict

    Example:
        >>> refs = extract_reference_samples(
        ...     translation_json=Path("data/temp/video_translation.json"),
        ...     audio_path=Path("data/temp/video_audio.wav"),
        ...     output_dir=Path("data/temp/references")
        ... )
        >>> print(refs)
        {'SPEAKER_01': Path('data/temp/references/reference_SPEAKER_01.wav'), ...}
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load translation JSON to get speaker segments
    print(f"Loading translation data from: {translation_json}")
    with open(translation_json, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)

    segments = translation_data.get('segments', [])

    if not segments:
        print("WARNING: No segments found in translation JSON")
        return {}

    # Group segments by speaker
    speaker_segments = {}
    for segment in segments:
        speaker_id = segment['speaker']

        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []

        speaker_segments[speaker_id].append({
            'id': segment['segment_id'],
            'speaker': speaker_id,
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['duration']
        })

    print(f"Found {len(speaker_segments)} unique speakers")

    # Initialize reference extractor
    extractor = ReferenceExtractor(audio_path, sample_rate=24000)

    # Extract reference for each speaker
    reference_paths = {}

    for speaker_id, segments_list in speaker_segments.items():
        print(f"\nProcessing speaker: {speaker_id} ({len(segments_list)} segments)")

        # Select best segment
        best_segment = select_best_segment(
            segments_list,
            min_duration=min_duration,
            max_duration=max_duration
        )

        if best_segment is None:
            total_duration = sum(s['duration'] for s in segments_list)
            print(f"  WARNING: No viable reference for {speaker_id} "
                  f"(total duration: {total_duration:.2f}s)")
            continue

        # Extract audio segment
        ref_filename = f"reference_{speaker_id}.wav"
        ref_path = output_dir / ref_filename

        try:
            extractor.extract_segment(
                start=best_segment['start'],
                end=best_segment['end'],
                output_path=ref_path
            )

            reference_paths[speaker_id] = ref_path

            concatenated_info = ""
            if best_segment.get('concatenated', False):
                concatenated_info = f" (concatenated {best_segment['num_segments']} segments)"

            print(f"  ✓ Extracted {best_segment['duration']:.2f}s reference{concatenated_info}")
            print(f"    Saved to: {ref_path}")

        except Exception as e:
            print(f"  ERROR: Failed to extract reference for {speaker_id}: {e}")
            continue

    print(f"\n✓ Successfully extracted {len(reference_paths)}/{len(speaker_segments)} speaker references")

    return reference_paths
