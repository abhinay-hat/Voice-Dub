"""
Speaker embedding generation and caching for XTTS-v2 voice cloning.
Generates conditioning latents once per speaker and caches for reuse.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch


class SpeakerEmbeddingCache:
    """
    Cache for XTTS speaker embeddings (conditioning latents).

    Stores speaker_id -> (gpt_cond_latent, speaker_embedding) tuples.
    Tracks GPU/CPU location for VRAM management.
    """

    def __init__(self):
        """Initialize empty embedding cache."""
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._locations: Dict[str, str] = {}  # Track 'cpu' or 'cuda'

    def put(
        self,
        speaker_id: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor
    ) -> None:
        """
        Store speaker embeddings in cache.

        Args:
            speaker_id: Unique speaker identifier
            gpt_cond_latent: GPT conditioning latent tensor
            speaker_embedding: Speaker embedding tensor
        """
        self._cache[speaker_id] = (gpt_cond_latent, speaker_embedding)

        # Detect location from tensor
        if gpt_cond_latent.is_cuda:
            self._locations[speaker_id] = 'cuda'
        else:
            self._locations[speaker_id] = 'cpu'

    def get(self, speaker_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve speaker embeddings from cache.

        Args:
            speaker_id: Unique speaker identifier

        Returns:
            (gpt_cond_latent, speaker_embedding) tuple or None if not found
        """
        return self._cache.get(speaker_id)

    def has(self, speaker_id: str) -> bool:
        """Check if speaker embeddings exist in cache."""
        return speaker_id in self._cache

    def move_to_cpu(self, speaker_id: str) -> None:
        """
        Move speaker embeddings from GPU to CPU.

        Useful for VRAM management when processing many speakers.

        Args:
            speaker_id: Speaker to move to CPU
        """
        if speaker_id not in self._cache:
            return

        gpt_cond_latent, speaker_embedding = self._cache[speaker_id]

        # Move to CPU if currently on GPU
        if gpt_cond_latent.is_cuda:
            gpt_cond_latent = gpt_cond_latent.cpu()
            speaker_embedding = speaker_embedding.cpu()

            self._cache[speaker_id] = (gpt_cond_latent, speaker_embedding)
            self._locations[speaker_id] = 'cpu'

    def move_to_gpu(self, speaker_id: str, device: str = 'cuda') -> None:
        """
        Move speaker embeddings from CPU to GPU.

        Args:
            speaker_id: Speaker to move to GPU
            device: Target device (default 'cuda')
        """
        if speaker_id not in self._cache:
            return

        gpt_cond_latent, speaker_embedding = self._cache[speaker_id]

        # Move to GPU if currently on CPU
        if not gpt_cond_latent.is_cuda:
            gpt_cond_latent = gpt_cond_latent.to(device)
            speaker_embedding = speaker_embedding.to(device)

            self._cache[speaker_id] = (gpt_cond_latent, speaker_embedding)
            self._locations[speaker_id] = device

    def clear(self) -> None:
        """Clear all cached embeddings and free memory."""
        # Move all to CPU first to ensure cleanup
        for speaker_id in list(self._cache.keys()):
            self.move_to_cpu(speaker_id)

        self._cache.clear()
        self._locations.clear()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_location(self, speaker_id: str) -> Optional[str]:
        """Get device location ('cpu' or 'cuda') for speaker embeddings."""
        return self._locations.get(speaker_id)

    def list_speakers(self) -> list:
        """Return list of all cached speaker IDs."""
        return list(self._cache.keys())

    def __len__(self) -> int:
        """Return number of cached speakers."""
        return len(self._cache)

    def __repr__(self) -> str:
        """String representation of cache."""
        gpu_count = sum(1 for loc in self._locations.values() if loc == 'cuda')
        cpu_count = len(self._cache) - gpu_count
        return f"SpeakerEmbeddingCache({len(self._cache)} speakers: {gpu_count} GPU, {cpu_count} CPU)"


def generate_single_embedding(
    xtts_model,
    audio_path: Path
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate XTTS speaker embedding from reference audio.

    Thin wrapper around model.get_conditioning_latents() with validation.

    Args:
        xtts_model: Loaded XTTS model instance
        audio_path: Path to reference audio file (6-10s sample)

    Returns:
        (gpt_cond_latent, speaker_embedding) tuple or None on error

    Example:
        >>> from TTS.api import TTS
        >>> xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        >>> embeddings = generate_single_embedding(xtts, Path("reference_speaker1.wav"))
        >>> gpt_cond, speaker_emb = embeddings
    """
    audio_path = Path(audio_path)

    # Validate audio file exists
    if not audio_path.exists():
        print(f"ERROR: Reference audio not found: {audio_path}")
        return None

    if not audio_path.is_file():
        print(f"ERROR: Path is not a file: {audio_path}")
        return None

    try:
        # XTTS get_conditioning_latents() accepts string path
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=str(audio_path)
        )

        return (gpt_cond_latent, speaker_embedding)

    except Exception as e:
        print(f"ERROR: Failed to generate embedding for {audio_path}: {e}")
        return None


def generate_speaker_embeddings(
    xtts_model,
    reference_paths: Dict[str, Path]
) -> SpeakerEmbeddingCache:
    """
    Generate XTTS embeddings for all speakers and cache results.

    Processes reference audio for each speaker, generates conditioning latents,
    and stores in SpeakerEmbeddingCache for reuse during synthesis.

    Args:
        xtts_model: Loaded XTTS model instance (from ModelManager)
        reference_paths: Dict mapping speaker_id -> reference_audio_path

    Returns:
        SpeakerEmbeddingCache containing all successfully generated embeddings

    Example:
        >>> from src.models.model_manager import ModelManager
        >>> from TTS.api import TTS
        >>>
        >>> manager = ModelManager()
        >>> xtts = manager.load_model("xtts", lambda: TTS(
        ...     "tts_models/multilingual/multi-dataset/xtts_v2", gpu=True
        ... ))
        >>>
        >>> reference_paths = {
        ...     "SPEAKER_01": Path("references/reference_SPEAKER_01.wav"),
        ...     "SPEAKER_02": Path("references/reference_SPEAKER_02.wav")
        ... }
        >>>
        >>> cache = generate_speaker_embeddings(xtts, reference_paths)
        >>> print(f"Generated {len(cache)} speaker embeddings")
    """
    cache = SpeakerEmbeddingCache()

    print(f"\nGenerating speaker embeddings for {len(reference_paths)} speakers...")

    successful = 0
    failed = 0

    for speaker_id, audio_path in reference_paths.items():
        print(f"  Processing {speaker_id}...")

        embeddings = generate_single_embedding(xtts_model, audio_path)

        if embeddings is not None:
            gpt_cond_latent, speaker_embedding = embeddings
            cache.put(speaker_id, gpt_cond_latent, speaker_embedding)
            successful += 1
            print(f"    ✓ Embedding generated (device: {cache.get_location(speaker_id)})")
        else:
            failed += 1
            print(f"    ✗ Failed to generate embedding")

    print(f"\n✓ Successfully generated {successful}/{len(reference_paths)} embeddings")

    if failed > 0:
        print(f"  WARNING: {failed} speakers failed embedding generation")

    # Report VRAM usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        print(f"  VRAM: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

    return cache
