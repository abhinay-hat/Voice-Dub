"""
Temporary file lifecycle management.
Provides context managers and utilities for creating and cleaning up temporary files
used during video processing pipeline.
"""
import tempfile
from pathlib import Path
from typing import Optional


class TempFileManager:
    """
    Context manager for temporary file lifecycle during video processing.

    Creates a temporary directory with pre-defined subdirectories for audio and frames.
    Automatically cleans up all files when exiting context.

    Attributes:
        temp_dir (Path): Root temporary directory path
        audio_path (Path): Pre-created path for audio.wav file
        frames_dir (Path): Pre-created directory for video frames

    Example:
        >>> with TempFileManager() as temp:
        ...     # Extract audio to temp.audio_path
        ...     extract_audio(video, temp.audio_path)
        ...     # Extract frames to temp.frames_dir
        ...     extract_frames(video, temp.frames_dir)
        ...     # Process...
        ... # Automatic cleanup when exiting context
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        prefix: str = "voicedub_"
    ):
        """
        Initialize temporary file manager.

        Args:
            base_dir: Base directory for temporary files. If None, uses system temp.
            prefix: Prefix for temporary directory name (default: "voicedub_")
        """
        self.base_dir = base_dir
        self.prefix = prefix
        self._temp_dir_obj = None
        self._temp_dir_path = None

    def __enter__(self) -> 'TempFileManager':
        """
        Create temporary directory and subdirectories.

        Returns:
            TempFileManager: Self for context manager protocol
        """
        # Create temporary directory
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir_obj = tempfile.TemporaryDirectory(
                prefix=self.prefix,
                dir=str(self.base_dir)
            )
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory(
                prefix=self.prefix
            )

        self._temp_dir_path = Path(self._temp_dir_obj.name)

        # Create subdirectories for common video processing needs
        self._frames_dir = self._temp_dir_path / "frames"
        self._frames_dir.mkdir(exist_ok=True)

        # Pre-define audio file path (file created by caller)
        self._audio_path = self._temp_dir_path / "audio.wav"

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up temporary directory and all contents.

        Args:
            exc_type: Exception type (if raised during context)
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            None: Exceptions propagate (cleanup happens regardless)
        """
        if self._temp_dir_obj:
            # TemporaryDirectory cleanup happens automatically
            # This cleans up all files and subdirectories
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None
            self._temp_dir_path = None

    @property
    def temp_dir(self) -> Path:
        """
        Get root temporary directory path.

        Returns:
            Path: Temporary directory path

        Raises:
            RuntimeError: If accessed outside context manager
        """
        if self._temp_dir_path is None:
            raise RuntimeError(
                "TempFileManager must be used as context manager "
                "(with TempFileManager() as temp:)"
            )
        return self._temp_dir_path

    @property
    def audio_path(self) -> Path:
        """
        Get pre-defined audio file path.

        Returns:
            Path: Path to audio.wav (file not created yet, use for output)

        Raises:
            RuntimeError: If accessed outside context manager
        """
        if self._audio_path is None:
            raise RuntimeError(
                "TempFileManager must be used as context manager"
            )
        return self._audio_path

    @property
    def frames_dir(self) -> Path:
        """
        Get pre-created frames directory path.

        Returns:
            Path: Directory for video frame storage (already created)

        Raises:
            RuntimeError: If accessed outside context manager
        """
        if self._frames_dir is None:
            raise RuntimeError(
                "TempFileManager must be used as context manager"
            )
        return self._frames_dir


def create_temp_directory(prefix: str = "voicedub_") -> Path:
    """
    Create temporary directory that caller must clean up manually.

    Use this function when context manager pattern doesn't fit your use case.
    Caller is responsible for deleting the directory when done.

    Args:
        prefix: Prefix for temporary directory name

    Returns:
        Path: Path to created temporary directory

    Example:
        >>> temp_dir = create_temp_directory()
        >>> # ... use temp_dir ...
        >>> shutil.rmtree(temp_dir)  # Manual cleanup required
    """
    temp_dir_obj = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir_obj)


def get_temp_file_path(temp_dir: Path, filename: str) -> Path:
    """
    Get path for a file within temporary directory.

    Creates parent directories if they don't exist.

    Args:
        temp_dir: Temporary directory path
        filename: Filename (can include subdirectories, e.g., "frames/frame_0001.png")

    Returns:
        Path: Full path to file within temp directory

    Example:
        >>> temp = Path("/tmp/voicedub_xyz")
        >>> frame_path = get_temp_file_path(temp, "frames/frame_0001.png")
        >>> # Creates /tmp/voicedub_xyz/frames/ if needed
        >>> # Returns /tmp/voicedub_xyz/frames/frame_0001.png
    """
    file_path = temp_dir / filename

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path
