"""
Voice Dub - Main entry point.
Launches the full quality-controls web UI (Phase 8).
"""
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ui.app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        max_file_size="500mb",
        share=False,
    )
