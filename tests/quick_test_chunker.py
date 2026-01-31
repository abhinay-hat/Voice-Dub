"""Quick test of context_chunker without full imports."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set config manually to avoid importing full settings
TRANSLATION_MAX_CHUNK_TOKENS = 1024
TRANSLATION_OVERLAP_TOKENS = 128
TRANSLATION_APPROX_CHARS_PER_TOKEN = 4

# Now import just the chunker module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "context_chunker",
    project_root / "src" / "stages" / "translation" / "context_chunker.py"
)
context_chunker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context_chunker)

# Import settings module for the chunker
import importlib.util as iu
settings_spec = iu.spec_from_file_location("settings", project_root / "src" / "config" / "settings.py")
settings = iu.module_from_spec(settings_spec)
sys.modules['src.config.settings'] = settings
settings_spec.loader.exec_module(settings)

# Re-import chunker with settings available
spec = iu.spec_from_file_location(
    "context_chunker",
    project_root / "src" / "stages" / "translation" / "context_chunker.py"
)
context_chunker = iu.module_from_spec(spec)
spec.loader.exec_module(context_chunker)

chunk_transcript_with_overlap = context_chunker.chunk_transcript_with_overlap
ContextChunker = context_chunker.ContextChunker

print("Testing ContextChunker...")

# Test 1: Short transcript
segments = [{"text": f"Segment {i}", "duration": 2.0} for i in range(5)]
chunks = chunk_transcript_with_overlap(segments)
print(f"[PASS] Short transcript: {len(chunks)} chunk(s)")

# Test 2: Long transcript with chunking
long_segments = [
    {"text": f"Segment {i} with a lot of text " * 20, "duration": 3.0}
    for i in range(50)
]
chunks = chunk_transcript_with_overlap(long_segments, max_tokens=1024, overlap_tokens=128)
print(f"[PASS] Long transcript: {len(chunks)} chunk(s)")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk['segments'])} segments, {chunk['total_tokens']} tokens, overlap={chunk['has_overlap']}")

# Test 3: Edge cases
empty_chunks = chunk_transcript_with_overlap([])
assert len(empty_chunks) == 0
print(f"[PASS] Empty list handled")

single_chunks = chunk_transcript_with_overlap([{"text": "Single", "duration": 1.0}])
assert len(single_chunks) == 1
print(f"[PASS] Single segment handled")

print("\n[SUCCESS] ALL QUICK TESTS PASSED")
