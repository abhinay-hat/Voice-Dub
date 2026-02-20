"""
Processing stages for the voice dubbing pipeline.
Each stage handles a specific part of the video dubbing workflow.
"""

from .assembly_stage import (
    run_assembly_stage,
    AssemblyResult,
    AssemblyStageFailed,
)
from .lip_sync_stage import (
    run_lip_sync_stage,
    LipSyncResult,
    LipSyncStageFailed,
)

__all__ = [
    # Assembly stage
    "run_assembly_stage",
    "AssemblyResult",
    "AssemblyStageFailed",
    # Lip sync stage
    "run_lip_sync_stage",
    "LipSyncResult",
    "LipSyncStageFailed",
]
