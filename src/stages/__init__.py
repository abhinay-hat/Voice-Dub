"""
Processing stages for the voice dubbing pipeline.
Each stage handles a specific part of the video dubbing workflow.
"""

from .assembly_stage import (
    run_assembly_stage,
    AssemblyResult,
    AssemblyStageFailed,
)

__all__ = [
    "run_assembly_stage",
    "AssemblyResult",
    "AssemblyStageFailed",
]
