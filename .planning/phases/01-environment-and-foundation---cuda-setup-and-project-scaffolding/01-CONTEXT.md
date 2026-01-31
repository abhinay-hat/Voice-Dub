# Phase 1: Environment & Foundation - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish verified CUDA environment on RTX 5090 with PyTorch and create project scaffolding that prevents silent CPU fallback and enables efficient GPU memory management for the ML pipeline. This is infrastructure setup, not user-facing features.

</domain>

<decisions>
## Implementation Decisions

### Environment Validation
- Run GPU validation on every startup (not just initial setup) to catch driver issues
- Display detailed GPU information: name, available VRAM, CUDA version for troubleshooting
- Verify compute capability (RTX 5090 requires compute 9.0) to prevent weird model errors

### Project Structure
- Claude's discretion on directory organization (flat vs stage-based)
- Claude's discretion on model storage location (project dir vs system cache)
- Claude's discretion on temp file organization (per-video vs per-stage)
- Claude's discretion on config file approach (YAML vs hardcoded constants)

### Memory Management
- Load models on-demand per stage (not preloaded at startup)
- Unload each model before loading the next to minimize VRAM usage
- Claude's discretion on memory monitoring approach (logging frequency, real-time display)
- Claude's discretion on PYTORCH_CUDA_ALLOC_CONF settings for fragmentation prevention
- Claude's discretion on OOM error handling (fail fast vs auto-retry vs cleanup)

### Development Setup
- Claude's discretion on virtual environment choice (venv/conda/none)
- Claude's discretion on dependency pinning strategy (exact versions vs flexible)
- Claude's discretion on installation approach (script vs docs vs both)
- Claude's discretion on post-install validation tests

### Claude's Discretion
- Best practice for GPU failure handling on startup
- Optimal directory structure for 11-phase ML pipeline
- Model storage strategy (local vs cache)
- Temp file organization pattern
- Configuration file approach
- Memory monitoring verbosity
- CUDA allocator configuration
- OOM recovery strategy
- Virtual environment tooling
- Dependency version management
- Installation documentation format
- Post-install validation depth

</decisions>

<specifics>
## Specific Ideas

No specific requirements - open to standard approaches for CUDA environment setup and Python project structure. User trusts Claude to make sensible choices for infrastructure decisions.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope (environment and foundation only).

</deferred>

---

*Phase: 01-environment-and-foundation*
*Context gathered: 2026-01-31*
