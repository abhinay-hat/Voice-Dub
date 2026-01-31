---
phase: 01-environment-and-foundation
plan: 02
subsystem: infrastructure
tags: [project-structure, vram-management, memory-monitoring, model-loading]
requires: [01-01]
provides:
  - ML pipeline directory structure (stage-based architecture)
  - ModelManager for sequential model loading/unloading
  - GPU memory monitoring utilities
  - Centralized configuration management
affects: [all-future-phases]
tech-stack:
  added: []
  patterns:
    - Sequential model loading (prevents VRAM exhaustion)
    - 4-step cleanup protocol (CPU move, delete, gc, empty_cache)
    - Centralized settings pattern
key-files:
  created:
    - src/models/model_manager.py
    - src/utils/memory_monitor.py
    - src/config/settings.py
    - data/raw/.gitkeep, data/temp/.gitkeep, data/outputs/.gitkeep
    - models/.gitkeep, tests/.gitkeep
    - src/pipeline/.gitkeep, src/stages/.gitkeep
  modified:
    - .gitignore
decisions:
  - id: sequential-model-loading
    title: Sequential model loading pattern
    impact: Prevents VRAM exhaustion on 32GB RTX 5090
    rationale: Loading all models simultaneously (Whisper + SeamlessM4T + XTTS + Wav2Lip) would exceed 32GB VRAM. Sequential loading per stage maximizes available VRAM for each model.
  - id: 4-step-cleanup
    title: 4-step model cleanup protocol
    impact: Ensures complete VRAM release without memory leaks
    rationale: Move to CPU → delete references → garbage collect → empty CUDA cache sequence prevents circular reference leaks and ensures PyTorch releases cached memory.
  - id: stage-based-directory-structure
    title: Stage-based src/ organization
    impact: Maps to 11-phase pipeline architecture
    rationale: Each pipeline phase corresponds to a stage module (src/stages/), enabling modular development and clear separation of concerns.
metrics:
  duration: 5.5 minutes
  completed: 2026-01-31
---

# Phase 01 Plan 02: Project Scaffolding and VRAM Management Summary

**One-liner:** ML pipeline directory structure with ModelManager implementing sequential loading pattern (prevents 32GB VRAM exhaustion) and real-time memory monitoring utilities.

## What Was Built

Created complete project scaffold for 11-phase ML pipeline and implemented VRAM management infrastructure for RTX 5090 (32GB).

### Directory Structure

```
voice-dub/
├── src/
│   ├── pipeline/          # Stage orchestration (future)
│   ├── stages/            # Individual processing stages (future)
│   ├── models/            # Model loading/unloading logic ✓
│   │   └── model_manager.py
│   ├── utils/             # GPU validation, memory monitoring ✓
│   │   └── memory_monitor.py
│   └── config/            # Configuration constants ✓
│       └── settings.py
├── data/
│   ├── raw/               # Original uploaded videos
│   ├── temp/              # Intermediate files (per-video subdirectories)
│   └── outputs/           # Final processed videos
├── models/                # Downloaded model weights (cache)
└── tests/                 # Test files
```

### ModelManager (src/models/model_manager.py)

**Purpose:** Prevents VRAM exhaustion by ensuring only one model loaded at a time.

**Key Features:**
- `load_model(name, loader_fn)` - Loads model, unloading previous first
- `unload_current_model()` - Manual unload without loading next
- `_unload_current_model()` - 4-step cleanup protocol
- `get_current_model_name()` - Returns currently loaded model
- Verbose logging tracks VRAM allocated/reserved

**4-Step Cleanup Protocol:**
1. Move model to CPU (safer for complex models)
2. Delete Python references
3. Force garbage collection (clears circular refs)
4. Clear CUDA cache (releases memory to GPU)

**Usage Pattern:**
```python
manager = ModelManager()
whisper = manager.load_model("whisper", lambda: load_whisper())
# ... use whisper ...
seamless = manager.load_model("seamless", lambda: load_seamless())
# Whisper automatically unloaded before Seamless loads
```

### Memory Monitoring (src/utils/memory_monitor.py)

**Purpose:** Real-time VRAM tracking during pipeline execution.

**Functions:**
- `get_gpu_memory_info()` - Returns dict with allocated/reserved/free/total (GB)
- `print_gpu_memory_summary(prefix)` - Formatted console output
- `get_memory_summary_string()` - String for logging integration

**Sample Output:**
```
GPU Memory: 8.24GB allocated | 10.50GB reserved | 21.50GB free | 31.84GB total
```

### Configuration (src/config/settings.py)

**Purpose:** Centralized constants (no magic numbers scattered in code).

**Key Settings:**
- Project paths (PROJECT_ROOT, MODELS_DIR, DATA_DIR subdirs)
- GPU config (RTX 5090: sm_120, 32GB VRAM)
- Memory management (PYTORCH_CUDA_ALLOC_CONF)
- Model names (Whisper large-v3, SeamlessM4T v2, XTTS v2, Wav2Lip)
- Pipeline parameters (max duration, supported formats, sample rate)

## Decisions Made

### 1. Sequential Model Loading Pattern
**Decision:** Load models on-demand per stage, unload before next stage.
**Rationale:** All models simultaneously (Whisper ~6GB + SeamlessM4T ~8GB + XTTS ~4GB + Wav2Lip ~8GB ≈ 26-30GB) approaches 32GB limit. Sequential loading provides full VRAM budget to each model.
**Impact:** Prevents OOM errors, enables larger batch sizes per stage, clearer memory profile.

### 2. 4-Step Cleanup Protocol
**Decision:** Move to CPU → delete refs → gc → empty_cache sequence.
**Rationale:** Python circular references can prevent GPU memory release. Garbage collection before cache clearing is critical.
**Impact:** Ensures VRAM returns to baseline after each stage (verified in tests: allocated drops to 0.00GB).

### 3. Stage-Based Directory Structure
**Decision:** src/pipeline/ for orchestration, src/stages/ for individual processing stages.
**Rationale:** Maps directly to 11-phase pipeline. Each phase adds new stage module (e.g., src/stages/speech_to_text.py).
**Impact:** Modular development, clear separation of concerns, easy to test stages independently.

## Technical Implementation

### Task 1: Directory Structure
**Commits:** 7019185
- Created src/pipeline, src/stages, src/models, src/config directories
- Created data/raw, data/temp, data/outputs subdirectories
- Added .gitkeep files to preserve empty directories in git
- Updated .gitignore to preserve structure but exclude content

### Task 2: ModelManager
**Commits:** e882993
- Implemented ModelManager class with load_model/unload_current_model methods
- 4-step cleanup ensures complete VRAM release
- Verbose logging tracks VRAM during operations
- Test script verified sequential loading works (unload → load cycle)

### Task 3: Memory Monitoring & Configuration
**Commits:** 65bf7f7
- get_gpu_memory_info() returns structured dict (allocated/reserved/free/total)
- print_gpu_memory_summary() for interactive debugging
- get_memory_summary_string() for logging integration
- settings.py centralizes all project paths and constants
- Test scripts verified memory tracking accuracy (±0.04GB for 40MB allocation)

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Recommendations:**
1. Next plan should validate environment end-to-end (GPU + PyTorch + directory structure)
2. Future plans can import ModelManager and use sequential loading pattern
3. All stage implementations should use memory_monitor for VRAM tracking
4. All paths should reference settings.py constants (no hardcoded paths)

## Integration Notes

**For Future Developers:**

1. **Using ModelManager:**
   ```python
   from src.models.model_manager import ModelManager

   manager = ModelManager(verbose=True)
   model = manager.load_model("stage-name", lambda: load_model_fn())
   # ModelManager automatically unloads previous model before loading
   ```

2. **Memory Monitoring:**
   ```python
   from src.utils.memory_monitor import print_gpu_memory_summary

   print_gpu_memory_summary("Before stage: ")
   # ... run stage ...
   print_gpu_memory_summary("After stage: ")
   ```

3. **Configuration:**
   ```python
   from src.config.settings import MODELS_DIR, WHISPER_MODEL

   model_path = MODELS_DIR / WHISPER_MODEL
   ```

**Dependencies:**
- Requires: 01-01 (PyTorch CUDA environment)
- Provides: Infrastructure for all future phases
- Affects: All phases (directory structure, ModelManager, memory monitoring)

## Validation Results

**Directory Structure:**
- ✅ All directories exist (src/pipeline, src/stages, src/models, src/utils, src/config, data/raw, data/temp, data/outputs, models, tests)
- ✅ .gitkeep files preserve empty directories
- ✅ .gitignore preserves structure but excludes content

**ModelManager:**
- ✅ Sequential loading works (test shows unload → load cycle)
- ✅ VRAM cleared after unloading (allocated drops to 0.00GB)
- ✅ get_current_model_name() returns correct model or None

**Memory Monitoring:**
- ✅ get_gpu_memory_info() returns dict with all keys (allocated/reserved/free/total)
- ✅ Memory tracking accurate (±0.04GB for ~40MB allocation)
- ✅ Shows baseline 31.84GB total VRAM (RTX 5090)

**Configuration:**
- ✅ settings.py loads successfully
- ✅ PROJECT_ROOT resolves correctly
- ✅ All paths defined (MODELS_DIR, DATA_DIR subdirs)

## Files Modified

**Created:**
- src/models/model_manager.py (132 lines)
- src/utils/memory_monitor.py (77 lines)
- src/config/settings.py (42 lines)
- src/__init__.py, src/models/__init__.py, src/utils/__init__.py, src/config/__init__.py
- data/raw/.gitkeep, data/temp/.gitkeep, data/outputs/.gitkeep
- models/.gitkeep, tests/.gitkeep
- src/pipeline/.gitkeep, src/stages/.gitkeep

**Modified:**
- .gitignore (added !.gitkeep exceptions)

## Commits

| Hash    | Type     | Description                                       |
|---------|----------|---------------------------------------------------|
| 7019185 | chore    | Create ML pipeline directory structure            |
| e882993 | feat     | Implement ModelManager for sequential loading     |
| 65bf7f7 | feat     | Add memory monitoring and configuration utilities |

**Total Duration:** 5.5 minutes
**Completed:** 2026-01-31
