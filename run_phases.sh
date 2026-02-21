#!/bin/bash
# =============================================================================
# run_phases.sh — Voice Dub: Run All Remaining Phases via GSD Workflow
# =============================================================================
#
# For each remaining phase this script runs the full cycle:
#   1. /gsd:discuss-phase  — gather context & clarify approach
#   2. /gsd:plan-phase     — create detailed PLAN.md files
#   3. /gsd:execute-phase  — execute every plan in the phase
#   4. /gsd:verify-work    — verify the phase goal was achieved
#
# Steps are skipped automatically when already done:
#   - Discuss  → skipped if {phase}-CONTEXT.md exists
#   - Plan     → skipped if {phase}-*-PLAN.md files exist
#   - Execute  → skipped if all SUMMARYs match all PLANs
#   - Verify   → skipped if {phase}-VERIFICATION.md exists  (phase fully done)
#
# Usage:
#   bash run_phases.sh              # run all remaining phases
#   bash run_phases.sh --from 7     # start from a specific phase
#   bash run_phases.sh --only 7 8   # run specific phases only
#   bash run_phases.sh --dry-run    # preview what would run without executing
#
# Requirements:
#   - claude CLI installed and authenticated
#   - Run from the project root directory
#   - Git Bash (Windows) or any bash-compatible shell
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNING_DIR="$SCRIPT_DIR/.planning"
PHASES_DIR="$PLANNING_DIR/phases"
LOG_FILE="$SCRIPT_DIR/run_phases.log"

# All phases in order — adjust if you add phases to the roadmap
ALL_PHASES=(1 2 3 4 5 6 7 8 9 10 11)

# Phases to process (overridden by --from / --only flags)
PHASES_TO_RUN=()

# Flags
DRY_RUN=false
FROM_PHASE=1
ONLY_PHASES=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --from)
            FROM_PHASE="$2"
            shift 2
            ;;
        --only)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ONLY_PHASES+=("$1")
                shift
            done
            ;;
        --help|-h)
            sed -n '/^# Usage:/,/^# =====/p' "$0" | grep -v '^# ====='
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Build the list of phases to run
if [[ ${#ONLY_PHASES[@]} -gt 0 ]]; then
    PHASES_TO_RUN=("${ONLY_PHASES[@]}")
else
    for p in "${ALL_PHASES[@]}"; do
        if [[ "$p" -ge "$FROM_PHASE" ]]; then
            PHASES_TO_RUN+=("$p")
        fi
    done
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    local msg="$1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

log_raw() {
    echo -e "$1"
    echo -e "$1" >> "$LOG_FILE"
}

pad() {
    printf "%02d" "$1"
}

# Get the phase directory path for a given phase number
get_phase_dir() {
    local num
    num=$(pad "$1")
    find "$PHASES_DIR" -maxdepth 1 -type d -name "${num}-*" 2>/dev/null | head -1
}

# Count files matching a glob inside a directory (safe with set -e)
count_files() {
    local dir="$1"
    local pattern="$2"
    shopt -s nullglob
    local files=("$dir"/$pattern)
    shopt -u nullglob
    echo "${#files[@]}"
}

# Check if phase is fully done (VERIFICATION.md exists)
is_verified() {
    local num
    num=$(pad "$1")
    local dir
    dir=$(get_phase_dir "$1")
    [[ -n "$dir" ]] && [[ -f "$dir/${num}-VERIFICATION.md" ]]
}

# Check if context/discussion is done
has_context() {
    local num
    num=$(pad "$1")
    local dir
    dir=$(get_phase_dir "$1")
    [[ -n "$dir" ]] && [[ -f "$dir/${num}-CONTEXT.md" ]]
}

# Check if plans exist
has_plans() {
    local num
    num=$(pad "$1")
    local dir
    dir=$(get_phase_dir "$1")
    [[ -n "$dir" ]] && [[ "$(count_files "$dir" "${num}-*-PLAN.md")" -gt 0 ]]
}

# Check if all plans have matching summaries
is_executed() {
    local num
    num=$(pad "$1")
    local dir
    dir=$(get_phase_dir "$1")
    [[ -z "$dir" ]] && return 1
    local plan_count summary_count
    plan_count=$(count_files "$dir" "${num}-*-PLAN.md")
    summary_count=$(count_files "$dir" "${num}-*-SUMMARY.md")
    [[ "$plan_count" -gt 0 ]] && [[ "$summary_count" -ge "$plan_count" ]]
}

# Run a claude GSD command (or print it in dry-run mode)
run_gsd() {
    local skill="$1"
    local phase="$2"
    local cmd="claude --dangerously-skip-permissions -p \"${skill} ${phase}\""

    log ""
    log "${CYAN}>>> ${skill} ${phase}${RESET}"
    log "${BLUE}────────────────────────────────────────────${RESET}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "${YELLOW}[DRY RUN] Would run: $cmd${RESET}"
        return 0
    fi

    # Run the claude CLI command
    if ! claude --dangerously-skip-permissions -p "${skill} ${phase}"; then
        log "${RED}ERROR: '${skill} ${phase}' exited with error.${RESET}"
        log "${YELLOW}Fix the error above, then re-run: bash run_phases.sh --from ${phase}${RESET}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

print_banner() {
    log ""
    log "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
    log "${BOLD}║       Voice Dub — Run All Phases             ║${RESET}"
    log "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
    log ""
    log "  Phases to process: ${PHASES_TO_RUN[*]}"
    log "  Dry run:           ${DRY_RUN}"
    log "  Log file:          $LOG_FILE"
    log ""

    if [[ "$DRY_RUN" == "true" ]]; then
        log "${YELLOW}DRY RUN MODE — no commands will execute${RESET}"
        log ""
    fi
}

# ---------------------------------------------------------------------------
# Phase status summary
# ---------------------------------------------------------------------------

print_status() {
    log "${BOLD}Current phase status:${RESET}"
    log ""
    log "  Phase  Status"
    log "  ─────  ──────────────────────────────────────────"
    for p in "${ALL_PHASES[@]}"; do
        local num status
        num=$(pad "$p")
        if is_verified "$p"; then
            status="${GREEN}✓ Complete${RESET}"
        elif is_executed "$p"; then
            status="${CYAN}⟳ Executed — needs verify${RESET}"
        elif has_plans "$p"; then
            local dir plan_count summary_count
            dir=$(get_phase_dir "$p")
            plan_count=$(count_files "$dir" "${num}-*-PLAN.md")
            summary_count=$(count_files "$dir" "${num}-*-SUMMARY.md")
            status="${YELLOW}▶ Planned ($summary_count/$plan_count done) — needs execute${RESET}"
        elif has_context "$p"; then
            status="${YELLOW}▶ Context gathered — needs plan${RESET}"
        else
            status="${RED}○ Not started${RESET}"
        fi
        log "  $num     $status"
    done
    log ""
}

# ---------------------------------------------------------------------------
# Main loop — one iteration per phase
# ---------------------------------------------------------------------------

run_phase() {
    local phase="$1"
    local num
    num=$(pad "$phase")

    log ""
    log "${BOLD}${BLUE}══════════════════════════════════════════════${RESET}"
    log "${BOLD}${BLUE}  PHASE $num${RESET}"
    log "${BOLD}${BLUE}══════════════════════════════════════════════${RESET}"

    # ── Already fully verified? ──────────────────────────────────────────
    if is_verified "$phase"; then
        log "${GREEN}  ✓ Phase $num already verified. Skipping.${RESET}"
        return 0
    fi

    local dir
    dir=$(get_phase_dir "$phase")

    # ── Step 1: Discuss ──────────────────────────────────────────────────
    if has_context "$phase"; then
        log "  [1/4] ${GREEN}Discuss: context exists. Skipping.${RESET}"
    else
        log "  [1/4] ${YELLOW}Discuss: gathering phase context...${RESET}"
        run_gsd "/gsd:discuss-phase" "$phase"
        # Refresh dir after potential creation
        dir=$(get_phase_dir "$phase")
    fi

    # ── Step 2: Plan ─────────────────────────────────────────────────────
    if has_plans "$phase"; then
        local plan_count
        dir=$(get_phase_dir "$phase")
        plan_count=$(count_files "$dir" "${num}-*-PLAN.md")
        log "  [2/4] ${GREEN}Plan: $plan_count plan(s) exist. Skipping.${RESET}"
    else
        log "  [2/4] ${YELLOW}Plan: creating execution plans...${RESET}"
        run_gsd "/gsd:plan-phase" "$phase"
    fi

    # ── Step 3: Execute ──────────────────────────────────────────────────
    if is_executed "$phase"; then
        log "  [3/4] ${GREEN}Execute: all plans complete. Skipping.${RESET}"
    else
        dir=$(get_phase_dir "$phase")
        local plan_count summary_count
        plan_count=$(count_files "$dir" "${num}-*-PLAN.md")
        summary_count=$(count_files "$dir" "${num}-*-SUMMARY.md")
        log "  [3/4] ${YELLOW}Execute: running plans ($summary_count/$plan_count done)...${RESET}"
        run_gsd "/gsd:execute-phase" "$phase"
    fi

    # ── Step 4: Verify ───────────────────────────────────────────────────
    log "  [4/4] ${YELLOW}Verify: checking phase goal achievement...${RESET}"
    run_gsd "/gsd:verify-work" "$phase"

    log ""
    log "  ${GREEN}${BOLD}Phase $num complete!${RESET}"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Initialise log
echo "=== run_phases.sh started $(date) ===" >> "$LOG_FILE"

print_banner
print_status

# Confirm before running (skip in dry-run)
if [[ "$DRY_RUN" == "false" ]]; then
    echo -e "${YELLOW}Ready to run phases: ${PHASES_TO_RUN[*]}${RESET}"
    echo -n "Press ENTER to continue, or Ctrl-C to cancel... "
    read -r
fi

PHASE_COUNT=0
SKIPPED_COUNT=0

for PHASE in "${PHASES_TO_RUN[@]}"; do
    if is_verified "$PHASE"; then
        ((SKIPPED_COUNT++)) || true
    fi
    run_phase "$PHASE"
    ((PHASE_COUNT++)) || true
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

log ""
log "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${RESET}"
log "${BOLD}${GREEN}║       ALL PHASES PROCESSED                   ║${RESET}"
log "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${RESET}"
log ""
log "  Phases processed : $PHASE_COUNT"
log "  Already complete : $SKIPPED_COUNT"
log ""
log "${CYAN}Next step:${RESET}"
log "  /gsd:complete-milestone  — archive milestone & start the next one"
log ""
echo "=== run_phases.sh finished $(date) ===" >> "$LOG_FILE"
