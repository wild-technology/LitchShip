#!/bin/bash
# =============================================================================
# Progressive Training Pipeline — Growing-Seed ADC Strategy
#
# Runs 3-stage Gaussian splatting training using tiered point clouds:
#   Stage 1 (seed):   Best points only → clean geometric scaffold
#   Stage 2 (expand): Medium+ points → fill spatial gaps
#   Stage 3 (refine): All filtered points → full detail recovery
#
# Each stage is a fresh ADC training run. The progressive benefit comes from
# the trainer seeing increasingly rich seed geometry in points3D.txt.
#
# Usage:
#   ./progressive_train.sh /path/to/staging           # staging/ from build_tiers.py
#   ./progressive_train.sh /path/to/staging --compare  # also run single-pass baseline
#   ./progressive_train.sh /path/to/staging --dry-run   # print commands only
#
# Requires: LichtFeld Studio binary built via setup_lichtfeld.sh
# =============================================================================
set -euo pipefail

# Locate the LitchShip root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITCHSHIP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$LITCHSHIP_ROOT/lib/common.sh"

# ─── Arguments ───────────────────────────────────────────────────────────────

STAGING_DIR=""
DRY_RUN=false
COMPARE=false
STAGE_ITERS=10000
RESIZE_FACTOR=2
REPO_DIR="$HOME/gaussian-splatting-cuda"

usage() {
    error "Usage: $0 <staging-dir> [options]"
    echo ""
    echo "  staging-dir    Directory with tier1/, tier2/, tier3/ from build_tiers.py"
    echo ""
    echo "Options:"
    echo "  --dry-run      Print training commands without executing"
    echo "  --compare      Also run single-pass 30K baseline for A/B comparison"
    echo "  --iters N      Iterations per stage (default: $STAGE_ITERS)"
    echo "  --resize N     Resize factor (default: $RESIZE_FACTOR)"
    echo "  --repo DIR     Path to gaussian-splatting-cuda (default: $REPO_DIR)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --compare)   COMPARE=true; shift ;;
        --iters)     STAGE_ITERS="$2"; shift 2 ;;
        --resize)    RESIZE_FACTOR="$2"; shift 2 ;;
        --repo)      REPO_DIR="$2"; shift 2 ;;
        -h|--help)   usage ;;
        -*)          error "Unknown option: $1"; usage ;;
        *)
            if [ -z "$STAGING_DIR" ]; then
                STAGING_DIR="$1"
            else
                error "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

[ -z "$STAGING_DIR" ] && usage
STAGING_DIR="$(cd "$STAGING_DIR" && pwd)"

# ─── Validate ─────────────────────────────────────────────────────────────────

for tier in tier1 tier2 tier3; do
    if [ ! -d "$STAGING_DIR/$tier/sparse/0" ]; then
        error "Missing $STAGING_DIR/$tier/sparse/0 — run build_tiers.py first"
        exit 1
    fi
    if [ ! -f "$STAGING_DIR/$tier/sparse/0/points3D.txt" ]; then
        error "Missing points3D.txt in $STAGING_DIR/$tier/sparse/0/"
        exit 1
    fi
done

# ─── Setup ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  Progressive Training — Growing Seed ADC${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""

setup_paths "$REPO_DIR"

if [ "$DRY_RUN" = false ]; then
    detect_gpu
    find_binary "$REPO_DIR"
else
    BINARY="<LichtFeld-Studio>"
    log "Dry run — skipping GPU detection and binary discovery"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_BASE="$HOME/output/progressive_${TIMESTAMP}"

echo ""
info "Configuration:"
echo "  Staging:      $STAGING_DIR"
echo "  Output base:  $OUTPUT_BASE"
echo "  Iters/stage:  $STAGE_ITERS"
echo "  Resize:       $RESIZE_FACTOR"
echo "  Strategy:     ADC (all stages)"
echo ""

# ─── Stage Definitions ────────────────────────────────────────────────────────

declare -A STAGE_NAMES=( [1]="seed" [2]="expand" [3]="refine" )
declare -A STAGE_MAXCAP=( [1]=500000 [2]=1000000 [3]=2000000 )
declare -A STAGE_DESC=(
    [1]="Core geometry scaffold from highest-confidence points"
    [2]="Fill spatial gaps with medium-confidence points"
    [3]="Full detail recovery, ADC prunes any floaters"
)

# ─── Run Stages ───────────────────────────────────────────────────────────────

RESULTS=()
TOTAL_START=$(date +%s)

for stage in 1 2 3; do
    name="${STAGE_NAMES[$stage]}"
    maxcap="${STAGE_MAXCAP[$stage]}"
    desc="${STAGE_DESC[$stage]}"
    tier_dir="$STAGING_DIR/tier${stage}"
    output_dir="${OUTPUT_BASE}/stage${stage}_${name}"
    log_file="${output_dir}/training.log"

    # Count points in this tier
    point_count=$(grep -vc '^#\|^$' "$tier_dir/sparse/0/points3D.txt" 2>/dev/null || echo "?")

    echo ""
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo -e "${BOLD}  Stage $stage: $name ($point_count points)${NC}"
    echo -e "${DIM}  $desc${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo ""

    CMD=("$BINARY"
        -d "$tier_dir"
        -o "$output_dir"
        --strategy adc
        --max-cap "$maxcap"
        -i "$STAGE_ITERS"
        -r "$RESIZE_FACTOR"
        --min-opacity 0.1
        --enable-sparsity
        --prune-ratio 0.6
        --enable-mip
        --headless
    )

    if [ "$DRY_RUN" = true ]; then
        info "[DRY RUN] Would execute:"
        echo "  ${CMD[*]}"
        echo ""
        RESULTS+=("Stage $stage ($name): DRY RUN")
        continue
    fi

    mkdir -p "$output_dir"

    STAGE_START=$(date +%s)
    info "Training stage $stage..."

    if "${CMD[@]}" 2>&1 | tee "$log_file"; then
        STAGE_END=$(date +%s)
        STAGE_ELAPSED=$((STAGE_END - STAGE_START))
        STAGE_MIN=$((STAGE_ELAPSED / 60))
        STAGE_SEC=$((STAGE_ELAPSED % 60))

        FINAL_PLY=$(find "$output_dir" -name '*.ply' -type f | sort | tail -1)
        PLY_SIZE=""
        [ -n "$FINAL_PLY" ] && PLY_SIZE="$(du -h "$FINAL_PLY" | cut -f1)"

        log "Stage $stage complete: ${STAGE_MIN}m${STAGE_SEC}s — $FINAL_PLY ($PLY_SIZE)"
        RESULTS+=("Stage $stage ($name): ${STAGE_MIN}m${STAGE_SEC}s — $PLY_SIZE")
    else
        error "Stage $stage FAILED (exit code $?)"
        RESULTS+=("Stage $stage ($name): FAILED")
        warn "Continuing to next stage..."
    fi
done

# ─── Optional: Single-Pass Baseline ──────────────────────────────────────────

if [ "$COMPARE" = true ]; then
    echo ""
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo -e "${BOLD}  Baseline: Single-pass 30K on tier3${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo ""

    TOTAL_ITERS=$((STAGE_ITERS * 3))
    baseline_dir="${OUTPUT_BASE}/baseline_singlepass"
    baseline_log="${baseline_dir}/training.log"

    CMD=("$BINARY"
        -d "$STAGING_DIR/tier3"
        -o "$baseline_dir"
        --strategy adc
        --max-cap 2000000
        -i "$TOTAL_ITERS"
        -r "$RESIZE_FACTOR"
        --min-opacity 0.1
        --enable-sparsity
        --prune-ratio 0.6
        --enable-mip
        --headless
    )

    if [ "$DRY_RUN" = true ]; then
        info "[DRY RUN] Baseline command:"
        echo "  ${CMD[*]}"
        RESULTS+=("Baseline: DRY RUN")
    else
        mkdir -p "$baseline_dir"
        BASE_START=$(date +%s)

        if "${CMD[@]}" 2>&1 | tee "$baseline_log"; then
            BASE_END=$(date +%s)
            BASE_ELAPSED=$((BASE_END - BASE_START))
            BASE_MIN=$((BASE_ELAPSED / 60))
            BASE_SEC=$((BASE_ELAPSED % 60))

            FINAL_PLY=$(find "$baseline_dir" -name '*.ply' -type f | sort | tail -1)
            PLY_SIZE=""
            [ -n "$FINAL_PLY" ] && PLY_SIZE="$(du -h "$FINAL_PLY" | cut -f1)"

            log "Baseline complete: ${BASE_MIN}m${BASE_SEC}s — $FINAL_PLY ($PLY_SIZE)"
            RESULTS+=("Baseline: ${BASE_MIN}m${BASE_SEC}s — $PLY_SIZE")
        else
            error "Baseline FAILED"
            RESULTS+=("Baseline: FAILED")
        fi
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${GREEN}  Progressive Training Summary${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "  Total time: ${TOTAL_MIN}m${TOTAL_SEC}s"
echo "  Output:     $OUTPUT_BASE"
echo ""

# Extract PSNR from logs if available
if [ "$DRY_RUN" = false ]; then
    echo -e "${BOLD}  PSNR Comparison:${NC}"
    for stage in 1 2 3; do
        name="${STAGE_NAMES[$stage]}"
        log_file="${OUTPUT_BASE}/stage${stage}_${name}/training.log"
        if [ -f "$log_file" ]; then
            psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$log_file" | tail -1 || echo "N/A")
            echo "    Stage $stage ($name): PSNR $psnr"
        fi
    done
    if [ "$COMPARE" = true ] && [ -f "${OUTPUT_BASE}/baseline_singlepass/training.log" ]; then
        psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "${OUTPUT_BASE}/baseline_singlepass/training.log" | tail -1 || echo "N/A")
        echo "    Baseline:          PSNR $psnr"
    fi
    echo ""
fi
