#!/bin/bash
# =============================================================================
# Progressive Training Pipeline — Growing-Seed ADC Strategy
#
# Runs 3-stage Gaussian splatting training using tiered point clouds:
#   Stage 1 (seed):   Best points only → clean geometric scaffold
#   Stage 2 (expand): Medium+ points → fill spatial gaps
#   Stage 3 (refine): All filtered points → full detail recovery
#
# Each stage is a fresh ADC training run seeded from progressively richer
# COLMAP point clouds (points3D.txt).
#
# Presets:
#   --preset maxquality   15K iters/stage, full res, fresh training (default)
#   --preset fast          7K iters/stage, half res, fresh training
#   --preset benchmark     10K iters/stage, half res, + single-pass baseline
#
# Usage:
#   ./progressive_train.sh /path/to/staging
#   ./progressive_train.sh /path/to/staging --preset fast
#   ./progressive_train.sh /path/to/staging --preset maxquality --compare
#   ./progressive_train.sh /path/to/staging --dry-run
#
# Requires: LichtFeld Studio binary built via setup_lichtfeld.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_root="$SCRIPT_DIR"; while [ "$_root" != "/" ] && [ ! -f "$_root/lib/common.sh" ]; do _root="$(dirname "$_root")"; done
source "$_root/lib/common.sh"
# LITCHSHIP_ROOT is now set automatically by common.sh

# ─── Arguments ───────────────────────────────────────────────────────────────

STAGING_DIR=""
DRY_RUN=false
COMPARE=false
NO_CLEAN=false
PRESET="maxquality"
STAGE_ITERS=0       # 0 = use preset default
RESIZE_FACTOR=0     # 0 = use preset default
REPO_DIR="$HOME/gaussian-splatting-cuda"

usage() {
    error "Usage: $0 <staging-dir> [options]"
    echo ""
    echo "  staging-dir    Directory with tier1/, tier2/, tier3/ from build_tiers.py"
    echo ""
    echo "Presets:"
    echo "  --preset maxquality  15K iters/stage, full res (-r 1), 2M max cap (default)"
    echo "  --preset fast        7K iters/stage, half res (-r 2), 1M max cap"
    echo "  --preset benchmark   10K iters/stage, half res (-r 2), + baseline comparison"
    echo ""
    echo "Options:"
    echo "  --dry-run      Print training commands without executing"
    echo "  --compare      Also run single-pass baseline for A/B comparison"
    echo "  --no-clean     Skip post-training cleaning"
    echo "  --iters N      Override iterations per stage"
    echo "  --resize N     Override resize factor for all stages"
    echo "  --repo DIR     Path to gaussian-splatting-cuda (default: $REPO_DIR)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --compare)   COMPARE=true; shift ;;
        --no-clean)  NO_CLEAN=true; shift ;;
        --preset)    PRESET="$2"; shift 2 ;;
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

# ─── Apply Preset ─────────────────────────────────────────────────────────────

case "$PRESET" in
    maxquality)
        [ "$STAGE_ITERS" -eq 0 ] 2>/dev/null && STAGE_ITERS=15000
        [ "$RESIZE_FACTOR" -eq 0 ] 2>/dev/null && RESIZE_FACTOR=1
        MAXCAP_1=500000; MAXCAP_2=1000000; MAXCAP_3=2000000
        ;;
    fast)
        [ "$STAGE_ITERS" -eq 0 ] 2>/dev/null && STAGE_ITERS=7000
        [ "$RESIZE_FACTOR" -eq 0 ] 2>/dev/null && RESIZE_FACTOR=2
        MAXCAP_1=500000; MAXCAP_2=1000000; MAXCAP_3=1000000
        ;;
    benchmark)
        [ "$STAGE_ITERS" -eq 0 ] 2>/dev/null && STAGE_ITERS=10000
        [ "$RESIZE_FACTOR" -eq 0 ] 2>/dev/null && RESIZE_FACTOR=2
        MAXCAP_1=500000; MAXCAP_2=1000000; MAXCAP_3=2000000
        COMPARE=true
        ;;
    *)
        error "Unknown preset: $PRESET (use maxquality, fast, or benchmark)"
        exit 1
        ;;
esac

# Adaptive CC keep-components based on tier3 point count
tier3_pts=$(grep -vc '^#\|^$' "$STAGING_DIR/tier3/sparse/0/points3D.txt" 2>/dev/null || echo 1000000)
if [ "$tier3_pts" -gt 10000000 ]; then
    CC_KEEP=15
elif [ "$tier3_pts" -gt 5000000 ]; then
    CC_KEEP=10
else
    CC_KEEP=5
fi

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
info "Preset: $PRESET"
info "Configuration:"
echo "  Staging:       $STAGING_DIR"
echo "  Output base:   $OUTPUT_BASE"
echo "  Iters/stage:   $STAGE_ITERS"
echo "  Resize:        $RESIZE_FACTOR"
echo "  Max gaussians: ${MAXCAP_1}→${MAXCAP_2}→${MAXCAP_3}"
echo "  CC components: --keep-components $CC_KEEP (stage 3)"
echo "  Strategy:      ADC (all stages, fresh training per stage)"
echo ""

# ─── Stage Definitions ────────────────────────────────────────────────────────

declare -A STAGE_NAMES=( [1]="seed" [2]="expand" [3]="refine" )
declare -A STAGE_MAXCAP=( [1]="$MAXCAP_1" [2]="$MAXCAP_2" [3]="$MAXCAP_3" )
declare -A STAGE_DESC=(
    [1]="Core geometry scaffold from highest-confidence points"
    [2]="Fill spatial gaps with medium-confidence points"
    [3]="Full detail recovery, ADC prunes any floaters"
)

# Per-stage cleaning configs (gentler for early stages since ADC already prunes)
CLEAN_SCRIPT="$LITCHSHIP_ROOT/training/clean_splat.py"
declare -A STAGE_CLEAN_FLAGS=(
    [1]="--opacity-min 0.05 -k 20 -s 3.5 -p 1"
    [2]="--opacity-min 0.05 -k 20 -s 3.0 -p 2"
    [3]="--opacity-min 0.05 -k 20 -s 3.0 -p 2 --connected --keep-components $CC_KEEP"
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

    point_count=$(grep -vc '^#\|^$' "$tier_dir/sparse/0/points3D.txt" 2>/dev/null || echo "?")

    echo ""
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo -e "${BOLD}  Stage $stage: $name ($point_count points)${NC}"
    echo -e "${DIM}  $desc${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        info "[DRY RUN] Would execute:"
        echo "  run_lf_training $BINARY $tier_dir $output_dir adc $maxcap $STAGE_ITERS $RESIZE_FACTOR"
        echo ""
        RESULTS+=("Stage $stage ($name): DRY RUN")
        continue
    fi

    STAGE_START=$(date +%s)
    info "Training stage $stage..."

    if run_lf_training "$BINARY" "$tier_dir" "$output_dir" adc "$maxcap" "$STAGE_ITERS" "$RESIZE_FACTOR"; then
        STAGE_END=$(date +%s)
        STAGE_ELAPSED=$((STAGE_END - STAGE_START))
        STAGE_MIN=$((STAGE_ELAPSED / 60))
        STAGE_SEC=$((STAGE_ELAPSED % 60))

        FINAL_PLY="$LF_LAST_PLY"
        PLY_SIZE=""
        [ -n "$FINAL_PLY" ] && PLY_SIZE="$(du -h "$FINAL_PLY" | cut -f1)"

        PSNR=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$LF_LAST_LOG" 2>/dev/null | tail -1 || echo "N/A")

        log "Stage $stage complete: ${STAGE_MIN}m${STAGE_SEC}s — $PLY_SIZE | PSNR=$PSNR"
        RESULTS+=("Stage $stage ($name): ${STAGE_MIN}m${STAGE_SEC}s — $PLY_SIZE")

        # Post-training cleaning
        if [ "$NO_CLEAN" = false ] && [ -n "$FINAL_PLY" ] && [ -f "$FINAL_PLY" ]; then
            clean_flags="${STAGE_CLEAN_FLAGS[$stage]}"
            info "Cleaning stage $stage (${clean_flags})..."
            CLEAN_START=$(date +%s)

            if run_lf_clean "$CLEAN_SCRIPT" "$FINAL_PLY" $clean_flags 2>&1 | tee "${output_dir}/cleaning.log"; then
                CLEAN_END=$(date +%s)
                CLEAN_ELAPSED=$((CLEAN_END - CLEAN_START))
                CLEANED_PLY="$LF_LAST_CLEANED_PLY"
                CLEAN_SIZE=""
                [ -f "$CLEANED_PLY" ] && CLEAN_SIZE="$(du -h "$CLEANED_PLY" | cut -f1)"
                RAW_COUNT=$(head -20 "$FINAL_PLY" | grep "element vertex" | awk '{print $3}')
                CLEAN_COUNT=$(head -20 "$CLEANED_PLY" | grep "element vertex" | awk '{print $3}')
                KEEP_PCT=0
                [ "$RAW_COUNT" -gt 0 ] 2>/dev/null && KEEP_PCT=$(( CLEAN_COUNT * 100 / RAW_COUNT ))
                log "Cleaned: ${RAW_COUNT} → ${CLEAN_COUNT} splats (${KEEP_PCT}% kept, $CLEAN_SIZE) in ${CLEAN_ELAPSED}s"
            else
                warn "Cleaning failed for stage $stage — raw PLY still available"
            fi
        fi
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
    echo -e "${BOLD}  Baseline: Single-pass $((STAGE_ITERS * 3)) iters on tier3${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo ""

    TOTAL_BASELINE=$((STAGE_ITERS * 3))
    baseline_dir="${OUTPUT_BASE}/baseline_singlepass"

    if [ "$DRY_RUN" = true ]; then
        info "[DRY RUN] Baseline command:"
        echo "  run_lf_training $BINARY $STAGING_DIR/tier3 $baseline_dir adc $MAXCAP_3 $TOTAL_BASELINE $RESIZE_FACTOR"
        RESULTS+=("Baseline: DRY RUN")
    else
        BASE_START=$(date +%s)

        if run_lf_training "$BINARY" "$STAGING_DIR/tier3" "$baseline_dir" adc "$MAXCAP_3" "$TOTAL_BASELINE" "$RESIZE_FACTOR"; then
            BASE_END=$(date +%s)
            BASE_ELAPSED=$((BASE_END - BASE_START))
            BASE_MIN=$((BASE_ELAPSED / 60))
            BASE_SEC=$((BASE_ELAPSED % 60))

            FINAL_PLY="$LF_LAST_PLY"
            PLY_SIZE=""
            [ -n "$FINAL_PLY" ] && PLY_SIZE="$(du -h "$FINAL_PLY" | cut -f1)"

            log "Baseline complete: ${BASE_MIN}m${BASE_SEC}s — $FINAL_PLY ($PLY_SIZE)"
            RESULTS+=("Baseline: ${BASE_MIN}m${BASE_SEC}s — $PLY_SIZE")

            if [ "$NO_CLEAN" = false ] && [ -n "$FINAL_PLY" ] && [ -f "$FINAL_PLY" ]; then
                info "Cleaning baseline (default config)..."
                if run_lf_clean "$CLEAN_SCRIPT" "$FINAL_PLY" \
                    --opacity-min 0.05 -k 20 -s 2.0 -p 3 2>&1 | tee "${baseline_dir}/cleaning.log"; then
                    CLEANED_PLY="$LF_LAST_CLEANED_PLY"
                    RAW_COUNT=$(head -20 "$FINAL_PLY" | grep "element vertex" | awk '{print $3}')
                    CLEAN_COUNT=$(head -20 "$CLEANED_PLY" | grep "element vertex" | awk '{print $3}')
                    log "Baseline cleaned: ${RAW_COUNT} → ${CLEAN_COUNT} splats"
                fi
            fi
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
echo "  Preset: $PRESET"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "  Total time: ${TOTAL_MIN}m${TOTAL_SEC}s"
echo "  Output:     $OUTPUT_BASE"
echo ""
