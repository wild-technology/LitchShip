#!/bin/bash
# =============================================================================
# LichtFeld Studio — Multi-Config Benchmark
#
# Usage:
#   ./benchmark.sh /path/to/colmap/dataset                  # results stay local
#   ./benchmark.sh /path/to/colmap/dataset /path/to/splats  # copy .ply outputs
#
# The dataset must already be on the WSL native filesystem in COLMAP layout:
#   dataset/images/*.jpg  +  dataset/sparse/0/{cameras,images,points3D}.txt
#
# Config format:  NAME|ITERATIONS|MAX_GAUSSIANS|RESIZE_FACTOR|STRATEGY|EXTRA_FLAGS
# EXTRA_FLAGS is optional — any additional CLI flags appended to the command.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# ─── Arguments ─────────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then
    error "Usage: $0 <dataset-path> [splat-output-dir]"
    error "  dataset-path: COLMAP dataset on WSL native filesystem"
    error "  splat-output-dir: optional directory to copy .ply results"
    exit 1
fi

WORK_DIR="$1"
SPLAT_DEST="${2:-}"
REPO_DIR="$HOME/gaussian-splatting-cuda"

# ─── Preflight ─────────────────────────────────────────────────────────────

setup_paths "$REPO_DIR"
detect_gpu
find_binary "$REPO_DIR"

if [ ! -d "$WORK_DIR/sparse/0" ]; then
    error "Dataset missing COLMAP files: $WORK_DIR/sparse/0/"
    exit 1
fi

# ─── Benchmark Configurations ──────────────────────────────────────────────
# Format: NAME|ITERS|MAX_GAUSS|RESIZE|STRATEGY|EXTRA_FLAGS
# ADC is the recommended strategy — built-in scale pruning, no bloom artifacts.
# MCMC configs kept for comparison but note: MCMC + non-default --min-opacity
# triggers a 17 PB memory allocation crash.

CONFIGS=(
    "adc_quick_lowres|3000|500000|4|adc|--min-opacity 0.1 --enable-sparsity --prune-ratio 0.6 --enable-mip"
    "adc_medium_halfres|7000|1000000|2|adc|--min-opacity 0.1 --enable-sparsity --prune-ratio 0.6 --enable-mip"
    "adc_full_halfres|30000|2000000|2|adc|--min-opacity 0.1 --enable-sparsity --prune-ratio 0.6 --enable-mip"
    "adc_full_highres|30000|3000000|1|adc|--min-opacity 0.1 --enable-sparsity --prune-ratio 0.6 --enable-mip"
    "mcmc_quick_lowres|3000|500000|4|mcmc|"
    "mcmc_medium_halfres|7000|1000000|2|mcmc|"
)

RESULTS_FILE="$HOME/output/benchmark_results_$(date +%Y%m%d_%H%M).txt"

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  LichtFeld Studio — Multi-Config Benchmark${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# ─── Dataset Info ──────────────────────────────────────────────────────────

TOTAL_IMAGES=$(find "$WORK_DIR/images" -type f | wc -l)
info "Dataset: $WORK_DIR ($TOTAL_IMAGES images)"
info "Points3D: $(grep -c -v '^#' "$WORK_DIR/sparse/0/points3D.txt" 2>/dev/null || echo '?') points"
echo ""

# ─── Run Benchmarks ───────────────────────────────────────────────────────

mkdir -p "$(dirname "$RESULTS_FILE")"
[ -n "$SPLAT_DEST" ] && mkdir -p "$SPLAT_DEST"

# Write results header
cat > "$RESULTS_FILE" << HEADER
================================================================================
LichtFeld Studio Benchmark Results
Date: $(date)
GPU: $GPU_NAME ($GPU_MEM)
Dataset: $(basename "$WORK_DIR") (${TOTAL_IMAGES} images)
Binary: $BINARY
================================================================================

Config                  | Iters | MaxGauss | Resize | Strategy | Time     | Exit | PLY Size
------------------------|-------|----------|--------|----------|----------|------|----------
HEADER

echo -e "${BOLD}Benchmark Configurations:${NC}"
for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r name iters maxg resize strategy extra <<< "${CONFIGS[$i]}"
    echo "  [$((i+1))] $name: ${iters} iters, ${maxg} max, resize=${resize}, ${strategy} ${extra}"
done
echo ""

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r name iters maxg resize strategy extra <<< "${CONFIGS[$i]}"

    OUTPUT_DIR="$HOME/output/bench_${name}_$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$OUTPUT_DIR/training.log"

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  [$((i+1))/${#CONFIGS[@]}] $name${NC}"
    echo -e "${BOLD}  Iters: $iters | Max: $maxg | Resize: $resize | Strategy: $strategy${NC}"
    [ -n "$extra" ] && echo -e "${BOLD}  Extra: $extra${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    info "GPU memory before:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

    START_TIME=$(date +%s)

    # Build command — extra flags are word-split intentionally
    # shellcheck disable=SC2086
    TRAIN_EXIT=0
    run_lf_training "$BINARY" "$WORK_DIR" "$OUTPUT_DIR" "$strategy" "$maxg" "$iters" "$resize" $extra || TRAIN_EXIT=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_FMT="$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"

    # Find output PLY and clean it
    PLY_FILE="$LF_LAST_PLY"
    PLY_SIZE="N/A"
    if [ -n "$PLY_FILE" ]; then
        PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)

        # Post-process: remove low-opacity floaters and spatial outliers
        CLEAN_SCRIPT="$SCRIPT_DIR/clean_splat.py"
        if [ -f "$CLEAN_SCRIPT" ]; then
            info "Cleaning splat (opacity + SOR)..."
            run_lf_clean "$CLEAN_SCRIPT" "$PLY_FILE" -k 20 -s 2.0 -p 3 --opacity-min 0.05 2>&1 | tail -5
            if [ -f "$LF_LAST_CLEANED_PLY" ]; then
                PLY_FILE="$LF_LAST_CLEANED_PLY"
                PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
            fi
        fi

        # Copy to splat destination if provided
        if [ -n "$SPLAT_DEST" ]; then
            DEST_NAME="${name}.ply"
            cp "$PLY_FILE" "$SPLAT_DEST/$DEST_NAME"
            log "Copied $DEST_NAME ($PLY_SIZE) -> $SPLAT_DEST/"
        fi
    fi

    # Record result
    printf "%-23s | %5s | %8s | %6s | %8s | %8s | %4s | %s\n" \
        "$name" "$iters" "$maxg" "$resize" "$strategy" "$ELAPSED_FMT" "$TRAIN_EXIT" "$PLY_SIZE" \
        >> "$RESULTS_FILE"

    if [ $TRAIN_EXIT -eq 0 ]; then
        log "$name completed in $ELAPSED_FMT (PLY: $PLY_SIZE)"
    else
        error "$name failed (exit $TRAIN_EXIT) after $ELAPSED_FMT"
    fi

    # Extract metrics from log
    FINAL_LOSS=$(grep -oP 'loss[=: ]*\K[0-9]+\.[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1)
    FINAL_PSNR=$(grep -oP '[Pp][Ss][Nn][Rr][=: ]*\K[0-9]+\.[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$FINAL_LOSS" ] || [ -n "$FINAL_PSNR" ]; then
        echo "  Metrics: loss=${FINAL_LOSS:-?} PSNR=${FINAL_PSNR:-?}" >> "$RESULTS_FILE"
    fi

    echo ""

    # Brief cooldown between runs
    if [ $i -lt $((${#CONFIGS[@]} - 1)) ]; then
        info "Cooldown 10s before next run..."
        sleep 10
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}  Benchmark Complete!${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

cat "$RESULTS_FILE"

echo ""
info "Results saved to: $RESULTS_FILE"
if [ -n "$SPLAT_DEST" ]; then
    info "Splats saved to: $SPLAT_DEST"
    ls -lh "$SPLAT_DEST"/*.ply 2>/dev/null || warn "No PLY files in $SPLAT_DEST"
fi
echo ""
