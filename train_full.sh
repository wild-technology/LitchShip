#!/bin/bash
# =============================================================================
# LichtFeld Studio — Full Resolution Training Pipeline
#
# Copies data from a source directory to the WSL native filesystem (if needed),
# prepares the COLMAP layout, generates a live monitor script, and runs
# training at full resolution.
#
# Usage:
#   ./train_full.sh /path/to/colmap/data        # required: dataset path
#
# WARNING: Do NOT use MCMC strategy with non-default --min-opacity.
# MCMC's cap_max parser interprets --min-opacity values as astronomical memory
# requests (e.g. 17 PB), causing an instant OOM crash. ADC handles opacity
# correctly and produces better quality (built-in scale pruning, no bloom).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# ─── Arguments ─────────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then
    error "Usage: $0 <dataset-path>"
    error "  dataset-path: directory containing images/ and sparse/0/ (or COLMAP text files at root)"
    exit 1
fi

DATA_SRC="$1"
DATASET_NAME="$(basename "$(dirname "$DATA_SRC")")"
[ "$DATASET_NAME" = "." ] && DATASET_NAME="$(basename "$DATA_SRC")"
WORK_DIR="$HOME/data/$DATASET_NAME"
REPO_DIR="$HOME/gaussian-splatting-cuda"
OUTPUT_DIR="$HOME/output/${DATASET_NAME}_fullres_$(date +%Y%m%d_%H%M)"
LOG_FILE="$OUTPUT_DIR/training.log"

# Training parameters — ADC is the recommended strategy
ITERATIONS=30000
MAX_GAUSSIANS=2000000    # 2M gaussians for large scene
RESIZE_FACTOR=1          # Full resolution
STRATEGY="adc"

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  LichtFeld Studio — Full Resolution Train${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo "  Dataset:    $DATASET_NAME"
echo "  Source:     $DATA_SRC"
echo "  Working:    $WORK_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Resolution: full (resize_factor=$RESIZE_FACTOR)"
echo "  Iterations: $ITERATIONS"
echo "  Max gauss:  $MAX_GAUSSIANS"
echo "  Strategy:   $STRATEGY"
echo ""

# ─── Preflight ─────────────────────────────────────────────────────────────

setup_paths "$REPO_DIR"
detect_gpu
find_binary "$REPO_DIR"

echo ""

# ─── Step 1: Copy Data to WSL Native Filesystem ────────────────────────────

echo "=== [1/3] Data Preparation ==="

if [ -f "$WORK_DIR/.data_ready" ]; then
    log "Data already prepared at $WORK_DIR"
    TOTAL_IMAGES=$(find "$WORK_DIR/images" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
    log "  Images: $TOTAL_IMAGES"
else
    if [ ! -d "$DATA_SRC" ]; then
        error "Source data not found: $DATA_SRC"
        exit 1
    fi

    warn "Copying data to WSL native filesystem..."
    warn "This may take several minutes (one-time operation)."
    echo ""

    mkdir -p "$WORK_DIR/images" "$WORK_DIR/sparse/0"

    # Copy images — check if they're in an images/ subdir or at the root
    if [ -d "$DATA_SRC/images" ] && [ "$(find "$DATA_SRC/images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -1)" ]; then
        info "Copying images from $DATA_SRC/images/ ..."
        IMG_SRC="$DATA_SRC/images"
    else
        info "Copying images from $DATA_SRC/ (flat layout)..."
        IMG_SRC="$DATA_SRC"
    fi

    info "Counting images..."
    TOTAL=$(find "$IMG_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
    info "Found $TOTAL images to copy"

    COUNT=0
    find "$IMG_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | while IFS= read -r img; do
        cp "$img" "$WORK_DIR/images/"
        COUNT=$((COUNT + 1))
        if [ $((COUNT % 500)) -eq 0 ]; then
            echo -ne "\r  Copied $COUNT / $TOTAL images..."
        fi
    done
    echo -ne "\r"

    TOTAL_IMAGES=$(find "$WORK_DIR/images" -type f | wc -l)
    log "Copied $TOTAL_IMAGES images"

    # Copy COLMAP files (handles case-insensitive names)
    copy_colmap_files "$DATA_SRC" "$WORK_DIR"

    # Fix extension mismatch (e.g. images.txt says .png but files are .jpeg)
    fix_extension_mismatch "$WORK_DIR"

    touch "$WORK_DIR/.data_ready"
    log "Data preparation complete"
fi

# Print data summary
echo ""
info "Dataset summary:"
TOTAL_IMAGES=$(find "$WORK_DIR/images" -type f | wc -l)
SAMPLE_IMG=$(find "$WORK_DIR/images" -type f | head -1)
if [ -n "$SAMPLE_IMG" ]; then
    IMG_SIZE=$(file "$SAMPLE_IMG" | grep -oP '\d+ x \d+' || echo "unknown")
    echo "  Images:     $TOTAL_IMAGES ($IMG_SIZE)"
fi
for f in cameras.txt images.txt points3D.txt; do
    if [ -f "$WORK_DIR/sparse/0/$f" ]; then
        LINES=$(wc -l < "$WORK_DIR/sparse/0/$f")
        SIZE=$(du -h "$WORK_DIR/sparse/0/$f" | cut -f1)
        echo "  $f: $LINES lines ($SIZE)"
    fi
done
DISK=$(du -sh "$WORK_DIR" | cut -f1)
echo "  Total disk: $DISK"

echo ""

# ─── Step 2: Prepare Monitoring ─────────────────────────────────────────────
# Generates a standalone monitor script that can be run in a second terminal
# to watch GPU stats, training progress, and output files in real time.

echo "=== [2/3] Setting Up Monitoring ==="

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/monitor.sh" << 'MONITOR_SCRIPT'
#!/bin/bash
# Live training monitor — run in a second terminal
# Usage: ./monitor.sh [log_file] [output_dir]
LOG="${1:-training.log}"
OUTDIR="${2:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

clear_screen() { printf '\033[2J\033[H'; }

while true; do
    clear_screen

    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         LichtFeld Studio — Training Monitor                 ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${CYAN}── GPU Status ──${NC}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader 2>/dev/null | while IFS=, read -r name temp util mem_used mem_total power; do
            echo -e "  GPU:   ${GREEN}$name${NC}"
            echo -e "  Temp:  $temp   Util: $util   Power: $power"
            echo -e "  VRAM:  $mem_used / $mem_total"
        done
    fi
    echo ""

    echo -e "${CYAN}── Training Progress ──${NC}"
    if [ -f "$LOG" ]; then
        LAST_ITER=$(grep -oP 'iter[ation]*\s*[\[: ]*\K\d+' "$LOG" 2>/dev/null | tail -1)
        TOTAL_ITER=$(grep -oP '(iterations|total_iter|max_iter)[=: ]*\K\d+' "$LOG" 2>/dev/null | tail -1)

        if [ -n "$LAST_ITER" ]; then
            if [ -n "$TOTAL_ITER" ] && [ "$TOTAL_ITER" -gt 0 ] 2>/dev/null; then
                PCT=$((LAST_ITER * 100 / TOTAL_ITER))
                BAR_LEN=40
                FILLED=$((PCT * BAR_LEN / 100))
                EMPTY=$((BAR_LEN - FILLED))
                BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null)
                SPACE=$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null)
                echo -e "  Progress: [${GREEN}${BAR}${DIM}${SPACE}${NC}] ${BOLD}${PCT}%${NC} ($LAST_ITER / $TOTAL_ITER)"
            else
                echo -e "  Iteration: ${BOLD}$LAST_ITER${NC}"
            fi
        else
            echo -e "  ${DIM}Waiting for training to start...${NC}"
        fi

        LAST_LOSS=$(grep -oP 'loss[=: ]*\K[0-9]+\.[0-9]+' "$LOG" 2>/dev/null | tail -1)
        LAST_PSNR=$(grep -oP '[Pp][Ss][Nn][Rr][=: ]*\K[0-9]+\.[0-9]+' "$LOG" 2>/dev/null | tail -1)
        LAST_SSIM=$(grep -oP '[Ss][Ss][Ii][Mm][=: ]*\K[0-9]+\.[0-9]+' "$LOG" 2>/dev/null | tail -1)
        NUM_GAUSS=$(grep -oP '(gaussians|splats|num_points)[=: ]*\K[0-9,]+' "$LOG" 2>/dev/null | tail -1)

        echo ""
        [ -n "$LAST_LOSS" ] && echo -e "  Loss:       ${BOLD}$LAST_LOSS${NC}"
        [ -n "$LAST_PSNR" ] && echo -e "  PSNR:       ${GREEN}${BOLD}$LAST_PSNR dB${NC}"
        [ -n "$LAST_SSIM" ] && echo -e "  SSIM:       ${BOLD}$LAST_SSIM${NC}"
        [ -n "$NUM_GAUSS" ] && echo -e "  Gaussians:  ${BOLD}$NUM_GAUSS${NC}"

        ITER_PER_SEC=$(grep -oP '(iter/s|it/s)[=: ]*\K[0-9]+\.?[0-9]*' "$LOG" 2>/dev/null | tail -1)
        if [ -n "$ITER_PER_SEC" ] && [ -n "$LAST_ITER" ] && [ -n "$TOTAL_ITER" ]; then
            REMAINING=$((TOTAL_ITER - LAST_ITER))
            ETA=$(echo "$REMAINING $ITER_PER_SEC" | awk '{
                secs = $1 / $2;
                if (secs > 3600) printf "%.1fh", secs/3600;
                else if (secs > 60) printf "%.0fm", secs/60;
                else printf "%.0fs", secs;
            }')
            echo -e "  Speed:      ${BOLD}$ITER_PER_SEC${NC} iter/s   ETA: ${YELLOW}$ETA${NC}"
        fi
    else
        echo -e "  ${DIM}Log file not found: $LOG${NC}"
    fi
    echo ""

    echo -e "${CYAN}── Output Files ──${NC}"
    if [ -d "$OUTDIR" ]; then
        TOTAL_SIZE=$(du -sh "$OUTDIR" 2>/dev/null | cut -f1)
        echo "  Total size: $TOTAL_SIZE"

        find "$OUTDIR" -name '*.ply' -type f 2>/dev/null | while read -r ply; do
            PLY_SIZE=$(du -h "$ply" | cut -f1)
            PLY_NAME=$(basename "$ply")
            echo -e "  ${GREEN}$PLY_NAME${NC} ($PLY_SIZE)"
        done

        CKPT_COUNT=$(find "$OUTDIR" -name '*.ckpt' -o -name '*.pt' -o -name 'checkpoint*' 2>/dev/null | wc -l)
        [ "$CKPT_COUNT" -gt 0 ] && echo "  Checkpoints: $CKPT_COUNT"

        RENDER_COUNT=$(find "$OUTDIR" -name '*.png' -o -name '*.jpg' 2>/dev/null | wc -l)
        [ "$RENDER_COUNT" -gt 0 ] && echo -e "  Renders: ${GREEN}$RENDER_COUNT images${NC}"
    fi
    echo ""

    echo -e "${CYAN}── Recent Log (last 15 lines) ──${NC}"
    if [ -f "$LOG" ]; then
        tail -15 "$LOG" 2>/dev/null | while IFS= read -r line; do
            line=$(echo "$line" | sed \
                -e "s/\(error\|ERROR\|Error\)/$(printf '\033[0;31m')&$(printf '\033[0m')/g" \
                -e "s/\(warning\|WARNING\|Warning\)/$(printf '\033[1;33m')&$(printf '\033[0m')/g" \
                -e "s/\(saving\|checkpoint\|PSNR\)/$(printf '\033[0;32m')&$(printf '\033[0m')/gI")
            echo "  $line"
        done
    fi
    echo ""
    echo -e "${DIM}  Refreshing every 5s... Press Ctrl+C to stop.${NC}"

    sleep 5
done
MONITOR_SCRIPT
chmod +x "$OUTPUT_DIR/monitor.sh"
log "Monitor script: $OUTPUT_DIR/monitor.sh"
info "Run in a second terminal:  $OUTPUT_DIR/monitor.sh $LOG_FILE $OUTPUT_DIR"

echo ""

# ─── Step 3: Run Training ──────────────────────────────────────────────────

echo "=== [3/3] Training ==="
echo ""

info "Configuration:"
echo "  Binary:       $BINARY"
echo "  Data:         $WORK_DIR"
echo "  Output:       $OUTPUT_DIR"
echo "  Log:          $LOG_FILE"
echo "  Strategy:     $STRATEGY"
echo "  Iterations:   $ITERATIONS"
echo "  Max Gauss:    $MAX_GAUSSIANS"
echo "  Resolution:   full (resize_factor=$RESIZE_FACTOR)"
echo ""

info "GPU memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

warn "Starting training... (this will take 15-60 minutes at full resolution)"
warn "Monitor in another terminal:  $OUTPUT_DIR/monitor.sh $LOG_FILE $OUTPUT_DIR"
echo ""

# ADC best config: scale pruning + sparsity + mip filtering
"$BINARY" \
    -d "$WORK_DIR" \
    -o "$OUTPUT_DIR" \
    --strategy "$STRATEGY" \
    --max-cap "$MAX_GAUSSIANS" \
    -i "$ITERATIONS" \
    -r "$RESIZE_FACTOR" \
    --min-opacity 0.1 \
    --enable-sparsity \
    --prune-ratio 0.6 \
    --enable-mip \
    --headless \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    log "Training completed successfully!"
    echo ""
    info "Output files:"
    find "$OUTPUT_DIR" -type f -name '*.ply' -exec ls -lh {} \;
    echo ""
    info "Total output size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
    echo ""

    FINAL_PLY=$(find "$OUTPUT_DIR" -name '*.ply' -type f | sort | tail -1)
    if [ -n "$FINAL_PLY" ]; then
        PLY_SIZE=$(du -h "$FINAL_PLY" | cut -f1)
        echo -e "${BOLD}============================================${NC}"
        echo -e "${GREEN}  Training Complete!${NC}"
        echo -e "${BOLD}============================================${NC}"
        echo ""
        echo "  Splat file: $FINAL_PLY ($PLY_SIZE)"
        echo "  Full log:   $LOG_FILE"
        echo ""
    fi
else
    error "Training failed with exit code $TRAIN_EXIT"
    echo ""
    warn "Last 30 lines of log:"
    tail -30 "$LOG_FILE" 2>/dev/null
fi
