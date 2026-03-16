#!/bin/bash
# =============================================================================
# LichtFeld Studio — Test Training Run
#
# Copies a small subset of images from a dataset, filters the COLMAP files
# to match, and runs a quick training to verify the full pipeline works.
#
# Usage:
#   ./run_test_training.sh /path/to/colmap/data
#
# The script creates a 30-image subset at ~/data/<name>_subset/ with filtered
# COLMAP files so the trainer only sees the selected images. COLMAP images.txt
# uses a paired-line format (pose line + 2D points line), so filtering must
# read two lines at a time and keep/discard both together.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# ─── Arguments ─────────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then
    error "Usage: $0 <dataset-path>"
    error "  dataset-path: directory with images/ and COLMAP sparse/ (or text files at root)"
    exit 1
fi

DATA_SRC="$1"
DATASET_NAME="$(basename "$DATA_SRC")"
FULL_DIR="$HOME/data/$DATASET_NAME"
TEST_DIR="$HOME/data/${DATASET_NAME}_subset"
REPO_DIR="$HOME/gaussian-splatting-cuda"
OUTPUT_DIR="$HOME/output/${DATASET_NAME}_test"
SUBSET_SIZE=30           # Number of images for quick test
TEST_ITERATIONS=3000     # Quick test (full is 30000)
MAX_GAUSSIANS=200000     # Conservative for test

echo ""
echo "============================================"
echo "  LichtFeld Studio — Test Training"
echo "  Subset: $SUBSET_SIZE images"
echo "  Iterations: $TEST_ITERATIONS"
echo "  Strategy: adc"
echo "============================================"
echo ""

# ─── Preflight ─────────────────────────────────────────────────────────────

setup_paths "$REPO_DIR"
detect_gpu

# Check CUDA
if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
    log "CUDA: $($CUDA_ROOT/bin/nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
else
    error "CUDA not found at $CUDA_ROOT"
    exit 1
fi

find_binary "$REPO_DIR"
echo ""

# ─── Step 1: Copy Full Data to WSL2 Native Filesystem ──────────────────────

echo "=== [1/3] Data Preparation ==="

if [ ! -d "$DATA_SRC" ]; then
    error "Source data not found: $DATA_SRC"
    exit 1
fi

if [ -d "$FULL_DIR/images" ]; then
    log "Full dataset already on WSL filesystem"
else
    info "Copying full dataset to WSL2 native filesystem..."
    info "This is a one-time operation for performance."
    mkdir -p "$FULL_DIR"

    # Copy images
    if [ -d "$DATA_SRC/images" ] && [ "$(ls -A "$DATA_SRC/images/" 2>/dev/null)" ]; then
        cp -r "$DATA_SRC/images" "$FULL_DIR/images"
    else
        ROOT_IMAGES=$(find "$DATA_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -1)
        if [ -n "$ROOT_IMAGES" ]; then
            mkdir -p "$FULL_DIR/images"
            info "Images found at root level, copying to images/ subdirectory..."
            find "$DATA_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -exec cp {} "$FULL_DIR/images/" \;
        else
            IMG_DIR=$(find "$DATA_SRC" -maxdepth 2 -type d -iname 'images' -o -iname 'input' | head -1)
            if [ -n "$IMG_DIR" ]; then
                cp -r "$IMG_DIR" "$FULL_DIR/images"
            else
                error "Cannot find images in $DATA_SRC"
                exit 1
            fi
        fi
    fi

    TOTAL_IMAGES=$(find "$FULL_DIR/images" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
    log "Copied $TOTAL_IMAGES images"

    # Copy COLMAP sparse reconstruction
    if [ -d "$DATA_SRC/sparse" ]; then
        cp -r "$DATA_SRC/sparse" "$FULL_DIR/sparse"
        log "Copied sparse/ directory"
    else
        mkdir -p "$FULL_DIR/sparse/0"
        FOUND_COLMAP=false

        for f in cameras.txt images.txt points3D.txt; do
            if [ -f "$DATA_SRC/$f" ]; then
                cp "$DATA_SRC/$f" "$FULL_DIR/sparse/0/"
                FOUND_COLMAP=true
            fi
        done

        if [ "$FOUND_COLMAP" = false ]; then
            for f in cameras.txt images.txt points3D.txt; do
                FOUND=$(find "$DATA_SRC" -maxdepth 3 -name "$f" -type f | head -1)
                if [ -n "$FOUND" ]; then
                    cp "$FOUND" "$FULL_DIR/sparse/0/"
                    FOUND_COLMAP=true
                fi
            done
        fi

        if [ "$FOUND_COLMAP" = true ]; then
            log "Organized COLMAP text files into sparse/0/"
        else
            error "Could not find COLMAP text files (cameras.txt, images.txt, points3D.txt)"
            exit 1
        fi
    fi
fi

echo ""

# ─── Step 2: Create Test Subset ────────────────────────────────────────────

echo "=== [2/3] Creating Test Subset ($SUBSET_SIZE images) ==="

if [ -d "$TEST_DIR" ] && [ -f "$TEST_DIR/.subset_ready" ]; then
    log "Test subset already prepared at $TEST_DIR"
else
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR/images" "$TEST_DIR/sparse/0"

    # Pick first N images sorted alphabetically
    FULL_IMAGE_DIR="$FULL_DIR/images"
    IMAGE_LIST=$(find "$FULL_IMAGE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | sort | head -n "$SUBSET_SIZE")

    ACTUAL_COUNT=0
    while IFS= read -r img; do
        cp "$img" "$TEST_DIR/images/"
        ACTUAL_COUNT=$((ACTUAL_COUNT + 1))
    done <<< "$IMAGE_LIST"
    log "Copied $ACTUAL_COUNT images to test subset"

    SELECTED_NAMES=$(ls "$TEST_DIR/images/" | sort)

    # Filter images.txt — COLMAP uses a paired-line format:
    #   Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   Line 2: POINTS2D[] as (X Y POINT3D_ID) ...
    # We must read both lines together and keep/discard the pair as a unit.
    COLMAP_SRC="$FULL_DIR/sparse/0"
    if [ -f "$COLMAP_SRC/images.txt" ]; then
        grep '^#' "$COLMAP_SRC/images.txt" > "$TEST_DIR/sparse/0/images.txt" 2>/dev/null || true

        grep -v '^#' "$COLMAP_SRC/images.txt" | while IFS= read -r line1; do
            IFS= read -r line2 || line2=""
            IMG_NAME=$(echo "$line1" | awk '{print $NF}')
            if echo "$SELECTED_NAMES" | grep -qx "$IMG_NAME"; then
                echo "$line1" >> "$TEST_DIR/sparse/0/images.txt"
                echo "$line2" >> "$TEST_DIR/sparse/0/images.txt"
            fi
        done
        log "Filtered images.txt for subset"
    fi

    # cameras.txt and points3D.txt are copied as-is
    if [ -f "$COLMAP_SRC/cameras.txt" ]; then
        cp "$COLMAP_SRC/cameras.txt" "$TEST_DIR/sparse/0/"
        log "Copied cameras.txt"
    fi

    if [ -f "$COLMAP_SRC/points3D.txt" ]; then
        cp "$COLMAP_SRC/points3D.txt" "$TEST_DIR/sparse/0/"
        log "Copied points3D.txt (trainer will filter unused points)"
    fi

    touch "$TEST_DIR/.subset_ready"
    log "Test subset ready at $TEST_DIR"
fi

echo ""
info "Subset contents:"
echo "  Images: $(find "$TEST_DIR/images" -type f | wc -l)"
echo "  COLMAP files:"
for f in cameras.txt images.txt points3D.txt; do
    if [ -f "$TEST_DIR/sparse/0/$f" ]; then
        LINES=$(grep -v '^#' "$TEST_DIR/sparse/0/$f" 2>/dev/null | wc -l)
        echo "    $f: $LINES data lines"
    else
        echo "    $f: MISSING"
    fi
done

echo ""

# ─── Step 3: Run Training ──────────────────────────────────────────────────

echo "=== [3/3] Training ($TEST_ITERATIONS iterations, ADC strategy) ==="
echo ""

mkdir -p "$OUTPUT_DIR"

info "Starting LichtFeld Studio training..."
info "Binary: $BINARY"
info "Data: $TEST_DIR"
info "Output: $OUTPUT_DIR"
info "Strategy: adc"
info "Max Gaussians: $MAX_GAUSSIANS"
info "Iterations: $TEST_ITERATIONS"
echo ""

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

info "Launching training..."
time "$BINARY" \
    -d "$TEST_DIR" \
    -o "$OUTPUT_DIR" \
    --strategy adc \
    --max-cap "$MAX_GAUSSIANS" \
    -i "$TEST_ITERATIONS" \
    -r 2 \
    --min-opacity 0.1 \
    --enable-sparsity \
    --prune-ratio 0.6 \
    --enable-mip \
    --headless

TRAIN_EXIT=$?

echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    log "Training completed successfully!"

    # Post-process: remove low-opacity floaters and spatial outliers
    FINAL_PLY=$(find "$OUTPUT_DIR" -name '*.ply' -type f -not -name '*_cleaned*' | sort | tail -1)
    CLEAN_SCRIPT="$SCRIPT_DIR/clean_splat.py"
    if [ -n "$FINAL_PLY" ] && [ -f "$CLEAN_SCRIPT" ]; then
        echo ""
        echo "=== Post-Processing: Outlier Removal ==="
        CLEANED_PLY="${FINAL_PLY%.ply}_cleaned.ply"
        python3 "$CLEAN_SCRIPT" "$FINAL_PLY" "$CLEANED_PLY" \
            -k 20 -s 2.0 -p 3 --opacity-min 0.05
        [ -f "$CLEANED_PLY" ] && log "Cleaned: $CLEANED_PLY ($(du -h "$CLEANED_PLY" | cut -f1))"
    fi

    echo ""
    info "Output files:"
    find "$OUTPUT_DIR" -type f | head -20
    echo ""
    info "Output size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
else
    error "Training exited with code $TRAIN_EXIT"
    echo ""
    warn "Common issues:"
    warn "  - 'no kernel image': rebuild with -DCMAKE_CUDA_ARCHITECTURES=120"
    warn "  - OOM: reduce --max-cap or increase -r (resize factor)"
    warn "  - COLMAP parse error: check sparse/0/ file format"
fi

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  Full training:  ./train_full.sh $DATA_SRC"
echo ""
