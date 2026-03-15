#!/bin/bash
# =============================================================================
# LichtFeld Studio — Test Training Run
# Copies a small subset of images and runs a quick training to verify the
# full pipeline works end-to-end on your RTX 5090 WSL2 setup.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; }
info()  { echo -e "${CYAN}[..]${NC} $1"; }

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_SRC="/mnt/c/Users/WildTech/Desktop/H2103d_test_colmap_workflow"
TEST_DIR="$HOME/data/H2103d_test_subset"
FULL_DIR="$HOME/data/H2103d_test_colmap_workflow"
REPO_DIR="$HOME/gaussian-splatting-cuda"
OUTPUT_DIR="$HOME/output/H2103d_test"
CUDA_ROOT="/usr/local/cuda"
SUBSET_SIZE=30           # Number of images for quick test
TEST_ITERATIONS=3000     # Quick test (full is 30000)
MAX_GAUSSIANS=200000     # Conservative for test

echo ""
echo "============================================"
echo "  LichtFeld Studio — Test Training"
echo "  Subset: $SUBSET_SIZE images"
echo "  Iterations: $TEST_ITERATIONS"
echo "============================================"
echo ""

# ─── Preflight ──────────────────────────────────────────────────────────────

# Check GPU
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log "GPU: $GPU_NAME ($GPU_MEM)"
else
    error "nvidia-smi failed. GPU not available."
    exit 1
fi

# Check CUDA
if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
    log "CUDA: $($CUDA_ROOT/bin/nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
else
    error "CUDA not found at $CUDA_ROOT"
    exit 1
fi

# Check LichtFeld binary
BINARY=""
for candidate in \
    "$REPO_DIR/build/LichtFeld-Studio" \
    "$REPO_DIR/build/bin/LichtFeld-Studio" \
    "$REPO_DIR/build/bin/lichtfeld-studio" \
    "$REPO_DIR/build/lichtfeld-studio" \
    "$REPO_DIR/build/bin/gaussian_splatting_cuda" \
    "$REPO_DIR/build/gaussian_splatting_cuda"; do
    if [ -x "$candidate" ]; then
        BINARY="$candidate"
        break
    fi
done

if [ -z "$BINARY" ]; then
    error "LichtFeld Studio binary not found. Run setup_lichtfeld.sh first."
    info "Looking in: $REPO_DIR/build/"
    info "Available executables:"
    find "$REPO_DIR/build/" -type f -executable 2>/dev/null | head -10
    exit 1
fi

log "Binary: $BINARY"
echo ""

# ─── Step 1: Copy Full Data to WSL2 Native Filesystem ──────────────────────

echo "=== [1/3] Data Preparation ==="

if [ ! -d "$DATA_SRC" ]; then
    error "Source data not found: $DATA_SRC"
    error "Check that the Windows path is correct."
    exit 1
fi

# First, copy the full dataset to WSL native filesystem (if not done)
if [ -d "$FULL_DIR/images" ]; then
    log "Full dataset already on WSL filesystem"
else
    info "Copying full dataset from Windows to WSL2 native filesystem..."
    info "This is a one-time operation for performance."
    mkdir -p "$FULL_DIR"

    # Copy images
    if [ -d "$DATA_SRC/images" ] && [ "$(ls -A "$DATA_SRC/images/" 2>/dev/null)" ]; then
        cp -r "$DATA_SRC/images" "$FULL_DIR/images"
    else
        # RealityScan may have images at root level (no images/ subdir)
        ROOT_IMAGES=$(find "$DATA_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -1)
        if [ -n "$ROOT_IMAGES" ]; then
            mkdir -p "$FULL_DIR/images"
            info "Images found at root level, copying to images/ subdirectory..."
            find "$DATA_SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -exec cp {} "$FULL_DIR/images/" \;
        else
            # Try to find the images directory
            IMG_DIR=$(find "$DATA_SRC" -maxdepth 2 -type d -iname 'images' -o -iname 'input' | head -1)
            if [ -n "$IMG_DIR" ]; then
                cp -r "$IMG_DIR" "$FULL_DIR/images"
            else
                error "Cannot find images in $DATA_SRC"
                info "Contents of source:"
                ls -la "$DATA_SRC/"
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
        # RealityScan exports COLMAP text files — find and organize them
        mkdir -p "$FULL_DIR/sparse/0"
        FOUND_COLMAP=false

        # Check root level
        for f in cameras.txt images.txt points3D.txt; do
            if [ -f "$DATA_SRC/$f" ]; then
                cp "$DATA_SRC/$f" "$FULL_DIR/sparse/0/"
                FOUND_COLMAP=true
            fi
        done

        # Check subdirectories if not found at root
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
            info "Contents of source:"
            find "$DATA_SRC" -maxdepth 2 -type f -name '*.txt' | head -20
            exit 1
        fi
    fi
fi

echo ""

# ─── Step 2: Create Test Subset ────────────────────────────────────────────

echo "=== [2/3] Creating Test Subset ($SUBSET_SIZE images) ==="

# We need to create a consistent subset: pick N images and filter the
# COLMAP files to only reference those images.

if [ -d "$TEST_DIR" ] && [ -f "$TEST_DIR/.subset_ready" ]; then
    log "Test subset already prepared at $TEST_DIR"
else
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR/images" "$TEST_DIR/sparse/0"

    # Get list of all images sorted, pick first N
    FULL_IMAGE_DIR="$FULL_DIR/images"
    IMAGE_LIST=$(find "$FULL_IMAGE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
        | sort | head -n "$SUBSET_SIZE")

    ACTUAL_COUNT=0
    while IFS= read -r img; do
        cp "$img" "$TEST_DIR/images/"
        ACTUAL_COUNT=$((ACTUAL_COUNT + 1))
    done <<< "$IMAGE_LIST"
    log "Copied $ACTUAL_COUNT images to test subset"

    # Build a set of selected filenames for filtering COLMAP files
    SELECTED_NAMES=$(ls "$TEST_DIR/images/" | sort)

    # Filter images.txt — keep only entries for selected images
    # images.txt format: pairs of lines
    #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   POINTS2D[] as (X Y POINT3D_ID) ...
    COLMAP_SRC="$FULL_DIR/sparse/0"
    if [ -f "$COLMAP_SRC/images.txt" ]; then
        # Copy header comments
        grep '^#' "$COLMAP_SRC/images.txt" > "$TEST_DIR/sparse/0/images.txt" 2>/dev/null || true

        # Process image entries (pairs of lines)
        # Read the file, skip comments, process in pairs
        grep -v '^#' "$COLMAP_SRC/images.txt" | while IFS= read -r line1; do
            IFS= read -r line2 || line2=""
            # Extract image name (last field on line1)
            IMG_NAME=$(echo "$line1" | awk '{print $NF}')
            # Check if this image is in our subset
            if echo "$SELECTED_NAMES" | grep -qx "$IMG_NAME"; then
                echo "$line1" >> "$TEST_DIR/sparse/0/images.txt"
                echo "$line2" >> "$TEST_DIR/sparse/0/images.txt"
            fi
        done
        log "Filtered images.txt for subset"
    fi

    # Copy cameras.txt as-is (camera models apply to all images)
    if [ -f "$COLMAP_SRC/cameras.txt" ]; then
        cp "$COLMAP_SRC/cameras.txt" "$TEST_DIR/sparse/0/"
        log "Copied cameras.txt"
    fi

    # Copy points3D.txt as-is (LichtFeld will filter unused points)
    if [ -f "$COLMAP_SRC/points3D.txt" ]; then
        cp "$COLMAP_SRC/points3D.txt" "$TEST_DIR/sparse/0/"
        log "Copied points3D.txt"
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

echo "=== [3/3] Training ($TEST_ITERATIONS iterations) ==="
echo ""

mkdir -p "$OUTPUT_DIR"

info "Starting LichtFeld Studio training..."
info "Binary: $BINARY"
info "Data: $TEST_DIR"
info "Output: $OUTPUT_DIR"
info "Strategy: mcmc"
info "Max Gaussians: $MAX_GAUSSIANS"
info "Iterations: $TEST_ITERATIONS"
echo ""

# Show GPU memory before training
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

# Run training
# Try with --help first to discover actual flags
info "Checking available flags..."
"$BINARY" --help 2>&1 | head -30 || true
echo ""

info "Launching training..."
time "$BINARY" \
    -d "$TEST_DIR" \
    -o "$OUTPUT_DIR" \
    --strategy mcmc \
    --max-cap "$MAX_GAUSSIANS" \
    -i "$TEST_ITERATIONS" \
    -r 2 \
    --headless

TRAIN_EXIT=$?

echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    log "Training completed successfully!"
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
    warn ""
    warn "Try running with PTX fallback:"
    warn "  cd $REPO_DIR/build"
    warn "  cmake .. -DBUILD_CUDA_PTX_ONLY=ON -DBUILD_CUDA_MIN_SM=75"
    warn "  cmake --build . -j\$(nproc)"
fi

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  Full training:  -i 30000 --max-cap 500000 -r 1"
echo "  Full dataset:   -d $FULL_DIR"
echo ""
