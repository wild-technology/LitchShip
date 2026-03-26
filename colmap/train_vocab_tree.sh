#!/bin/bash
# =============================================================================
# Vocabulary Tree Trainer — Domain-Specific for Underwater Imagery
#
# Trains a custom COLMAP vocabulary tree from underwater shipwreck imagery.
# A domain-specific tree produces better image retrieval than generic Flickr
# trees — fewer missed loop closures and better pair selection for sequential
# and vocabulary tree matching.
#
# The trained tree is reusable across all underwater projects. Train once
# from a diverse set of dive sites, reuse everywhere.
#
# Usage:
#   ./train_vocab_tree.sh /path/to/training_images
#   ./train_vocab_tree.sh /path/to/training_images --num-words 262144
#   ./train_vocab_tree.sh /path/to/training_images --output ~/colmap-vocab/my_tree.bin
#
# The input directory should contain a large, diverse set of underwater images
# (ideally 10K-50K from multiple sites). You can use symlinks to combine images
# from multiple dataset directories.
#
# Requires: COLMAP with CUDA (~/.local/bin/colmap)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_root="$SCRIPT_DIR"; while [ "$_root" != "/" ] && [ ! -f "$_root/lib/common.sh" ]; do _root="$(dirname "$_root")"; done
source "$_root/lib/common.sh"
# LITCHSHIP_ROOT is now set automatically by common.sh

# ─── Defaults ────────────────────────────────────────────────────────────────

IMAGE_DIR=""
OUTPUT_TREE="$HOME/colmap-vocab/vocab_tree_underwater_256K.bin"
DB_PATH=""                  # auto-generated in workspace
NUM_WORDS=262144            # 256K — sweet spot for 20-50K image datasets
NUM_ITERATIONS=100          # COLMAP 4.x default for FAISS k-means
NUM_ROUNDS=3                # Number of clustering rounds (best kept)
MAX_DESCRIPTORS=-1          # -1 = use all (190GB RAM handles 50K images)
MAX_FEATURES=16384
PEAK_THRESHOLD=0.004
GPU_INDEX=0
NUM_THREADS=$(nproc)
BATCH=false
DRY_RUN=false
COLMAP_BIN="${COLMAP_BIN:-$HOME/.local/bin/colmap}"

# ─── Usage ───────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
Usage: train_vocab_tree.sh <image_dir> [options]

Arguments:
  image_dir    Directory of training images (10K-50K from multiple dive sites)

Options:
  --output PATH        Output tree path (default: ~/colmap-vocab/vocab_tree_underwater_256K.bin)
  --database PATH      Use existing COLMAP database (skip feature extraction)
  --num-words N        Number of visual words (default: 262144 = 256K)
  --iterations N       K-means iterations per step (default: 100)
  --rounds N           Clustering rounds, best kept (default: 3)
  --max-descriptors N  Limit total descriptors, -1 for all (default: -1)
  --max-features N     SIFT features per image (default: 16384)
  --gpu-index N        GPU for feature extraction (default: 0)
  --num-threads N      CPU threads (default: all cores)
  --batch              Non-interactive mode
  --dry-run            Print commands without executing
  -h, --help           Show this help

Memory guide (at 16K features/image, 128 bytes/descriptor):
  10K images ≈  21 GB working memory
  20K images ≈  42 GB working memory
  50K images ≈ 105 GB working memory + ~50 GB overhead = ~155 GB total
USAGE
    exit 1
}

# ─── Argument Parsing ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)       OUTPUT_TREE="$2"; shift 2 ;;
        --database)     DB_PATH="$2"; shift 2 ;;
        --num-words)    NUM_WORDS="$2"; shift 2 ;;
        --iterations)   NUM_ITERATIONS="$2"; shift 2 ;;
        --rounds)       NUM_ROUNDS="$2"; shift 2 ;;
        --max-descriptors) MAX_DESCRIPTORS="$2"; shift 2 ;;
        --max-features) MAX_FEATURES="$2"; shift 2 ;;
        --gpu-index)    GPU_INDEX="$2"; shift 2 ;;
        --num-threads)  NUM_THREADS="$2"; shift 2 ;;
        --batch)        BATCH=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        -*)             error "Unknown option: $1"; usage ;;
        *)
            if [ -z "$IMAGE_DIR" ]; then
                IMAGE_DIR="$1"
            else
                error "Unexpected argument: $1"; usage
            fi
            shift ;;
    esac
done

[ -z "$IMAGE_DIR" ] && { error "Missing <image_dir>"; usage; }
[ -d "$IMAGE_DIR" ] || { error "Image directory not found: $IMAGE_DIR"; exit 1; }

IMAGE_DIR="$(cd "$IMAGE_DIR" && pwd)"

# ─── Environment ─────────────────────────────────────────────────────────────

export OMP_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"

# ─── Helper Functions ────────────────────────────────────────────────────────

prompt_continue() {
    local msg="$1"
    if [ "$BATCH" = true ]; then return 0; fi
    echo ""
    echo -e "${BOLD}$msg${NC}"
    read -rp "  Continue? [Y/n/q] " reply
    case "$reply" in
        n|N) info "Skipping."; return 1 ;;
        q|Q) info "Quitting."; exit 0 ;;
        *)   return 0 ;;
    esac
}

set_stage() {
    echo "$1" > "$STAGE_FILE" 2>/dev/null || true
}

run_colmap() {
    local desc="$1"; shift
    if [ "$DRY_RUN" = true ]; then
        echo -e "${DIM}[dry-run] $COLMAP_BIN $*${NC}"
        return 0
    fi
    info "$desc"
    local start_time=$SECONDS
    "$COLMAP_BIN" "$@" \
        --log_target stderr_and_file \
        --log_path "$LOGDIR"
    local elapsed=$((SECONDS - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    log "$desc completed in ${mins}m ${secs}s"
}

count_images() {
    find "$1" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l
}

# ─── Detect Environment ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Vocabulary Tree Trainer — Underwater Domain${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify COLMAP
if [ ! -x "$COLMAP_BIN" ]; then
    error "COLMAP not found at: $COLMAP_BIN"
    exit 1
fi

setup_paths
detect_gpu

NUM_IMAGES=$(count_images "$IMAGE_DIR")
if [ "$NUM_IMAGES" -eq 0 ]; then
    error "No images found in: $IMAGE_DIR"
    exit 1
fi

RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo)

# Estimate memory requirements
TRAIN_IMAGES="$NUM_IMAGES"
EST_DESCRIPTORS=$((TRAIN_IMAGES * MAX_FEATURES))
if [ "$MAX_DESCRIPTORS" -gt 0 ] && [ "$MAX_DESCRIPTORS" -lt "$EST_DESCRIPTORS" ]; then
    EST_DESCRIPTORS="$MAX_DESCRIPTORS"
fi
EST_RAW_GB=$((EST_DESCRIPTORS * 128 / 1073741824))
EST_WORKING_GB=$((EST_RAW_GB * 3 / 2))  # ~1.5x overhead

echo -e "  ${CYAN}COLMAP:${NC}      $($COLMAP_BIN help 2>&1 | head -1)"
echo -e "  ${CYAN}GPU:${NC}         $GPU_NAME"
echo -e "  ${CYAN}CPU:${NC}         $NUM_THREADS threads"
echo -e "  ${CYAN}RAM:${NC}         ${RAM_GB} GB"
echo ""
echo -e "  ${CYAN}Training images:${NC}  $NUM_IMAGES available"
if [ "$MAX_DESCRIPTORS" -gt 0 ]; then
    echo -e "  ${CYAN}Max descriptors:${NC} $MAX_DESCRIPTORS"
else
    echo -e "  ${CYAN}Using:${NC}           all $NUM_IMAGES images"
fi
echo -e "  ${CYAN}Features/image:${NC}  $MAX_FEATURES"
echo -e "  ${CYAN}Visual words:${NC}    $NUM_WORDS"
echo -e "  ${CYAN}Rounds:${NC}          $NUM_ROUNDS"
echo ""
echo -e "  ${CYAN}Est. descriptors:${NC}  ~$((EST_DESCRIPTORS / 1000000))M"
echo -e "  ${CYAN}Est. memory:${NC}       ~${EST_WORKING_GB} GB (${RAM_GB} GB available)"
echo -e "  ${CYAN}Output:${NC}           $OUTPUT_TREE"
echo ""

# Memory warning
if [ "$EST_WORKING_GB" -gt "$((RAM_GB * 9 / 10))" ]; then
    warn "Estimated memory usage (${EST_WORKING_GB} GB) is close to available RAM (${RAM_GB} GB)"
    warn "Consider using --max-images to subsample, or close other applications"
    if [ "$BATCH" = false ]; then
        read -rp "  Proceed anyway? [y/N] " reply
        [ "$reply" = "y" ] || [ "$reply" = "Y" ] || exit 0
    fi
fi

# ─── Workspace ───────────────────────────────────────────────────────────────

EXTERNAL_DB=false
if [ -n "$DB_PATH" ] && [ -f "$DB_PATH" ]; then
    # User provided an existing database — skip extraction entirely
    EXTERNAL_DB=true
    SKIP_EXTRACT=true
    EXISTING_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo "0")
    log "Using existing database: $DB_PATH ($EXISTING_COUNT images with features)"
elif [ -n "$DB_PATH" ] && [ ! -f "$DB_PATH" ]; then
    error "Database not found: $DB_PATH"
    exit 1
else
    WORKSPACE="$HOME/colmap-vocab/workspace"
    DB_PATH="$WORKSPACE/training.db"
    mkdir -p "$WORKSPACE"
fi

mkdir -p "$(dirname "$OUTPUT_TREE")"

# Log and stage tracking
if [ "$EXTERNAL_DB" = true ]; then
    LOGDIR="$(dirname "$DB_PATH")/logs"
else
    LOGDIR="$WORKSPACE/logs"
fi
STAGE_FILE="${LOGDIR}/../.colmap_stage"
MONITOR_PID=""
mkdir -p "$LOGDIR"

# Launch dashboard
MONITOR_SCRIPT="$SCRIPT_DIR/monitor.py"
if [ -f "$MONITOR_SCRIPT" ] && [ "$DRY_RUN" = false ]; then
    python3 "$MONITOR_SCRIPT" \
        --db "$DB_PATH" \
        --log-dir "$LOGDIR" \
        --stage-file "$STAGE_FILE" \
        --image-count "$NUM_IMAGES" \
        --port 8080 &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null; wait $MONITOR_PID 2>/dev/null" EXIT
    log "Dashboard started at http://localhost:8080 (PID $MONITOR_PID)"
fi

set_stage "INITIALIZING"

# Check for existing training database (only when not using external DB)
if [ "$EXTERNAL_DB" = false ] && [ -f "$DB_PATH" ] && [ "$DRY_RUN" = false ]; then
    EXISTING_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo "0")
    if [ "$EXISTING_COUNT" -gt 0 ]; then
        echo -e "  ${YELLOW}Existing training database found with $EXISTING_COUNT images.${NC}"
        if [ "$BATCH" = false ]; then
            echo "  1) Reuse existing features (skip extraction)"
            echo "  2) Delete and re-extract from scratch"
            read -rp "  Choice [1/2]: " choice
            if [ "$choice" = "1" ]; then
                SKIP_EXTRACT=true
            else
                rm -f "$DB_PATH"
                SKIP_EXTRACT=false
            fi
        else
            info "Reusing existing training database"
            SKIP_EXTRACT=true
        fi
    else
        SKIP_EXTRACT=false
    fi
elif [ "$EXTERNAL_DB" = false ]; then
    SKIP_EXTRACT=false
fi

# ─── Stage 1: Feature Extraction ────────────────────────────────────────────

set_stage "FEATURE_EXTRACTION"
if [ "${SKIP_EXTRACT:-false}" = false ]; then
    echo ""
    echo -e "${BOLD}── Stage 1: Feature Extraction (GPU) ──────────────────────────${NC}"
    echo -e "  Extracting $MAX_FEATURES SIFT features per image with underwater-tuned parameters."
    echo -e "  This stage is GPU-accelerated and typically fast."
    echo ""

    if prompt_continue "Extract features from $NUM_IMAGES images?"; then
        run_colmap "Feature extraction for vocab tree training" feature_extractor \
            --database_path "$DB_PATH" \
            --image_path "$IMAGE_DIR" \
            --ImageReader.camera_model OPENCV \
            --FeatureExtraction.use_gpu 1 \
            --FeatureExtraction.gpu_index "$GPU_INDEX" \
            --FeatureExtraction.max_image_size 4096 \
            --FeatureExtraction.num_threads "$NUM_THREADS" \
            --SiftExtraction.max_num_features "$MAX_FEATURES" \
            --SiftExtraction.first_octave -1 \
            --SiftExtraction.peak_threshold "$PEAK_THRESHOLD" \
            --SiftExtraction.edge_threshold 16

        if [ "$DRY_RUN" = false ] && [ -f "$DB_PATH" ]; then
            FEAT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo "?")
            log "Features extracted for $FEAT_COUNT images"
        fi
    fi
fi

set_stage "VOCAB_TREE_BUILDING"
# ─── Stage 2: Build Vocabulary Tree ─────────────────────────────────────────

echo ""
echo -e "${BOLD}── Stage 2: Vocabulary Tree Construction (CPU) ────────────────${NC}"
echo -e "  This is the main compute-intensive step. It runs hierarchical k-means"
echo -e "  clustering on all extracted SIFT descriptors."
echo ""
echo -e "  ${YELLOW}This is CPU-only and may take several hours for large datasets.${NC}"
echo -e "  ${YELLOW}The GPU is not used during tree construction.${NC}"
echo ""

# Estimate time
if [ "$TRAIN_IMAGES" -le 10000 ]; then
    EST_TIME="30 minutes to 2 hours"
elif [ "$TRAIN_IMAGES" -le 25000 ]; then
    EST_TIME="1 to 4 hours"
else
    EST_TIME="4 to 12 hours"
fi
echo -e "  ${CYAN}Estimated time:${NC}  $EST_TIME"
echo ""

if prompt_continue "Build vocabulary tree ($NUM_WORDS words, $NUM_ROUNDS rounds, FAISS)?"; then
    TREE_START=$SECONDS

    run_colmap "Vocabulary tree construction" vocab_tree_builder \
        --database_path "$DB_PATH" \
        --vocab_tree_path "$OUTPUT_TREE" \
        --num_visual_words "$NUM_WORDS" \
        --num_iterations "$NUM_ITERATIONS" \
        --num_rounds "$NUM_ROUNDS" \
        --max_num_descriptors "$MAX_DESCRIPTORS"

    if [ "$DRY_RUN" = false ]; then
        TREE_ELAPSED=$((SECONDS - TREE_START))
        TREE_HOURS=$((TREE_ELAPSED / 3600))
        TREE_MINS=$(( (TREE_ELAPSED % 3600) / 60 ))

        if [ -f "$OUTPUT_TREE" ]; then
            TREE_SIZE=$(du -h "$OUTPUT_TREE" | cut -f1)
            echo ""
            log "Vocabulary tree built successfully!"
            echo -e "  ${CYAN}Output:${NC}    $OUTPUT_TREE"
            echo -e "  ${CYAN}Size:${NC}      $TREE_SIZE"
            echo -e "  ${CYAN}Words:${NC}     $NUM_WORDS"
            echo -e "  ${CYAN}Time:${NC}      ${TREE_HOURS}h ${TREE_MINS}m"
        else
            error "Vocabulary tree file not created"
            exit 1
        fi
    fi
fi

set_stage "COMPLETE"
# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Vocabulary Tree Training Complete${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Tree:${NC}   $OUTPUT_TREE"
echo -e "  ${CYAN}Words:${NC}  $NUM_WORDS"
echo ""
echo -e "  ${DIM}Usage in colmap_reconstruct.sh:${NC}"
echo -e "  ${DIM}  --mode sequential --vocab-tree $OUTPUT_TREE${NC}"
echo ""
echo -e "  ${DIM}This tree is reusable across all underwater projects.${NC}"
echo -e "  ${DIM}No need to retrain per-dataset.${NC}"
echo ""
