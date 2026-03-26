#!/bin/bash
# =============================================================================
# COLMAP Reconstruction Pipeline — Underwater-Tuned Interactive Workflow
#
# Runs GPU-accelerated Structure-from-Motion on raw underwater imagery.
# Produces sparse (and optionally dense) reconstructions compatible with
# the LitchShip progressive training pipeline.
#
# Matching modes:
#   exhaustive   All pairs — robust but O(n²), best for ≤3000 images
#   sequential   Ordered frames — fast, with vocab-tree loop detection
#   spatial      GPS/USBL priors — matches nearby images only
#
# Usage:
#   ./colmap_reconstruct.sh /path/to/images /path/to/output
#   ./colmap_reconstruct.sh /path/to/images /path/to/output --mode sequential
#   ./colmap_reconstruct.sh /path/to/images /path/to/output --mode spatial --gps-file nav.csv
#   ./colmap_reconstruct.sh /path/to/images /path/to/output --dense
#   ./colmap_reconstruct.sh /path/to/images /path/to/output --batch
#
# Requires: COLMAP with CUDA (~/.local/bin/colmap)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITCHSHIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$LITCHSHIP_ROOT/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────────────

IMAGE_DIR=""
OUTPUT_DIR=""
MODE=""                     # empty = prompt user
MAPPER="global"             # global (uses gravity priors) or incremental
DENSE=false
CAMERA_MODEL="OPENCV"
GPS_FILE=""
MAX_FEATURES=16384
NUM_THREADS=$(nproc)
GPU_INDEX=0
VOCAB_TREE=""
CAMERA_CONFIG=""
FEATURE_TYPE="SIFT"         # SIFT or ALIKED
BATCH=false
DRY_RUN=false
COLMAP_BIN="${COLMAP_BIN:-$HOME/.local/bin/colmap}"

# Underwater-tuned SIFT defaults
PEAK_THRESHOLD=0.004
EDGE_THRESHOLD=16
MAX_IMAGE_SIZE=4096

# ─── Usage ───────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
Usage: colmap_reconstruct.sh <image_dir> <output_dir> [options]

Arguments:
  image_dir    Directory containing source images (JPEG/PNG)
  output_dir   Output workspace (will be created)

Matching modes:
  --mode exhaustive    All-pairs matching (≤3000 images recommended)
  --mode sequential    Ordered frames + loop detection (ROV/video)
  --mode spatial       GPS/USBL-guided matching (requires --gps-file)

Options:
  --mapper M           Mapper: global (default, uses gravity priors) or incremental
  --features TYPE      Feature type: SIFT (default) or ALIKED (learned, more robust)
  --dense              Also run PatchMatch dense stereo + fusion
  --camera-model M     Camera model (default: OPENCV, single-camera mode)
  --camera-config F    Multi-camera config file (prefix;model;focal_mm;sensor_width_mm)
  --gps-file PATH      Nav file (CSV/semicolon) with positions and optional orientation
  --max-features N     SIFT features per image (default: 16384)
  --peak-threshold F   SIFT peak threshold (default: 0.004, lower=more features)
  --max-image-size N   Max image dimension for extraction (default: 4096)
  --num-threads N      CPU threads (default: all cores)
  --gpu-index N        GPU device index (default: 0)
  --vocab-tree PATH    Vocabulary tree for sequential matching
  --batch              Non-interactive mode (skip all prompts)
  --dry-run            Print commands without executing
  -h, --help           Show this help
USAGE
    exit 1
}

# ─── Argument Parsing ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           MODE="$2"; shift 2 ;;
        --mapper)         MAPPER="$2"; shift 2 ;;
        --features)       FEATURE_TYPE="$2"; shift 2 ;;
        --dense)          DENSE=true; shift ;;
        --camera-model)   CAMERA_MODEL="$2"; shift 2 ;;
        --camera-config)  CAMERA_CONFIG="$2"; shift 2 ;;
        --gps-file)       GPS_FILE="$2"; shift 2 ;;
        --max-features)   MAX_FEATURES="$2"; shift 2 ;;
        --peak-threshold) PEAK_THRESHOLD="$2"; shift 2 ;;
        --max-image-size) MAX_IMAGE_SIZE="$2"; shift 2 ;;
        --num-threads)    NUM_THREADS="$2"; shift 2 ;;
        --gpu-index)      GPU_INDEX="$2"; shift 2 ;;
        --vocab-tree)     VOCAB_TREE="$2"; shift 2 ;;
        --batch)          BATCH=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        -h|--help)        usage ;;
        -*)               error "Unknown option: $1"; usage ;;
        *)
            if [ -z "$IMAGE_DIR" ]; then
                IMAGE_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                error "Unexpected argument: $1"; usage
            fi
            shift ;;
    esac
done

[ -z "$IMAGE_DIR" ] && { error "Missing <image_dir>"; usage; }
[ -z "$OUTPUT_DIR" ] && { error "Missing <output_dir>"; usage; }
[ -d "$IMAGE_DIR" ] || { error "Image directory not found: $IMAGE_DIR"; exit 1; }

# Resolve to absolute paths
IMAGE_DIR="$(cd "$IMAGE_DIR" && pwd)"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

# ─── Environment ─────────────────────────────────────────────────────────────

export OMP_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"

# Log and stage tracking directories (set after OUTPUT_DIR resolved)
LOGDIR="$OUTPUT_DIR/logs"
STAGE_FILE="$OUTPUT_DIR/.colmap_stage"
MONITOR_PID=""

# ─── Helper Functions ────────────────────────────────────────────────────────

set_stage() {
    echo "$1" > "$STAGE_FILE" 2>/dev/null || true
}

prompt_continue() {
    local msg="$1"
    if [ "$BATCH" = true ]; then return 0; fi
    echo ""
    echo -e "${BOLD}$msg${NC}"
    read -rp "  Continue? [Y/n/q] " reply
    case "$reply" in
        n|N) info "Skipping stage."; return 1 ;;
        q|Q) info "Quitting."; exit 0 ;;
        *)   return 0 ;;
    esac
}

prompt_choice() {
    local prompt="$1"
    shift
    local options=("$@")
    if [ "$BATCH" = true ]; then echo "${options[0]}"; return; fi
    echo "" >&2
    echo -e "${BOLD}$prompt${NC}" >&2
    for i in "${!options[@]}"; do
        echo "  $((i+1))) ${options[$i]}" >&2
    done
    while true; do
        read -rp "  Choice [1-${#options[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo "${options[$((choice-1))]}"
            return
        fi
        echo "  Invalid choice. Enter 1-${#options[@]}." >&2
    done
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
echo -e "${BOLD}  COLMAP Reconstruction Pipeline — Underwater-Tuned${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Verify COLMAP binary
if [ ! -x "$COLMAP_BIN" ]; then
    error "COLMAP not found at: $COLMAP_BIN"
    error "Build with CUDA first, or set COLMAP_BIN=/path/to/colmap"
    exit 1
fi

COLMAP_VERSION=$("$COLMAP_BIN" help 2>&1 | head -2 || true)
if echo "$COLMAP_VERSION" | grep -q "without CUDA"; then
    error "COLMAP was built without CUDA — GPU acceleration unavailable"
    error "Rebuild from source with -DCUDA_ENABLED=ON"
    exit 1
fi

# Detect hardware
setup_paths
detect_gpu

NUM_IMAGES=$(count_images "$IMAGE_DIR")
if [ "$NUM_IMAGES" -eq 0 ]; then
    error "No images found in: $IMAGE_DIR"
    error "Supported formats: jpg, jpeg, png, tif, tiff"
    exit 1
fi

RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo -e "  ${CYAN}COLMAP:${NC}  $(echo "$COLMAP_VERSION" | head -1)"
echo -e "  ${CYAN}GPU:${NC}     $GPU_NAME ($VRAM_MB MiB VRAM)"
echo -e "  ${CYAN}CPU:${NC}     $NUM_THREADS threads"
echo -e "  ${CYAN}RAM:${NC}     ${RAM_GB} GB"
echo ""
DISK_FREE_GB=$(df --output=avail "$OUTPUT_DIR" | tail -1 | awk '{printf "%.0f", $1/1048576}')
EST_DB_GB=$((NUM_IMAGES * MAX_FEATURES * 128 * 2 / 1073741824))

echo -e "  ${CYAN}Disk:${NC}    ${DISK_FREE_GB} GB free"
echo ""
echo -e "  ${CYAN}Images:${NC}  $NUM_IMAGES in $(basename "$IMAGE_DIR")"
echo -e "  ${CYAN}Output:${NC}  $OUTPUT_DIR"
if [ -n "$CAMERA_CONFIG" ]; then
    echo -e "  ${CYAN}Cameras:${NC} multi-camera config: $(basename "$CAMERA_CONFIG")"
else
    echo -e "  ${CYAN}Camera:${NC}  $CAMERA_MODEL (single model for all images)"
fi
echo ""

# Disk space warning
if [ "$EST_DB_GB" -gt "$((DISK_FREE_GB * 8 / 10))" ]; then
    warn "Estimated database size (~${EST_DB_GB} GB) may exceed available disk (${DISK_FREE_GB} GB free)"
    warn "Consider reducing --max-features or using a different output location"
fi

# ─── Mode Selection ──────────────────────────────────────────────────────────

if [ -z "$MODE" ]; then
    # Suggest based on image count
    if [ "$NUM_IMAGES" -le 3000 ]; then
        suggestion="exhaustive (≤3000 images, manageable O(n²))"
    else
        suggestion="sequential (>3000 images, faster than exhaustive)"
    fi
    info "Suggested mode: $suggestion"

    MODE=$(prompt_choice "Select matching mode:" \
        "exhaustive" "sequential" "spatial")
fi

# Validate mode
case "$MODE" in
    exhaustive|sequential|spatial) ;;
    *) error "Invalid mode: $MODE (use exhaustive, sequential, or spatial)"; exit 1 ;;
esac

# Spatial mode requires GPS file
if [ "$MODE" = "spatial" ] && [ -z "$GPS_FILE" ]; then
    if [ "$BATCH" = true ]; then
        error "Spatial mode requires --gps-file"; exit 1
    fi
    echo ""
    read -rp "  GPS file path (CSV: image_name,lat,lon,alt): " GPS_FILE
    [ -f "$GPS_FILE" ] || { error "GPS file not found: $GPS_FILE"; exit 1; }
fi

# Sequential mode: find vocab tree
if [ "$MODE" = "sequential" ] && [ -z "$VOCAB_TREE" ]; then
    # Prefer domain-specific tree, fall back to generic
    if [ -f "$HOME/colmap-vocab/vocab_tree_underwater_256K.bin" ]; then
        VOCAB_TREE="$HOME/colmap-vocab/vocab_tree_underwater_256K.bin"
        info "Using domain-trained underwater vocabulary tree"
    elif [ -f "$HOME/colmap-vocab/vocab_tree_flickr100K_words256K.bin" ]; then
        VOCAB_TREE="$HOME/colmap-vocab/vocab_tree_flickr100K_words256K.bin"
        warn "Using generic Flickr vocabulary tree (train a domain-specific tree for better results)"
    else
        warn "No vocabulary tree found — loop detection will be disabled"
        warn "Download: wget -O ~/colmap-vocab/vocab_tree_flickr100K_words256K.bin https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
    fi
fi

# Time estimate
if [ "$MODE" = "exhaustive" ] && [ "$NUM_IMAGES" -gt 3000 ]; then
    warn "Exhaustive matching on $NUM_IMAGES images = ~$((NUM_IMAGES * NUM_IMAGES / 2000000)) million pairs"
    warn "This could take many hours. Consider --mode sequential for ordered imagery."
    prompt_continue "Proceed with exhaustive matching?" || { MODE="sequential"; info "Switched to sequential mode."; }
fi

log "Mode: $MODE"
echo ""

# ─── Workspace Setup ────────────────────────────────────────────────────────

DB_PATH="$OUTPUT_DIR/database.db"
SPARSE_PATH="$OUTPUT_DIR/sparse"
mkdir -p "$LOGDIR"

# Symlink images into workspace
if [ ! -e "$OUTPUT_DIR/images" ]; then
    ln -s "$IMAGE_DIR" "$OUTPUT_DIR/images"
    log "Linked images → $IMAGE_DIR"
fi

# ─── Launch Monitoring Dashboard ─────────────────────────────────────────────

MONITOR_SCRIPT="$SCRIPT_DIR/colmap_monitor.py"
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

# ─── Stage 1: Feature Extraction ────────────────────────────────────────────

set_stage "FEATURE_EXTRACTION"
echo -e "${BOLD}── Stage 1: Feature Extraction (GPU) ──────────────────────────${NC}"
echo -e "  Feature type:    $FEATURE_TYPE"
echo -e "  Features/image:  $MAX_FEATURES"
echo -e "  First octave:    0 (native resolution, no upsample)"
echo -e "  Max image size:  $MAX_IMAGE_SIZE px"
echo -e "  Camera model:    $CAMERA_MODEL"

if prompt_continue "Extract features from $NUM_IMAGES images?"; then
    EXTRACT_ARGS=(
        --database_path "$DB_PATH"
        --image_path "$IMAGE_DIR"
        --ImageReader.camera_model "$CAMERA_MODEL"
    )

    if [ "$FEATURE_TYPE" = "ALIKED" ]; then
        EXTRACT_ARGS+=(
            --FeatureExtraction.type ALIKED
            --AlikedExtraction.max_num_features "$MAX_FEATURES"
        )
    else
        EXTRACT_ARGS+=(
            --FeatureExtraction.type SIFT
            --FeatureExtraction.use_gpu 1
            --FeatureExtraction.gpu_index "$GPU_INDEX"
            --FeatureExtraction.max_image_size "$MAX_IMAGE_SIZE"
            --FeatureExtraction.num_threads "$NUM_THREADS"
            --SiftExtraction.max_num_features "$MAX_FEATURES"
            --SiftExtraction.first_octave 0
            --SiftExtraction.edge_threshold "$EDGE_THRESHOLD"
        )
    fi

    run_colmap "Feature extraction ($FEATURE_TYPE)" feature_extractor "${EXTRACT_ARGS[@]}"

    if [ "$DRY_RUN" = false ] && [ -f "$DB_PATH" ]; then
        FEAT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo "?")
        log "Features extracted for $FEAT_COUNT images"
    fi
fi

set_stage "CAMERA_ASSIGNMENT"
# ─── Stage 1.5a: Multi-Camera Assignment ─────────────────────────────────────

if [ -n "$CAMERA_CONFIG" ] && [ -f "$CAMERA_CONFIG" ]; then
    echo ""
    echo -e "${BOLD}── Stage 1.5a: Multi-Camera Assignment ────────────────────────${NC}"
    echo -e "  Config: $CAMERA_CONFIG"

    ASSIGN_SCRIPT="$SCRIPT_DIR/assign_cameras.py"
    if [ -f "$ASSIGN_SCRIPT" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${DIM}[dry-run] python3 $ASSIGN_SCRIPT $DB_PATH $CAMERA_CONFIG${NC}"
        elif [ "$BATCH" = true ]; then
            python3 "$ASSIGN_SCRIPT" "$DB_PATH" "$CAMERA_CONFIG" --batch
            log "Camera models assigned by prefix"
        else
            python3 "$ASSIGN_SCRIPT" "$DB_PATH" "$CAMERA_CONFIG"
        fi
    else
        error "Camera assignment script not found: $ASSIGN_SCRIPT"
        exit 1
    fi
fi

# ─── Stage 1.5b: Nav/GPS Import ─────────────────────────────────────────────

if [ -n "$GPS_FILE" ] && [ -f "$GPS_FILE" ]; then
    echo ""
    echo -e "${BOLD}── Stage 1.5b: Nav/Position Import ────────────────────────────${NC}"
    echo -e "  Nav file: $GPS_FILE"

    if prompt_continue "Import position/orientation priors into COLMAP database?"; then
        GPS_IMPORT_SCRIPT="$SCRIPT_DIR/import_gps_to_colmap.py"
        if [ -f "$GPS_IMPORT_SCRIPT" ]; then
            IMPORT_FLAGS=()
            [ "$BATCH" = true ] && IMPORT_FLAGS+=(--batch)
            if [ "$DRY_RUN" = true ]; then
                echo -e "${DIM}[dry-run] python3 $GPS_IMPORT_SCRIPT $DB_PATH $GPS_FILE${NC}"
            else
                python3 "$GPS_IMPORT_SCRIPT" "$DB_PATH" "$GPS_FILE" "${IMPORT_FLAGS[@]}"
                log "Nav priors imported"
            fi
        else
            error "Nav import script not found: $GPS_IMPORT_SCRIPT"
            exit 1
        fi
    fi
fi

# ─── Stage 2: Feature Matching ──────────────────────────────────────────────

set_stage "MATCHING"
echo ""
echo -e "${BOLD}── Stage 2: Feature Matching ($MODE) ─────────────────────────${NC}"

# Set matcher type based on feature type
if [ "$FEATURE_TYPE" = "ALIKED" ]; then
    MATCH_TYPE="ALIKED_LIGHTGLUE"
    echo -e "  Matcher:     ALIKED + LightGlue (learned)"
else
    MATCH_TYPE="SIFT_BRUTEFORCE"
    echo -e "  Matcher:     SIFT brute-force (GPU)"
fi

MATCH_ARGS=(--FeatureMatching.type "$MATCH_TYPE")
case "$MODE" in
    exhaustive)
        echo -e "  Block size:  100"

        MATCH_CMD="exhaustive_matcher"
        MATCH_ARGS+=(
            --ExhaustiveMatching.block_size 100
        )
        MATCH_ARGS+=(--FeatureMatching.num_threads "$NUM_THREADS")
        if [ "$FEATURE_TYPE" = "SIFT" ]; then
            MATCH_ARGS+=(
                --SiftMatching.max_ratio 0.85
                --SiftMatching.max_distance 0.7
                --SiftMatching.cross_check 1
            )
        fi
        ;;
    sequential)
        echo -e "  Overlap:     20 frames"
        echo -e "  Loop detect: enabled"
        [ -n "$VOCAB_TREE" ] && echo -e "  Vocab tree:  $(basename "$VOCAB_TREE")"

        MATCH_CMD="sequential_matcher"
        MATCH_ARGS+=(
            --SequentialMatching.overlap 20
            --SequentialMatching.quadratic_overlap 1
        )
        MATCH_ARGS+=(--FeatureMatching.num_threads "$NUM_THREADS")
        if [ "$FEATURE_TYPE" = "SIFT" ]; then
            MATCH_ARGS+=(
                --SiftMatching.max_ratio 0.85
                --SiftMatching.max_distance 0.7
                --SiftMatching.cross_check 1
            )
        fi
        if [ -n "$VOCAB_TREE" ] && [ -f "$VOCAB_TREE" ]; then
            MATCH_ARGS+=(
                --SequentialMatching.loop_detection 1
                --SequentialMatching.loop_detection_period 10
                --SequentialMatching.loop_detection_num_images 100
                --SequentialMatching.loop_detection_num_nearest_neighbors 5
                --SequentialMatching.vocab_tree_path "$VOCAB_TREE"
            )
        fi
        ;;
    spatial)
        echo -e "  Max neighbors: 100"
        echo -e "  Max distance:  50 meters"
        echo -e "  Coordinates:   from pose_priors table"

        MATCH_CMD="spatial_matcher"
        MATCH_ARGS+=(
            --SpatialMatching.max_num_neighbors 100
            --SpatialMatching.max_distance 50
        )
        MATCH_ARGS+=(--FeatureMatching.num_threads "$NUM_THREADS")
        if [ "$FEATURE_TYPE" = "SIFT" ]; then
            MATCH_ARGS+=(
                --SiftMatching.max_ratio 0.85
                --SiftMatching.cross_check 1
            )
        fi
        ;;
esac

if prompt_continue "Run $MODE matching?"; then
    run_colmap "Feature matching ($MODE)" "$MATCH_CMD" \
        --database_path "$DB_PATH" \
        "${MATCH_ARGS[@]}"

    if [ "$DRY_RUN" = false ] && [ -f "$DB_PATH" ]; then
        MATCH_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0;" 2>/dev/null || echo "?")
        log "Verified image pairs with matches: $MATCH_COUNT"
    fi
fi

# ─── Stage 3: Sparse Reconstruction ─────────────────────────────────────────

set_stage "RECONSTRUCTION"
echo ""
echo -e "${BOLD}── Stage 3: Sparse Reconstruction ($MAPPER mapper) ───────────${NC}"
echo -e "  Mapper:           $MAPPER"
echo -e "  CPU threads:      $NUM_THREADS"
if [ "$MAPPER" = "global" ]; then
    echo -e "  Gravity priors:   used in rotation averaging (if imported)"
    echo -e "  Position priors:  used in global positioning"
else
    echo -e "  Min matches:      30 (raised for underwater)"
    echo -e "  Init min inliers: 200"
fi

if prompt_continue "Run $MAPPER mapper?"; then
    mkdir -p "$SPARSE_PATH"

    if [ "$MAPPER" = "global" ]; then
        run_colmap "Sparse reconstruction (global)" global_mapper \
            --database_path "$DB_PATH" \
            --image_path "$IMAGE_DIR" \
            --output_path "$SPARSE_PATH" \
            --GlobalMapper.num_threads "$NUM_THREADS" \
            --GlobalMapper.min_num_matches 30 \
            --GlobalMapper.ba_refine_focal_length 1 \
            --GlobalMapper.ba_refine_extra_params 1
    else
        run_colmap "Sparse reconstruction (incremental)" mapper \
            --database_path "$DB_PATH" \
            --image_path "$IMAGE_DIR" \
            --output_path "$SPARSE_PATH" \
            --Mapper.ba_global_max_num_iterations 50 \
            --Mapper.ba_global_max_refinements 5 \
            --Mapper.ba_local_max_num_iterations 25 \
            --Mapper.min_num_matches 30 \
            --Mapper.init_min_num_inliers 200 \
            --Mapper.abs_pose_min_num_inliers 50 \
            --Mapper.filter_max_reproj_error 4.0 \
            --Mapper.num_threads "$NUM_THREADS"
    fi

    # Convert to text format for downstream pipeline
    # Global mapper writes directly to SPARSE_PATH; incremental writes to SPARSE_PATH/0/
    if [ "$DRY_RUN" = false ]; then
        # Find the model directory (handle both global and incremental output)
        MODEL_DIR=""
        if [ -f "$SPARSE_PATH/0/cameras.bin" ]; then
            MODEL_DIR="$SPARSE_PATH/0"
        elif [ -f "$SPARSE_PATH/cameras.bin" ]; then
            # Global mapper output — move into 0/ subdirectory for consistency
            mkdir -p "$SPARSE_PATH/0"
            mv "$SPARSE_PATH"/*.bin "$SPARSE_PATH"/*.txt "$SPARSE_PATH"/project.ini "$SPARSE_PATH/0/" 2>/dev/null
            MODEL_DIR="$SPARSE_PATH/0"
        fi

        if [ -n "$MODEL_DIR" ]; then
            info "Converting sparse model to text format..."
            "$COLMAP_BIN" model_converter \
                --input_path "$MODEL_DIR" \
                --output_path "$MODEL_DIR" \
                --output_type TXT

            # Report reconstruction stats
            if [ -f "$MODEL_DIR/images.txt" ]; then
                REG_IMAGES=$(grep -c "^[0-9]" "$MODEL_DIR/images.txt" 2>/dev/null || echo "?")
                log "Registered images: $REG_IMAGES / $NUM_IMAGES"
            fi
            if [ -f "$MODEL_DIR/points3D.txt" ]; then
                NUM_POINTS=$(grep -c "^[0-9]" "$MODEL_DIR/points3D.txt" 2>/dev/null || echo "?")
                log "3D points: $NUM_POINTS"
            fi
            if [ -f "$MODEL_DIR/cameras.txt" ]; then
                NUM_CAMERAS=$(grep -c "^[0-9]" "$MODEL_DIR/cameras.txt" 2>/dev/null || echo "?")
                log "Camera models: $NUM_CAMERAS"
            fi
        else
            error "No sparse model found in $SPARSE_PATH"
        fi

        # Check for multiple sub-models (incremental mapper only)
        SUBMODELS=$(find "$SPARSE_PATH" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ "$SUBMODELS" -gt 1 ]; then
            warn "Mapper produced $SUBMODELS sub-models (disconnected components)"
            warn "Model 0 is typically the largest. Inspect in COLMAP GUI: colmap gui"
        fi
    fi
fi

set_stage "DENSE_RECONSTRUCTION"
# ─── Stage 4: Dense Reconstruction (optional) ───────────────────────────────

if [ "$DENSE" = true ]; then
    echo ""
    echo -e "${BOLD}── Stage 4: Dense Reconstruction (GPU PatchMatch) ────────────${NC}"
    echo -e "  Max image size: 2000 px (PatchMatch)"
    echo -e "  Window radius:  7 (larger for low-texture underwater)"
    echo -e "  GPU:            $GPU_NAME"

    DENSE_PATH="$OUTPUT_DIR/dense"

    if prompt_continue "Run dense stereo reconstruction? (GPU-intensive)"; then
        # Undistort images
        run_colmap "Image undistortion" image_undistorter \
            --image_path "$IMAGE_DIR" \
            --input_path "$SPARSE_PATH/0" \
            --output_path "$DENSE_PATH" \
            --output_type COLMAP

        # PatchMatch stereo
        run_colmap "PatchMatch stereo (GPU)" patch_match_stereo \
            --workspace_path "$DENSE_PATH" \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true \
            --PatchMatchStereo.gpu_index "$GPU_INDEX" \
            --PatchMatchStereo.max_image_size 2000 \
            --PatchMatchStereo.window_radius 7 \
            --PatchMatchStereo.num_iterations 7 \
            --PatchMatchStereo.filter_min_ncc 0.05 \
            --PatchMatchStereo.filter_min_num_consistent 2

        # Stereo fusion
        run_colmap "Stereo fusion" stereo_fusion \
            --workspace_path "$DENSE_PATH" \
            --workspace_format COLMAP \
            --output_path "$DENSE_PATH/fused.ply" \
            --StereoFusion.min_num_pixels 3 \
            --StereoFusion.max_reproj_error 2.0

        if [ "$DRY_RUN" = false ] && [ -f "$DENSE_PATH/fused.ply" ]; then
            FUSED_SIZE=$(du -h "$DENSE_PATH/fused.ply" | cut -f1)
            log "Dense point cloud: $DENSE_PATH/fused.ply ($FUSED_SIZE)"
        fi
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

set_stage "COMPLETE"
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Reconstruction Complete${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Output:${NC}  $OUTPUT_DIR"
echo -e "  ${CYAN}Sparse:${NC}  $SPARSE_PATH/0/"
[ "$DENSE" = true ] && echo -e "  ${CYAN}Dense:${NC}   $DENSE_PATH/fused.ply"
echo ""
# Generate benchmark report
REPORT_SCRIPT="$SCRIPT_DIR/generate_report.py"
if [ -f "$REPORT_SCRIPT" ] && [ "$DRY_RUN" = false ]; then
    info "Generating benchmark report..."
    python3 "$REPORT_SCRIPT" "$OUTPUT_DIR" 2>&1 || warn "Report generation failed"
fi

echo -e "  ${DIM}Next steps:${NC}"
echo -e "  ${DIM}  • View report:  $OUTPUT_DIR/report.pdf${NC}"
echo -e "  ${DIM}  • Inspect in GUI:  colmap gui (File -> Import Model -> $SPARSE_PATH/0)${NC}"
echo -e "  ${DIM}  • Progressive training:  analyze_colmap.py -> build_tiers.py -> progressive_train.sh${NC}"
echo -e "  ${DIM}  • Direct training:  LichtFeld-Studio -d $OUTPUT_DIR ...${NC}"
echo ""
