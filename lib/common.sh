#!/bin/bash
# =============================================================================
# LitchShip — Shared library for all training/benchmark scripts
#
# Usage: source "$SCRIPT_DIR/lib/common.sh"
#
# Provides:
#   Colors:    RED GREEN YELLOW CYAN BOLD DIM NC
#   Logging:   log, warn, error, info
#   GPU:       detect_gpu (sets GPU_NAME, GPU_MEM)
#   Paths:     setup_paths (sets PATH, LD_LIBRARY_PATH, CUDA_ROOT)
#   Binary:    find_binary (sets BINARY)
#   Data:      fix_extension_mismatch, copy_colmap_files
# =============================================================================

# ─── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Logging ─────────────────────────────────────────────────────────────────

log()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; }
info()  { echo -e "${CYAN}[..]${NC} $1"; }

# ─── GPU Detection ───────────────────────────────────────────────────────────
# Sets: GPU_NAME, GPU_MEM
# Exits on failure unless ALLOW_NO_GPU=true

detect_gpu() {
    if ! nvidia-smi &>/dev/null; then
        if [ "${ALLOW_NO_GPU:-false}" = true ]; then
            warn "GPU not available (ALLOW_NO_GPU=true, continuing)"
            GPU_NAME="none"
            GPU_MEM="0"
            return 0
        fi
        error "GPU not available (nvidia-smi failed)"
        exit 1
    fi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log "GPU: $GPU_NAME ($GPU_MEM)"
}

# ─── PATH / CUDA Setup ──────────────────────────────────────────────────────
# Sets: CUDA_ROOT, PATH, LD_LIBRARY_PATH

setup_paths() {
    local repo_dir="${1:-$HOME/gaussian-splatting-cuda}"
    CUDA_ROOT="/usr/local/cuda"

    # Deduplicate: only prepend paths not already present
    local -a new_path_dirs=("$HOME/.local/bin" "$CUDA_ROOT/bin")
    for d in "${new_path_dirs[@]}"; do
        case ":$PATH:" in
            *":$d:"*) ;;
            *) PATH="$d:$PATH" ;;
        esac
    done
    export PATH

    local -a new_ld_dirs=("$repo_dir/build" "$CUDA_ROOT/lib64" "$HOME/.local/lib")
    for d in "${new_ld_dirs[@]}"; do
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$d:"*) ;;
            *) LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}" ;;
        esac
    done
    export LD_LIBRARY_PATH
}

# ─── Binary Discovery ────────────────────────────────────────────────────────
# Sets: BINARY
# Searches common build output paths for the LichtFeld Studio binary.

find_binary() {
    local repo_dir="${1:-$HOME/gaussian-splatting-cuda}"
    BINARY=""
    for candidate in \
        "$repo_dir/build/LichtFeld-Studio" \
        "$repo_dir/build/bin/LichtFeld-Studio" \
        "$repo_dir/build/bin/lichtfeld-studio" \
        "$repo_dir/build/lichtfeld-studio" \
        "$repo_dir/build/bin/gaussian_splatting_cuda" \
        "$repo_dir/build/gaussian_splatting_cuda"; do
        if [ -x "$candidate" ]; then
            BINARY="$candidate"
            break
        fi
    done

    if [ -z "$BINARY" ]; then
        error "LichtFeld Studio binary not found in $repo_dir/build/"
        error "Run setup_lichtfeld.sh first."
        info "Available executables:"
        find "$repo_dir/build/" -type f -executable 2>/dev/null | head -10
        exit 1
    fi
    log "Binary: $BINARY"
}

# ─── Extension Mismatch Fix ─────────────────────────────────────────────────
# RealityScan exports often reference .png in images.txt but files are .jpeg.
# This checks the first non-comment entry and fixes all extensions if needed.
#
# Usage: fix_extension_mismatch /path/to/work_dir

fix_extension_mismatch() {
    local work_dir="$1"
    local images_txt="$work_dir/sparse/0/images.txt"

    [ -f "$images_txt" ] || return 0

    local sample_ref
    sample_ref=$(grep -v '^#' "$images_txt" | head -1 | awk '{print $NF}')
    [ -n "$sample_ref" ] || return 0

    # If the referenced file already exists, no mismatch
    [ -f "$work_dir/images/$sample_ref" ] && {
        log "Extension check passed: images.txt matches actual files"
        return 0
    }

    local sample_base="${sample_ref%.*}"
    local ref_ext="${sample_ref##*.}"
    local actual
    actual=$(find "$work_dir/images" -maxdepth 1 -name "${sample_base}.*" -type f | head -1)

    if [ -n "$actual" ]; then
        local actual_ext="${actual##*.}"
        warn "Extension mismatch: images.txt says .$ref_ext but files are .$actual_ext"
        warn "Fixing images.txt: replacing .$ref_ext -> .$actual_ext"
        sed -i "s/\\.${ref_ext}\$/\\.${actual_ext}/g" "$images_txt"
        log "Fixed extension mismatch in images.txt"
    else
        error "Cannot find image for: $sample_ref"
        exit 1
    fi
}

# ─── COLMAP File Helpers ─────────────────────────────────────────────────────
# Copy COLMAP text files from source to work_dir/sparse/0/, handling
# case-insensitive filenames (Cameras.txt vs cameras.txt).
#
# Usage: copy_colmap_files /path/to/source /path/to/work_dir

# ─── Training Invocation ────────────────────────────────────────────────────
# Build and run a LichtFeld Studio training command with standard ADC flags.
#
# Usage: run_lf_training <binary> <data_dir> <output_dir> <strategy> \
#            <max_cap> <iterations> <resize_factor> [extra_flags...]
#
# Returns: exit code from the training binary.
# Sets: LF_LAST_LOG to the log file path, LF_LAST_PLY to the output PLY.

run_lf_training() {
    local binary="$1" data_dir="$2" output_dir="$3" strategy="$4"
    local max_cap="$5" iterations="$6" resize_factor="$7"
    shift 7
    local extra_flags=("$@")

    mkdir -p "$output_dir"
    local log_file="$output_dir/training.log"
    LF_LAST_LOG="$log_file"

    local cmd=("$binary"
        -d "$data_dir"
        -o "$output_dir"
        --strategy "$strategy"
        --max-cap "$max_cap"
        -i "$iterations"
        -r "$resize_factor"
        --headless
    )

    # Standard ADC flags (only added for adc strategy)
    if [ "$strategy" = "adc" ]; then
        cmd+=(--min-opacity 0.1 --enable-sparsity --prune-ratio 0.6 --enable-mip)
    fi

    # Append any extra flags
    cmd+=("${extra_flags[@]}")

    "${cmd[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    # Find output PLY
    LF_LAST_PLY=$(find "$output_dir" -name '*.ply' -type f ! -name '*cleaned*' | sort | tail -1)

    return "$exit_code"
}

# ─── Post-Training Cleanup ─────────────────────────────────────────────────
# Run clean_splat.py on a PLY file.
#
# Usage: run_lf_clean <clean_script> <input_ply> [clean_flags...]
# Sets: LF_LAST_CLEANED_PLY

run_lf_clean() {
    local clean_script="$1" input_ply="$2"
    shift 2
    local clean_flags=("$@")

    local cleaned_ply="${input_ply%.ply}_cleaned.ply"
    LF_LAST_CLEANED_PLY="$cleaned_ply"

    python3 "$clean_script" "$input_ply" "$cleaned_ply" "${clean_flags[@]}"
}

# ─── COLMAP File Helpers ─────────────────────────────────────────────────────
# Copy COLMAP text files from source to work_dir/sparse/0/, handling
# case-insensitive filenames (Cameras.txt vs cameras.txt).
#
# Usage: copy_colmap_files /path/to/source /path/to/work_dir

copy_colmap_files() {
    local src="$1"
    local work_dir="$2"

    mkdir -p "$work_dir/sparse/0"

    # cameras.txt
    for src_name in cameras.txt Cameras.txt; do
        if [ -f "$src/$src_name" ]; then
            cp "$src/$src_name" "$work_dir/sparse/0/cameras.txt"
            log "Copied $src_name -> sparse/0/cameras.txt"
            break
        fi
    done

    # images.txt
    for src_name in images.txt Images.txt; do
        if [ -f "$src/$src_name" ]; then
            cp "$src/$src_name" "$work_dir/sparse/0/images.txt"
            log "Copied $src_name -> sparse/0/images.txt"
            break
        fi
    done

    # points3D.txt
    for src_name in points3D.txt Points3D.txt; do
        if [ -f "$src/$src_name" ]; then
            cp "$src/$src_name" "$work_dir/sparse/0/points3D.txt"
            log "Copied $src_name -> sparse/0/points3D.txt"
            break
        fi
    done
}
