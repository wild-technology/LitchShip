#!/bin/bash
# train_h2102.sh — Train H2102 shipwreck sections with maritime-tuned post-processing.
#                   Bow: single-pass ADC. Amid: progressive + baseline comparison.
set -euo pipefail
trap '' PIPE  # Ignore SIGPIPE from pipes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_root="$SCRIPT_DIR"; while [ "$_root" != "/" ] && [ ! -f "$_root/lib/common.sh" ]; do _root="$(dirname "$_root")"; done
source "$_root/lib/common.sh"
# LITCHSHIP_ROOT is now set automatically by common.sh

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_BASE="$HOME/data"
OUTPUT_BASE="$HOME/output"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
CLEAN_SCRIPT="$LITCHSHIP_ROOT/training/clean_splat.py"
PROGRESSIVE_SCRIPT="$LITCHSHIP_ROOT/training/progressive/progressive_train.sh"
VIEWER_SPLATS="$LITCHSHIP_ROOT/viewer/splats"

# Maritime-specific clean flags
MARITIME_CLEAN_FLAGS=(
    --opacity-min 0.1
    --remove-hue 180,240 --sat-min 0.3
    -s 1.5 -p 3
    --connected --keep-components 5
)

# ── Setup ─────────────────────────────────────────────────────────────────────

setup_paths
detect_gpu
find_binary

echo ""
echo -e "${BOLD}═══ H2102 Shipwreck — Training Pipeline ═══${NC}"
echo -e "  GPU: $GPU_NAME ($GPU_MEM)"
echo -e "  Binary: $BINARY"
echo ""

# Verify data is prepared
for section in bow amid; do
    if [ ! -f "$DATA_BASE/h2102_${section}/.data_ready" ]; then
        error "Data not prepared for $section. Run ./prepare_h2102.sh first."
        exit 1
    fi
done

if [ ! -d "$DATA_BASE/h2102_amid/staging" ]; then
    error "Progressive tiers not built for amid. Run ./prepare_h2102.sh first."
    exit 1
fi

# ── Maritime cleaning helper ──────────────────────────────────────────────────

maritime_clean() {
    local input_ply="$1"
    local output_ply="$2"
    info "Maritime cleaning: $(basename "$input_ply")"
    python3 "$CLEAN_SCRIPT" "$input_ply" "$output_ply" "${MARITIME_CLEAN_FLAGS[@]}"
}

# ── Step 1: Bow single-pass ──────────────────────────────────────────────────

BOW_DATA="$DATA_BASE/h2102_bow"
BOW_OUTPUT="$OUTPUT_BASE/h2102_bow_adc_${TIMESTAMP}"

echo -e "${BOLD}── Step 1: Bow single-pass (ADC, 30K, half-res) ──${NC}"
info "Output: $BOW_OUTPUT"

run_lf_training "$BINARY" "$BOW_DATA" "$BOW_OUTPUT" adc 2000000 30000 2

BOW_RAW="$LF_LAST_PLY"
BOW_CLEANED="${BOW_RAW%.ply}_maritime.ply"
maritime_clean "$BOW_RAW" "$BOW_CLEANED"
log "Bow training + cleaning complete"

# ── Step 2: Amid progressive ─────────────────────────────────────────────────

AMID_STAGING="$DATA_BASE/h2102_amid/staging"

echo ""
echo -e "${BOLD}── Step 2: Amid progressive (fast preset + baseline) ──${NC}"

"$PROGRESSIVE_SCRIPT" "$AMID_STAGING" \
    --preset fast \
    --no-clean \
    --compare

# Find progressive output directory (most recent)
AMID_PROG_DIR=$(find "$OUTPUT_BASE" -maxdepth 1 -name "progressive_*" -type d | sort -r | head -1)

if [ -z "$AMID_PROG_DIR" ]; then
    error "Progressive training output not found"
    exit 1
fi

log "Progressive output: $AMID_PROG_DIR"

# Maritime clean all progressive outputs
echo ""
info "Applying maritime cleaning to progressive outputs..."

declare -A AMID_CLEANED_PLYS=()

for stage_dir in "$AMID_PROG_DIR"/stage*/ "$AMID_PROG_DIR"/baseline*/; do
    [ -d "$stage_dir" ] || continue
    stage_name=$(basename "$stage_dir")

    # Find the PLY (newest, excluding *_cleaned*)
    raw_ply=$(find "$stage_dir" -name "*.ply" ! -name "*_cleaned*" ! -name "*_maritime*" -type f | sort -t/ -k999 | tail -1)
    if [ -z "$raw_ply" ]; then
        warn "No PLY found in $stage_name"
        continue
    fi

    cleaned="${raw_ply%.ply}_maritime.ply"
    maritime_clean "$raw_ply" "$cleaned"
    AMID_CLEANED_PLYS[$stage_name]="$cleaned"
done

log "Amid progressive training + cleaning complete"

# ── Results comparison ────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══ Results Summary ═══${NC}"
echo ""

printf "%-25s %12s %12s\n" "Output" "Raw splats" "Cleaned"
printf "%-25s %12s %12s\n" "-------------------------" "------------" "------------"

print_ply_stats() {
    local label="$1"
    local raw="$2"
    local cleaned="$3"
    local raw_count="-"
    local clean_count="-"

    if [ -f "$raw" ]; then
        raw_count=$(python3 -c "
from pathlib import Path
import struct
p = Path('$raw')
with open(p, 'rb') as f:
    header = b''
    while True:
        line = f.readline()
        header += line
        if b'element vertex' in line:
            print(line.decode().split()[-1])
            break
" 2>/dev/null || echo "?")
    fi

    if [ -f "$cleaned" ]; then
        clean_count=$(python3 -c "
from pathlib import Path
with open('$cleaned', 'rb') as f:
    for line in f:
        if b'element vertex' in line:
            print(line.decode().split()[-1])
            break
" 2>/dev/null || echo "?")
    fi

    printf "%-25s %12s %12s\n" "$label" "$raw_count" "$clean_count"
}

print_ply_stats "Bow single-pass" "$BOW_RAW" "$BOW_CLEANED"

for stage_name in "${!AMID_CLEANED_PLYS[@]}"; do
    raw="${AMID_CLEANED_PLYS[$stage_name]}"
    raw="${raw%_maritime.ply}.ply"
    print_ply_stats "Amid $stage_name" "$raw" "${AMID_CLEANED_PLYS[$stage_name]}"
done

# ── Copy to viewer ───────────────────────────────────────────────────────────

echo ""
info "Copying cleaned PLYs to viewer..."
mkdir -p "$VIEWER_SPLATS"

if [ -f "$BOW_CLEANED" ]; then
    cp "$BOW_CLEANED" "$VIEWER_SPLATS/bow_cleaned.ply"
    log "  bow_cleaned.ply → viewer/splats/"
fi

# Use stage3_refine as the amid viewer PLY (best progressive output)
AMID_FINAL="${AMID_CLEANED_PLYS[stage3_refine]:-}"
if [ -n "$AMID_FINAL" ] && [ -f "$AMID_FINAL" ]; then
    cp "$AMID_FINAL" "$VIEWER_SPLATS/amid_cleaned.ply"
    log "  amid_cleaned.ply → viewer/splats/"
fi

echo ""
log "Training pipeline complete"
echo ""
echo -e "  ${CYAN}Train stern:${NC}   (copy the bow training block, adjust paths)"
echo ""
echo -e "  ${DIM}Stern command (after reviewing quality):${NC}"
echo -e "  ${DIM}run_lf_training \$BINARY ~/data/h2102_stern ~/output/h2102_stern_adc adc 2000000 30000 2${NC}"
echo ""

# Auto-launch viewer
info "Launching viewer..."
exec bash "$LITCHSHIP_ROOT/viewer/serve.sh"
