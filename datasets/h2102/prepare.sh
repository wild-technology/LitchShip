#!/bin/bash
# prepare_h2102.sh — Copy H2102 shipwreck sections from Windows FS to WSL native,
#                     restructure COLMAP layout, run analysis for progressive training.
set -euo pipefail
trap '' PIPE  # Ignore SIGPIPE (benign from head/wc in pipes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_root="$SCRIPT_DIR"; while [ "$_root" != "/" ] && [ ! -f "$_root/lib/common.sh" ]; do _root="$(dirname "$_root")"; done
source "$_root/lib/common.sh"
# LITCHSHIP_ROOT is now set automatically by common.sh

# ── Configuration ─────────────────────────────────────────────────────────────
WIN_BASE="/mnt/c/Users/Public/Documents/h2102"
DATA_BASE="$HOME/data"

declare -A SECTIONS=(
    [amid]="amid_colmap"
    [bow]="bow_colmap"
    [stern]="Stern_colmap"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

copy_section() {
    local name="$1"
    local src_folder="$2"
    local src_dir="$WIN_BASE/$src_folder"
    local work_dir="$DATA_BASE/h2102_${name}"

    if [ -f "$work_dir/.data_ready" ]; then
        log "[$name] Data already prepared at $work_dir"
        return 0
    fi

    info "[$name] Preparing $work_dir from $src_dir"

    if [ ! -d "$src_dir" ]; then
        error "[$name] Source not found: $src_dir"
        return 1
    fi

    mkdir -p "$work_dir/images"

    # Copy COLMAP sparse files (handles capitalization)
    copy_colmap_files "$src_dir" "$work_dir"

    # Copy images with progress (source images are at folder root)
    local TOTAL
    TOTAL=$(find "$src_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
    info "[$name] Copying $TOTAL images..."

    local COUNT=0
    while IFS= read -r img; do
        cp "$img" "$work_dir/images/"
        COUNT=$((COUNT + 1))
        if [ $((COUNT % 500)) -eq 0 ]; then
            echo -ne "\r  Copied $COUNT / $TOTAL images..."
        fi
    done < <(find "$src_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \))
    echo -ne "\r"
    log "[$name] Copied $COUNT / $TOTAL images"

    # Fix any extension mismatches in images.txt
    fix_extension_mismatch "$work_dir"

    touch "$work_dir/.data_ready"
    log "[$name] Data preparation complete"
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══ H2102 Shipwreck — Data Preparation ═══${NC}"
echo ""

# Verify source exists
if [ ! -d "$WIN_BASE" ]; then
    error "Windows dataset not found at $WIN_BASE"
    error "Expected: C:\\Users\\Public\\Documents\\h2102"
    exit 1
fi

mkdir -p "$DATA_BASE"

# Copy all three sections
for name in amid bow stern; do
    copy_section "$name" "${SECTIONS[$name]}"
done

# Verify image counts
echo ""
info "Verifying image counts..."
for name in amid bow stern; do
    local_count=$(find "$DATA_BASE/h2102_${name}/images" -type f | wc -l)
    echo "  $name: $local_count images"
done

# Run analysis + tier building on amid section for progressive training
AMID_DIR="$DATA_BASE/h2102_amid"
TOOLS_DIR="$LITCHSHIP_ROOT/training/progressive"

if [ -f "$AMID_DIR/analysis.json" ] && [ -d "$AMID_DIR/staging" ]; then
    log "Analysis and tiers already exist for amid section"
else
    echo ""
    info "Running COLMAP analysis on amid section..."
    python3 "$TOOLS_DIR/analyze_colmap.py" "$AMID_DIR" --output "$AMID_DIR/analysis.json"
    log "Analysis complete: $AMID_DIR/analysis.json"

    info "Building progressive tiers for amid section..."
    python3 "$TOOLS_DIR/build_tiers.py" "$AMID_DIR" \
        --analysis "$AMID_DIR/analysis.json" \
        --output "$AMID_DIR/staging"
    log "Tiers built: $AMID_DIR/staging/"

    # Show tier summary
    if [ -f "$AMID_DIR/staging/tier_manifest.json" ]; then
        echo ""
        info "Tier manifest:"
        python3 -c "
import json, sys
m = json.load(open('$AMID_DIR/staging/tier_manifest.json'))
tc = m['tier_counts']
print(f\"  Tier 1: {tc['tier1']:,} points\")
print(f\"  Tier 2: {tc['tier2']:,} points (cumulative)\")
print(f\"  Tier 3: {tc['tier3']:,} points (cumulative)\")
print(f\"  Dropped: {tc['dropped']:,} points\")
print(f\"  Total:   {tc['total']:,} points\")
"
    fi
fi

echo ""
log "All H2102 sections prepared successfully"
echo ""
echo -e "  ${CYAN}Next:${NC} ./train_h2102.sh"
echo ""
