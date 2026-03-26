#!/bin/bash
# =============================================================================
# Progressive Training Benchmark — VRAM, Speed, Quality, Cleaning Comparison
#
# Runs all 3 progressive stages + single-pass baseline, monitors VRAM usage
# and timing, then runs clean_splat.py with multiple configs on each output.
# Produces a comprehensive results JSON + summary text.
#
# Usage:
#   ./benchmark_progressive.sh /path/to/staging
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_root="$SCRIPT_DIR"; while [ "$_root" != "/" ] && [ ! -f "$_root/lib/common.sh" ]; do _root="$(dirname "$_root")"; done
source "$_root/lib/common.sh"
# LITCHSHIP_ROOT is now set automatically by common.sh

STAGING_DIR="${1:?Usage: $0 <staging-dir>}"
STAGING_DIR="$(cd "$STAGING_DIR" && pwd)"
REPO_DIR="$HOME/gaussian-splatting-cuda"
STAGE_ITERS=10000
RESIZE_FACTOR=2
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_BASE="$HOME/output/progressive_bench_${TIMESTAMP}"

mkdir -p "$OUTPUT_BASE"

# ─── Setup ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  Progressive Benchmark${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""

setup_paths "$REPO_DIR"
detect_gpu
find_binary "$REPO_DIR"

RESULTS_JSON="$OUTPUT_BASE/benchmark_results.json"
RESULTS_TXT="$OUTPUT_BASE/benchmark_summary.txt"
VRAM_LOG="$OUTPUT_BASE/vram_monitor.csv"

# VRAM monitor function — polls nvidia-smi every 2s, writes CSV
start_vram_monitor() {
    local label="$1"
    echo "timestamp,label,used_mib,free_mib,total_mib" >> "$VRAM_LOG"
    while true; do
        local ts=$(date +%s)
        local vals=$(nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        echo "${ts},${label},${vals}" >> "$VRAM_LOG"
        sleep 2
    done
}

stop_vram_monitor() {
    if [ -n "${VRAM_PID:-}" ]; then
        kill "$VRAM_PID" 2>/dev/null || true
        wait "$VRAM_PID" 2>/dev/null || true
        unset VRAM_PID
    fi
}

# Get peak VRAM for a label from the CSV
get_peak_vram() {
    local label="$1"
    grep ",${label}," "$VRAM_LOG" | awk -F',' '{print $3}' | sort -n | tail -1
}

# Get baseline VRAM (before training starts)
BASELINE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
info "Baseline VRAM usage: ${BASELINE_VRAM} MiB"

# ─── Training Runs ─────────────────────────────────────────────────────────

declare -A STAGE_NAMES=( [1]="seed" [2]="expand" [3]="refine" )
declare -A STAGE_MAXCAP=( [1]=500000 [2]=1000000 [3]=2000000 )

# Arrays to collect results
declare -a RUN_LABELS=()
declare -a RUN_TIMES=()
declare -a RUN_PEAK_VRAM=()
declare -a RUN_PLY_FILES=()
declare -a RUN_PLY_SIZES=()
declare -a RUN_SPLAT_COUNTS=()

run_training() {
    local label="$1"
    local data_dir="$2"
    local output_dir="$3"
    local maxcap="$4"
    local iters="$5"

    echo ""
    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo -e "${BOLD}  Running: $label${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"

    local point_count=$(grep -vc '^#\|^$' "$data_dir/sparse/0/points3D.txt" 2>/dev/null || echo "?")
    echo "  Points: $point_count | Max cap: $maxcap | Iters: $iters"

    mkdir -p "$output_dir"
    local log_file="$output_dir/training.log"

    # Start VRAM monitor
    start_vram_monitor "$label" &
    VRAM_PID=$!

    local t_start=$(date +%s)

    "$BINARY" \
        -d "$data_dir" \
        -o "$output_dir" \
        --strategy adc \
        --max-cap "$maxcap" \
        -i "$iters" \
        -r "$RESIZE_FACTOR" \
        --min-opacity 0.1 \
        --enable-sparsity \
        --prune-ratio 0.6 \
        --enable-mip \
        --headless \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local t_end=$(date +%s)
    local elapsed=$((t_end - t_start))

    stop_vram_monitor

    local peak_vram=$(get_peak_vram "$label")
    local net_vram=$((peak_vram - BASELINE_VRAM))

    # Find output PLY
    local ply_file=$(find "$output_dir" -name '*.ply' -type f | sort | tail -1)
    local ply_size="0"
    local splat_count="0"
    if [ -n "$ply_file" ]; then
        ply_size=$(stat -c%s "$ply_file" 2>/dev/null || echo "0")
        # Count vertices in PLY header
        splat_count=$(head -20 "$ply_file" | grep "element vertex" | awk '{print $3}' || echo "0")
    fi

    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    log "$label: ${mins}m${secs}s | Peak VRAM: ${peak_vram} MiB (net +${net_vram}) | Splats: ${splat_count} | PLY: $(numfmt --to=iec $ply_size 2>/dev/null || echo "${ply_size}B")"

    RUN_LABELS+=("$label")
    RUN_TIMES+=("$elapsed")
    RUN_PEAK_VRAM+=("$peak_vram")
    RUN_PLY_FILES+=("$ply_file")
    RUN_PLY_SIZES+=("$ply_size")
    RUN_SPLAT_COUNTS+=("$splat_count")
}

# Stage 1-3
for stage in 1 2 3; do
    name="${STAGE_NAMES[$stage]}"
    run_training \
        "stage${stage}_${name}" \
        "$STAGING_DIR/tier${stage}" \
        "$OUTPUT_BASE/stage${stage}_${name}" \
        "${STAGE_MAXCAP[$stage]}" \
        "$STAGE_ITERS"
done

# Baseline: single-pass 30K on tier3
TOTAL_ITERS=$((STAGE_ITERS * 3))
run_training \
    "baseline_singlepass" \
    "$STAGING_DIR/tier3" \
    "$OUTPUT_BASE/baseline_singlepass" \
    "2000000" \
    "$TOTAL_ITERS"

# ─── Cleaning Evaluation ──────────────────────────────────────────────────

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  Cleaning Evaluation${NC}"
echo -e "${BOLD}============================================${NC}"

CLEAN_SCRIPT="$LITCHSHIP_ROOT/training/clean_splat.py"

# Cleaning configs to test
declare -A CLEAN_CONFIGS
CLEAN_CONFIGS["default"]="--opacity-min 0.05 -s 2.0 -p 3"
CLEAN_CONFIGS["gentle"]="--opacity-min 0.03 -s 2.5 -p 2"
CLEAN_CONFIGS["aggressive"]="--opacity-min 0.1 -s 1.5 -p 3 --scale-max 0.5"
CLEAN_CONFIGS["connected"]="--opacity-min 0.05 -s 2.0 -p 3 --connected"

# For each training output, run all cleaning configs
CLEAN_RESULTS_FILE="$OUTPUT_BASE/cleaning_results.csv"
echo "source,clean_config,input_splats,output_splats,removed_pct,time_sec,output_file" > "$CLEAN_RESULTS_FILE"

for idx in "${!RUN_LABELS[@]}"; do
    label="${RUN_LABELS[$idx]}"
    ply="${RUN_PLY_FILES[$idx]}"

    if [ -z "$ply" ] || [ ! -f "$ply" ]; then
        echo "  Skipping $label — no PLY output"
        continue
    fi

    echo ""
    echo -e "${BOLD}  Cleaning: $label${NC}"

    for config_name in default gentle aggressive connected; do
        config_flags="${CLEAN_CONFIGS[$config_name]}"
        output_ply="$OUTPUT_BASE/${label}_clean_${config_name}.ply"

        echo "    Config: $config_name ($config_flags)"
        t_start=$(date +%s)

        clean_output=$(python3 "$CLEAN_SCRIPT" "$ply" "$output_ply" $config_flags 2>&1)
        t_end=$(date +%s)
        clean_time=$((t_end - t_start))

        # Parse splat counts from clean output
        input_count=$(echo "$clean_output" | grep -oP 'Loaded \K[\d,]+' | tr -d ',')
        final_count=$(echo "$clean_output" | grep -oP 'Remaining: \K[\d,]+' | tr -d ',')
        removed_pct=$(echo "$clean_output" | grep -oP 'Removed:.*\(\K[\d.]+')

        echo "      ${input_count:-?} → ${final_count:-?} splats (${removed_pct:-?}% removed) in ${clean_time}s"
        echo "${label},${config_name},${input_count:-0},${final_count:-0},${removed_pct:-0},${clean_time},${output_ply}" >> "$CLEAN_RESULTS_FILE"
    done
done

# ─── Extract PSNR from logs ───────────────────────────────────────────────

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  Quality Metrics (PSNR from logs)${NC}"
echo -e "${BOLD}============================================${NC}"

declare -a RUN_PSNR=()
for idx in "${!RUN_LABELS[@]}"; do
    label="${RUN_LABELS[$idx]}"
    log_file="$OUTPUT_BASE/${label}/training.log"
    psnr="N/A"
    if [ -f "$log_file" ]; then
        psnr=$(grep -oP 'PSNR[:\s]+\K[\d.]+' "$log_file" | tail -1 || echo "N/A")
    fi
    RUN_PSNR+=("$psnr")
    echo "  $label: PSNR $psnr"
done

# ─── Build JSON results ───────────────────────────────────────────────────

cat > "$RESULTS_JSON" << HEADER
{
  "timestamp": "$TIMESTAMP",
  "dataset": "$STAGING_DIR",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader)",
  "baseline_vram_mib": $BASELINE_VRAM,
  "resize_factor": $RESIZE_FACTOR,
  "runs": [
HEADER

for idx in "${!RUN_LABELS[@]}"; do
    comma=""
    [ "$idx" -gt 0 ] && comma=","
    cat >> "$RESULTS_JSON" << ENTRY
    ${comma}{
      "label": "${RUN_LABELS[$idx]}",
      "time_sec": ${RUN_TIMES[$idx]},
      "peak_vram_mib": ${RUN_PEAK_VRAM[$idx]:-0},
      "net_vram_mib": $((${RUN_PEAK_VRAM[$idx]:-0} - BASELINE_VRAM)),
      "splat_count": ${RUN_SPLAT_COUNTS[$idx]:-0},
      "ply_size_bytes": ${RUN_PLY_SIZES[$idx]:-0},
      "psnr": "${RUN_PSNR[$idx]:-N/A}"
    }
ENTRY
done

echo "  ]" >> "$RESULTS_JSON"
echo "}" >> "$RESULTS_JSON"

# ─── Summary ──────────────────────────────────────────────────────────────

TOTAL_PROGRESSIVE=$((${RUN_TIMES[0]} + ${RUN_TIMES[1]} + ${RUN_TIMES[2]}))

DATASET_NAME="$(basename "$STAGING_DIR")"
DATASET_IMAGES=$(find "$STAGING_DIR/tier3/images" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
DATASET_POINTS=$(grep -vc '^#\|^$' "$STAGING_DIR/tier3/sparse/0/points3D.txt" 2>/dev/null || echo "?")

cat > "$RESULTS_TXT" << SUMMARY
================================================================================
  ${DATASET_NAME} Progressive Training Benchmark Results
  $(date)
  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
  Dataset: ${DATASET_IMAGES} images | ${DATASET_POINTS} COLMAP points | resize=$RESIZE_FACTOR
================================================================================

TRAINING PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
  Run                    Time      Peak VRAM    Net VRAM    Splats       PSNR
  ─────────────────────  ────────  ──────────   ─────────   ──────────   ─────
SUMMARY

for idx in "${!RUN_LABELS[@]}"; do
    label="${RUN_LABELS[$idx]}"
    mins=$((${RUN_TIMES[$idx]} / 60))
    secs=$((${RUN_TIMES[$idx]} % 60))
    peak="${RUN_PEAK_VRAM[$idx]:-0}"
    net=$((peak - BASELINE_VRAM))
    splats="${RUN_SPLAT_COUNTS[$idx]:-0}"
    psnr="${RUN_PSNR[$idx]:-N/A}"
    printf "  %-23s %3dm%02ds   %6d MiB   %+5d MiB   %10s   %s\n" \
        "$label" "$mins" "$secs" "$peak" "$net" "$splats" "$psnr" >> "$RESULTS_TXT"
done

cat >> "$RESULTS_TXT" << FOOTER

  Progressive total: $((TOTAL_PROGRESSIVE / 60))m$((TOTAL_PROGRESSIVE % 60))s
  Baseline total:    $((${RUN_TIMES[3]} / 60))m$((${RUN_TIMES[3]} % 60))s

CLEANING EVALUATION
────────────────────────────────────────────────────────────────────────────────
FOOTER

# Append cleaning results as formatted table
if [ -f "$CLEAN_RESULTS_FILE" ]; then
    printf "  %-23s %-14s %10s %10s %8s %5s\n" "Source" "Config" "Input" "Output" "Removed" "Time" >> "$RESULTS_TXT"
    printf "  %-23s %-14s %10s %10s %8s %5s\n" "───────────────────────" "──────────────" "──────────" "──────────" "────────" "─────" >> "$RESULTS_TXT"
    tail -n+2 "$CLEAN_RESULTS_FILE" | while IFS=',' read -r src cfg inp out pct tm _; do
        printf "  %-23s %-14s %10s %10s %7s%% %4ss\n" "$src" "$cfg" "$inp" "$out" "$pct" "$tm" >> "$RESULTS_TXT"
    done
fi

cat >> "$RESULTS_TXT" << EOF2

OUTPUT FILES
────────────────────────────────────────────────────────────────────────────────
  Results JSON:     $RESULTS_JSON
  Cleaning CSV:     $CLEAN_RESULTS_FILE
  VRAM monitor:     $VRAM_LOG
  Training outputs: $OUTPUT_BASE/stage*/  $OUTPUT_BASE/baseline_singlepass/
  Cleaned PLYs:     $OUTPUT_BASE/*_clean_*.ply
================================================================================
EOF2

cat "$RESULTS_TXT"
echo ""
info "Full results at: $OUTPUT_BASE"
