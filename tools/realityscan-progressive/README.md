# RealityScan Progressive Training Pipeline

Quality-aware growing-seed approach for Gaussian splatting. Uses COLMAP reprojection error and track-length data (optionally enhanced with RealityScan quality reports) to tier 3D seed points by confidence, then trains progressively from the cleanest points outward.

## Quick Start

```bash
# 1. Analyze your COLMAP export
python3 analyze_colmap.py /path/to/my_scan_export

# 2. Build tiered datasets
python3 build_tiers.py /path/to/my_scan_export --output /path/to/staging

# 3. Run progressive training
./progressive_train.sh /path/to/staging

# Or dry-run first:
./progressive_train.sh /path/to/staging --dry-run
```

## What to Export from RealityScan

### Required: COLMAP Text Format + Undistorted Images

In RealityScan 2.1:

1. **Set distortion model** to `Brown3 with Tangential2` — ensures exported cameras use PINHOLE-equivalent intrinsics (`fx, fy, cx, cy`) and images are fully undistorted

2. **Export Registration** as COLMAP Text Format:
   - Outputs `cameras.txt`, `images.txt`, `points3D.txt`

3. **Export Undistorted Images** separately:
   - **Fit:** Inner Region
   - **Resolution:** Fit
   - **Image Format:** JPEG
   - **Naming Convention:** Original file name
   - Place in an `images/` subdirectory alongside the sparse files

### Optional: Quality Report JSON

For enhanced tiering, export a per-image quality report using the provided template.
This extracts `numPoints`, `imageCoverage`, and `reprojError` stats per camera.

**Automated install (on Windows where RealityScan is installed):**

```powershell
# Auto-detects RealityScan installation
python install_template.py

# Or specify the path explicitly
python install_template.py --rs-path "D:\Epic Games\RealityScan"

# Preview what it would do
python install_template.py --dry-run
```

**Manual install:**

1. Copy `templates/quality_report.json.tpl` to `C:\Program Files\Epic Games\RealityScan\Reports\`
2. Add this entry to `Report.xml` (inside the `<formats>` element):
   ```xml
   <format id="{8F3A1B2C-4D5E-6F78-9A0B-C1D2E3F4A5B6}" mask="*.json"
           desc="Quality Report (JSON) for LitchShip"
           writer="RealityScan.Export.ReportWriter">
     <hint>Per-image quality metrics for progressive Gaussian splatting training</hint>
     <body>$Include("Reports\quality_report.json.tpl")</body>
   </format>
   ```

**Export the report:**

```
GUI:  ALIGNMENT tab -> Export -> Report -> "Quality Report (JSON) for LitchShip"
CLI:  RealityScan.exe -load project.rsproj -exportReport "C:\output\quality_report.json" "{8F3A1B2C-4D5E-6F78-9A0B-C1D2E3F4A5B6}"
```

**Post-process (fixes JSON formatting from template engine quirks):**

```bash
python fix_report_json.py quality_report.json --in-place
```

Note: `analyze_colmap.py` auto-cleans the JSON on load, so this step is optional but useful for inspection.

Place the resulting `quality_report.json` in your dataset root — `analyze_colmap.py` auto-detects it.

### Expected Directory Layout

```
my_scan_export/
├── images/
│   ├── IMG_0001.jpg
│   └── ...
├── sparse/0/          # or files at root level
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
└── quality_report.json  # optional
```

## How It Works

### Stage 1: Analyze (`analyze_colmap.py`)

Parses COLMAP text files and computes per-point quality metrics:
- **Reprojection error** from `points3D.txt` ERROR field (lower = better)
- **Track length** — number of images observing each point (higher = more confident)
- **Per-image derived scores** from cross-referencing images.txt and points3D.txt

If a RealityScan `quality_report.json` is present, merges in per-image `imageCoverage` and `reprojError.median` for enhanced scoring.

### Stage 2: Build Tiers (`build_tiers.py`)

Creates three COLMAP dataset directories with **growing seed point clouds**:

| Tier | Points Included | Default Thresholds |
|------|----------------|-------------------|
| T1 (seed) | Highest confidence only | track ≥ 5, error < median |
| T2 (expand) | Medium+ confidence | track ≥ 3, error < 2× median |
| T3 (full) | All except outliers | error < 3× median |

All images stay in every tier (symlinked). Only `points3D.txt` differs. Observation references in `images.txt` to removed points are set to `-1` (COLMAP convention).

### Stage 3: Progressive Training (`progressive_train.sh`)

Runs 3 fresh ADC training passes, each seeding from a richer point cloud:

```
Stage 1 (seed):    10K iter, 500K max gaussians, tier1 points
Stage 2 (expand):  10K iter, 1M max gaussians,   tier2 points
Stage 3 (refine):  10K iter, 2M max gaussians,   tier3 points
```

Use `--compare` to also run a single-pass 30K baseline for A/B quality comparison.

## CLI Reference

### install_template.py

```
python install_template.py [--rs-path DIR] [--dry-run]

  --rs-path DIR  RealityScan installation directory (auto-detected if omitted)
  --dry-run      Show what would be done without making changes
```

### fix_report_json.py

```
python fix_report_json.py <input> [--output FILE] [--in-place]

  input          Raw quality_report.json from RealityScan
  --output FILE  Write cleaned JSON to FILE (default: stdout)
  --in-place     Overwrite input file with cleaned JSON
```

### analyze_colmap.py

```
python3 analyze_colmap.py <dataset> [--rs-report FILE] [--output FILE]

  dataset      Path to COLMAP dataset directory
  --rs-report  Path to quality_report.json (auto-detected if in dataset root)
  --output     Output path (default: <dataset>/analysis.json)
```

### build_tiers.py

```
python3 build_tiers.py <dataset> [options]

  dataset              Path to COLMAP dataset directory
  --output DIR         Staging output directory (default: <dataset>/staging)
  --analysis FILE      Path to analysis.json (default: <dataset>/analysis.json)
  --t1-min-track N     Tier 1 min track length (default: 5)
  --t1-max-error-pct N Tier 1 max error as % of median (default: 100)
  --t2-min-track N     Tier 2 min track length (default: 3)
  --t2-max-error-pct N Tier 2 max error as % of median (default: 200)
  --t3-max-error-pct N Tier 3 max error as % of median (default: 300)
```

### progressive_train.sh

```
./progressive_train.sh <staging-dir> [options]

  staging-dir    Directory with tier1/, tier2/, tier3/ from build_tiers.py
  --dry-run      Print commands without executing
  --compare      Also run single-pass baseline
  --iters N      Iterations per stage (default: 10000)
  --resize N     Resize factor (default: 2)
  --repo DIR     Path to gaussian-splatting-cuda repo
```

## Tuning Tips

- **Fewer tier1 points = cleaner scaffold** but may miss thin structures. If tier1 has < 5% of total points, lower `--t1-min-track` to 3.
- **Higher `--t3-max-error-pct`** keeps more points but may introduce floaters. ADC's built-in pruning handles this well at `--prune-ratio 0.6`.
- For very large scenes (>100K points), consider `--iters 15000` per stage to give ADC more time to converge.
- Always use `--compare` on your first run to validate that progressive actually improves over single-pass.
