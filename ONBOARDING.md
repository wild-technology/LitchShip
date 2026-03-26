# LitchShip COLMAP Pipeline — Team Onboarding & Handoff Guide

## Project Overview

This project reconstructs 3D sparse point clouds from underwater ROV imagery of shipwrecks using COLMAP, then trains Gaussian Splat models for visualization. The pipeline handles 20,000–50,000+ images per survey from a multi-camera ROV rig with GPS/orientation navigation data.

**Current state**: Pipeline is code-complete, tested on 100-image subset (100% registration, 0.001px reprojection error). Full 48K run on H2103d Northampton dataset was blocked by RAM — the global mapper requires ~200GB and the previous machine had 190GB physical. The COLMAP database with all features extracted and matches computed is preserved and can be reused on a machine with sufficient RAM.

**Target machine requirements**: Ubuntu 24.04, NVIDIA GPU with CUDA, 256GB+ RAM, 1TB+ NVMe for workspace.

---

## Team Structure & Responsibilities

### Team 1: Orientation & Feature/Bug Review

**Goal**: Understand the codebase, document known issues, verify assumptions.

#### Step 1: Read these files in order
1. `CLAUDE.md` — project conventions, build system, training strategy, known bugs
2. `ONBOARDING.md` — this file
3. `tools/colmap_reconstruct.sh` — the main pipeline script (~670 lines)
4. `tools/import_gps_to_colmap.py` — navigation/pose prior import
5. `tools/assign_cameras.py` — multi-camera rig assignment
6. `tools/train_vocab_tree.sh` — vocabulary tree trainer
7. `tools/colmap_monitor.py` — real-time web dashboard
8. `tools/generate_report.py` — PDF benchmark report generator
9. `tools/camera_configs/h2103d_northampton.conf` — camera definitions
10. `lib/common.sh` — shared shell utilities

#### Step 2: Understand the pipeline flow
```
Raw images + Nav file + Camera config
    |
    v
Feature Extraction (GPU SIFT) -----> database.db (keypoints, descriptors)
    |
    v
Camera Assignment (per prefix) -----> database.db (cameras, rigs, frames)
    |
    v
Nav/GPS Import (position + gravity) -> database.db (pose_priors)
    |
    v
Feature Matching (spatial/sequential) -> database.db (matches, two_view_geometries)
    |
    v
Sparse Reconstruction (global or incremental mapper) -> sparse/0/ (cameras, images, points3D)
    |
    v
[Optional] Dense Reconstruction (PatchMatch) -> dense/fused.ply
    |
    v
PDF Report -> report.pdf
    |
    v
Gaussian Splat Training (LichtFeld Studio) -> .ply splat model
```

#### Step 3: Known bugs to verify are fixed
| Bug | File | Status | Description |
|-----|------|--------|-------------|
| Display-only | `colmap_reconstruct.sh` | Fixed | Was showing "Peak threshold: 0.004" but not passing it — now shows "First octave: 0" |
| Global mapper output path | `colmap_reconstruct.sh` | Fixed | Global mapper writes to `sparse/` not `sparse/0/` — script now handles both |
| Dashboard SQLite timeout | `colmap_monitor.py` | Fixed | Increased to 30s timeout + query_only mode for locked DB during mapper |
| Docstring | `assign_cameras.py` | Fixed | Examples updated from SIMPLE_PINHOLE to OPENCV/OPENCV_FISHEYE |
| HERC scatter grouping | `colmap_monitor.py` | Fixed | HERC images now detected by substring match, not prefix split |

#### Step 4: Known limitations
- **Global mapper OOM at ~190GB** for 45K images — needs 256GB+ RAM
- **Incremental mapper is very slow** — global BA every 500 frames takes hours at scale
- **Ceres lacks CUDA** — BA runs on CPU only (see Team 3 for fix)
- **Dashboard goes dark during long BA** — SQLite locked by COLMAP
- **COLMAP 4.1.0.dev0 is a dev build from main** — not a stable release

---

### Team 2: Settings Research & Documentation Verification

**Goal**: Verify every COLMAP parameter matches documentation and intended purpose. Cross-reference with COLMAP 4.x docs at https://colmap.github.io/.

#### Feature Extraction Parameters
| Parameter | Current Value | Default | Verify Against |
|-----------|--------------|---------|----------------|
| `--FeatureExtraction.type` | SIFT | SIFT | Also available: ALIKED (learned features, potentially better for underwater) |
| `--FeatureExtraction.use_gpu` | 1 | 1 | Must be 1 for GPU SIFT |
| `--FeatureExtraction.max_image_size` | 4096 | -1 (unlimited) | Images are 3840x2160, so 4096 means no downscale. Verify -1 doesn't cause VRAM OOM |
| `--SiftExtraction.max_num_features` | 16384 | 8192 | Doubled for underwater. Verify this doesn't cause excessive DB size or matching time |
| `--SiftExtraction.first_octave` | 0 | -1 | 0 = native resolution. -1 = 2x upsample (4x compute, sub-pixel features are noise for underwater). **Must be 0** |
| `--SiftExtraction.edge_threshold` | 16 | 10 | Raised for soft underwater edges. Verify this doesn't admit too many false features |

#### Matching Parameters
| Parameter | Current Value | Default | Verify Against |
|-----------|--------------|---------|----------------|
| `--SiftMatching.max_ratio` | 0.85 | 0.8 | Relaxed for underwater low-contrast features. Check if 0.8 produces better verification rate |
| `--SpatialMatching.max_distance` | 50 | 100 | 50m radius in UTM meters. GPS accuracy is +/-10m. Verify 50m captures all relevant pairs |
| `--SpatialMatching.max_num_neighbors` | 100 | 50 | Doubled. More pairs = better connectivity but slower matching |

#### Mapper Parameters
| Parameter | Current Value | Default | Verify Against |
|-----------|--------------|---------|----------------|
| `--Mapper.min_num_matches` | 30 | 15 | Raised for underwater false-positive resilience. May reject valid pairs — test with 15 |
| `--Mapper.init_min_num_inliers` | 200 | 100 | Higher initialization threshold. If mapper fails to initialize, lower this |
| `--Mapper.ba_global_max_num_iterations` | 50 | 50 | Default. Consider reducing to 20-30 for speed |
| `--Mapper.ba_global_frames_freq` | 500 (default) | 500 | Global BA every 500 frames. For 45K images, consider 2000+ |
| `--Mapper.filter_max_reproj_error` | 4.0 | 4.0 | Default. Appropriate for underwater |
| `--GlobalMapper.gp_optimize_points` | 1 (default) | 1 | Optimizes 3D points during global positioning — major RAM consumer. Try 0 if OOM |

#### Camera Models — CRITICAL
| Camera | Config Model | Verify |
|--------|-------------|--------|
| camlower (Canon 16-35mm f/2.8L III @ 16mm) | OPENCV | 8 params: fx, fy, cx, cy, k1, k2, p1, p2. Correct for rectilinear wide-angle with barrel distortion |
| cammid (Canon 8-12mm f/4 @ 12mm fisheye) | OPENCV_FISHEYE | 8 params: fx, fy, cx, cy, k1, k2, k3, k4. Correct for full-frame fisheye |
| camupper (Canon 8-12mm f/4 @ 12mm fisheye) | OPENCV_FISHEYE | Same as cammid |
| HERC (video ~24mm) | OPENCV | Verify this is correct for the HERC camera — may need OPENCV_FISHEYE if it has fisheye distortion |

**NEVER use SIMPLE_PINHOLE or SIMPLE_RADIAL** — all source images are distorted.

#### Gravity Vector Convention
The import script computes gravity (down direction) in COLMAP camera frame from pitch/roll:
```
COLMAP camera: X-right, Y-down, Z-forward
pitch = degrees below horizontal (0=level, 90=straight down)
gravity_x = -sin(roll) * cos(pitch)
gravity_y = cos(roll) * cos(pitch)
gravity_z = sin(pitch)
```
**Verify this matches the ROV navigation convention.** The test data shows pitch ~89 degrees (near-vertical downward), producing gravity ~(0, 0.02, 1.0) which is correct (mostly along Z-forward = world down when camera looks down).

#### Coordinate System
- Nav data is UTM Zone 57 South
- COLMAP 4.x coordinate system: CARTESIAN (1) for UTM
- **Verify the import script sets coordinate_system=1 (CARTESIAN)**, not WGS84 (0)

#### ALIKED + LightGlue (experimental alternative)
COLMAP 4.1 supports learned features. Test whether `--features ALIKED` produces better results for underwater:
```bash
./colmap_reconstruct.sh <images> <output> --features ALIKED --mode spatial ...
```
ALIKED auto-downloads ONNX models. LightGlue matcher is used automatically with ALIKED features.

---

### Team 3: Machine Setup (Fresh Ubuntu 24.04)

**Goal**: Set up a new machine from scratch to run the pipeline.

#### Hardware Requirements
- **GPU**: NVIDIA with CUDA compute capability 7.0+ (tested on RTX 5090, sm_120)
- **RAM**: 256GB+ (global mapper needs ~200GB for 45K images)
- **Storage**: 1TB+ NVMe for workspace (feature DB is ~80GB for 45K images at 16K features)
- **CPU**: As many threads as possible (Ceres BA is CPU-bound)

#### Step 1: System packages
```bash
sudo apt-get update && sudo apt-get install -y \
    git ninja-build cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libglew-dev libcgal-dev libceres-dev \
    qtbase5-dev libqt5opengl5-dev \
    python3 python3-pip sqlite3
pip install fpdf2
```

#### Step 2: CUDA toolkit
Install CUDA 12.8+ from https://developer.nvidia.com/cuda-downloads. Verify:
```bash
nvcc --version
nvidia-smi
```

#### Step 3: Build COLMAP from source (CRITICAL — with CUDA and GPU Ceres)

**IMPORTANT**: The apt COLMAP package has NO CUDA support. You MUST build from source.

**ALSO IMPORTANT**: Build Ceres with CUDA support FIRST to enable GPU bundle adjustment. The previous machine used system Ceres without CUDA, causing the "Falling back to CPU-based dense solvers" warning that makes BA ~10x slower.

```bash
# Step 3a: Build Ceres with CUDA
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver && mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF
cmake --build . --config Release -j$(nproc)
sudo cmake --install .
cd ../..

# Step 3b: Build COLMAP from main branch
git clone https://github.com/colmap/colmap.git colmap-src
cd colmap-src
git checkout main
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    -DGUI_ENABLED=ON \
    -DTESTS_ENABLED=OFF \
    -DCUDA_ENABLED=ON \
    -DFETCH_FAISS=ON \
    -DONNX_ENABLED=ON \
    -DFETCH_ONNX=ON
cmake --build . --config Release -j$(nproc)
cmake --install .
```

**Verify the build**:
```bash
~/.local/bin/colmap help 2>&1 | head -2
# Should show: "COLMAP 4.x.x (Commit ... with CUDA)"

# Verify GPU BA support:
~/.local/bin/colmap mapper -h 2>&1 | grep ba_use_gpu
# Should show: --Mapper.ba_use_gpu arg (=0)
# This means GPU BA is available (set to 1 to enable)
```

If `ba_use_gpu` option exists, Ceres was compiled with CUDA. If the global mapper still shows "Falling back to CPU-based dense solvers", Ceres was NOT built with CUDA — rebuild it.

#### Step 4: Clone the project
```bash
git clone git@github.com:wild-technology/LitchShip.git
cd LitchShip
git checkout claude/setup-lichtfeld-wsl-SPLoG
```

#### Step 5: Transfer the COLMAP database (optional — saves ~6 hours)
If you copy `database.db` from the previous run, you can skip feature extraction and matching:
```
colmap_48k/
  database.db          (~80GB — has features + matches for 45K images)
  logs/                (COLMAP log files)
```
Place this on the NVMe workspace. The image directory must also be accessible at the same path, or re-symlinked.

#### Step 6: WSL2-specific notes (if applicable)
- Set `.wslconfig` memory to at least 240GB: `memory=240GB`
- NVMe drives must be attached with `wsl --mount \\.\PhysicalDriveN --bare` then formatted ext4 inside WSL for native I/O speed
- Windows NTFS mounts via 9p are ~10x slower than native ext4

---

### Team 4: Testing

**Goal**: Validate the pipeline produces correct results before the full run.

#### Test 1: 100-image smoke test
```bash
# Create a 100-image subset (60 camlower + 20 cammid + 20 camupper)
# from images that appear in the flight log
mkdir -p /tmp/test_100/images
NAV=/path/to/NA173_H2103d_UTM57S.txt
IMGDIR=/path/to/vocab_tree_training_50k
for cam in camlower cammid camupper; do
    COUNT=$( [ "$cam" = "camlower" ] && echo 60 || echo 20 )
    grep "^${cam}" "$NAV" | cut -d';' -f1 | while read name; do
        [ -f "$IMGDIR/$name" ] && echo "$name"
    done | head -$COUNT | while read name; do
        cp "$IMGDIR/$name" /tmp/test_100/images/
    done
done

# Run full pipeline
./tools/colmap_reconstruct.sh \
    /tmp/test_100/images \
    /tmp/test_100/output \
    --mode spatial \
    --mapper global \
    --gps-file /path/to/NA173_H2103d_UTM57S.txt \
    --camera-config tools/camera_configs/h2103d_northampton.conf \
    --batch
```

#### Test 1 — Expected results
| Metric | Expected |
|--------|----------|
| Images registered | 100 (100%) |
| Camera models | 3 (OPENCV + 2x OPENCV_FISHEYE) |
| 3D points | 40,000–50,000 |
| Reprojection error (mean) | < 1.0 px |
| Feature extraction | < 30 seconds (GPU) |
| Total time | < 5 minutes |

#### Test 2: Verify GPU SIFT (not CPU)
Check the log for "SIFT GPU feature extractor" — NOT "Covariant SIFT CPU feature extractor". CPU fallback happens if `domain_size_pooling` is enabled.

#### Test 3: Verify camera models
```bash
sqlite3 output/database.db "SELECT camera_id, model FROM cameras;"
# model 4 = OPENCV, model 5 = OPENCV_FISHEYE
# Should NOT see model 0 (SIMPLE_PINHOLE) or model 2 (SIMPLE_RADIAL)
```

#### Test 4: Verify pose priors with gravity
```bash
sqlite3 output/database.db "SELECT COUNT(*) FROM pose_priors WHERE gravity IS NOT NULL;"
# Should equal the number of matched images (100 for test)
```

#### Test 5: Verify cross-camera matching
```python
# Run in Python against the database
import sqlite3
conn = sqlite3.connect("output/database.db")
pairs = conn.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0 ORDER BY RANDOM() LIMIT 50").fetchall()
# Decode pair_ids and check that different camera types match each other
# (camlower <-> cammid, camlower <-> HERC, etc.)
```

#### Test 6: Dashboard launches and shows data
Open http://localhost:8080 during the pipeline run. Verify:
- Stage indicator updates
- Feature count populates
- Position scatter shows all 4 camera types in different colors
- Log tail auto-scrolls

#### Test 7: PDF report generates
Check `output/report.pdf` after completion. Verify timing, camera models, and recommendations sections are populated.

---

### Team 5: Post-Test Code Review & Debugging

**Goal**: Review all code for correctness, defensive coding, edge cases.

#### Checklist
- [ ] All COLMAP parameter names are valid for 4.1.0 (no 3.x leftovers)
  ```bash
  grep -rn "SiftExtraction.use_gpu\|SiftExtraction.gpu_index\|SiftExtraction.max_image_size\|SiftExtraction.num_threads\|SiftMatching.num_threads\|SiftMatching.use_gpu\|SpatialMatching.is_gps" tools/
  # Should return ZERO results
  ```
- [ ] No SIMPLE_PINHOLE anywhere except comments/docs
  ```bash
  grep -rn "SIMPLE_PINHOLE" tools/ | grep -v "^.*:#\|^.*:.*#\|docstring"
  ```
- [ ] SQLite error handling in Python scripts — all DB operations wrapped in try/except
- [ ] Shell scripts use `set -euo pipefail` — verify errors propagate correctly
- [ ] `run_colmap()` captures exit codes — verify COLMAP failures stop the pipeline
- [ ] Symlink handling — `count_images()` uses `-type f -o -type l`
- [ ] Image name matching — `import_gps_to_colmap.py` matches by basename fallback (handles COLMAP storing relative paths)
- [ ] Camera assignment — `assign_cameras.py` rebuilds rigs/frames/frame_data tables correctly for COLMAP 4.x
- [ ] `prompt_choice()` outputs menu to stderr, value to stdout (no capture pollution)
- [ ] Dashboard handles DB lock gracefully (30s timeout, silent retry)
- [ ] Report generator handles missing data (no crash if stages were skipped)

#### Edge cases to test
- What happens if no images match the nav file? (should error clearly)
- What happens if `--camera-config` has a prefix that matches zero images? (should warn)
- What happens if COLMAP crashes mid-pipeline? (dashboard should show last known state)
- What happens with zero GPS data and `--mode spatial`? (should error before matching)
- What happens with duplicate image filenames? (COLMAP UNIQUE constraint — should fail at extraction)

---

### Team 6: Efficiency Review & Monitoring

**Goal**: Establish what to monitor, how to observe trends, and identify optimization opportunities.

#### Key metrics to track during runs
| Metric | Source | Healthy Range | Action if outside |
|--------|--------|--------------|-------------------|
| **Registration rate** | `grep -c "num_reg_frames=" mapper.log` | 5-50 img/min | <5 = check if stuck in BA |
| **Memory (RSS)** | `ps -p PID -o rss` | <80% of total | >90% = OOM risk, consider reducing features |
| **GPU utilization** | `nvidia-smi` | >50% during extraction/matching | <10% = COLMAP using CPU fallback |
| **Verification rate** | `verified_pairs / total_matches` | >50% | <30% = features too permissive, lower max_ratio |
| **Points per image** | `image sees X / Y points` in log | >50% of Y | <30% = poor overlap or wrong camera model |
| **Reprojection error** | report.pdf or points3D.txt | <2.0 px mean | >4.0 px = camera model issue |
| **DB size** | `ls -lh database.db` | ~2 GB/1000 images at 16K features | Growing faster = check max_features |

#### Monitoring tools
1. **Dashboard**: http://localhost:8080 — real-time web UI (auto-launches with pipeline)
2. **Trends CSV**: `/path/to/output/mapper_trends.csv` — hourly snapshots (use `/tmp/check_mapper.sh`)
3. **COLMAP logs**: `/path/to/output/logs/colmap.*.INFO.*` — full COLMAP output per stage
4. **PDF report**: auto-generated on pipeline completion

#### Setting up the hourly trend tracker
```bash
# Create the checker script
cat > /tmp/check_mapper.sh << 'SCRIPT'
#!/bin/bash
LOGFILE=/path/to/output/mapper_trends.csv
MAPPER_LOG=/path/to/output/incremental_mapper.log
PID=$(pgrep -f "colmap mapper")
# ... (see /tmp/check_mapper.sh on the previous machine for full script)
SCRIPT
chmod +x /tmp/check_mapper.sh

# Run hourly
while true; do sleep 3600; /tmp/check_mapper.sh; done &
```

#### Optimization opportunities to investigate

1. **GPU-accelerated Ceres BA** (`--Mapper.ba_use_gpu 1`)
   - Requires Ceres built with CUDA (see Team 3 setup)
   - Could speed up BA by 5-10x, making global mapper viable
   - Test on 100-image subset first

2. **ALIKED + LightGlue features** (`--features ALIKED`)
   - Learned features, more robust to illumination/contrast changes
   - Potentially fewer but higher-quality matches
   - Auto-downloads ONNX models, runs on GPU

3. **Reduce global BA frequency** (`--Mapper.ba_global_frames_freq 2000`)
   - Default 500 means BA every 500 frames — takes hours at 45K scale
   - 2000 = less frequent BA, more drift but much faster
   - Only relevant for incremental mapper

4. **Global mapper with `--GlobalMapper.gp_optimize_points 0`**
   - Skips 3D point optimization during global positioning
   - Significantly reduces memory — points refined in later BA pass
   - Could make global mapper fit in less RAM

5. **Feature count** (`--max-features 8192` vs 16384)
   - Halves DB size, matching time, and BA memory
   - COLMAP default is 8192 — 16384 may be overkill for 3840x2160 underwater
   - Test registration rate with both on 100-image subset

6. **Parallel feature extraction** for multi-GPU systems
   - COLMAP supports `--FeatureExtraction.gpu_index` for specific GPU
   - Could split images across GPUs with separate databases, then merge

---

## H2103d Northampton Dataset Reference

| Property | Value |
|----------|-------|
| **Survey** | NA173 H2103d, HMS Northampton shipwreck |
| **Location** | UTM Zone 57 South |
| **Depth** | ~650m |
| **Total images** | 45,166 |
| **Cameras** | 4 (camlower 18,053 / HERC 16,643 / camupper 5,258 / cammid 5,212) |
| **Resolution** | 3840x2160 (16:9) |
| **Nav file** | `NA173_H2103d_UTM57S.txt` (semicolon-delimited, 48,590 entries) |
| **Nav accuracy** | Position +/-10m, Orientation +/-5 degrees |
| **Pitch** | ~89 degrees (near-vertical, looking down) |
| **Nav format** | `filename;X (East);Y (North);Alt;X Accuracy;Y Accuracy;Alt Accuracy;Yaw;Pitch;Roll;...` |
| **Connected images** | 44,446 / 45,166 (720 isolated) |
| **Verified match pairs** | 2,070,906 |
| **Existing DB** | Features + matches computed, pose priors imported (transferable) |

## Quick Reference: Running the Full Pipeline

```bash
# With nav data (spatial matching + global mapper with gravity):
./tools/colmap_reconstruct.sh \
    /path/to/images \
    /path/to/output \
    --mode spatial \
    --mapper global \
    --gps-file /path/to/nav_file.txt \
    --camera-config tools/camera_configs/h2103d_northampton.conf

# Without nav data (sequential matching + incremental mapper):
./tools/colmap_reconstruct.sh \
    /path/to/images \
    /path/to/output \
    --mode sequential \
    --mapper incremental

# Train vocab tree from existing database:
./tools/train_vocab_tree.sh \
    /path/to/images \
    --database /path/to/output/database.db \
    --output ~/colmap-vocab/vocab_tree_underwater_256K.bin
```
