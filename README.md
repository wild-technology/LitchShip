# LitchShip

Automated build and training pipeline for [LichtFeld Studio](https://github.com/MrNeRF/gaussian-splatting-cuda) (C++23/CUDA gaussian splatting) on WSL2.

Takes COLMAP datasets in, produces `.ply` gaussian splats out. Designed to run on dedicated GPU machines with minimal intervention.

## Quick Start

```bash
# Clone this repo
git clone https://github.com/wild-technology/LitchShip.git
cd LitchShip

# Build everything (20-40 min first time)
chmod +x setup_lichtfeld.sh
./setup_lichtfeld.sh

# Train on your COLMAP dataset
~/gaussian-splatting-cuda/build/LichtFeld-Studio \
    -d /path/to/your/dataset \
    -o ~/output/my_scene \
    --strategy mcmc \
    --max-cap 1000000 \
    -i 30000 \
    -r 2 \
    --min-opacity 0.05 \
    --enable-mip \
    --headless
```

## What It Does

`setup_lichtfeld.sh` handles the full build chain automatically:

1. Detects CUDA toolkit and GPU compute capability
2. Installs build dependencies (GCC 14, CMake 3.30+, Ninja, vcpkg)
3. Builds autoconf 2.72 from source if system version is too old
4. Installs autoconf-archive locally if unavailable via apt
5. Clones [gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda) with submodules
6. Applies vcpkg overlay ports (SDL3 fixes for minimal installs)
7. Downloads LibTorch (~2 GB, tries multiple CUDA channels)
8. Configures and builds with Ninja (parallel, ~10-30 min)
9. Verifies the binary runs

The script is idempotent — re-running skips completed steps.

## Requirements

### Hardware
- NVIDIA GPU (tested on RTX 5090, should work on RTX 20-series and up)
- 16+ GB RAM recommended
- 20+ GB disk for build artifacts

### Software
- **WSL2** with Ubuntu 20.04+ (or native Linux)
- **NVIDIA driver 570+** on Windows (WSL2) or Linux
- **CUDA Toolkit** (auto-installed by setup if sudo available)
- `sudo` access for apt packages, OR pre-installed dependencies (use `--no-sudo`)

### Running Without sudo

If you don't have sudo (e.g., CI, containers, restricted environments):

```bash
# Pre-install these packages on the base image:
apt install build-essential git curl wget zip unzip tar \
    pkg-config ninja-build python3 python3-dev python3-pip \
    gcc-14 g++-14 libssl-dev libx11-dev libxrandr-dev libxi-dev \
    libxtst-dev libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev \
    libxinerama-dev libwayland-dev libxkbcommon-dev libegl1-mesa-dev \
    autoconf automake autoconf-archive libtool nasm yasm

# Then run setup without sudo:
./setup_lichtfeld.sh --no-sudo
```

## Data Format

The trainer expects COLMAP text format:

```
dataset/
├── images/
│   ├── 00000.jpg  (or .jpeg, .png)
│   ├── 00001.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```

**Important:** Image filenames in `images.txt` must match actual files in `images/`. If your COLMAP references `.png` but files are `.jpeg`, fix one or the other.

If images are at the root level (common with RealityScan exports), move them into an `images/` subdirectory.

## Training Options

| Flag | Description | Recommended |
|------|-------------|-------------|
| `--strategy mcmc` | Optimization strategy | `mcmc` for most scenes |
| `--max-cap N` | Max gaussian count | `500000`-`1000000` |
| `-i N` | Iterations | `30000` for production |
| `-r N` | Resolution downscale | `2` (fast) or `1` (full) |
| `--min-opacity F` | Prune transparent splats | `0.05` (reduces haze) |
| `--enable-mip` | Anti-aliasing filter | Yes (reduces sparkle) |
| `--headless` | No GUI window | Required for WSL2/servers |
| `--bilateral-grid` | Bilateral filtering | Try if haze persists |

### Performance Reference (RTX 5090, 32 GB VRAM)

| Dataset | Images | Resolution | Gaussians | Time | Iter/s |
|---------|--------|-----------|-----------|------|--------|
| Small test | 30 | half | 200K | 19s | 161 |
| Medium | 6,821 | half | 500K | 8 min | 63 |
| Medium | 6,821 | full | 500K | 19 min | 26 |
| Large | 23,630 | half | 1M | 8 min | 60 |

## Reducing Specular Haze

Gaussian splatting can produce a halo/haze around objects, especially on reflective surfaces. To minimize it:

**At training time (in this pipeline):**
- `--min-opacity 0.05` — prunes near-transparent floater splats
- `--enable-mip` — mip filtering reduces aliasing artifacts
- Higher `--max-cap` with more iterations helps MCMC converge

**At render time (in Unreal Engine, Unity, etc.):**
- Use the 3DGS plugin's opacity threshold/cutoff slider
- This is non-destructive and gives real-time control per scene
- Generally the better place for final cleanup

## Files

| File | Purpose |
|------|---------|
| `setup_lichtfeld.sh` | Full build pipeline (run this first) |
| `run_test_training.sh` | Quick 30-image test run to verify the build |
| `overlay-ports/sdl3/` | vcpkg overlay port (SDL3 without XTEST/ibus) |
| `setup_claude_wsl.sh` | Bootstrap script for Claude Code in WSL |
| `SETUP_LICHTFELD_WSL.md` | Detailed manual setup reference |
| `CLAUDE.md` | Build notes and known issues |

## Known Issues

1. **GLIBC 2.31 (Ubuntu 20.04):** Kitware apt repo won't work for CMake. The script uses `pip install cmake` instead.
2. **Autoconf < 2.71:** vcpkg's Python3 port needs 2.71+. The script builds 2.72 from source automatically.
3. **Missing libxtst-dev / libibus-1.0-dev:** SDL3 build fails. The included overlay port disables these features.
4. **WSL2 has no display:** Always use `--headless` for training.
5. **LibTorch URLs change:** The script tries 4 different download channels.
6. **Images.txt extension mismatch:** RealityScan may export COLMAP referencing `.png` while actual files are `.jpeg`. Fix manually.

## License

Scripts in this repo are provided as-is. LichtFeld Studio itself is licensed under [GPL-3.0](https://github.com/MrNeRF/gaussian-splatting-cuda/blob/main/LICENSE).
