# LitchShip

Automated build and training pipeline for [LichtFeld Studio](https://github.com/MrNeRF/gaussian-splatting-cuda) (C++23/CUDA gaussian splatting) on WSL2 and Linux.

Takes COLMAP datasets in, produces `.ply` gaussian splats out. Designed to run on dedicated GPU machines with minimal intervention, including restricted environments without `sudo`.

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

1. Detects CUDA toolkit and GPU compute capability (sm_75 through sm_120)
2. Installs build dependencies (GCC 14, CMake 3.30+, Ninja, vcpkg)
3. Builds autotools from source if system versions are too old
4. Installs all required dev libraries (OpenGL, X11, etc.)
5. Clones [gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda) with submodules
6. Applies vcpkg overlay ports (SDL3 fixes for minimal installs)
7. Downloads LibTorch (~2 GB, tries multiple CUDA channels)
8. Configures and builds with Ninja (parallel, ~10-30 min)
9. Verifies the binary runs

The script is idempotent -- re-running skips completed steps.

## Requirements

### Hardware
- NVIDIA GPU (tested on RTX 5090, should work on RTX 20-series and up)
- 16+ GB RAM recommended
- 20+ GB disk for build artifacts

### Software
- **WSL2** with Ubuntu 24.04 recommended (20.04+ supported), or native Linux
- **NVIDIA driver 570+** on Windows (WSL2) or Linux
- **CUDA Toolkit 12.8+** (auto-installed by setup if sudo available)

## Running With sudo

```bash
./setup_lichtfeld.sh
```

With sudo, the script uses `apt` to install all dependencies including GCC 14, CMake, OpenGL/X11 dev libs, and autotools. This is the simplest path.

## Running Without sudo

```bash
./setup_lichtfeld.sh --no-sudo
```

Without sudo, the script automatically:

- Installs **cmake** and **ninja** via pip
- Extracts **zip**, **unzip**, **pkg-config**, **nasm**, **yasm** from `.deb` packages
- Extracts **GCC 14** (compiler, cc1plus, C++23 headers) from `.deb` packages
- Extracts **OpenGL** dev libraries (Mesa, libglvnd) from `.deb` packages
- Extracts **X11** dev libraries and headers from `.deb` packages
- Builds **m4**, **autoconf 2.72**, **automake 1.17**, **libtool 2.5.4** from source
- Installs **autoconf-archive** and **pkg.m4** macros from `.deb` packages
- Fixes the **libgomp.so** symlink (broken in GCC 14 deb extraction)

Everything installs to `~/.local`. The only prerequisites are:

- `gcc` and `g++` (any version, just for bootstrapping autotools)
- `git`, `curl`, `wget`, `python3`, `pip3`
- CUDA toolkit pre-installed

If you're setting up a base image (Docker, CI), pre-install the full list:

```bash
apt install build-essential git curl wget zip unzip tar \
    pkg-config ninja-build python3 python3-dev python3-pip \
    gcc-14 g++-14 libssl-dev libx11-dev libxrandr-dev libxi-dev \
    libxtst-dev libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev \
    libxinerama-dev libwayland-dev libxkbcommon-dev libegl1-mesa-dev \
    libxext-dev libxau-dev libxdmcp-dev libxcb1-dev x11proto-dev \
    autoconf automake autoconf-archive libtool nasm yasm
```

## Data Format

The trainer expects COLMAP text format:

```
dataset/
+-- images/
|   +-- 00000.jpg  (or .jpeg, .png)
|   +-- 00001.jpg
|   +-- ...
+-- sparse/
    +-- 0/
        +-- cameras.txt
        +-- images.txt
        +-- points3D.txt
```

Image filenames in `images.txt` must match actual files in `images/`. If images are at the root level (common with RealityScan exports), move them into an `images/` subdirectory.

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
- `--min-opacity 0.05` -- prunes near-transparent floater splats
- `--enable-mip` -- mip filtering reduces aliasing artifacts
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

## Known Issues

1. **GCC 14 required:** C++23 `<print>` header is only available in GCC 14+. GCC 13 (Ubuntu 24.04 default) will not work. The script installs GCC 14 automatically (via PPA with sudo, or from debs without sudo).
2. **libgomp.so broken symlink:** When GCC 14 is extracted from debs without sudo, the `libgomp.so` symlink points to a non-existent path. The script detects and fixes this automatically.
3. **Missing OpenGL/X11 in WSL2:** Minimal WSL2 installs lack GL and X11 dev headers. The script installs them from debs to `~/.local` when sudo is unavailable.
4. **pkg.m4 needed for libb2:** vcpkg's libb2 port runs `autoreconf` which needs `pkg.m4` for the `PKG_CHECK_MODULES` macro. Without it, the build fails with `pkgconfigdir is undefined`. The script installs this macro.
5. **Autoconf < 2.71 (Ubuntu 20.04):** vcpkg's Python3 port needs 2.71+. The script builds 2.72 from source.
6. **Missing libxtst-dev / libibus-1.0-dev:** SDL3 build fails without these. The included overlay port disables XTEST and ibus.
7. **WSL2 has no display:** Always use `--headless` for training.
8. **LibTorch URLs change:** The script tries 4 different download channels.
9. **RealityScan image layout:** RealityScan exports images at root level, not in `images/`. `run_test_training.sh` handles this.
10. **Ubuntu 20.04 (GLIBC 2.31):** Kitware apt repo for cmake won't work. The script falls back to `pip3 install cmake`.

## License

Scripts in this repo are provided as-is. LichtFeld Studio itself is licensed under [GPL-3.0](https://github.com/MrNeRF/gaussian-splatting-cuda/blob/main/LICENSE).
