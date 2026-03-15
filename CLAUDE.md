# LitchShip — LichtFeld Studio WSL2 Build Project

## Environment (auto-detected)
- **WSL2 Ubuntu** (user: otter, host: Honeybadger)
- **GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
- **CUDA:** 13.2 at /usr/local/cuda
- **GCC:** gcc (Ubuntu 14.3.0-12ubuntu1~20~ppa1) 14.3.0
- **CMake:** cmake version 4.2.3
- **libc:** 2.31 (affects which apt repos and pre-built binaries work)
- **Python:** Python 3.8.10

## Data
- **Source (Windows):** /mnt/c/Users/WildTech/Desktop/H2103d_test_colmap_workflow
- **Working copy:** ~/data/H2103d_test_colmap_workflow
- **Format:** RealityScan 2.1 export (undistorted images + COLMAP text format)
- **Size:** 200+ images

## Current Task

Build MrNeRF/gaussian-splatting-cuda (LichtFeld Studio) and run a test
training on a 30-image subset. This is a C++23/CUDA project, not the
Python gsplat implementation.

### What's done
- setup_lichtfeld.sh exists and handles: CUDA detection, GCC 14, CMake 3.30+,
  vcpkg, repo clone, LibTorch download, cmake build
- run_test_training.sh: creates 30-image subset, filters COLMAP files, runs
  3K iteration MCMC training
- CUDA 13.2, GCC 14, CMake, vcpkg all confirmed working on this system

### What's done (BUILD COMPLETE)
- LichtFeld Studio v0.4.2 built successfully for sm_120 (RTX 5090)
- Binary: ~/gaussian-splatting-cuda/build/LichtFeld-Studio
- Test training: 30 images, 3K iterations, 18.6s, 161.3 iter/s
- Output: ~/output/H2103d_test/splat_3000.ply (200K gaussians)

### Known Issues (discovered iteratively)
1. **libc is 2.31** — Kitware apt repo for cmake won't work, use `pip3 install cmake`
2. **CUDA path** is /usr/local/cuda (symlink), not /usr/local/cuda-12.8
3. **LibTorch URLs** change frequently — the setup script tries multiple channels
4. **sm_120** (Blackwell) CUDA kernels may need PTX fallback if native build crashes
   Use `-DBUILD_CUDA_PTX_ONLY=ON` as fallback
5. **Do NOT install nvidia drivers inside WSL2** — the Windows driver provides libcuda.so
6. **sudo not available** in Claude Code sessions — workarounds used:
   - autoconf 2.72 built locally at ~/.local/bin (needed for vcpkg python3)
   - autoconf-archive m4 macros at ~/.local/share/aclocal (needed for libb2)
   - SDL3 overlay port at overlay-ports/sdl3/ (disables XTEST, ibus)
   - `libxtst-dev` and `libibus-1.0-dev` not installed, worked around
7. **WSL2 has no display** — must use `--headless` flag for training
8. **Data structure**: RealityScan exports images at root level (not in images/ subdir)
   — run_test_training.sh handles this by copying into images/

## How to Proceed

Run the existing scripts in order:

```bash
# 1. Build everything (idempotent — skips completed steps)
./setup_lichtfeld.sh

# 2. Test training (30 images, 3K iterations)
./run_test_training.sh
```

If setup_lichtfeld.sh fails, debug the specific step that failed.
The script is in ~/LitchShip/setup_lichtfeld.sh.

## Key Files
- `setup_lichtfeld.sh` — full environment + build setup
- `run_test_training.sh` — test training with 30-image subset
- `setup_claude_wsl.sh` — this bootstrap script
- `SETUP_LICHTFELD_WSL.md` — reference documentation

## Build Commands (LichtFeld Studio)

```bash
cd ~/gaussian-splatting-cuda
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_PREFIX_PATH=$HOME/gaussian-splatting-cuda/external/libtorch
cmake --build . --config Release -j$(nproc)
```

## Training Command

```bash
cd ~/gaussian-splatting-cuda
# Find binary (name may vary after build)
find build/ -type f -executable | head -10

# Test run (adjust binary path as needed)
./build/bin/lichtfeld-studio \
    -d ~/data/H2103d_test_colmap_workflow \
    -o ~/output/H2103d_test \
    --strategy mcmc \
    --max-cap 200000 \
    -i 3000 -r 2
```
