# LitchShip — LichtFeld Studio WSL2 Build Project

## Environment
- **WSL2 Ubuntu 24.04** (also tested on 20.04)
- **GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
- **CUDA:** 12.8 at /usr/local/cuda
- **GCC:** 14.2 required (C++23 `<print>` header needs GCC 14+)
- **CMake:** 3.30+ required

## Key Files
- `setup_lichtfeld.sh` — full environment + build setup (supports `--no-sudo`)
- `run_test_training.sh` — test training with 30-image subset
- `setup_claude_wsl.sh` — bootstrap script for Claude Code in WSL
- `overlay-ports/sdl3/` — vcpkg overlay port (disables XTEST/ibus)
- `SETUP_LICHTFELD_WSL.md` — detailed manual setup reference

## How to Build

```bash
# With sudo:
./setup_lichtfeld.sh

# Without sudo (CI, containers, Claude Code sessions):
./setup_lichtfeld.sh --no-sudo
```

The script is idempotent — re-running skips completed steps.

## Build Commands (Manual)

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
~/gaussian-splatting-cuda/build/LichtFeld-Studio \
    -d ~/data/H2103d_test_colmap_workflow \
    -o ~/output/H2103d_test \
    --strategy mcmc \
    --max-cap 200000 \
    -i 3000 -r 2 \
    --headless
```

## Known Issues

1. **GCC 14 required** — GCC 13 fails with `fatal error: print: No such file or directory`
2. **libgomp.so broken symlink** — when GCC 14 extracted from debs, `libgomp.so` is a dangling symlink. Script auto-fixes.
3. **Missing OpenGL/X11 in WSL2** — minimal installs lack dev headers. Script extracts from debs.
4. **pkg.m4 missing** — libb2 vcpkg build fails with `pkgconfigdir is undefined`. Script installs the macro.
5. **LibTorch URLs change** — script tries 4 different download channels
6. **Do NOT install nvidia drivers inside WSL2** — the Windows driver provides libcuda.so
7. **WSL2 has no display** — must use `--headless` flag for training
8. **sm_120** (Blackwell) — use `-DBUILD_CUDA_PTX_ONLY=ON` as fallback if native kernels crash
