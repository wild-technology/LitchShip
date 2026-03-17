# LitchShip — LichtFeld Studio WSL2 Build Project

## Environment
- **WSL2 Ubuntu 24.04** (also tested on 20.04)
- **GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
- **CUDA:** 12.8 at /usr/local/cuda
- **GCC:** 14.2 required (C++23 `<print>` header needs GCC 14+)
- **CMake:** 3.30+ required

## Strategy Recommendation: ADC

**Use ADC, not MCMC.** After extensive benchmarking:

- ADC has built-in scale pruning — removes bloom/haze automatically
- ADC produces higher visual quality with fewer artifacts
- ADC handles `--min-opacity` correctly (MCMC does not — see known bugs)

**Best training config:**
```bash
~/gaussian-splatting-cuda/build/LichtFeld-Studio \
    -d /path/to/dataset \
    -o ~/output/my_scene \
    --strategy adc \
    --max-cap 1000000 \
    -i 30000 \
    -r 2 \
    --min-opacity 0.1 \
    --enable-sparsity \
    --prune-ratio 0.6 \
    --enable-mip \
    --headless
```

## Key Files
| File | Purpose |
|------|---------|
| `lib/common.sh` | Shared functions (colors, logging, GPU, paths, binary discovery) |
| `clean_splat.py` | Post-processing: opacity filter + iterative statistical outlier removal |
| `setup_lichtfeld.sh` | Full environment + build setup (supports `--no-sudo`) |
| `run_test_training.sh` | Quick 30-image test run to verify the build |
| `train_full.sh` | Full resolution production training pipeline |
| `benchmark.sh` | Multi-config benchmark runner (ADC + MCMC configs) |
| `setup_claude_wsl.sh` | Bootstrap script for Claude Code in WSL |
| `overlay-ports/sdl3/` | vcpkg overlay port (disables XTEST/ibus) |
| `SETUP_LICHTFELD_WSL.md` | Detailed manual setup reference |

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

## Post-Processing

All training scripts automatically run `clean_splat.py` after training to remove
low-opacity floaters and spatial outliers. To run manually:

```bash
# Default: opacity filter (< 5%) + 3 passes of statistical outlier removal
python3 clean_splat.py input.ply output.ply

# More aggressive (tighter threshold, higher opacity cutoff)
python3 clean_splat.py input.ply output.ply -s 1.5 --opacity-min 0.1

# With bounding box clip and scale filter
python3 clean_splat.py input.ply output.ply --bbox -10,-10,-10,10,10,10 --scale-max 0.5
```

## Training Scripts

```bash
# Quick test (30 images, 3K iterations, ADC):
./run_test_training.sh /path/to/dataset

# Full production training (30K iterations, ADC):
./train_full.sh /path/to/dataset

# Benchmark multiple configs:
./benchmark.sh /path/to/dataset [/path/to/splat/output]
```

## Known Bugs

1. **MCMC + non-default `--min-opacity` = 17 PB crash** — MCMC's cap_max parser misinterprets `--min-opacity` values as memory sizes, requesting astronomically large allocations. **Do not use MCMC with `--min-opacity`**. ADC handles this correctly.
2. **LichtFeld CLI parser quirk** — some flag combinations cause silent misparse. Always test new flag combos with a small dataset first.
3. **GCC 14 required** — GCC 13 fails with `fatal error: print: No such file or directory`
4. **libgomp.so broken symlink** — when GCC 14 extracted from debs, `libgomp.so` is a dangling symlink. Script auto-fixes.
5. **Missing OpenGL/X11 in WSL2** — minimal installs lack dev headers. Script extracts from debs.
6. **pkg.m4 missing** — libb2 vcpkg build fails with `pkgconfigdir is undefined`. Script installs the macro.
7. **LibTorch URLs change** — script tries 4 different download channels
8. **Do NOT install nvidia drivers inside WSL2** — the Windows driver provides libcuda.so
9. **WSL2 has no display** — must use `--headless` flag for training
10. **sm_120** (Blackwell) — use `-DBUILD_CUDA_PTX_ONLY=ON` as fallback if native kernels crash
11. **Full-res OOM at high gaussian counts** — 4.6M gaussians at 3702x2091 OOMs on 32 GB VRAM. At full resolution (`-r 1`), keep `--max-cap` at ~3M or below. Use `-r 2` for large point clouds (>3M COLMAP points) to stay within VRAM.

## Benchmark Results (RTX 5090, 32 GB VRAM)

| Config | Strategy | Images | Res | Iters | Time | Quality Notes |
|--------|----------|--------|-----|-------|------|---------------|
| Quick test | ADC | 30 | half | 3K | ~20s | Good for validation |
| Medium | ADC | 6,821 | half | 7K | ~8 min | Production-ready |
| Full halfres | ADC | 6,821 | half | 30K | ~35 min | Best quality/time |
| Full highres | ADC | 6,821 | full | 30K | ~90 min | Maximum quality |

ADC consistently produces cleaner splats than MCMC — no bloom, no opacity artifacts, built-in scale pruning removes floaters automatically.
