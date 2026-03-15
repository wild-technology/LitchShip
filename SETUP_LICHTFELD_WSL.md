# LichtFeld Studio (gaussian-splatting-cuda) — WSL2 Build & Run Guide

> **System:** RTX 5090 (Blackwell, sm_120) · Threadripper · 196 GB RAM · WSL2 Ubuntu 24.04
> **Data location:** `C:\Users\WildTech\Desktop\H2103d_Northampton\images`

---

## Table of Contents

1. [Prerequisites (Windows Side)](#1-prerequisites-windows-side)
2. [WSL2 Configuration](#2-wsl2-configuration)
3. [CUDA Toolkit Installation (Inside WSL2)](#3-cuda-toolkit-installation-inside-wsl2)
4. [Verify GPU & CUDA](#4-verify-gpu--cuda)
5. [Install Build Dependencies](#5-install-build-dependencies)
6. [Clone & Build LichtFeld Studio](#6-clone--build-lichtfeld-studio)
7. [Prepare Your Data](#7-prepare-your-data)
8. [Run Training](#8-run-training)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites (Windows Side)

### NVIDIA Driver

Your RTX 5090 requires driver **570+** on Windows. Install from
[nvidia.com/drivers](https://www.nvidia.com/drivers).

> **Avoid driver 572.16** — it has been linked to black screens and detection
> failures. Use the latest Game Ready or Studio driver instead.

**Do NOT install any NVIDIA Linux GPU driver inside WSL2.**
The Windows driver automatically provides `libcuda.so` to WSL2.

### WSL2

Ensure you have WSL2 (not WSL1). From PowerShell (Admin):

```powershell
wsl --update
wsl --set-default-version 2
```

---

## 2. WSL2 Configuration

With 196 GB system RAM and a large GPU workload, tune WSL2 memory limits.

Create/edit `C:\Users\WildTech\.wslconfig`:

```ini
[wsl2]
memory=128GB
swap=32GB
processors=32
```

Then restart WSL from PowerShell:

```powershell
wsl --shutdown
```

---

## 3. CUDA Toolkit Installation (Inside WSL2)

Open your WSL2 terminal (`otter@Honeybadger`).

> **Critical:** Install **only** `cuda-toolkit`, never the `cuda` or
> `cuda-drivers` meta-packages — those try to install Linux GPU drivers and
> will break WSL2's driver passthrough.

```bash
# 1. Add NVIDIA WSL-Ubuntu repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2. Install CUDA Toolkit (NOT cuda or cuda-drivers)
sudo apt-get install -y cuda-toolkit

# 3. Add CUDA to your PATH — append to ~/.bashrc
echo '' >> ~/.bashrc
echo '# CUDA Toolkit' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

This installs the latest CUDA toolkit (12.8+ or 13.x) which supports your
RTX 5090's Blackwell architecture (sm_120 / compute capability 12.0).

---

## 4. Verify GPU & CUDA

Run these commands **in order** to confirm everything works:

```bash
# Check GPU visibility (uses Windows driver stub)
nvidia-smi
# → Should show "NVIDIA GeForce RTX 5090", driver version, CUDA version

# Check CUDA compiler version
nvcc --version
# → Should show CUDA 12.8+ or 13.x

# Definitive test — compile and run deviceQuery
cd /usr/local/cuda/extras/demo_suite/
./deviceQuery
# → Should show "Result = PASS" and list RTX 5090 properties

# Optional — test memory bandwidth
./bandwidthTest
```

> **Note:** `nvidia-smi` shows the *driver's maximum supported* CUDA version.
> `nvcc --version` shows your *installed toolkit* version. They may differ —
> this is normal.

---

## 5. Install Build Dependencies

LichtFeld Studio requires **C++23** (needs GCC 14+), **CMake 3.30+**, vcpkg,
and LibTorch. Ubuntu 24.04 ships GCC 13 and CMake 3.28, so both need upgrading.

```bash
# Essential build tools
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    zip \
    unzip \
    tar \
    pkg-config \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    software-properties-common

# Libraries needed by vcpkg dependencies
sudo apt-get install -y \
    libssl-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxcursor-dev \
    libxinerama-dev \
    libwayland-dev \
    libxkbcommon-dev \
    autoconf \
    automake \
    libtool \
    nasm \
    yasm
```

### Install GCC 14 (Required — C++23)

Ubuntu 24.04 ships GCC 13, but LichtFeld Studio's C++23 codebase requires
**GCC 14 or newer**. Install it via the Ubuntu toolchain PPA:

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-14 g++-14

# Set GCC 14 as the default compiler
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14

# Verify
gcc --version
# → Should show gcc 14.x
```

### Install CMake 3.30+ (Required)

Ubuntu 24.04 ships CMake 3.28, but LichtFeld Studio requires **3.30+**.
Install the latest via Kitware's APT repository:

```bash
# Remove old cmake if installed
sudo apt-get remove -y cmake 2>/dev/null || true

# Add Kitware APT repository
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
    | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt-get update
sudo apt-get install -y cmake

# Verify
cmake --version
# → Should show 3.30+
```

### Install vcpkg

```bash
cd ~
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh

# Set environment variable (add to ~/.bashrc)
echo '' >> ~/.bashrc
echo '# vcpkg' >> ~/.bashrc
echo 'export VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Clone & Build LichtFeld Studio

### Clone the Repository

```bash
cd ~
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda.git
cd gaussian-splatting-cuda
```

> **Note:** Use `--recursive` to also clone submodules (gsplat rasterization
> backend, etc.).

### Download LibTorch

LichtFeld Studio requires LibTorch (the C++ distribution of PyTorch) built
for CUDA 12.8. **Do NOT install LibTorch via vcpkg** — there is a
[known vcpkg bug](https://github.com/microsoft/vcpkg/issues/36844) that causes
"Found two conflicting CUDA installs" errors.

```bash
mkdir -p external && cd external

# Download LibTorch for CUDA 12.8 (Linux, cxx11 ABI)
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.cpu.zip -O libtorch.zip
unzip libtorch.zip
rm libtorch.zip

cd ..
```

> **Note:** Check https://pytorch.org/get-started/locally/ for the latest
> LibTorch download link matching your CUDA version. Select:
> PyTorch Build = Stable, OS = Linux, Package = LibTorch, Language = C++,
> Compute Platform = CUDA 12.8.

### Configure & Build

```bash
# Create build directory
cmake --preset linux-release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DVCPKG_ROOT=$HOME/vcpkg

# If the preset doesn't exist, use manual configuration:
mkdir -p build && cd build
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_PREFIX_PATH=$HOME/gaussian-splatting-cuda/external/libtorch

# Build (use -j to parallelize across your Threadripper cores)
cmake --build . --config Release -j$(nproc)
```

**Key flags explained:**

| Flag | Purpose |
|------|---------|
| `CMAKE_CUDA_ARCHITECTURES=120` | Targets RTX 5090 Blackwell (sm_120) |
| `CUDAToolkit_ROOT` | Points to your CUDA installation |
| `CMAKE_TOOLCHAIN_FILE` | Lets vcpkg manage C++ dependencies |
| `CMAKE_PREFIX_PATH` | Points to LibTorch |
| `-j$(nproc)` | Uses all CPU cores (Threadripper advantage) |

**Additional CMake options:**

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA_PTX_ONLY` | OFF | PTX-only build (smaller binary, JIT at runtime — portable across GPU archs) |
| `BUILD_PORTABLE` | OFF | Self-contained distributable build (implies PTX-only) |
| `BUILD_CUDA_MIN_SM` | 75 | Minimum SM version for PTX builds |
| `BUILD_TESTS` | OFF | Build the test suite |
| `ENABLE_CUDA_GL_INTEROP` | ON | Enable CUDA/OpenGL interop (needed for viewer) |

> **Tip:** If you only want to train (no interactive viewer), you can build
> without OpenGL interop to avoid X11/display dependencies.

### vcpkg Dependencies (Installed Automatically)

The build will automatically fetch via vcpkg:

- args, boost-preprocessor, boost-regex, freetype, glad, sdl3, glm
- gtest, implot, libarchive, libwebp, nlohmann-json, spdlog, tbb
- nanobind, assimp
- ffmpeg (avcodec, avformat, swscale, nvcodec, x264)
- imgui (docking-experimental, freetype, opengl3, sdl3)
- openimageio (webp), python3 (extensions), rmlui (svg)

First build will take a while as vcpkg compiles these from source.

---

## 7. Prepare Your Data

### Copy Data from Windows to WSL2 (Strongly Recommended)

File I/O across `/mnt/c/` is **very slow**. Copy your data to the WSL2
native filesystem:

```bash
# Create a working directory
mkdir -p ~/data/H2103d_Northampton

# Copy your images and COLMAP text files
cp -r "/mnt/c/Users/WildTech/Desktop/H2103d_Northampton/images" \
      ~/data/H2103d_Northampton/images
```

### Expected COLMAP Text Data Structure

LichtFeld Studio expects a standard COLMAP reconstruction layout. Since you
have COLMAP text files, organize them like this:

```
~/data/H2103d_Northampton/
├── images/                    # Your photographs
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt        # Camera intrinsics
        ├── images.txt         # Camera extrinsics (poses)
        └── points3D.txt       # 3D point cloud
```

If your COLMAP text files are loose in the data directory, move them:

```bash
mkdir -p ~/data/H2103d_Northampton/sparse/0

# Move/copy your COLMAP text files into the sparse directory
# Adjust these paths based on where your .txt files actually are
cp /mnt/c/Users/WildTech/Desktop/H2103d_Northampton/cameras.txt \
   ~/data/H2103d_Northampton/sparse/0/
cp /mnt/c/Users/WildTech/Desktop/H2103d_Northampton/images.txt \
   ~/data/H2103d_Northampton/sparse/0/
cp /mnt/c/Users/WildTech/Desktop/H2103d_Northampton/points3D.txt \
   ~/data/H2103d_Northampton/sparse/0/
```

### COLMAP Text File Format Reference

**cameras.txt** — one line per camera:
```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 1920 1080 1500.0 1500.0 960.0 540.0
```

**images.txt** — two lines per image:
```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 0.9 0.1 0.05 0.02 1.0 2.0 3.0 1 IMG_0001.jpg
100.0 200.0 1 300.0 400.0 2 ...
```

**points3D.txt** — one line per point:
```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 1.5 2.3 4.1 128 200 50 0.5 1 0 2 1
```

---

## 8. Run Training

The binary is built as `lichtfeld-studio` (the project was rebranded from
gaussian-splatting-cuda). Training parameters can be passed via CLI flags
or via JSON config files in the `parameter/` directory.

```bash
cd ~/gaussian-splatting-cuda

# First, check available options
./build/bin/lichtfeld-studio --help

# Basic training run
./build/bin/lichtfeld-studio \
    -d ~/data/H2103d_Northampton \
    -o ~/output/H2103d_Northampton \
    --strategy mcmc \
    --max-cap 500000 \
    -i 30000
```

> **Note:** The exact binary path may vary — check `build/bin/` or just
> `build/` for the executable. Run `find build/ -type f -executable` if unsure.

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-d, --data-path PATH` | Path to COLMAP data directory | (required) |
| `-o, --output-path PATH` | Output directory for results | `./output` |
| `-i, --iter NUM` | Number of training iterations | 30000 |
| `-r, --resize_factor NUM` | Image downscale factor (1=full res) | 1 |
| `--strategy [mcmc\|default]` | Optimization strategy | mcmc |
| `--max-cap NUM` | Maximum number of Gaussians (MCMC) | 1000000 |

> Run `--help` for the full list — additional parameters may be available for
> the interactive viewer mode, bilateral grid settings, etc.

### JSON Config Files

The `parameter/` directory contains JSON configuration files with default
training parameters. You can copy and modify these for your scene:

```bash
ls ~/gaussian-splatting-cuda/parameter/
# Inspect defaults and adjust as needed
```

### Strategy Notes

- **`mcmc`** (recommended): Monte Carlo Markov Chain — better convergence,
  controls Gaussian count via `--max-cap`
- **`default`**: Traditional densification via duplication and splitting

### Performance Tips for Your Hardware

- **RTX 5090 (32 GB VRAM):** Start with `--max-cap 500000`. You can push to
  1M+ Gaussians given your VRAM headroom.
- **196 GB RAM:** No memory concerns for data loading. Use `--resize_factor 1`
  for full resolution.
- **Threadripper:** The multi-core advantage mainly helps during the build
  phase. Training is GPU-bound.

### RTX 5090 Blackwell Compatibility Warning

The RTX 5090 (sm_120) is a new architecture. While the project compiles with
CUDA 12.8+ targeting sm_120, the custom CUDA kernels in `gsplat/` and
`fastgs/` were originally tuned for Ampere/Ada architectures. Blackwell has
different warp scheduling and shared memory layouts. If you encounter runtime
errors or incorrect rendering:

1. Try building with PTX-only mode for JIT compilation:
   ```bash
   cmake .. -DBUILD_CUDA_PTX_ONLY=ON -DBUILD_CUDA_MIN_SM=75
   ```
2. Check the project's GitHub issues for sm_120-specific fixes.
3. The PTX build produces a much smaller binary (~66 MB vs ~750 MB) but has
   a 5-15 second JIT compilation delay on first launch.

---

## 9. Troubleshooting

### "no kernel image is available for execution on the device"

Your CUDA code wasn't compiled for sm_120. Rebuild with:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . -j$(nproc)
```

### nvidia-smi works but nvcc not found

CUDA toolkit PATH not set:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### nvidia-smi shows "No devices found"

1. Update Windows NVIDIA driver to 570+
2. Run `wsl --shutdown` from PowerShell, then reopen WSL
3. Check BIOS: ensure IOMMU / virtualization is enabled

### Slow file operations

You're probably reading from `/mnt/c/`. Copy data to `~/`:
```bash
cp -r /mnt/c/Users/WildTech/Desktop/H2103d_Northampton ~/data/
```

### CMake can't find CUDA

```bash
# Verify CUDA is installed
ls /usr/local/cuda/bin/nvcc

# Explicitly tell CMake where CUDA lives
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda
```

### vcpkg build failures

```bash
# Ensure vcpkg is up to date
cd $VCPKG_ROOT && git pull && ./bootstrap-vcpkg.sh

# Clean and retry
rm -rf build/
cmake .. [your flags]
```

### PCIe crashes with RTX 5090

Some Threadripper + RTX 5090 systems have PCIe 5.0 signal integrity issues.
Set PCIe to **Gen 4** mode in BIOS — less than 1% performance impact for
GPU compute.

### GCC version incompatibility / C++23 errors

LichtFeld Studio requires C++23, which needs **GCC 14+**. Ubuntu 24.04 ships
GCC 13 — you must upgrade:
```bash
gcc --version
# If < 14, install GCC 14:
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-14 g++-14
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
```

### LibTorch "two conflicting CUDA installs" error

Do NOT install LibTorch via vcpkg. Download it manually into `external/`
as described in the build section. See
[vcpkg issue #36844](https://github.com/microsoft/vcpkg/issues/36844).

---

## Quick Reference — Full Setup Script

For convenience, here's the entire setup condensed into a single script.
**Read through it before running** — adjust paths as needed:

```bash
#!/bin/bash
set -euo pipefail

echo "=== [1/6] Installing CUDA Toolkit ==="
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit
rm cuda-keyring_1.1-1_all.deb

# Add CUDA to PATH
grep -q 'cuda/bin' ~/.bashrc || {
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
}
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== [2/7] Installing Build Dependencies + GCC 14 + CMake 3.30+ ==="
sudo apt-get install -y \
    build-essential git curl zip unzip tar \
    pkg-config ninja-build python3 python3-dev python3-pip \
    software-properties-common \
    libssl-dev libx11-dev libxrandr-dev libxi-dev \
    libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev \
    libwayland-dev libxkbcommon-dev autoconf automake libtool nasm yasm

# GCC 14 (required for C++23)
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-14 g++-14
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14

# CMake 3.30+ (required)
sudo apt-get remove -y cmake 2>/dev/null || true
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
    | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt-get update
sudo apt-get install -y cmake

echo "=== [3/7] Installing vcpkg ==="
if [ ! -d "$HOME/vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
    ~/vcpkg/bootstrap-vcpkg.sh
fi
export VCPKG_ROOT=$HOME/vcpkg
grep -q 'VCPKG_ROOT' ~/.bashrc || {
    echo 'export VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
    echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
}

echo "=== [4/7] Cloning LichtFeld Studio ==="
if [ ! -d "$HOME/gaussian-splatting-cuda" ]; then
    git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda.git ~/gaussian-splatting-cuda
fi
cd ~/gaussian-splatting-cuda

echo "=== [5/7] Downloading LibTorch ==="
mkdir -p external && cd external
if [ ! -d "libtorch" ]; then
    wget -q "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.cpu.zip" -O libtorch.zip
    unzip -q libtorch.zip
    rm libtorch.zip
fi
cd ..

echo "=== [6/7] Building ==="
mkdir -p build && cd build
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_PREFIX_PATH=$HOME/gaussian-splatting-cuda/external/libtorch
cmake --build . --config Release -j$(nproc)

echo ""
echo "=== BUILD COMPLETE ==="
echo "Binary: find ~/gaussian-splatting-cuda/build/ -type f -executable"
echo ""
echo "Next steps:"
echo "  1. Copy your data:  cp -r /mnt/c/Users/WildTech/Desktop/H2103d_Northampton ~/data/"
echo "  2. Run training:    ./build/bin/lichtfeld-studio -d ~/data/H2103d_Northampton -o ~/output/H2103d --strategy mcmc"
```

Save this as `~/setup_lichtfeld.sh`, then run:

```bash
chmod +x ~/setup_lichtfeld.sh
./setup_lichtfeld.sh
```

---

## References

- [MrNeRF/gaussian-splatting-cuda (GitHub)](https://github.com/MrNeRF/gaussian-splatting-cuda)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [CUDA Toolkit Downloads (WSL-Ubuntu)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [CUDA Toolkit 12.8 Blackwell Support Blog](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support)
- [PyTorch LibTorch Downloads](https://pytorch.org/get-started/locally/)
