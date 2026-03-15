#!/bin/bash
# =============================================================================
# LichtFeld Studio (gaussian-splatting-cuda) — WSL2 Setup Script
# Target: RTX 5090 (Blackwell, sm_120) · Ubuntu 24.04 WSL2
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

DATA_SRC="/mnt/c/Users/WildTech/Desktop/H2103d_test_colmap_workflow"
DATA_DST="$HOME/data/H2103d_test_colmap_workflow"
REPO_DIR="$HOME/gaussian-splatting-cuda"
CUDA_ROOT="/usr/local/cuda"

# ─── Preflight Checks ───────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "  LichtFeld Studio — WSL2 Setup"
echo "  Target GPU: RTX 5090 (Blackwell sm_120)"
echo "============================================"
echo ""

# Check we're in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    error "This script is designed for WSL2. Exiting."
    exit 1
fi

# Check GPU is visible
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        log "GPU detected via nvidia-smi"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        error "nvidia-smi found but failed. Check Windows NVIDIA driver (570+ required)."
        error "Try: wsl --shutdown from PowerShell, then reopen WSL."
        exit 1
    fi
else
    warn "nvidia-smi not found. It should appear after CUDA toolkit install."
fi

echo ""

# ─── Step 1: CUDA Toolkit ───────────────────────────────────────────────────

echo "=== [1/7] CUDA Toolkit ==="

# Check for CUDA at the known path first
if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
    log "CUDA toolkit found at $CUDA_ROOT: $($CUDA_ROOT/bin/nvcc --version | grep release)"
    # Ensure it's on PATH
    if ! echo "$PATH" | grep -q "$CUDA_ROOT/bin"; then
        export PATH="$CUDA_ROOT/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
    fi
elif command -v nvcc &>/dev/null; then
    log "CUDA toolkit already installed: $(nvcc --version | grep release)"
    CUDA_ROOT="$(dirname $(dirname $(which nvcc)))"
else
    warn "Installing CUDA Toolkit (WSL-Ubuntu package)..."
    warn "This installs cuda-toolkit ONLY — no Linux GPU driver."

    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y cuda-toolkit
    rm -f cuda-keyring_1.1-1_all.deb

    log "CUDA Toolkit installed"
fi

# Ensure CUDA is on PATH
if ! echo "$PATH" | grep -q 'cuda'; then
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
    grep -q 'cuda' ~/.bashrc 2>/dev/null || {
        echo '' >> ~/.bashrc
        echo '# CUDA Toolkit' >> ~/.bashrc
        echo "export PATH=$CUDA_ROOT/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    }
    log "CUDA added to PATH"
fi

echo ""

# ─── Step 2: Build Dependencies ─────────────────────────────────────────────

echo "=== [2/7] Build Dependencies ==="

sudo apt-get update -qq
sudo apt-get install -y \
    build-essential git curl zip unzip tar \
    pkg-config ninja-build python3 python3-dev python3-pip \
    software-properties-common \
    libssl-dev libx11-dev libxrandr-dev libxi-dev \
    libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev \
    libwayland-dev libxkbcommon-dev autoconf automake libtool nasm yasm \
    2>&1 | tail -1

log "Base dependencies installed"

# GCC 14+ required for C++23
if gcc --version 2>/dev/null | grep -q ' 1[4-9]\.\| [2-9][0-9]\.'; then
    log "GCC 14+ already installed: $(gcc --version | head -1)"
else
    warn "Installing GCC 14 (required for C++23)..."
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update -qq
    sudo apt-get install -y gcc-14 g++-14
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
    log "GCC 14 installed and set as default"
fi

# CMake 3.30+ required
CMAKE_OK=false
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "0.0.0")
    CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -gt 3 ] 2>/dev/null || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 30 ]; } 2>/dev/null; then
        CMAKE_OK=true
        log "CMake $CMAKE_VER already meets requirement (3.30+)"
    fi
fi

if [ "$CMAKE_OK" = false ]; then
    warn "Installing CMake 3.30+ from Kitware repository..."
    sudo apt-get remove -y cmake 2>/dev/null || true
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
        | sudo tee /etc/apt/sources.list.d/kitware.list
    sudo apt-get update -qq
    sudo apt-get install -y cmake
    log "CMake $(cmake --version | head -1) installed"
fi

echo ""

# ─── Step 3: vcpkg ──────────────────────────────────────────────────────────

echo "=== [3/7] vcpkg ==="

export VCPKG_ROOT="$HOME/vcpkg"

if [ -d "$VCPKG_ROOT" ] && [ -x "$VCPKG_ROOT/vcpkg" ]; then
    log "vcpkg already installed at $VCPKG_ROOT"
else
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
    log "vcpkg installed"
fi

grep -q 'VCPKG_ROOT' ~/.bashrc 2>/dev/null || {
    echo '' >> ~/.bashrc
    echo '# vcpkg' >> ~/.bashrc
    echo "export VCPKG_ROOT=$HOME/vcpkg" >> ~/.bashrc
    echo 'export PATH=$VCPKG_ROOT:$PATH' >> ~/.bashrc
}

echo ""

# ─── Step 4: Clone Repository ───────────────────────────────────────────────

echo "=== [4/7] Clone LichtFeld Studio ==="

if [ -d "$REPO_DIR/.git" ]; then
    log "Repository already cloned at $REPO_DIR"
    cd "$REPO_DIR"
    git pull --ff-only || warn "Could not fast-forward; using existing state"
else
    git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda.git "$REPO_DIR"
    log "Repository cloned (with submodules)"
fi

cd "$REPO_DIR"
echo ""

# ─── Step 5: LibTorch ───────────────────────────────────────────────────────

echo "=== [5/7] LibTorch (C++ PyTorch for CUDA 12.8) ==="

mkdir -p external
cd external

if [ -d "libtorch" ]; then
    log "LibTorch already present"
else
    warn "Downloading LibTorch (~2 GB)... this may take a few minutes."
    # Check PyTorch website for latest link: https://pytorch.org/get-started/locally/
    wget -q --show-progress \
        "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.cpu.zip" \
        -O libtorch.zip
    unzip -q libtorch.zip
    rm libtorch.zip
    log "LibTorch downloaded and extracted"
fi

cd "$REPO_DIR"
echo ""

# ─── Step 6: Build ──────────────────────────────────────────────────────────

echo "=== [6/7] Build LichtFeld Studio ==="
echo "  Targeting: sm_120 (RTX 5090 Blackwell)"
echo "  Build type: Release"
echo "  Parallelism: $(nproc) cores"
echo ""

mkdir -p build && cd build

# Try preset first, fall back to manual cmake
if cmake --preset linux-release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DVCPKG_ROOT="$VCPKG_ROOT" 2>/dev/null; then
    log "CMake configured via preset"
else
    warn "Preset not found, using manual CMake configuration"
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DCUDAToolkit_ROOT="$CUDA_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DCMAKE_PREFIX_PATH="$REPO_DIR/external/libtorch"
    log "CMake configured manually"
fi

echo ""
warn "Building... (first build takes a while due to vcpkg dependency compilation)"
cmake --build . --config Release -j"$(nproc)"

log "Build complete!"
cd "$REPO_DIR"
echo ""

# ─── Step 7: Copy Data ──────────────────────────────────────────────────────

echo "=== [7/7] Data Preparation ==="

if [ -d "$DATA_DST/images" ]; then
    log "Data already copied to $DATA_DST"
else
    if [ -d "$DATA_SRC" ]; then
        warn "Copying data from Windows filesystem to WSL2 native filesystem..."
        warn "This avoids slow cross-filesystem I/O during training."
        mkdir -p "$DATA_DST"

        # Copy images
        if [ -d "$DATA_SRC/images" ]; then
            cp -r "$DATA_SRC/images" "$DATA_DST/images"
            IMAGE_COUNT=$(find "$DATA_DST/images" -type f | wc -l)
            log "Copied $IMAGE_COUNT images"
        else
            warn "No 'images' subdirectory found in $DATA_SRC"
            warn "Check your data path and copy images manually."
        fi

        # Copy COLMAP sparse reconstruction if present
        if [ -d "$DATA_SRC/sparse" ]; then
            cp -r "$DATA_SRC/sparse" "$DATA_DST/sparse"
            log "Copied COLMAP sparse reconstruction"
        elif [ -f "$DATA_SRC/cameras.txt" ]; then
            # COLMAP text files are loose — organize them
            mkdir -p "$DATA_DST/sparse/0"
            for f in cameras.txt images.txt points3D.txt; do
                if [ -f "$DATA_SRC/$f" ]; then
                    cp "$DATA_SRC/$f" "$DATA_DST/sparse/0/"
                fi
            done
            log "Organized COLMAP text files into sparse/0/"
        else
            warn "No COLMAP data found. You'll need to add sparse/0/ with:"
            warn "  cameras.txt, images.txt, points3D.txt"
        fi
    else
        warn "Source data not found at: $DATA_SRC"
        warn "Copy your data manually:"
        warn "  mkdir -p $DATA_DST"
        warn "  cp -r /mnt/c/Users/WildTech/Desktop/H2103d_Northampton/images $DATA_DST/"
    fi
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Expected data structure:"
echo "  $DATA_DST/"
echo "  ├── images/"
echo "  │   ├── IMG_0001.jpg"
echo "  │   └── ..."
echo "  └── sparse/"
echo "      └── 0/"
echo "          ├── cameras.txt"
echo "          ├── images.txt"
echo "          └── points3D.txt"
echo ""
echo "To run training:"
echo "  cd $REPO_DIR"
echo ""
echo "  # Find the binary (name may vary):"
echo "  find build/ -type f -executable -name 'lichtfeld*' -o -name 'gaussian*' 2>/dev/null | head -3"
echo ""
echo "  # Example training command:"
echo "  ./build/bin/lichtfeld-studio \\"
echo "      -d $DATA_DST \\"
echo "      -o ~/output/H2103d_Northampton \\"
echo "      --strategy mcmc \\"
echo "      --max-cap 500000 \\"
echo "      -i 30000"
echo ""
