#!/bin/bash
# =============================================================================
# LichtFeld Studio (gaussian-splatting-cuda) — WSL2 Setup Script
# Target: RTX 5090 (Blackwell, sm_120) · Ubuntu 24.04 WSL2 (20.04+ supported)
#
# This script is designed to run with minimal intervention on a fresh WSL2
# instance. It handles both sudo and non-sudo environments gracefully.
#
# Usage:
#   ./setup_lichtfeld.sh              # Full setup (needs sudo for apt)
#   ./setup_lichtfeld.sh --no-sudo    # Skip apt steps (deps pre-installed)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

REPO_DIR="$HOME/gaussian-splatting-cuda"
CUDA_ROOT="/usr/local/cuda"
NO_SUDO=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --no-sudo) NO_SUDO=true ;;
    esac
done

# Auto-detect if sudo is available
CAN_SUDO=true
if [ "$NO_SUDO" = true ]; then
    CAN_SUDO=false
elif ! command -v sudo &>/dev/null; then
    CAN_SUDO=false
elif ! sudo -n true 2>/dev/null; then
    # sudo exists but needs a password — try interactively
    if [ -t 0 ]; then
        # We have a terminal, sudo will prompt
        CAN_SUDO=true
    else
        warn "sudo requires a password but no terminal is available."
        warn "Continuing without sudo (--no-sudo mode)."
        CAN_SUDO=false
    fi
fi

run_sudo() {
    if [ "$CAN_SUDO" = true ]; then
        sudo "$@"
    else
        warn "Skipping (no sudo): $*"
        return 1
    fi
}

# ─── Preflight Checks ───────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "  LichtFeld Studio — WSL2 Setup"
echo "  Target GPU: RTX 5090 (Blackwell sm_120)"
echo "  sudo: $([ "$CAN_SUDO" = true ] && echo "available" || echo "unavailable")"
echo "============================================"
echo ""

# Check we're in WSL (optional — allow native Linux too)
if grep -qi microsoft /proc/version 2>/dev/null; then
    log "Running in WSL2"
elif [ -f /proc/version ]; then
    warn "Not WSL2 — running on native Linux (should still work)"
else
    warn "Cannot detect OS type"
fi

# Check GPU is visible
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        log "GPU detected via nvidia-smi"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        error "nvidia-smi found but failed. Check NVIDIA driver."
        error "WSL2: Check Windows NVIDIA driver (570+ required). Try: wsl --shutdown"
        exit 1
    fi
else
    warn "nvidia-smi not found. It should appear after CUDA toolkit install."
fi

# Detect GPU compute capability for CUDA arch
CUDA_ARCH="120"  # Default: Blackwell
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$GPU_NAME" in
        *5090*|*5080*|*5070*|*5060*|*B200*|*B100*|*GB*) CUDA_ARCH="120" ;;  # Blackwell
        *4090*|*4080*|*4070*|*4060*|*L40*|*H100*|*H200*) CUDA_ARCH="89" ;;   # Ada/Hopper
        *3090*|*3080*|*3070*|*3060*|*A100*|*A6000*) CUDA_ARCH="86" ;;         # Ampere
        *2080*|*2070*|*2060*|*T4*) CUDA_ARCH="75" ;;                           # Turing
        *) warn "Unknown GPU '$GPU_NAME', defaulting to sm_$CUDA_ARCH" ;;
    esac
    log "GPU: $GPU_NAME → targeting sm_$CUDA_ARCH"
fi

echo ""

# ─── Step 1: CUDA Toolkit ───────────────────────────────────────────────────

echo "=== [1/8] CUDA Toolkit ==="

if [ -x "$CUDA_ROOT/bin/nvcc" ]; then
    log "CUDA toolkit found at $CUDA_ROOT: $($CUDA_ROOT/bin/nvcc --version | grep release)"
elif command -v nvcc &>/dev/null; then
    log "CUDA toolkit found: $(nvcc --version | grep release)"
    CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
else
    if [ "$CAN_SUDO" = true ]; then
        warn "Installing CUDA Toolkit (WSL-Ubuntu package)..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y cuda-toolkit
        rm -f cuda-keyring_1.1-1_all.deb
        log "CUDA Toolkit installed"
    else
        error "CUDA toolkit not found and cannot install without sudo."
        error "Install CUDA manually: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
fi

# Ensure CUDA is on PATH for this session and future ones
export PATH="$CUDA_ROOT/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
grep -q 'cuda' ~/.bashrc 2>/dev/null || {
    echo '' >> ~/.bashrc
    echo '# CUDA Toolkit' >> ~/.bashrc
    echo "export PATH=$CUDA_ROOT/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
}

echo ""

# ─── Step 2: Build Dependencies (apt) ───────────────────────────────────────

echo "=== [2/8] Build Dependencies ==="

APT_PACKAGES=(
    build-essential git curl wget zip unzip tar
    pkg-config ninja-build python3 python3-dev python3-pip
    software-properties-common
    libssl-dev libx11-dev libxrandr-dev libxi-dev libxtst-dev
    libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev
    libwayland-dev libxkbcommon-dev libegl1-mesa-dev
    autoconf automake autoconf-archive libtool nasm yasm
)

if [ "$CAN_SUDO" = true ]; then
    sudo apt-get update -qq
    sudo apt-get install -y "${APT_PACKAGES[@]}" 2>&1 | tail -1
    log "Base dependencies installed via apt"
else
    warn "Skipping apt install (no sudo). Checking if key tools exist..."
    MISSING=()
    for cmd in gcc g++ cmake ninja git python3 curl wget; do
        command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
    done
    if [ ${#MISSING[@]} -gt 0 ]; then
        error "Missing required tools: ${MISSING[*]}"
        error "Run with sudo or install them manually."
        exit 1
    fi
    log "Key build tools found"

    # Install autoconf-archive locally if missing (needed by vcpkg libb2)
    if ! dpkg -l autoconf-archive &>/dev/null 2>&1; then
        if [ ! -d "$HOME/.local/share/aclocal" ] || [ "$(ls -A "$HOME/.local/share/aclocal" 2>/dev/null | wc -l)" -lt 100 ]; then
            warn "Installing autoconf-archive macros locally..."
            mkdir -p "$HOME/.local/share/aclocal"
            cd /tmp
            apt-get download autoconf-archive 2>/dev/null || true
            if ls autoconf-archive*.deb 1>/dev/null 2>&1; then
                dpkg -x autoconf-archive*.deb autoconf-archive-tmp
                cp autoconf-archive-tmp/usr/share/aclocal/*.m4 "$HOME/.local/share/aclocal/" 2>/dev/null || true
                rm -rf autoconf-archive-tmp autoconf-archive*.deb
                log "autoconf-archive macros installed locally"
            fi
            cd -
        fi
    fi
fi

export ACLOCAL_PATH="$HOME/.local/share/aclocal:${ACLOCAL_PATH:-}"

# GCC 14+ required for C++23
if gcc --version 2>/dev/null | grep -q ' 1[4-9]\.\| [2-9][0-9]\.'; then
    log "GCC 14+ already installed: $(gcc --version | head -1)"
elif [ "$CAN_SUDO" = true ]; then
    warn "Installing GCC 14 (required for C++23)..."
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update -qq
    sudo apt-get install -y gcc-14 g++-14
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
    log "GCC 14 installed and set as default"
else
    error "GCC 14+ required but not found. Install with sudo."
    exit 1
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
    warn "Installing CMake 3.30+ via pip..."
    run_sudo apt-get remove -y cmake 2>/dev/null || true
    pip3 install --user cmake --upgrade 2>/dev/null || pip install --user cmake --upgrade
    PIP_BIN="$HOME/.local/bin"
    if [ -d "$PIP_BIN" ] && ! echo "$PATH" | grep -q "$PIP_BIN"; then
        export PATH="$PIP_BIN:$PATH"
        grep -q '.local/bin' ~/.bashrc 2>/dev/null || {
            echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
        }
    fi
    log "CMake $(cmake --version | head -1) installed via pip"
fi

# Autoconf 2.71+ required for vcpkg python3 build
AUTOCONF_OK=false
if command -v autoconf &>/dev/null; then
    AC_VER=$(autoconf --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' || echo "0.0")
    AC_MAJOR=$(echo "$AC_VER" | cut -d. -f1)
    AC_MINOR=$(echo "$AC_VER" | cut -d. -f2)
    if [ "$AC_MAJOR" -gt 2 ] 2>/dev/null || { [ "$AC_MAJOR" -eq 2 ] && [ "$AC_MINOR" -ge 71 ]; } 2>/dev/null; then
        AUTOCONF_OK=true
        log "Autoconf $AC_VER meets requirement (2.71+)"
    fi
fi

if [ "$AUTOCONF_OK" = false ]; then
    if [ -x "$HOME/.local/bin/autoconf" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        log "Using locally installed autoconf: $($HOME/.local/bin/autoconf --version | head -1)"
    else
        warn "Building autoconf 2.72 from source (needed for vcpkg python3)..."
        cd /tmp
        wget -q https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.xz
        tar xf autoconf-2.72.tar.xz
        cd autoconf-2.72
        ./configure --prefix="$HOME/.local" --quiet
        make -j"$(nproc)" --quiet
        make install --quiet
        cd /tmp && rm -rf autoconf-2.72 autoconf-2.72.tar.xz
        export PATH="$HOME/.local/bin:$PATH"
        log "Autoconf 2.72 installed locally"
    fi
fi

echo ""

# ─── Step 3: vcpkg ──────────────────────────────────────────────────────────

echo "=== [3/8] vcpkg ==="

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

echo "=== [4/8] Clone LichtFeld Studio ==="

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

# ─── Step 5: Overlay Ports ──────────────────────────────────────────────────

echo "=== [5/8] vcpkg Overlay Ports ==="

# SDL3 overlay: disables XTEST (needs libxtst-dev) and ibus (needs libibus-1.0-dev)
# These may not be available on minimal installs
if [ -d "$SCRIPT_DIR/overlay-ports/sdl3" ]; then
    mkdir -p "$REPO_DIR/overlay-ports"
    cp -r "$SCRIPT_DIR/overlay-ports/sdl3" "$REPO_DIR/overlay-ports/"
    log "SDL3 overlay port installed (XTEST disabled, ibus disabled)"
else
    warn "No overlay-ports/sdl3 found in $SCRIPT_DIR — SDL3 may fail if libxtst-dev is missing"
fi

echo ""

# ─── Step 6: LibTorch ───────────────────────────────────────────────────────

echo "=== [6/8] LibTorch (C++ PyTorch for CUDA 12.8+) ==="

mkdir -p external
cd external

if [ -d "libtorch" ]; then
    log "LibTorch already present"
else
    warn "Downloading LibTorch (~2 GB)... this may take a few minutes."

    LIBTORCH_URLS=(
        "https://download.pytorch.org/libtorch/nightly/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip"
        "https://download.pytorch.org/libtorch/test/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip"
        "https://download.pytorch.org/libtorch/nightly/cu128/libtorch-shared-with-deps-latest.zip"
        "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-latest.zip"
    )

    DOWNLOADED=false
    for LIBTORCH_URL in "${LIBTORCH_URLS[@]}"; do
        warn "Trying: $LIBTORCH_URL"
        if wget --progress=bar:force:noscroll -O libtorch.zip "$LIBTORCH_URL" 2>&1; then
            if file libtorch.zip | grep -qi zip; then
                DOWNLOADED=true
                log "Downloaded from: $LIBTORCH_URL"
                break
            else
                warn "Downloaded file is not a valid zip — trying next URL..."
                rm -f libtorch.zip
            fi
        else
            warn "Failed — trying next URL..."
            rm -f libtorch.zip
        fi
    done

    if [ "$DOWNLOADED" = false ]; then
        error "Could not download LibTorch from any channel."
        error "Download manually from https://pytorch.org/get-started/locally/"
        exit 1
    fi

    unzip -q libtorch.zip
    rm libtorch.zip
    log "LibTorch downloaded and extracted"
fi

cd "$REPO_DIR"
echo ""

# ─── Step 7: Build ──────────────────────────────────────────────────────────

echo "=== [7/8] Build LichtFeld Studio ==="
echo "  Targeting: sm_$CUDA_ARCH"
echo "  Build type: Release"
echo "  Parallelism: $(nproc) cores"
echo ""

mkdir -p build && cd build

# Build overlay ports arg (if overlay exists)
OVERLAY_ARG=""
if [ -d "$REPO_DIR/overlay-ports" ]; then
    OVERLAY_ARG="-DVCPKG_OVERLAY_PORTS=$REPO_DIR/overlay-ports"
fi

cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_PREFIX_PATH="$REPO_DIR/external/libtorch" \
    $OVERLAY_ARG

log "CMake configured"

echo ""
warn "Building... (first build takes 10-30 min due to vcpkg dependency compilation)"
cmake --build . --config Release -j"$(nproc)"

log "Build complete!"

# Verify binary exists
BINARY=""
for candidate in \
    "$REPO_DIR/build/LichtFeld-Studio" \
    "$REPO_DIR/build/bin/LichtFeld-Studio"; do
    if [ -x "$candidate" ]; then
        BINARY="$candidate"
        break
    fi
done

if [ -n "$BINARY" ]; then
    log "Binary: $BINARY"
else
    error "Build succeeded but binary not found!"
    find "$REPO_DIR/build/" -type f -executable 2>/dev/null | head -5
    exit 1
fi

cd "$REPO_DIR"
echo ""

# ─── Step 8: Verify ─────────────────────────────────────────────────────────

echo "=== [8/8] Verification ==="

# Quick smoke test — just check it starts and prints version
if "$BINARY" --version 2>&1 | head -1; then
    log "Binary runs successfully"
else
    warn "Binary may have runtime issues (missing libs?)"
    warn "Try: LD_LIBRARY_PATH=$REPO_DIR/build:\$LD_LIBRARY_PATH $BINARY --version"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Binary: $BINARY"
echo ""
echo "To train on a COLMAP dataset:"
echo ""
echo "  $BINARY \\"
echo "      -d /path/to/colmap/dataset \\"
echo "      -o ~/output/my_scene \\"
echo "      --strategy mcmc \\"
echo "      --max-cap 1000000 \\"
echo "      -i 30000 \\"
echo "      -r 2 \\"
echo "      --min-opacity 0.05 \\"
echo "      --enable-mip \\"
echo "      --headless"
echo ""
echo "Expected data structure:"
echo "  dataset/"
echo "  ├── images/"
echo "  │   ├── 00000.jpg (or .jpeg, .png)"
echo "  │   └── ..."
echo "  └── sparse/"
echo "      └── 0/"
echo "          ├── cameras.txt"
echo "          ├── images.txt"
echo "          └── points3D.txt"
echo ""
echo "Notes:"
echo "  - Use --headless for WSL2 or headless servers (no display)"
echo "  - Use -r 2 for half-resolution (faster), -r 1 for full"
echo "  - COLMAP image names in images.txt must match actual filenames"
echo "  - If images are at root level (not in images/), move them to images/"
echo ""
