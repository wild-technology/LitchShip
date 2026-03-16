#!/bin/bash
# =============================================================================
# LichtFeld Studio (gaussian-splatting-cuda) — WSL2 Setup Script
# Target: NVIDIA GPUs (Turing through Blackwell) · Ubuntu 24.04 WSL2 (20.04+)
#
# This script handles both sudo and non-sudo environments. Without sudo it
# installs everything possible into ~/.local via pip, deb extraction, or
# building from source.
#
# Usage:
#   ./setup_lichtfeld.sh              # Full setup (uses sudo for apt)
#   ./setup_lichtfeld.sh --no-sudo    # No sudo (CI, containers, restricted)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

REPO_DIR="$HOME/gaussian-splatting-cuda"
CUDA_ROOT="/usr/local/cuda"
LOCAL_PREFIX="$HOME/.local"
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
    if [ -t 0 ]; then
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

# Helper: extract a binary from a .deb package into ~/.local/bin
# Usage: install_deb_bin <package-name> <binary-name> [extra-packages...]
install_deb_bin() {
    local pkg="$1"
    local bin="$2"
    shift 2
    local extra_pkgs=("$@")

    if command -v "$bin" &>/dev/null; then
        return 0
    fi
    if [ -x "$LOCAL_PREFIX/bin/$bin" ]; then
        return 0
    fi

    local tmpdir
    tmpdir=$(mktemp -d)
    cd "$tmpdir"

    local all_pkgs=("$pkg" "${extra_pkgs[@]}")
    apt-get download "${all_pkgs[@]}" 2>/dev/null || { cd /; rm -rf "$tmpdir"; return 1; }

    mkdir -p extract
    for deb in *.deb; do
        dpkg -x "$deb" extract 2>/dev/null || true
    done

    # Copy binaries
    if [ -d extract/usr/bin ]; then
        find extract/usr/bin -type f -executable -exec cp {} "$LOCAL_PREFIX/bin/" \; 2>/dev/null || true
        # Also copy symlinks (resolve them)
        find extract/usr/bin -type l -exec cp -L {} "$LOCAL_PREFIX/bin/" \; 2>/dev/null || true
    fi
    # Copy libraries
    if [ -d extract/usr/lib/x86_64-linux-gnu ]; then
        find extract/usr/lib/x86_64-linux-gnu -maxdepth 1 \( -name '*.so*' -o -name '*.a' \) \
            -exec cp -a {} "$LOCAL_PREFIX/lib/" \; 2>/dev/null || true
        # pkgconfig
        if [ -d extract/usr/lib/x86_64-linux-gnu/pkgconfig ]; then
            cp -a extract/usr/lib/x86_64-linux-gnu/pkgconfig/* "$LOCAL_PREFIX/lib/pkgconfig/" 2>/dev/null || true
        fi
    fi
    # Copy headers
    if [ -d extract/usr/include ]; then
        cp -a extract/usr/include/* "$LOCAL_PREFIX/include/" 2>/dev/null || true
    fi
    # Copy aclocal macros
    if [ -d extract/usr/share/aclocal ]; then
        cp -a extract/usr/share/aclocal/* "$LOCAL_PREFIX/share/aclocal/" 2>/dev/null || true
    fi

    cd /
    rm -rf "$tmpdir"
}

# Helper: build a GNU tool from source
# Usage: build_gnu_tool <name> <version> <url-suffix>
build_gnu_tool() {
    local name="$1"
    local version="$2"
    local tarball="$3"
    local url="https://ftp.gnu.org/gnu/$name/$tarball"

    warn "Building $name $version from source..."
    local tmpdir
    tmpdir=$(mktemp -d)
    cd "$tmpdir"
    wget -q "$url"
    tar xf "$tarball"
    cd "${tarball%.tar.*}"
    ./configure --prefix="$LOCAL_PREFIX" --quiet
    make -j"$(nproc)" --quiet
    make install --quiet
    cd /
    rm -rf "$tmpdir"
    log "$name $version installed to $LOCAL_PREFIX"
}

# ─── Setup ~/.local directories ─────────────────────────────────────────────

mkdir -p "$LOCAL_PREFIX/bin" "$LOCAL_PREFIX/lib" "$LOCAL_PREFIX/lib/pkgconfig" \
         "$LOCAL_PREFIX/include" "$LOCAL_PREFIX/share/aclocal" \
         "$LOCAL_PREFIX/libexec"

# Ensure ~/.local/bin is on PATH for the rest of this script
export PATH="$LOCAL_PREFIX/bin:$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$LOCAL_PREFIX/lib:${LIBRARY_PATH:-}"
export CPATH="$LOCAL_PREFIX/include:${CPATH:-}"
export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export ACLOCAL_PATH="$LOCAL_PREFIX/share/aclocal:${ACLOCAL_PATH:-}"

# ─── Preflight Checks ───────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "  LichtFeld Studio — WSL2 Setup"
echo "  sudo: $([ "$CAN_SUDO" = true ] && echo "available" || echo "unavailable (--no-sudo)")"
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

# Detect GPU compute capability for CUDA arch.
# Maps GPU model name patterns to SM architecture versions:
#   sm_120 = Blackwell (RTX 50xx, B-series)
#   sm_89  = Ada Lovelace / Hopper (RTX 40xx, L40, H100/H200)
#   sm_86  = Ampere (RTX 30xx, A100, A6000)
#   sm_75  = Turing (RTX 20xx, T4)
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

# Persist CUDA PATH for future sessions
grep -q 'cuda' ~/.bashrc 2>/dev/null || {
    echo '' >> ~/.bashrc
    echo '# CUDA Toolkit' >> ~/.bashrc
    echo "export PATH=$CUDA_ROOT/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
}

echo ""

# ─── Step 2: Build Dependencies ─────────────────────────────────────────────

echo "=== [2/8] Build Dependencies ==="

APT_PACKAGES=(
    build-essential git curl wget zip unzip tar
    pkg-config ninja-build python3 python3-dev python3-pip
    software-properties-common
    libssl-dev libx11-dev libxrandr-dev libxi-dev libxtst-dev
    libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev
    libwayland-dev libxkbcommon-dev libegl1-mesa-dev
    libxext-dev libxau-dev libxdmcp-dev libxcb1-dev x11proto-dev
    autoconf automake autoconf-archive libtool nasm yasm
)

if [ "$CAN_SUDO" = true ]; then
    sudo apt-get update -qq
    sudo apt-get install -y "${APT_PACKAGES[@]}" 2>&1 | tail -1
    log "Base dependencies installed via apt"
else
    warn "No sudo available. Installing dependencies locally..."

    # ── Essential CLI tools (from deb) ──
    for tool in zip unzip nasm yasm; do
        if ! command -v "$tool" &>/dev/null && [ ! -x "$LOCAL_PREFIX/bin/$tool" ]; then
            install_deb_bin "$tool" "$tool"
            log "Installed $tool locally"
        fi
    done

    # ── pkg-config (needs pkgconf-bin + libpkgconf3) ──
    if ! command -v pkg-config &>/dev/null && [ ! -x "$LOCAL_PREFIX/bin/pkg-config" ]; then
        install_deb_bin pkgconf-bin pkgconf libpkgconf3
        ln -sf pkgconf "$LOCAL_PREFIX/bin/pkg-config" 2>/dev/null || true
        log "Installed pkg-config locally"
    fi

    # ── cmake + ninja (via pip) ──
    if ! command -v cmake &>/dev/null; then
        warn "Installing cmake + ninja via pip..."
        pip3 install --user cmake ninja --upgrade 2>/dev/null \
            || pip install --user cmake ninja --upgrade
        log "cmake + ninja installed via pip"
    fi

    # ── autoconf-archive macros (from deb) ──
    if [ "$(ls -A "$LOCAL_PREFIX/share/aclocal" 2>/dev/null | wc -l)" -lt 100 ]; then
        install_deb_bin autoconf-archive false
        log "autoconf-archive macros installed locally"
    fi

    # ── pkg.m4 (from pkgconf deb, needed for libb2 build) ──
    if [ ! -f "$LOCAL_PREFIX/share/aclocal/pkg.m4" ]; then
        install_deb_bin pkgconf false
        log "pkg.m4 macro installed locally"
    fi

    # ── OpenGL dev libraries (from debs) ──
    if [ ! -f "$LOCAL_PREFIX/lib/libOpenGL.so" ]; then
        warn "Installing OpenGL development libraries locally..."
        install_deb_bin libgl-dev false \
            libglx-dev libegl-dev libopengl-dev libglvnd-dev \
            libgl1-mesa-dev mesa-common-dev libglu1-mesa-dev \
            libglvnd0 libglx0 libgl1 libegl1 libopengl0 \
            libglx-mesa0 libglu1-mesa
        log "OpenGL dev libraries installed locally"
    fi

    # ── X11 dev libraries (from debs) ──
    if [ ! -f "$LOCAL_PREFIX/include/X11/Xatom.h" ]; then
        warn "Installing X11 development libraries locally..."
        install_deb_bin libx11-dev false \
            libx11-6 libxcb1-dev libxau-dev libxdmcp-dev \
            x11proto-dev libxext-dev libxrandr-dev libxi-dev \
            libxcursor-dev libxinerama-dev
        log "X11 dev libraries installed locally"
    fi

    # Verify key tools now exist
    MISSING=()
    for cmd in gcc g++ cmake ninja git python3 curl wget zip unzip pkg-config; do
        command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
    done
    if [ ${#MISSING[@]} -gt 0 ]; then
        error "Missing required tools: ${MISSING[*]}"
        error "Install them manually or run with sudo."
        exit 1
    fi
    log "Key build tools verified"
fi

# ── GCC 14 (required for C++23 <print> header) ──
GCC14_OK=false
if gcc --version 2>/dev/null | grep -q ' 1[4-9]\.\| [2-9][0-9]\.'; then
    GCC14_OK=true
    log "GCC 14+ already installed: $(gcc --version | head -1)"
fi

if [ "$GCC14_OK" = false ]; then
    if [ "$CAN_SUDO" = true ]; then
        warn "Installing GCC 14 (required for C++23)..."
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update -qq
        sudo apt-get install -y gcc-14 g++-14
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14
        log "GCC 14 installed and set as default"
    else
        # Install GCC 14 locally from Ubuntu debs
        if [ ! -x "$LOCAL_PREFIX/bin/gcc-14" ]; then
            warn "Installing GCC 14 locally from packages..."
            local_tmpdir=$(mktemp -d)
            cd "$local_tmpdir"
            apt-get download \
                gcc-14-x86-64-linux-gnu g++-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu \
                libgcc-14-dev libstdc++-14-dev 2>/dev/null

            mkdir -p extract
            for deb in *.deb; do dpkg -x "$deb" extract 2>/dev/null || true; done

            # Binaries
            cp extract/usr/bin/x86_64-linux-gnu-gcc-14 "$LOCAL_PREFIX/bin/gcc-14"
            cp extract/usr/bin/x86_64-linux-gnu-g++-14 "$LOCAL_PREFIX/bin/g++-14"
            ln -sf gcc-14 "$LOCAL_PREFIX/bin/gcc"
            ln -sf g++-14 "$LOCAL_PREFIX/bin/g++"
            ln -sf gcc-14 "$LOCAL_PREFIX/bin/cc"
            ln -sf g++-14 "$LOCAL_PREFIX/bin/c++"

            # Compiler internals (cc1, cc1plus, etc.)
            mkdir -p "$LOCAL_PREFIX/libexec/gcc/x86_64-linux-gnu/14"
            cp -a extract/usr/libexec/gcc/x86_64-linux-gnu/14/* \
                "$LOCAL_PREFIX/libexec/gcc/x86_64-linux-gnu/14/" 2>/dev/null || true

            # Libraries (libgcc, libstdc++)
            mkdir -p "$LOCAL_PREFIX/lib/gcc/x86_64-linux-gnu"
            cp -a extract/usr/lib/gcc/x86_64-linux-gnu/14 \
                "$LOCAL_PREFIX/lib/gcc/x86_64-linux-gnu/" 2>/dev/null || true

            # C++ headers
            mkdir -p "$LOCAL_PREFIX/include/c++"
            cp -a extract/usr/include/c++/14 "$LOCAL_PREFIX/include/c++/" 2>/dev/null || true

            # Target-specific headers (bits/c++config.h)
            mkdir -p "$LOCAL_PREFIX/include/x86_64-linux-gnu/c++"
            cp -a extract/usr/include/x86_64-linux-gnu/c++/14 \
                "$LOCAL_PREFIX/include/x86_64-linux-gnu/c++/" 2>/dev/null || true

            cd /
            rm -rf "$local_tmpdir"

            # Fix broken libgomp.so symlink (common with deb extraction)
            if [ -L "$LOCAL_PREFIX/lib/gcc/x86_64-linux-gnu/14/libgomp.so" ] && \
               [ ! -e "$LOCAL_PREFIX/lib/gcc/x86_64-linux-gnu/14/libgomp.so" ]; then
                SYSTEM_GOMP=$(find /usr/lib -name 'libgomp.so.1' 2>/dev/null | head -1)
                if [ -n "$SYSTEM_GOMP" ]; then
                    ln -sf "$SYSTEM_GOMP" "$LOCAL_PREFIX/lib/gcc/x86_64-linux-gnu/14/libgomp.so"
                fi
            fi

            log "GCC 14 installed locally"
        fi
    fi
fi

# ── CMake 3.30+ ──
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
    log "CMake $(cmake --version | head -1) installed via pip"
fi

# ── Autotools (m4, autoconf, automake, libtool) ──
# These are needed by vcpkg to build libb2, libffi, python3, etc.

if ! command -v m4 &>/dev/null; then
    build_gnu_tool m4 1.4.19 "m4-1.4.19.tar.xz"
else
    log "m4 found: $(m4 --version | head -1)"
fi

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
    build_gnu_tool autoconf 2.72 "autoconf-2.72.tar.xz"
fi

if ! command -v automake &>/dev/null; then
    build_gnu_tool automake 1.17 "automake-1.17.tar.xz"
else
    log "automake found: $(automake --version | head -1)"
fi

if ! command -v libtool &>/dev/null; then
    build_gnu_tool libtool 2.5.4 "libtool-2.5.4.tar.xz"
else
    log "libtool found: $(libtool --version | head -1)"
fi

# Persist local paths for future sessions
grep -q '.local/bin' ~/.bashrc 2>/dev/null || {
    echo '' >> ~/.bashrc
    echo '# Local tools' >> ~/.bashrc
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
}

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

# OpenGL hints for non-standard locations (e.g. ~/.local)
OPENGL_ARGS=""
if [ -f "$LOCAL_PREFIX/lib/libOpenGL.so" ]; then
    OPENGL_ARGS="-DOPENGL_opengl_LIBRARY=$LOCAL_PREFIX/lib/libOpenGL.so"
    OPENGL_ARGS="$OPENGL_ARGS -DOPENGL_glx_LIBRARY=$LOCAL_PREFIX/lib/libGLX.so"
    OPENGL_ARGS="$OPENGL_ARGS -DOPENGL_INCLUDE_DIR=$LOCAL_PREFIX/include"
    OPENGL_ARGS="$OPENGL_ARGS -DOPENGL_egl_LIBRARY=$LOCAL_PREFIX/lib/libEGL.so"
    log "Using OpenGL from $LOCAL_PREFIX/lib"
fi

# Use locally-installed GCC 14 if available
CC_ARGS=""
if [ -x "$LOCAL_PREFIX/bin/gcc-14" ]; then
    CC_ARGS="-DCMAKE_C_COMPILER=$LOCAL_PREFIX/bin/gcc-14 -DCMAKE_CXX_COMPILER=$LOCAL_PREFIX/bin/g++-14"
    log "Using GCC 14 from $LOCAL_PREFIX/bin"
fi

# CMAKE_CUDA_ARCHITECTURES: target the detected GPU (see CUDA_ARCH above)
# CUDAToolkit_ROOT: explicit CUDA path avoids cmake finding wrong toolkit
# CMAKE_TOOLCHAIN_FILE: vcpkg manages C++ deps (SDL3, glm, spdlog, etc.)
# CMAKE_PREFIX_PATH: manually-installed LibTorch (not vcpkg — see known issues)
# OVERLAY_ARG: our SDL3 port disables XTEST/ibus for headless builds
# OPENGL_ARGS: points to ~/.local if GL was extracted from debs
# CC_ARGS: uses locally-installed GCC 14 if present
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_PREFIX_PATH="$REPO_DIR/external/libtorch" \
    $OVERLAY_ARG $OPENGL_ARGS $CC_ARGS

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
export LD_LIBRARY_PATH="$REPO_DIR/build:$CUDA_ROOT/lib64:$LOCAL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
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
echo "      --strategy adc \\"
echo "      --max-cap 1000000 \\"
echo "      -i 30000 \\"
echo "      -r 2 \\"
echo "      --min-opacity 0.1 \\"
echo "      --enable-sparsity \\"
echo "      --prune-ratio 0.6 \\"
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
