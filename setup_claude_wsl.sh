#!/bin/bash
# =============================================================================
# Install Claude Code in WSL2 and bootstrap the LitchShip project
#
# Run this in your WSL terminal:
#   chmod +x setup_claude_wsl.sh && ./setup_claude_wsl.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; }

REPO_DIR="$HOME/LitchShip"

echo ""
echo "============================================"
echo "  Claude Code + LitchShip WSL2 Setup"
echo "============================================"
echo ""

# ─── Step 1: Install Claude Code CLI ────────────────────────────────────────

echo "=== [1/3] Claude Code CLI ==="

if command -v claude &>/dev/null; then
    log "Claude Code already installed: $(claude --version 2>/dev/null || echo 'unknown version')"
else
    warn "Installing Claude Code CLI..."
    curl -fsSL https://claude.ai/install.sh | bash

    # Ensure it's on PATH for this session
    export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"

    if command -v claude &>/dev/null; then
        log "Claude Code installed: $(claude --version 2>/dev/null || echo 'installed')"
    else
        error "Claude Code install failed. Try manually:"
        error "  curl -fsSL https://claude.ai/install.sh | bash"
        exit 1
    fi
fi

echo ""

# ─── Step 2: Set up LitchShip repo ─────────────────────────────────────────

echo "=== [2/3] LitchShip Repository ==="

if [ -d "$REPO_DIR/.git" ]; then
    log "Repo exists at $REPO_DIR"
    cd "$REPO_DIR"
    git pull --ff-only 2>/dev/null || warn "Could not pull; using existing state"
else
    warn "Cloning LitchShip..."
    git clone https://github.com/wild-technology/LitchShip.git "$REPO_DIR"
    cd "$REPO_DIR"
    log "Cloned to $REPO_DIR"
fi

echo ""

# ─── Step 3: Create CLAUDE.md ──────────────────────────────────────────────

echo "=== [3/3] Project Context (CLAUDE.md) ==="

# Don't overwrite CLAUDE.md if it already exists — it may contain
# project-specific instructions that have been refined over time.
if [ -f "$REPO_DIR/CLAUDE.md" ]; then
    log "CLAUDE.md already exists, not overwriting"
    echo ""
    echo "============================================"
    echo "  Setup Complete!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "  cd $REPO_DIR && claude"
    echo ""
    exit 0
fi

# Gather live environment info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
CUDA_VER="not found"
CUDA_PATH="not found"
for p in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.2 /usr/local/cuda-12.8; do
    if [ -x "$p/bin/nvcc" ]; then
        CUDA_PATH="$p"
        CUDA_VER=$("$p/bin/nvcc" --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
        break
    fi
done
GCC_VER=$(gcc --version 2>/dev/null | head -1 || echo "not found")
CMAKE_VER=$(cmake --version 2>/dev/null | head -1 || echo "not found")
LIBC_VER=$(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$' || echo "unknown")
WHOAMI=$(whoami)
HOSTNAME_VAL=$(hostname)

cat > "$REPO_DIR/CLAUDE.md" << HEREDOC
# LitchShip — LichtFeld Studio WSL2 Build Project

## Environment (auto-detected)
- **WSL2 Ubuntu** (user: $WHOAMI, host: $HOSTNAME_VAL)
- **GPU:** $GPU_NAME ($GPU_MEM)
- **CUDA:** $CUDA_VER at $CUDA_PATH
- **GCC:** $GCC_VER
- **CMake:** $CMAKE_VER
- **libc:** $LIBC_VER (affects which apt repos and pre-built binaries work)
- **Python:** $(python3 --version 2>/dev/null || echo "not found")

## Data
- **Source:** Copy your COLMAP dataset to ~/data/ for best performance
- **Format:** RealityScan 2.1 export (undistorted images + COLMAP text format)
- **Layout:** images/ + sparse/0/{cameras,images,points3D}.txt

## Current Task

Build MrNeRF/gaussian-splatting-cuda (LichtFeld Studio) and run a test
training on a 30-image subset. This is a C++23/CUDA project, not the
Python gsplat implementation.

### What's done
- setup_lichtfeld.sh exists and handles: CUDA detection, GCC 14, CMake 3.30+,
  vcpkg, repo clone, LibTorch download, cmake build
- run_test_training.sh: creates 30-image subset, filters COLMAP files, runs
  3K iteration ADC training
- CUDA $CUDA_VER, GCC 14, CMake, vcpkg all confirmed working on this system

### What's in progress
- LibTorch download may fail (URL changes frequently, GLIBC $LIBC_VER compat)
- LichtFeld build not yet completed
- Test training not yet run

### Known Issues (discovered iteratively)
1. **libc is $LIBC_VER** — Kitware apt repo for cmake won't work, use \`pip3 install cmake\`
2. **CUDA path** is $CUDA_PATH (symlink), not /usr/local/cuda-12.8
3. **LibTorch URLs** change frequently — the setup script tries multiple channels
4. **sm_120** (Blackwell) CUDA kernels may need PTX fallback if native build crashes
   Use \`-DBUILD_CUDA_PTX_ONLY=ON\` as fallback
5. **Do NOT install nvidia drivers inside WSL2** — the Windows driver provides libcuda.so

## How to Proceed

Run the existing scripts in order:

\`\`\`bash
# 1. Build everything (idempotent — skips completed steps)
./setup_lichtfeld.sh

# 2. Test training (30 images, 3K iterations)
./run_test_training.sh
\`\`\`

If setup_lichtfeld.sh fails, debug the specific step that failed.
The script is in ~/LitchShip/setup_lichtfeld.sh.

## Key Files
- \`setup_lichtfeld.sh\` — full environment + build setup
- \`run_test_training.sh\` — test training with 30-image subset
- \`setup_claude_wsl.sh\` — this bootstrap script
- \`SETUP_LICHTFELD_WSL.md\` — reference documentation

## Build Commands (LichtFeld Studio)

\`\`\`bash
cd ~/gaussian-splatting-cuda
mkdir -p build && cd build
cmake .. -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_CUDA_ARCHITECTURES=120 \\
    -DCUDAToolkit_ROOT=$CUDA_PATH \\
    -DCMAKE_TOOLCHAIN_FILE=\$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake \\
    -DCMAKE_PREFIX_PATH=\$HOME/gaussian-splatting-cuda/external/libtorch
cmake --build . --config Release -j\$(nproc)
\`\`\`

## Training Command

\`\`\`bash
cd ~/gaussian-splatting-cuda
# Find binary (name may vary after build)
find build/ -type f -executable | head -10

# Test run (adjust binary path as needed)
./build/bin/lichtfeld-studio \\
    -d ~/data/H2103d_test_colmap_workflow \\
    -o ~/output/H2103d_test \\
    --strategy adc \\
    --max-cap 200000 \\
    -i 3000 -r 2
\`\`\`
HEREDOC

log "CLAUDE.md created with live environment details"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Your environment:"
echo "  GPU:   $GPU_NAME ($GPU_MEM)"
echo "  CUDA:  $CUDA_VER ($CUDA_PATH)"
echo "  GCC:   $GCC_VER"
echo "  CMake: $CMAKE_VER"
echo ""
echo "Next steps:"
echo ""
echo "  1. Authenticate Claude Code:"
echo "     cd $REPO_DIR && claude"
echo ""
echo "  2. When Claude Code starts, tell it:"
echo "     'Run setup_lichtfeld.sh, debug any failures,"
echo "      then run run_test_training.sh'"
echo ""
echo "  Claude Code will have full access to your GPU,"
echo "  filesystem, and can iterate on build issues in"
echo "  real-time without copy-pasting output."
echo ""
