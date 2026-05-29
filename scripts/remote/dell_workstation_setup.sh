#!/usr/bin/env bash
# Dell Precision 7960 Tower — High-Performance Workstation Setup for ROSClaw v1.0
# Run this inside: ssh dell@dell-precision-7960-tower
# Then: tmux attach -t for_rosclaw
# Then: bash ~/rosclaw_dell_setup.sh

set -euo pipefail

echo "========================================"
echo "  ROSClaw v1.0 — Dell 7960 Setup"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x RTX A6000"
echo "========================================"

# ── 1. System deps ──
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git git-lfs build-essential cmake ninja-build \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libosmesa6-dev libglfw3 patchelf \
    python3-pip python3-venv python3-dev \
    ffmpeg

# ── 2. MuJoCo 3.3.0 ──
echo "[2/7] Installing MuJoCo 3.3.0..."
MUJOCO_VERSION="3.3.0"
MUJOCO_DIR="$HOME/.mujoco"
mkdir -p "$MUJOCO_DIR"
if [ ! -d "$MUJOCO_DIR/mujoco-${MUJOCO_VERSION}" ]; then
    cd /tmp
    wget -q "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
    tar -xzf "mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz" -C "$MUJOCO_DIR"
    rm "mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
fi
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${MUJOCO_DIR}/mujoco-${MUJOCO_VERSION}/lib:${LD_LIBRARY_PATH:-}"
echo "export MUJOCO_GL=egl" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"${MUJOCO_DIR}/mujoco-${MUJOCO_VERSION}/lib:\${LD_LIBRARY_PATH:-}\"" >> ~/.bashrc

# ── 3. ROSClaw repo ──
echo "[3/7] Cloning ROSClaw..."
cd ~
if [ ! -d "rosclaw" ]; then
    git clone https://github.com/ROSClaw/rosclaw.git || echo "(clone skipped or private repo)"
fi
# If local copy needed, rsync from ubuntu server later

# ── 4. Python venv ──
echo "[4/7] Creating Python 3.10 venv..."
cd ~/rosclaw/rosclaw-v1.0 2>/dev/null || mkdir -p ~/rosclaw-work
cd ~/rosclaw-work
python3 -m venv venv --system-site-packages || true
source venv/bin/activate

pip install -q --upgrade pip setuptools wheel

# ── 5. Core dependencies ──
echo "[5/7] Installing ROSClaw dependencies..."
pip install -q \
    numpy scipy torch torchvision torchaudio \
    transformers accelerate sentencepiece \
    opencv-python pillow imageio \
    pyyaml aiohttp aiofiles fastapi uvicorn \
    pytest pytest-asyncio pytest-cov \
    websockets redis sqlalchemy \
    mujoco==${MUJOCO_VERSION} \
    dm-control gymnasium \
    stable-baselines3 \
    2>&1 | tail -5

# ── 6. GPU verification ──
echo "[6/7] Verifying GPU + MuJoCo..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')

import mujoco
print(f'MuJoCo: {mujoco.__version__}')

# Quick smoke test
xml = '<mujoco><worldbody><body name=\"test\" pos=\"0 0 1\"><freejoint/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print('MuJoCo smoke test: PASSED')
"

# ── 7. tmux helper ──
echo "[7/7] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. tmux attach -t for_rosclaw"
echo "  2. source ~/rosclaw-work/venv/bin/activate"
echo "  3. cd ~/rosclaw/rosclaw-v1.0  (or rsync from ubuntu server)"
echo "  4. Run high-performance tests:"
echo "     pytest tests/integration/test_v1_0_closed_loop.py -v"
echo ""
echo "========================================"
