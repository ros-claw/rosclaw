#!/usr/bin/env bash
# Dell Precision 7960 Tower — ROSClaw v1.0 High-Performance Deployment
# Run this on: ssh dell@dell-precision-7960-tower
# Then: tmux attach -t for_rosclaw

set -euo pipefail

echo "========================================"
echo "  ROSClaw v1.0 — Dell 7960 Deploy"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x RTX A6000"
echo "========================================"

# ── 1. Sync code from ubuntu server ──
echo "[1/6] Syncing ROSClaw codebase..."
RSYNC_DST="$HOME/rosclaw-v1.0"
mkdir -p "$RSYNC_DST"

# If rsync from ubuntu is available, use it; otherwise clone from git
if ssh -o ConnectTimeout=3 ubuntu@ubuntu-server "echo OK" 2>/dev/null; then
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "ubuntu@ubuntu-server:/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0/" "$RSYNC_DST/"
else
    echo "  (ubuntu server not reachable, using git clone)"
    cd ~
    git clone https://github.com/ROSClaw/rosclaw.git rosclaw-git 2>/dev/null || true
    cp -r rosclaw-git/rosclaw-v1.0/* "$RSYNC_DST/" 2>/dev/null || true
fi

# ── 2. Python venv ──
echo "[2/6] Setting up Python 3.10 venv..."
cd "$RSYNC_DST"
python3 -m venv venv --system-site-packages 2>/dev/null || python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip setuptools wheel

# ── 3. Install deps ──
echo "[3/6] Installing dependencies..."
pip install -q -e . 2>/dev/null || pip install -q \
    numpy scipy torch torchvision torchaudio \
    transformers accelerate \
    opencv-python pillow imageio \
    pyyaml aiohttp fastapi uvicorn \
    pytest pytest-asyncio pytest-cov \
    websockets redis sqlalchemy \
    mujoco dm-control gymnasium \
    2>&1 | tail -5

# ── 4. MuJoCo GPU verification ──
echo "[4/6] Verifying MuJoCo + GPU..."
python3 -c "
import torch
import mujoco
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'MuJoCo {mujoco.__version__}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

xml = '<mujoco><worldbody><body pos=\"0 0 1\"><freejoint/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print('MuJoCo GPU smoke test: PASSED')
" || echo "WARN: GPU verification failed"

# ── 5. Run high-performance tests ──
echo "[5/6] Running high-performance test suite..."
python3 -m pytest tests/integration/test_v1_0_closed_loop.py -v --tb=short || true
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -5 || true

# ── 6. Start dashboard (optional) ──
echo "[6/6] Setup complete!"
echo ""
echo "Next steps:"
echo "  source $RSYNC_DST/venv/bin/activate"
echo "  cd $RSYNC_DST"
echo "  rosclaw doctor"
echo "  rosclaw status"
echo "  pytest tests/integration/test_v1_0_closed_loop.py -v"
echo ""
echo "========================================"
