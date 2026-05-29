# ROSClaw v1.0 Installation Guide

## System Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for GPU acceleration)
- Linux (x86_64 or ARM64/aarch64)

## Quick Install

### Option 1: Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Option 2: System Python (Ubuntu 22.04+/24.04+)

On systems with PEP 668 (externally-managed-environment):

```bash
pip install -e . --break-system-packages
```

Or use `--user` flag:

```bash
pip install -e . --user
export PATH=$HOME/.local/bin:$PATH
```

## Platform-Specific Notes

### NVIDIA Spark (ARM64)

The system may have `python3` pointing to a uv-installed Python 3.11.
Use the system Python 3.12 explicitly:

```bash
/usr/bin/python3 -m pip install -e . --break-system-packages
/usr/bin/python3 -m pytest tests/
```

### Dell Precision 7960 (x86_64)

Standard installation works. For GPU-accelerated MuJoCo:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mujoco
```

## Verify Installation

```bash
rosclaw doctor
rosclaw status
pytest tests/ -q
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named torch` | Check `which python3` matches `which pip`. Use `/usr/bin/python3` explicitly. |
| `jieba not found` | `pip install jieba` or reinstall with updated pyproject.toml |
| MuJoCo rendering fails | `sudo apt install libglfw3 libglew2.2` |
