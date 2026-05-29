# ROSClaw v1.0 Quick Start

## 1. Install

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
pip install -e . --break-system-packages  # or use venv
```

## 2. Verify

```bash
rosclaw doctor
rosclaw status
```

## 3. Run Tests

```bash
pytest tests/integration/test_v1_0_physical_simulation.py -v
```

## 4. Use a Robot

```python
from rosclaw.runtime.eurdf_loader import EURDFLoader
loader = EURDFLoader()
profile = loader.load("ur5e")
print(profile.embodiment.dof)
```

## 5. Physical Simulation

```python
import mujoco
from rosclaw.runtime.eurdf_loader import EURDFLoader

loader = EURDFLoader()
profile = loader.load("ur5e")
# Use profile to build MuJoCo simulation
```
