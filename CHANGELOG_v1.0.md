# ROSClaw v1.0 Release Notes

**Version**: 1.0.0  
**Release Date**: 2026-05-30  
**Status**: General Availability (GA)  
**Codename**: Grounding Runtime

---

## 🌟 Highlights

ROSClaw v1.0 is the **first general-availability release** of the open
infrastructure for physical intelligence. After 20 sprints of architecture
freeze, integration audits, and **417 empirical benchmark runs** across 52
Frontier-Engineering tasks, the foundation is stable enough for production
embodied-AI deployments.

### Headline numbers

| Metric | Value |
|--------|-------|
| Public modules | **16** |
| Unit + integration tests | **419** (zero regression) |
| Public API surface | locked under semver |
| Wheel size | **177 KB** |
| Source dist size | 1.3 MB |
| Supported Python | 3.10, 3.11, 3.12 |
| Optional ROS 2 distros | Humble, Jazzy |

### Six Grounding Engines (all production-ready)

1. **Physical (`e_urdf`)** — robot DNA: kinematic + dynamic envelopes, soft
   limits, sensor mounts
2. **Action (`firewall`)** — `DigitalTwinFirewall` validates every motion in
   MuJoCo *before* execution; STRICT / MODERATE / PERMISSIVE safety levels
3. **Timeline (`practice`)** — 1 kHz sensorimotor + 1 Hz CoT capture on a
   unified time axis with 100× MCAP compression
4. **Experience (`memory`)** — SeekDB-backed hippocampus with Object
   Permanence, DTW trajectory search, three-axis cognitive query
5. **Skill (`skill_manager`)** — embodiment-portable skills with
   precondition checking and JSON / programmed loaders
6. **Collaboration (`swarm`)** — multi-robot EventBus coordination with FIFO
   task allocation (auction-based in v1.1)

### 🧠 Know-How Skill System (validated)

The v1.0 know-how layer is the project's biggest empirical win. After 16
iterations (v1 → v16, plus the rosclaw-how Phase 1 implementation), the
champion is a single **600-character task-agnostic deadlock-breaker toolkit**:

```text
[STUCK TOOLKIT — consult ONLY if your improvements have stalled]
• Score oscillates / diverges → damping (lower lr, gradient clip, anti-windup)
• NaN/Inf appears → numerical safeguards (epsilon, log1p, clip values)
• Memory blows up → bounded buffer (sliding window, circular cache)
• Slow convergence → schedule annealing (cosine lr, warmup, restart)
• Local optima → diversify (multi-start, perturbation, basin hopping)
• Saturation / clipping → soft limit (tanh, sigmoid bound)
Otherwise: ignore this toolkit and keep exploring.
```

Measured impact on Frontier-Engineering (mean of multiple runs):

| Task | NoKB baseline | v15 deadlock-breaker | **Δ** |
|------|--------------|---------------------|-------|
| **MallocLab** | 51.33 (n=3) | **78.75** (n=4) | **+53.4 %** |
| **PIDTuning** | 0.09 (n=5) | **0.10** (n=2) | **+13.3 %** |
| **AES-128** | 31.53 | **35.08** | **+11.3 %** |
| CVaR Stress | 95.76 | 100.00 | +4.4 % |

Full research log: [`docs/ROSCLAW_V1_RESEARCH.md`](docs/ROSCLAW_V1_RESEARCH.md).

---

## 📦 What's in the Box

### Code

- 16 production modules under `src/rosclaw/`
- 419 tests under `tests/` (run with `make test`)
- 2 demo scripts under `examples/`
- 4 tutorials under `tutorials/`

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Vision, architecture, quickstart |
| `CHANGELOG.md` | Per-release change log |
| `CHANGELOG_v1.0.md` | **This document** — GA release notes |
| `docs/API_REFERENCE.md` | Full public API |
| `docs/MODULES.md` | Per-module usage guide |
| `docs/DELL_DEPLOY.md` | Dell 7960 deployment quickstart |
| `docs/DEPLOY_DELL_7960.md` | Long-form Dell 7960 deployment guide |
| `docs/ROSCLAW_V1_RESEARCH.md` | KB-injection 19-version research log |
| `docs/BENCHMARK.md` | Performance baselines |
| `docs/SECURITY_AUDIT.md` | Threat model |
| `docs/INTEGRATION_GUIDE.md` | Agent integration |
| `docs/OPENCLAW_INTEGRATION.md` | OpenClaw bridge |
| `docs/RELEASE_CHECKLIST_v1.0.md` | GA release checklist |

### Artifacts

```
dist/rosclaw-1.0.0-py3-none-any.whl   (177 KB)
dist/rosclaw-1.0.0.tar.gz             (1.3 MB)
```

Smoke test: **16 / 16 modules import successfully** in a clean venv after
`pip install rosclaw==1.0.0`.

---

## 🚀 Quickstart

```bash
# Option A — from PyPI (after GA push)
pip install rosclaw==1.0.0

# Option B — from source
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
bash scripts/install.sh
./rosclaw doctor
```

Minimal Python entry point:

```python
from rosclaw.core import Runtime, RuntimeConfig

config = RuntimeConfig(
    robot_id="ur5e",
    robot_zoo_path="./e-urdf-zoo",
    default_eurdf_robot="ur5e",
    enable_firewall=True,
    enable_memory=True,
    enable_practice=True,
    enable_how=True,
)
runtime = Runtime(config); runtime.initialize(); runtime.start()
```

MCP wiring for Claude Code / OpenClaw — see [`README.md` § Quick Start](README.md#quick-start).

---

## ⚠️ Known Limitations

These are the limits we know about. v1.1 will address most of them.

1. **v16 monkey-patch state-aware injection** does not propagate across
   `ProcessPoolExecutor` workers. Use `kb_rosclaw_how_api` HTTP service for
   true state-aware injection in multi-worker setups.
2. **`rosclaw.swarm`** uses FIFO task allocation only; auction-based ships
   in v1.1.
3. **`ROS2Driver` and `SerialDriver`** are stubs ready for community
   contribution; full hardware drivers ship in v1.1.
4. **Single-run benchmark variance is ~30 %**. Production evaluations should
   average ≥ 3 runs.
5. **MuJoCo OpenGL via SSH**: set `MUJOCO_GL=osmesa` to avoid X11 errors.

---

## 🔐 Security

- All joint positions validated for type, finiteness, and bounds (≤ 1e5).
- `SkillRegistry` rejects empty names and non-`SkillEntry` types.
- LLM API keys read from environment; never logged.
- Firewall enforces hardware soft limits even in PERMISSIVE mode.
- MCP rate limiting and per-tenant usage tracking enabled by default.

Full threat model: [`docs/SECURITY_AUDIT.md`](docs/SECURITY_AUDIT.md).

---

## 🛣️ What's Next

### v1.1 (target Q3 2026)

- Sprint 7: Evolution Loop (Flywheel + Auto)
- Sprint 8: Swarm DDS-native reflex handshake
- Production ROS 2 driver with TF tree
- Auction-based swarm allocation
- Memory replay learning

### v1.2 (target Q4 2026)

- Sprint 9: Darwin Arena evaluation harness
- Cross-embodiment skill transfer benchmark
- Federated skill marketplace prototype

---

## 🙏 Acknowledgements

- The Frontier-Engineering team for the rigorous benchmark suite
- MuJoCo project for the simulation backbone
- Anthropic for MCP (Model Context Protocol) and Claude Code
- Every developer who filed an issue during the v0.x sprint cycle

---

## 📞 Get Help

- Issues: https://github.com/ros-claw/rosclaw/issues
- Docs: https://rosclaw.io
- Email: team@rosclaw.io

When filing an issue, please attach:

```bash
rosclaw --version
uname -a
python3 --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader  # if GPU
```

---

*Released 2026-05-30 — Grounding AGI into the Physical World.*
