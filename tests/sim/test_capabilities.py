"""MuJoCo 运行时能力探测测试（PR-MH0，ADR-0014，红→绿）。

原则：探测必须是真探测（属性/子模块/import 检查），不是版本字符串
比较，也不是硬编码真值表；必需能力缺失 fail-fast，可选能力缺失只
记录、不静默降级。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rosclaw.sim import capabilities as capsmod
from rosclaw.sim.capabilities import (
    ALL_CAPABILITIES,
    SIM_CAPABILITY_UNAVAILABLE,
    SimCapabilityError,
    probe_mujoco_capabilities,
)
from rosclaw.sim.contracts import SimulationBackendCapabilities


def test_probe_returns_structured_contract_on_this_machine() -> None:
    result = probe_mujoco_capabilities()

    assert isinstance(result, SimulationBackendCapabilities)
    assert result.schema_version == "rosclaw.sim.backend_capabilities.v1"
    assert result.backend == "mujoco"

    import mujoco

    assert result.backend_version == str(mujoco.__version__)
    assert result.platform == sys.platform
    # 12 项能力键齐全，值均为 bool。
    assert set(result.capabilities) == set(ALL_CAPABILITIES)
    assert all(isinstance(v, bool) for v in result.capabilities.values())
    # 本机（mujoco>=3.0）必需能力必须全在，否则 probe 已 fail-fast。
    assert result.missing_required == []
    # digest 已盖章。
    assert result.digest.startswith("simcap_")


def test_genuine_detection_not_hardcoded() -> None:
    """探测结果必须与解释器里真实符号存在性一致——证明不是硬编码。"""
    import mujoco

    result = probe_mujoco_capabilities()
    assert result.capabilities["mjspec"] == hasattr(mujoco, "MjSpec")
    # surfacevel 是 geom 属性（非传感器枚举）。
    expected_surfacevel = hasattr(mujoco.MjSpec().worldbody.add_geom(), "surfacevel")
    assert result.capabilities["surfacevel"] == expected_surfacevel
    assert result.capabilities["pid_actuator"] == hasattr(mujoco.mjtGain, "mjGAIN_PID")


def test_missing_required_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import mujoco

    monkeypatch.delattr(mujoco, "MjSpec")
    with pytest.raises(SimCapabilityError) as excinfo:
        probe_mujoco_capabilities()
    err = excinfo.value
    assert err.code == SIM_CAPABILITY_UNAVAILABLE
    assert "mjspec" in err.missing
    assert err.details.get("mjspec")


def test_unknown_required_name_rejected() -> None:
    with pytest.raises(ValueError, match="unknown capability"):
        probe_mujoco_capabilities(required=("no_such_capability",))


def test_optional_missing_no_silent_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """可选能力缺失：结果 available=False 且 details 记录原因，绝不静默。"""
    real_gl = capsmod._gl_backend_available

    def fake_gl(name: str) -> tuple[bool, str]:
        if name == "egl":
            return False, "libEGL not found (simulated)"
        return real_gl(name)

    monkeypatch.setattr(capsmod, "_gl_backend_available", fake_gl)
    result = probe_mujoco_capabilities(required=())
    assert result.capabilities["egl"] is False
    assert "egl" in result.details
    assert "egl" not in result.missing_required


def test_gl_probe_has_no_import_side_effect() -> None:
    """GL 探测不得比 `import mujoco` 本身多加载任何 GL 后端模块
    （#547 CI 实证：GL 模块导入会影响后续渲染的后端选择）。

    注意：`import mujoco` 自身会按 MUJOCO_GL 环境加载对应后端
    （3.13 实测），不变量是 **probe 前后集合不变**。
    """
    code = (
        "import sys; "
        "import mujoco; "
        "before = {m for m in sys.modules if m.startswith('mujoco.') and m.rsplit('.', 1)[-1] in ('egl', 'osmesa', 'glfw')}; "
        "from rosclaw.sim.capabilities import probe_mujoco_capabilities; "
        "probe_mujoco_capabilities(); "
        "after = {m for m in sys.modules if m.startswith('mujoco.') and m.rsplit('.', 1)[-1] in ('egl', 'osmesa', 'glfw')}; "
        "assert after == before, f'probe added GL modules: {after - before}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_probe_importable_without_eager_mujoco() -> None:
    """import rosclaw.sim.capabilities 不得提前加载 mujoco（子进程断言）。"""
    code = (
        "import sys; "
        "import rosclaw.sim.capabilities; "
        "assert 'mujoco' not in sys.modules, 'mujoco eagerly imported'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
