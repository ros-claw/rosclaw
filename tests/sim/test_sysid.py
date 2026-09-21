"""SysID / Digital Twin 测试（MH17，0916 优化 §二十五-§二十六，红→绿）。

G26 闭环：合成"真实"数据（已知参数偏差的 truth 模型）→ 官方
sysid 工具箱（nonlinear least squares + box bounds）参数识别 →
候选模型（patch 血缘派生）→ **holdout 独立复算**（绝不 train
residual 降了就升级）。

诚实纪律：识别不出来就 NO_IMPROVEMENT/identifiability warning，
绝不把 train fit 冒充 twin 升级（§26.3）。
"""

from __future__ import annotations

import pytest

#: 基座模型（"以为的"参数：阻尼 0.01）。
PENDULUM_BASE = """<mujoco model="pend_base">
  <option timestep="0.002"/>
  <worldbody>
    <body name="rod" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="rod_g" type="capsule" size="0.02 0.25" pos="0 0 -0.25" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

#: "真实世界"（同一结构，真阻尼 0.3）——SysID 要恢复的真相。
PENDULUM_TRUTH = PENDULUM_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
    'model="pend_base"', 'model="pend_truth"'
)

#: 真阻尼超界（3.0 > max 2.0）——bounds_hit/identifiability 红测试。
PENDULUM_TRUTH_BEYOND = PENDULUM_BASE.replace('damping="0.01"', 'damping="3.0"').replace(
    'model="pend_base"', 'model="pend_beyond"'
)


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "base.xml").write_text(PENDULUM_BASE, encoding="utf-8")
    (tmp_path / "truth.xml").write_text(PENDULUM_TRUTH, encoding="utf-8")
    (tmp_path / "beyond.xml").write_text(PENDULUM_TRUTH_BEYOND, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _record_dataset(backend, truth_file: str, qpos0_list: list[float]) -> str:
    """从 truth 模型录制数据集（合成真实数据）：每个初始角一段
    自由摆动轨迹。"""
    truth = backend.load_model(truth_file)
    return backend.record_dataset(
        truth.model_ref,
        sequences=[
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [q0]} for q0 in qpos0_list
        ],
    )


def test_g26_synthetic_recovery_and_holdout(backend) -> None:
    """G26 主门：阻尼 0.01 → 恢复到 0.3（±0.05）；holdout 独立
    复算显著改进；候选模型血缘根 = 基座。"""
    base = backend.load_model("base.xml")
    dataset_ref = _record_dataset(backend, "truth.xml", [0.6, -0.4, 0.9])

    receipt = backend.run_sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base.model_ref,
            "dataset_ref": dataset_ref,
            "parameters": [
                {"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0},
            ],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )

    assert receipt["schema_version"] == "rosclaw.sim.sysid_receipt.v1"
    assert receipt["verdict"] == "IMPROVED", receipt
    recovered = receipt["parameters_after"]["hinge_damping"]
    assert recovered == pytest.approx(0.3, abs=0.05)
    # holdout 独立复算（§26.3：绝不 train fit 就升级）。
    assert receipt["holdout_improvement"] > 0.5
    assert receipt["holdout_optimized_residual"] < receipt["holdout_baseline_residual"]
    # 血缘：候选是基座的 patch 派生。
    candidate = receipt["candidate_model_ref"]
    assert candidate.startswith("simmdl_")
    manifest = backend.store.get(candidate)
    assert manifest["parent_model_ref"] == base.model_ref
    # receipt 落库（可追溯）。
    assert receipt["receipt_ref"].startswith("simexp_")


def test_g26_unidentifiable_is_honest(backend) -> None:
    """诚实负例：数据集零运动（qpos0=0 静止）→ 阻尼不可识别，
    绝不升级 twin（verdict != IMPROVED）。"""
    base = backend.load_model("base.xml")
    dataset_ref = _record_dataset(backend, "truth.xml", [0.0, 0.0, 0.0])

    receipt = backend.run_sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base.model_ref,
            "dataset_ref": dataset_ref,
            "parameters": [
                {"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0},
            ],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    assert receipt["verdict"] in ("NO_IMPROVEMENT", "NOT_IDENTIFIABLE")
    assert receipt.get("candidate_model_ref") in (None, "")


def test_g26_bounds_hit_identifiability_warning(backend) -> None:
    """真值越界（3.0 > max 2.0）：恢复到边界 + bounds_hit +
    identifiability warning（不假装收敛到真相）。"""
    base = backend.load_model("base.xml")
    dataset_ref = _record_dataset(backend, "beyond.xml", [0.6, -0.4, 0.9])

    receipt = backend.run_sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base.model_ref,
            "dataset_ref": dataset_ref,
            "parameters": [
                {"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0},
            ],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    assert receipt["parameters_after"]["hinge_damping"] == pytest.approx(2.0, abs=0.01)
    assert "hinge_damping" in receipt["bounds_hit"]
    assert receipt["identifiability_warning"] is True


def test_sysid_spec_validation(backend) -> None:
    """契约校验：未知参数类型/缺字段 fail-closed。"""
    base = backend.load_model("base.xml")
    dataset_ref = _record_dataset(backend, "truth.xml", [0.6, 0.9])
    with pytest.raises(ValueError, match="SYSID_PARAM_UNSUPPORTED"):
        backend.run_sysid(
            {
                "schema_version": "rosclaw.sim.sysid_spec.v1",
                "base_model_ref": base.model_ref,
                "dataset_ref": dataset_ref,
                "parameters": [{"type": "teleport", "joint": "hinge", "min": 0, "max": 1}],
                "train_sequences": [0],
                "holdout_sequences": [1],
            }
        )
    with pytest.raises(ValueError, match="SYSID_SPEC_INVALID"):
        backend.run_sysid(
            {
                "schema_version": "rosclaw.sim.sysid_spec.v1",
                "base_model_ref": base.model_ref,
                "dataset_ref": dataset_ref,
                "parameters": [],
                "train_sequences": [0],
                "holdout_sequences": [1],
            }
        )


def test_sysid_contracts() -> None:
    """SysIDSpec/SysIDReceipt 契约：schema literal + digest 稳定。"""
    from rosclaw.sim.contracts import SysIDReceipt, SysIDSpec

    spec = SysIDSpec(base_model_ref="simmdl_" + "0" * 16)
    assert spec.schema_version == "rosclaw.sim.sysid_spec.v1"
    assert SysIDSpec(base_model_ref="simmdl_" + "0" * 16).with_digest().digest == (
        spec.with_digest().digest
    )
    receipt = SysIDReceipt()
    assert receipt.schema_version == "rosclaw.sim.sysid_receipt.v1"
    assert receipt.trust_level == "SIMULATED"
    assert receipt.usable_for_real_execution is False


def test_record_dataset_shape(backend) -> None:
    """record_dataset：内容寻址 dataset_ref + 序列轨迹齐全 +
    幂等（同参数重录同 ref）。"""
    truth = backend.load_model("truth.xml")
    ref1 = backend.record_dataset(
        truth.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": 0.2, "qpos0": [0.5]}],
    )
    ref2 = backend.record_dataset(
        truth.model_ref,
        sequences=[{"controller": {"hold": True}, "duration_s": 0.2, "qpos0": [0.5]}],
    )
    assert ref1 == ref2
    dataset = backend.store.get(ref1)
    assert dataset["kind"] == "sysid_dataset"
    assert len(dataset["sequences"]) == 1
    assert dataset["sequences"][0]["trace_ref"].startswith("simtrc_")
