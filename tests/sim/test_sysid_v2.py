"""SysID v2 / Twin Qualification 测试（MH22，讨论总纲 §20-§27，红→绿）。

MH17 证明了单参数可恢复；MH22 测真正困难的 identifiability：
- 多参数联合（S21 damping+mass / S23 kp+damping）；
- 参数相关性诊断（Jacobian rank/condition/相关对——不只
  bounds_hit；探针实证：自由摆 damping+mass 轨迹完美拟合但
  参数错（0.19/0.95 vs 0.3/1.5）→ 必须 WEAKLY_IDENTIFIABLE）；
- 噪声鲁棒（1%/5% 高斯噪声下恢复在容差内或诚实降级——绝不
  巨大错误参数却 PASS）；
- excitation 纪律（train/holdout 同激励 = 数据泄漏 → 拒绝）；
- twin promotion 门（holdout→physical audit→TWIN_CANDIDATE，
  不自动覆盖 e-URDF）。
"""

from __future__ import annotations

import numpy as np
import pytest

PEND_BASE = """<mujoco model="pend_base">
  <option timestep="0.002"/>
  <worldbody>
    <body name="rod" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="rod_g" type="capsule" size="0.02 0.25" pos="0 0 -0.25" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

SERVO_BASE = """<mujoco model="servo_base">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0.4">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="g" type="capsule" size="0.04 0.2" pos="0 0 -0.2" mass="1.2"/>
    </body>
  </worldbody>
  <actuator><position name="srv" joint="j1" kp="10" ctrlrange="-2 2"/></actuator>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "base.xml").write_text(PEND_BASE, encoding="utf-8")
    (tmp_path / "servo.xml").write_text(SERVO_BASE, encoding="utf-8")
    return MujocoBackend(tmp_path)


def _dataset(backend, truth_xml: str, qpos0_list: list[float], controller=None, noise: float = 0.0, seed: int = 42) -> str:
    """录制 truth 数据集；noise>0 时注入乘性高斯噪声（探针同款）。"""
    from pathlib import Path

    name = f"truth_{abs(hash(truth_xml)) % 99999}.xml"
    Path(backend._task_root / name).write_text(truth_xml, encoding="utf-8")
    truth = backend.load_model(name)
    sequences = []
    for q0 in qpos0_list:
        sequences.append(
            {"controller": controller or {"hold": True}, "duration_s": 0.8, "qpos0": list(q0)}
        )
    dataset_ref = backend.record_dataset(truth.model_ref, sequences=sequences)
    if noise <= 0:
        return dataset_ref
    dataset = backend.store.get(dataset_ref)
    rng = np.random.default_rng(seed)
    new_seqs = []
    for seq in dataset["sequences"]:
        trace = backend.store.get(seq["trace_ref"])
        noisy = []
        for row in trace["states"]:
            qpos = np.asarray(row["qpos"], dtype=float) * (1 + rng.normal(0, noise))
            qvel = np.asarray(row["qvel"], dtype=float) * (1 + rng.normal(0, noise))
            noisy.append({**row, "qpos": qpos.tolist(), "qvel": qvel.tolist()})
        new_ref = backend.store.put("traces", {**trace, "states": noisy})
        new_seqs.append({**seq, "trace_ref": new_ref})
    return backend.store.put("traces", {**dataset, "sequences": new_seqs})


def _spec(base_ref: str, dataset_ref: str, parameters: list[dict], train=(0, 1), holdout=(2,)) -> dict:
    return {
        "schema_version": "rosclaw.sim.sysid_spec.v1",
        "base_model_ref": base_ref,
        "dataset_ref": dataset_ref,
        "parameters": parameters,
        "train_sequences": list(train),
        "holdout_sequences": list(holdout),
    }


def test_s21_damping_mass_weakly_identifiable(backend) -> None:
    """S21：自由摆 damping+mass——轨迹完美拟合但参数错（探针：
    0.19/0.95 vs 0.3/1.5）。必须 WEAKLY_IDENTIFIABLE 且不得
    晋升 twin（弱可识别 ≠ 可信参数）。"""
    truth_xml = PEND_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
        'mass="1.0"', 'mass="1.5"'
    ).replace('pend_base', 'pend_truth')
    base = backend.load_model("base.xml")
    ds = _dataset(backend, truth_xml, [[0.6], [-0.4], [0.9]])
    receipt = backend.run_sysid(
        _spec(
            base.model_ref,
            ds,
            [
                {"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0},
                {"type": "geom_mass", "geom": "rod_g", "min": 0.5, "max": 3.0},
            ],
        )
    )
    ident = receipt["identifiability"]
    assert ident["classification"] == "WEAKLY_IDENTIFIABLE", receipt
    assert ident["weak_parameter_pairs"]  # damping↔mass 相关被点名
    assert receipt["promotion"]["twin_candidate"] is False


def test_s23_kp_damping_identifiable_and_promoted(backend) -> None:
    """S23：伺服 kp+damping 精确恢复（探针：0.6/25 精确）→
    IDENTIFIABLE + audit 过 → TWIN_CANDIDATE。"""
    truth_xml = SERVO_BASE.replace('damping="0.1"', 'damping="0.6"').replace(
        'kp="10"', 'kp="25"'
    ).replace('servo_base', 'servo_truth')
    base = backend.load_model("servo.xml")
    ds = _dataset(
        backend,
        truth_xml,
        [[0.0], [0.2], [-0.3]],
        controller={"position_targets": [0.5]},
    )
    receipt = backend.run_sysid(
        _spec(
            base.model_ref,
            ds,
            [
                {"type": "joint_damping", "joint": "j1", "min": 0.05, "max": 2.0},
                {"type": "actuator_kp", "actuator": "srv", "min": 5.0, "max": 50.0},
            ],
        )
    )
    assert receipt["identifiability"]["classification"] == "IDENTIFIABLE"
    assert receipt["parameters_after"]["j1_damping"] == pytest.approx(0.6, abs=0.05)
    assert receipt["parameters_after"]["srv_kp"] == pytest.approx(25.0, abs=1.0)
    assert receipt["promotion"]["twin_candidate"] is True
    assert receipt["promotion"]["audit_status"] in ("PASS", "WARN")
    # §27：SimulationProfile 落进 receipt（不自动覆盖 e-URDF）。
    profile = receipt["simulation_profile"]
    assert profile["simulation"]["mujoco"]["calibration"]["method"] == "mujoco_sysid"
    assert "identified" in str(profile)


def test_noise_robustness_1pct_5pct(backend) -> None:
    """§25：1%/5% 噪声下恢复在 5% 容差内（探针：0.3001/0.3005）。
    绝不巨大错误参数却 PASS。"""
    truth_xml = PEND_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
        'pend_base', 'pend_truth'
    )
    base = backend.load_model("base.xml")
    for noise in (0.01, 0.05):
        ds = _dataset(backend, truth_xml, [[0.6], [-0.4], [0.9]], noise=noise)
        receipt = backend.run_sysid(
            _spec(
                base.model_ref,
                ds,
                [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
            )
        )
        recovered = receipt["parameters_after"]["hinge_damping"]
        assert recovered == pytest.approx(0.3, rel=0.05), f"noise={noise}: {recovered}"
        assert receipt["verdict"] == "IMPROVED"


def test_excitation_identical_rejected(backend) -> None:
    """§24：train/holdout 同激励同初值 = 数据泄漏 → 拒绝。"""
    truth_xml = PEND_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
        'pend_base', 'pend_truth'
    )
    base = backend.load_model("base.xml")
    ds = _dataset(backend, truth_xml, [[0.6], [0.6], [0.6]])  # 三序列同激励
    with pytest.raises(ValueError, match="EXCITATION_INSUFFICIENT"):
        backend.run_sysid(
            _spec(
                base.model_ref,
                ds,
                [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
                train=(0, 1),
                holdout=(2,),
            )
        )


def test_identifiability_diagnostics_structure(backend) -> None:
    """诊断结构：rank/condition_number/parameter_sensitivity/
    classification 全在（§23）。"""
    truth_xml = PEND_BASE.replace('damping="0.01"', 'damping="0.3"').replace(
        'pend_base', 'pend_truth'
    )
    base = backend.load_model("base.xml")
    ds = _dataset(backend, truth_xml, [[0.6], [-0.4], [0.9]])
    receipt = backend.run_sysid(
        _spec(
            base.model_ref,
            ds,
            [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
        )
    )
    ident = receipt["identifiability"]
    assert ident["classification"] == "IDENTIFIABLE"
    assert ident["jacobian_rank"] == 1
    assert ident["condition_number"] > 0
    assert "hinge_damping" in ident["parameter_sensitivity"]
