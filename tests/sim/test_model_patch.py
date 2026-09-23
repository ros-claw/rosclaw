"""MjSpec 结构化模型补丁测试（PR-MH2，ADR-0014，规格 §12.3/§8/§39，红→绿）。

P0 只开放 set 白名单；add/remove/attach 显式拒绝。母模型不可变。
"""

from __future__ import annotations

import math

import pytest


@pytest.fixture
def loaded(loaded_backend):
    return loaded_backend


def _patch(op="set", target=None, field="damping", value=5.0):
    return {
        "op": op,
        "target": target or {"type": "joint", "name": "shoulder"},
        "field": field,
        "value": value,
    }


# --- set 白名单 roundtrip ---------------------------------------------------


def test_set_joint_damping_roundtrip(loaded) -> None:
    backend, ref = loaded
    parent_manifest_bytes = backend.store.resolve(ref.model_ref).read_bytes()

    result = backend.patch_model(ref.model_ref, [_patch(value=5.0)])
    assert result.ok is True
    assert result.new_model_ref and result.new_model_ref != ref.model_ref

    child = backend.compile_model(result.new_model_ref)
    shoulder = next(j for j in child.detail["joints_detail"] if j["name"] == "shoulder")
    assert shoulder["damping"] == pytest.approx(5.0)

    # 母模型分毫不动。
    assert backend.store.resolve(ref.model_ref).read_bytes() == parent_manifest_bytes
    assert backend.inspect_model(ref.model_ref).detail["joints_detail"][0][
        "damping"
    ] == pytest.approx(0.5)


def test_set_joint_range(loaded) -> None:
    backend, ref = loaded
    result = backend.patch_model(ref.model_ref, [_patch(field="range", value=[-1.0, 1.0])])
    child = backend.compile_model(result.new_model_ref)
    shoulder = next(j for j in child.detail["joints_detail"] if j["name"] == "shoulder")
    assert shoulder["range"] == pytest.approx([-1.0, 1.0])


def test_set_actuator_kp_kv(loaded) -> None:
    backend, ref = loaded
    patches = [
        _patch(target={"type": "actuator", "name": "shoulder_servo"}, field="kp", value=40.0),
        _patch(target={"type": "actuator", "name": "shoulder_servo"}, field="kv", value=4.0),
    ]
    result = backend.patch_model(ref.model_ref, patches)
    child = backend.compile_model(result.new_model_ref)
    servo = next(a for a in child.detail["actuators_detail"] if a["name"] == "shoulder_servo")
    assert servo["kp"] == pytest.approx(40.0)
    assert servo["kv"] == pytest.approx(4.0)


def test_set_geom_friction_and_rgba(loaded) -> None:
    backend, ref = loaded
    patches = [
        _patch(
            target={"type": "geom", "name": "base_geom"}, field="friction", value=[0.8, 0.01, 0.001]
        ),
        _patch(
            target={"type": "geom", "name": "base_geom"}, field="rgba", value=[1.0, 0.0, 0.0, 1.0]
        ),
    ]
    result = backend.patch_model(ref.model_ref, patches)
    child = backend.compile_model(result.new_model_ref)
    geom = next(g for g in child.detail["geoms"] if g["name"] == "base_geom")
    assert geom["friction"][0] == pytest.approx(0.8)
    assert geom["rgba"][0] == pytest.approx(1.0)


def test_set_body_pos_and_option_timestep(loaded) -> None:
    backend, ref = loaded
    patches = [
        _patch(target={"type": "body", "name": "forearm"}, field="pos", value=[0.0, 0.0, 0.5]),
        _patch(target={"type": "option"}, field="timestep", value=0.001),
    ]
    result = backend.patch_model(ref.model_ref, patches)
    child = backend.compile_model(result.new_model_ref)
    assert child.detail["options"]["timestep"] == pytest.approx(0.001)
    forearm = next(b for b in child.detail["bodies"] if b["name"] == "forearm")
    assert forearm["pos"][2] == pytest.approx(0.5)


def test_set_geom_pos(loaded) -> None:
    """geom.pos 入白名单（G39 live 试点实证：hidden_overlap 类修复
    需要移动 geom——patch 白名单不含 geom.pos 时 R01 B 腿不可赢，
    诚实 Agent 只能写文件 → 血缘拒绝 → 必然 false_success）。
    加白后修复可经 patch 血缘表达。"""
    backend, ref = loaded
    patches = [
        _patch(target={"type": "geom", "name": "base_geom"}, field="pos", value=[0.1, 0.0, 0.0])
    ]
    result = backend.patch_model(ref.model_ref, patches)
    assert result.new_model_ref != ref.model_ref
    parent = backend.store.get(result.new_model_ref)["parent_model_ref"]
    assert parent == ref.model_ref  # 血缘可追溯（修复必须基于原模型）


def test_set_geom_pos_invalid_shape(loaded) -> None:
    backend, ref = loaded
    with pytest.raises(ValueError, match="MODEL_PATCH_INVALID|MODEL_FIELD_UNSUPPORTED"):
        backend.patch_model(
            ref.model_ref,
            [_patch(target={"type": "geom", "name": "base_geom"}, field="pos", value=[0.1, 0.2])],
        )


def test_patch_lineage_recorded(loaded) -> None:
    backend, ref = loaded
    patches = [_patch(value=2.5)]
    result = backend.patch_model(ref.model_ref, patches)
    assert result.patch_digest.startswith("simpat_")

    manifest = backend.store.get(result.new_model_ref)
    assert manifest["parent_model_ref"] == ref.model_ref
    assert manifest["patches"] == patches
    assert manifest["backend"] == "mujoco"
    assert manifest["backend_version"]
    # created_at 来自不可变 manifest 的落盘时间。
    child_ref = backend.describe_model(result.new_model_ref)
    assert child_ref.created_at


def test_patch_idempotent(loaded) -> None:
    backend, ref = loaded
    patches = [_patch(value=3.3)]
    r1 = backend.patch_model(ref.model_ref, patches)
    r2 = backend.patch_model(ref.model_ref, patches)
    assert r1.new_model_ref == r2.new_model_ref


# --- 拒绝面（fail closed，规格 §39） ---------------------------------------


def test_reject_add_remove_attach(loaded) -> None:
    backend, ref = loaded
    for op in ("add", "remove", "attach"):
        with pytest.raises(ValueError, match="MODEL_FIELD_UNSUPPORTED"):
            backend.patch_model(ref.model_ref, [_patch(op=op)])


def test_reject_unknown_op(loaded) -> None:
    backend, ref = loaded
    with pytest.raises(ValueError, match="MODEL_PATCH_INVALID"):
        backend.patch_model(ref.model_ref, [_patch(op="mutate")])


def test_reject_field_outside_whitelist(loaded) -> None:
    backend, ref = loaded
    with pytest.raises(ValueError, match="MODEL_FIELD_UNSUPPORTED"):
        backend.patch_model(ref.model_ref, [_patch(field="stiffness", value=1.0)])


def test_target_not_found(loaded) -> None:
    backend, ref = loaded
    with pytest.raises(ValueError, match="MODEL_TARGET_NOT_FOUND"):
        backend.patch_model(ref.model_ref, [_patch(target={"type": "joint", "name": "ghost"})])


def test_reject_nan_inf(loaded) -> None:
    backend, ref = loaded
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="MODEL_PATCH_INVALID"):
            backend.patch_model(ref.model_ref, [_patch(value=bad)])


def test_reject_bad_quaternion(loaded) -> None:
    backend, ref = loaded
    patch = _patch(
        target={"type": "body", "name": "forearm"}, field="quat", value=[1.0, 1.0, 1.0, 1.0]
    )
    with pytest.raises(ValueError, match="MODEL_PATCH_INVALID"):
        backend.patch_model(ref.model_ref, [patch])


def test_reject_inverted_range(loaded) -> None:
    backend, ref = loaded
    with pytest.raises(ValueError, match="MODEL_PATCH_INVALID"):
        backend.patch_model(ref.model_ref, [_patch(field="range", value=[1.0, -1.0])])


def test_reject_rgba_out_of_bounds(loaded) -> None:
    backend, ref = loaded
    patch = _patch(
        target={"type": "geom", "name": "base_geom"}, field="rgba", value=[2.0, 0.0, 0.0, 1.0]
    )
    with pytest.raises(ValueError, match="MODEL_PATCH_INVALID"):
        backend.patch_model(ref.model_ref, [patch])


def test_compile_failure_preserves_parent(loaded, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, ref = loaded
    parent_manifest_bytes = backend.store.resolve(ref.model_ref).read_bytes()
    before = set(backend.store.list_children("models"))

    import mujoco

    original = mujoco.MjSpec.compile

    def boom(self):  # noqa: ANN001, ANN202
        raise ValueError("Error: simulated compile failure")

    monkeypatch.setattr(mujoco.MjSpec, "compile", boom)
    try:
        with pytest.raises(ValueError, match="MODEL_COMPILE_FAILED"):
            backend.patch_model(ref.model_ref, [_patch(value=5.0)])
    finally:
        monkeypatch.setattr(mujoco.MjSpec, "compile", original)

    assert backend.store.resolve(ref.model_ref).read_bytes() == parent_manifest_bytes
    assert set(backend.store.list_children("models")) == before
