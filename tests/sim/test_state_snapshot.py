"""状态快照测试（PR-MH3，ADR-0014，规格 §9/§14，红→绿）。"""

from __future__ import annotations

import pytest


def test_initial_state_deterministic(loaded_backend) -> None:
    backend, ref = loaded_backend
    s1 = backend.initial_state(ref.model_ref)
    s2 = backend.initial_state(ref.model_ref)
    assert s1 == s2  # 内容寻址幂等
    assert s1.startswith("simsta_")


def test_initial_state_payload_complete(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    snap = backend.store.get(state_ref)

    assert snap["kind"] == "state_snapshot"
    assert snap["model_ref"] == ref.model_ref
    assert snap["model_digest"] == ref.model_digest
    assert snap["time"] == 0.0
    # 完整可继续仿真状态（规格 §9），不只 qpos/qvel。
    assert snap["qpos"] == [0.0, 0.0]
    assert snap["qvel"] == [0.0, 0.0]
    assert snap["ctrl"] == [0.0, 0.0]
    for key in ("act", "mocap_pos", "mocap_quat"):
        assert key in snap


def test_snapshot_restore_recapture_equal(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    snap = backend.store.get(state_ref)
    snap["qpos"] = [0.3, -0.2]
    snap["ctrl"] = [0.3, -0.2]
    moved_ref = backend.snapshot_state(ref.model_ref, snap)
    assert moved_ref != state_ref

    model, data = backend.restore_state(ref.model_ref, moved_ref)
    assert list(data.qpos) == pytest.approx([0.3, -0.2])
    assert list(data.ctrl) == pytest.approx([0.3, -0.2])

    recaptured_ref = backend.capture_and_store(ref.model_ref, model, data)
    assert recaptured_ref == moved_ref  # 往返一致


def test_snapshot_validates_dimensions(loaded_backend) -> None:
    backend, ref = loaded_backend
    snap = backend.store.get(backend.initial_state(ref.model_ref))
    snap["qpos"] = [0.1]  # nq=2
    with pytest.raises(ValueError, match="STATE_DIMENSION"):
        backend.snapshot_state(ref.model_ref, snap)

    snap = backend.store.get(backend.initial_state(ref.model_ref))
    snap["ctrl"] = [0.0]  # nu=2
    with pytest.raises(ValueError, match="CTRL_DIMENSION"):
        backend.snapshot_state(ref.model_ref, snap)


def test_snapshot_rejects_non_finite(loaded_backend) -> None:
    backend, ref = loaded_backend
    snap = backend.store.get(backend.initial_state(ref.model_ref))
    snap["qpos"] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="STATE_INVALID"):
        backend.snapshot_state(ref.model_ref, snap)
    snap["qpos"] = [float("inf"), 0.0]
    with pytest.raises(ValueError, match="STATE_INVALID"):
        backend.snapshot_state(ref.model_ref, snap)
