"""状态恢复与跨模型拒绝测试（PR-MH3，ADR-0014，规格 §9/§39，红→绿）。"""

from __future__ import annotations

import pytest


def test_restore_applies_full_state(loaded_backend) -> None:
    backend, ref = loaded_backend
    snap = backend.store.get(backend.initial_state(ref.model_ref))
    snap.update({"time": 1.5, "qpos": [0.4, 0.2], "qvel": [0.1, -0.1], "ctrl": [0.4, 0.2]})
    state_ref = backend.snapshot_state(ref.model_ref, snap)

    model, data = backend.restore_state(ref.model_ref, state_ref)
    assert data.time == pytest.approx(1.5)
    assert list(data.qvel) == pytest.approx([0.1, -0.1])


def test_cross_model_state_rejected(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)

    # patch 产生新 model（nq 相同但 digest 不同）——跨模型仍拒绝。
    result = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": "shoulder"},
                "field": "damping",
                "value": 3.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.restore_state(result.new_model_ref, state_ref)


def test_cross_model_reject_on_digest_mismatch_even_same_file(
    loaded_backend, tiny_task_root
) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    # 手工构造 digest 不符的 state（不改维度）。
    snap = backend.store.get(state_ref)
    snap["model_digest"] = "sha256:" + "0" * 64
    import hashlib

    from rosclaw.contracts.common import canonical_json
    from rosclaw.sim.refs import make_ref

    forged = make_ref("simsta", hashlib.sha256(canonical_json(snap).encode()).hexdigest())
    backend.store.put("states", snap, ref=forged)
    with pytest.raises(ValueError, match="CROSS_MODEL_REF"):
        backend.restore_state(ref.model_ref, forged)


def test_fork_state_creates_identical_branches(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    fork = backend.fork_state(ref.model_ref, state_ref, 4)

    assert fork["kind"] == "fork"
    assert fork["n"] == 4
    assert fork["base_state_ref"] == state_ref
    assert len(fork["branch_refs"]) == 4
    # 所有 branch 初始 digest 必须一致（内容寻址 ⇒ 同一 ref）。
    assert len(set(fork["branch_refs"])) == 1
    assert fork["branch_refs"][0] == state_ref
    assert [b["branch_id"] for b in fork["branches"]] == ["b0", "b1", "b2", "b3"]

    # fork 记录不可变且幂等。
    fork2 = backend.fork_state(ref.model_ref, state_ref, 4)
    assert fork["fork_ref"] == fork2["fork_ref"]


def test_fork_branch_count_budget(loaded_backend) -> None:
    backend, ref = loaded_backend
    state_ref = backend.initial_state(ref.model_ref)
    with pytest.raises(ValueError, match="SIM_BUDGET_EXCEEDED"):
        backend.fork_state(ref.model_ref, state_ref, 10_000)
    with pytest.raises(ValueError, match="SIM_BUDGET_EXCEEDED"):
        backend.fork_state(ref.model_ref, state_ref, 0)
