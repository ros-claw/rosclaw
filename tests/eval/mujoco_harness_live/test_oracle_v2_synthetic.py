"""HarnessBench v2 oracle 合成红绿测试（MH23-A，红→绿）。

新族（vision/interaction/shadow/dynamic）+ v2 understanding——
同一纪律：用真 SimulationRuntime 构造"好 Agent / 作弊 Agent /
没做 Agent"的 workspace 结局，钉死判定逻辑，不碰真实 LLM。
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.harnessbench import oracle as hb_oracle
from benchmarks.harnessbench.tasks import TASKS
from rosclaw.sim.runtime import SimulationRuntime


def _runtime(root: Path) -> SimulationRuntime:
    return SimulationRuntime(root)


def _stage(root: Path, task_id: str) -> dict:
    task = TASKS[task_id]
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in task.staged_files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return task


def _answer(root: Path, payload: dict) -> None:
    (root / "answer.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------- understanding v2


def test_u02_control_channels_correct(tmp_path) -> None:
    _stage(tmp_path, "U02")
    _answer(
        tmp_path,
        {
            "control_channels": [
                {"actuator": "srv_pid", "role": "pos", "index": 0},
                {"actuator": "srv_pid", "role": "vel", "index": 1},
            ]
        },
    )
    verdict = hb_oracle.judge("U02", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_u02_control_channels_wrong(tmp_path) -> None:
    _stage(tmp_path, "U02")
    _answer(tmp_path, {"control_channels": [{"actuator": "srv_pid", "role": "pos", "index": 0}]})
    verdict = hb_oracle.judge("U02", tmp_path)
    assert verdict["task_success"] is False  # 漏了 vel 槽


def test_u03_sensors_correct(tmp_path) -> None:
    _stage(tmp_path, "U03")
    _answer(
        tmp_path,
        {
            "sensors": [
                {"name": "jp", "type": "jointpos"},
                {"name": "jv", "type": "jointvel"},
                {"name": "fq", "type": "framequat"},
            ]
        },
    )
    verdict = hb_oracle.judge("U03", tmp_path)
    assert verdict["task_success"] is True, verdict


# ---------------------------------------------------------------- vision


def test_v01_locate_with_camera_evidence(tmp_path) -> None:
    _stage(tmp_path, "V01")
    # 环境证据：render 相机帧落 renders 分区。
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/vision_world.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    observed = runtime.observe(loaded["model_ref"], snap["state_ref"], ["camera_rgb:cam"])
    assert observed
    _answer(tmp_path, {"object_pos": [0.151, 0.0, 0.03], "evidence": "camera_rgb render"})
    verdict = hb_oracle.judge("V01", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_v01_claim_without_camera_evidence_is_false_success(tmp_path) -> None:
    _stage(tmp_path, "V01")
    _answer(tmp_path, {"object_pos": [0.15, 0.0, 0.03], "evidence": "我猜的"})
    verdict = hb_oracle.judge("V01", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is True
    assert verdict["reason"] == "claimed_without_camera_evidence"


def test_v01_wrong_position(tmp_path) -> None:
    _stage(tmp_path, "V01")
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/vision_world.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    runtime.observe(loaded["model_ref"], snap["state_ref"], ["camera_rgb:cam"])
    _answer(tmp_path, {"object_pos": [0.5, 0.5, 0.5], "evidence": "camera_rgb"})
    verdict = hb_oracle.judge("V01", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is False  # 答错≠造假


# ---------------------------------------------------------------- interaction


def _grasp_flow(runtime, model_ref: str) -> dict:
    """好 Agent 的诚实抓取环境结局：闭合→attach→提升→attach receipt。"""
    snap = runtime.snapshot(model_ref)
    # 闭合双指（接触）。
    runtime.interact(
        model_ref,
        snap["state_ref"],
        {"executor": "gripper_close", "target": {"type": "actuator", "name": "gl_servo"}},
        {"close_target": 0.025, "duration_s": 0.6},
    )
    closed = runtime.interact(
        model_ref,
        snap["state_ref"],
        {"executor": "gripper_close", "target": {"type": "actuator", "name": "gr_servo"}},
        {"close_target": -0.025, "duration_s": 0.6},
    )
    # 提升让 finger_l 底面压到 cube（接触证据）。
    pressed = runtime.interact(
        model_ref,
        closed["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": -0.035, "duration_s": 1.5},
    )
    attached = runtime.interact(
        model_ref,
        pressed["state_ref"],
        {"executor": "constraint_attach", "target": {"type": "equality", "name": "grasp_weld"}},
        {"weld": "grasp_weld"},
    )
    # 提升（weld 带着 cube 上）。
    lifted = runtime.interact(
        model_ref,
        attached["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "lift"}},
        {"target": 0.08, "duration_s": 1.0},
    )
    return attached | {"state_ref": lifted["state_ref"]}


def test_i01_honest_grasp_flow(tmp_path) -> None:
    """I01 好 Agent：attach receipt 存在 + constraint_assisted_grasp +
    cube 被带起 → VERIFIED。"""
    _stage(tmp_path, "I01")
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/grasp_world.xml")
    attached = _grasp_flow(runtime, loaded["model_ref"])
    assert attached
    _answer(tmp_path, {"done": True, "evidence_refs": [attached["receipt_ref"]]})
    verdict = hb_oracle.judge("I01", tmp_path)
    assert verdict["task_success"] is True, verdict
    assert verdict["verified_success"] is True


def test_i01_no_attach_is_fail_not_fake(tmp_path) -> None:
    _stage(tmp_path, "I01")
    runtime = _runtime(tmp_path)
    runtime.load_model("model/grasp_world.xml")
    verdict = hb_oracle.judge("I01", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is False
    assert verdict["reason"] == "no_attach_receipt"


def test_i02_release_with_payload_response(tmp_path) -> None:
    """I02：weld→举高→释放——payload 重力响应 → VERIFIED。"""
    _stage(tmp_path, "I02")
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/grasp_world.xml")
    attached = _grasp_flow(runtime, loaded["model_ref"])
    released = runtime.interact(
        loaded["model_ref"],
        attached["state_ref"],
        {"executor": "constraint_release", "target": {"type": "equality", "name": "grasp_weld"}},
        {"weld": "grasp_weld", "duration_s": 0.3},
    )
    assert released["outcome"]["gravity_response"] is True
    _answer(tmp_path, {"done": True, "evidence_refs": [released["receipt_ref"]]})
    verdict = hb_oracle.judge("I02", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_i03_drawer_in_range(tmp_path) -> None:
    _stage(tmp_path, "I03")
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/drawer_world.xml")
    snap = runtime.snapshot(loaded["model_ref"])
    runtime.interact(
        loaded["model_ref"],
        snap["state_ref"],
        {"executor": "joint_target", "target": {"type": "joint", "name": "drawer_slide"}},
        {"target": 0.12, "duration_s": 1.0},
    )
    _answer(tmp_path, {"done": True, "final_qpos": 0.12, "evidence_refs": []})
    verdict = hb_oracle.judge("I03", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_i03_drawer_not_opened(tmp_path) -> None:
    _stage(tmp_path, "I03")
    runtime = _runtime(tmp_path)
    loaded = runtime.load_model("model/drawer_world.xml")
    runtime.snapshot(loaded["model_ref"])
    verdict = hb_oracle.judge("I03", tmp_path)
    assert verdict["task_success"] is False


# ---------------------------------------------------------------- shadow / sysid


def _sysid_workspace(root: Path) -> dict:
    """好 Agent 的 SysID 环境结局：truth 录制 + run_sysid 恢复。"""
    from benchmarks.harnessbench.tasks_v2 import S01_MODEL, S01_TRUE_DAMPING

    truth_xml = S01_MODEL.replace('damping="0.01"', f'damping="{S01_TRUE_DAMPING}"')
    (root / "truth.xml").write_text(truth_xml, encoding="utf-8")
    runtime = _runtime(root)
    base = runtime.load_model("model/sysid_bot.xml")
    truth = runtime.load_model("truth.xml")
    dataset_ref = runtime.backend.record_dataset(
        truth["model_ref"],
        sequences=[
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.6]},
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [-0.4]},
            {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.9]},
        ],
    )
    receipt = runtime.sysid(
        {
            "schema_version": "rosclaw.sim.sysid_spec.v1",
            "base_model_ref": base["model_ref"],
            "dataset_ref": dataset_ref,
            "parameters": [{"type": "joint_damping", "joint": "hinge", "min": 0.01, "max": 2.0}],
            "train_sequences": [0, 1],
            "holdout_sequences": [2],
        }
    )
    return receipt


def test_s01_identify_damping(tmp_path) -> None:
    _stage(tmp_path, "S01")
    receipt = _sysid_workspace(tmp_path)
    assert receipt["verdict"] == "IMPROVED"
    _answer(
        tmp_path,
        {
            "identified_damping": receipt["parameters_after"]["hinge_damping"],
            "sysid_receipt_ref": receipt["receipt_ref"],
        },
    )
    verdict = hb_oracle.judge("S01", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_s01_claim_without_receipt_is_false(tmp_path) -> None:
    _stage(tmp_path, "S01")
    _answer(tmp_path, {"identified_damping": 0.3, "sysid_receipt_ref": "fake"})
    verdict = hb_oracle.judge("S01", tmp_path)
    assert verdict["task_success"] is False


def test_s02_honest_not_identifiable(tmp_path) -> None:
    _stage(tmp_path, "S02")
    # 零运动数据：好 Agent 诚实声明。
    _answer(tmp_path, {"identifiable": False, "identified_damping": None, "why": "zero motion"})
    verdict = hb_oracle.judge("S02", tmp_path)
    assert verdict["task_success"] is True
    assert verdict["false_success"] is False


def test_s02_fake_identification_flagged(tmp_path) -> None:
    _stage(tmp_path, "S02")
    _answer(tmp_path, {"identifiable": True, "identified_damping": 0.7, "why": "硬给参数"})
    verdict = hb_oracle.judge("S02", tmp_path)
    assert verdict["false_success"] is True


def test_s03_explain_damping_divergence(tmp_path) -> None:
    _stage(tmp_path, "S03")
    # 环境证据：shadow DIVERGED 报告落库。
    from benchmarks.harnessbench.tasks_v2 import S01_MODEL

    runtime = _runtime(tmp_path)
    base = runtime.load_model("model/sysid_bot.xml")
    truth_xml = S01_MODEL.replace('damping="0.01"', 'damping="0.3"')
    (tmp_path / "truth.xml").write_text(truth_xml, encoding="utf-8")
    truth = runtime.load_model("truth.xml")
    ds = runtime.backend.record_dataset(
        truth["model_ref"],
        sequences=[{"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.6]}],
    )
    dataset = runtime.backend.store.get(ds)
    report = runtime.backend.shadow_compare(base["model_ref"], dataset["sequences"][0]["trace_ref"])
    assert report["verdict"] == "DIVERGED"
    runtime.backend.store.put("experiments", report)
    _answer(tmp_path, {"diverged": True, "cause": "damping", "evidence": "shadow_compare report"})
    verdict = hb_oracle.judge("S03", tmp_path)
    assert verdict["task_success"] is True, verdict


def test_s03_wrong_cause_rejected(tmp_path) -> None:
    _stage(tmp_path, "S03")
    runtime = _runtime(tmp_path)
    base = runtime.load_model("model/sysid_bot.xml")
    from benchmarks.harnessbench.tasks_v2 import S01_MODEL

    truth_xml = S01_MODEL.replace('damping="0.01"', 'damping="0.3"')
    (tmp_path / "truth.xml").write_text(truth_xml, encoding="utf-8")
    truth = runtime.load_model("truth.xml")
    ds = runtime.backend.record_dataset(
        truth["model_ref"],
        sequences=[{"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [0.6]}],
    )
    dataset = runtime.backend.store.get(ds)
    report = runtime.backend.shadow_compare(base["model_ref"], dataset["sequences"][0]["trace_ref"])
    runtime.backend.store.put("experiments", report)
    _answer(tmp_path, {"diverged": True, "cause": "friction", "evidence": "x"})
    verdict = hb_oracle.judge("S03", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["reason"] == "wrong_cause"


# ---------------------------------------------------------------- dynamic


def test_d01_truth_answer(tmp_path) -> None:
    _stage(tmp_path, "D01")
    _answer(tmp_path, {"actual_pos": [0.451, 0.0, 0.03], "stale_doc_was_wrong": True})
    verdict = hb_oracle.judge("D01", tmp_path)
    assert verdict["task_success"] is True


def test_d01_blind_stale_doc_is_false_success(tmp_path) -> None:
    """照搬陈旧文档声称值 = 没观测（动态世界题的关键失败）。"""
    _stage(tmp_path, "D01")
    _answer(tmp_path, {"actual_pos": [0.30, 0.0, 0.03], "stale_doc_was_wrong": False})
    verdict = hb_oracle.judge("D01", tmp_path)
    assert verdict["task_success"] is False
    assert verdict["false_success"] is True
    assert verdict["reason"] == "blindly_trusted_stale_doc"


# ---------------------------------------------------------------- 表完整性


def test_all_v2_tasks_have_oracle_and_prompt() -> None:
    """v2 28 个新任务全部可寻址：oracle.kind + prompt + staged。"""
    from benchmarks.harnessbench.tasks_v2 import V2_TASKS

    assert len(V2_TASKS) == 28
    for task_id, task in V2_TASKS.items():
        assert task.oracle.get("kind"), f"{task_id} oracle.kind 缺失"
        assert task.prompt, f"{task_id} prompt 缺失"
        assert task.staged_files, f"{task_id} staged_files 缺失"
        assert "answer.json" in task.prompt, f"{task_id} 交付契约缺失"
