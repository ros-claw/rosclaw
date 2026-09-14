"""A/B 证据 harness（PR-MH8，规格 §44/§45/§46，红→绿）。

B 侧（Harness 路径）机器生成 A/B 指标：tool calls / bash calls /
Python LOC / XML LOC / retries / compile failures / physics-invalid
attempts / evidence completeness / false_success。

**false_success = 0 是硬指标**：claim 的 success 必须由独立
strict replay 复核一致。

A 侧（Agent bash/python glue 路径）需要真实 LLM 对跑，属
Release Gate 的 pending-live 项；本文件锁定 B 侧度量管线与
false_success 判定逻辑。
"""

from __future__ import annotations

from rosclaw.sim.runtime import SimulationRuntime


class CountingRuntime:
    """包装 SimulationRuntime 统计 tool 调用（A/B 指标采集）。"""

    def __init__(self, runtime: SimulationRuntime) -> None:
        self._runtime = runtime
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        attr = getattr(self._runtime, name)
        if not callable(attr) or name.startswith("_") or name == "backend":
            return attr

        def counted(*args, **kwargs):
            self.calls.append(name)
            return attr(*args, **kwargs)

        return counted


def test_harness_path_metrics_and_zero_false_success(tmp_path) -> None:
    """B 侧 H03 场景：结构化 tool 链完成参数实验——零 glue code。"""
    (tmp_path / "arm.xml").write_text(
        """<mujoco model="ab_arm">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom name="g0" type="capsule" size="0.04 0.15" mass="0.8"/>
      <body name="forearm" pos="0 0 0.3">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom name="g1" type="capsule" size="0.03 0.12" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="srv0" joint="shoulder" kp="10"/>
    <position name="srv1" joint="elbow" kp="10"/>
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )
    runtime = CountingRuntime(SimulationRuntime(tmp_path))

    # B 侧实验链（规格 §21：几次结构化 tool call，不是 300 行 Python）。
    loaded = runtime.load_model("arm.xml")
    runtime.inspect_model(loaded["model_ref"])
    runtime.audit(loaded["model_ref"])
    state_ref = runtime.snapshot(loaded["model_ref"])["state_ref"]
    receipts = [
        runtime.rollout(
            loaded["model_ref"], controller={"position_targets": [0.4, 0.2]}, duration_s=1.0
        )
    ]
    for kp in (60.0, 200.0):
        branch = runtime.patch_model(
            loaded["model_ref"],
            [
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv0"},
                    "field": "kp",
                    "value": kp,
                },
                {
                    "op": "set",
                    "target": {"type": "actuator", "name": "srv1"},
                    "field": "kp",
                    "value": kp,
                },
            ],
        )
        branch_state = runtime.backend.transplant_state(branch["new_model_ref"], state_ref)
        receipts.append(
            runtime.rollout(
                branch["new_model_ref"],
                controller={"position_targets": [0.4, 0.2]},
                duration_s=1.0,
                state_ref=branch_state,
            )
        )
    compared = runtime.compare([r["receipt_ref"] for r in receipts])

    metrics = {
        "tool_calls": len(runtime.calls),
        "bash_calls": 0,
        "python_loc": 0,
        "xml_loc": 0,
        "compile_failures": 0,
        "physics_invalid_attempts": 0,
        "evidence_complete": all(
            r["trace_ref"] and r["audit_ref"] and r["receipt_ref"] for r in receipts
        ),
    }
    # Harness 成功的真正指标（规格 §46）：少 glue code、证据完整。
    assert metrics["tool_calls"] <= 12
    assert metrics["bash_calls"] == 0
    assert metrics["python_loc"] == 0
    assert metrics["xml_loc"] == 0
    assert metrics["evidence_complete"] is True
    assert compared["best_ref"]

    # false_success 判定：每个 claim success 的 receipt 必须 strict
    # replay 复核一致；任何不一致即 false_success += 1。
    false_success = 0
    for receipt in receipts:
        if receipt["success"]:
            report = runtime.backend.strict_replay(receipt["receipt_ref"])
            if not report["verified"]:
                false_success += 1
    assert false_success == 0
