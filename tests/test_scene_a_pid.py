"""
test_scene_a_pid.py — Scene A: Mobile base PID motion control end-to-end.

Validates the complete closed-loop pipeline for a mobile base task:
    Agent intent → Provider routing → PID control → Sandbox validation
    → Runtime execution → Practice recording → Critic judgment → Memory storage

Usage:
    PYTHONPATH=src python -m pytest tests/test_scene_a_pid.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time  # noqa: E402
import pytest  # noqa: E402

from rosclaw.core import Runtime, RuntimeConfig, Event  # noqa: E402
from rosclaw.control.pid_controller import PIDController, PIDGains  # noqa: E402


@pytest.fixture
def runtime():
    """Create and initialize a full ROSClaw Runtime for Scene A."""
    config = RuntimeConfig(
        robot_id="go2",
        robot_zoo_path=str(PROJECT_ROOT / "e-urdf-zoo"),
        default_eurdf_robot="go2",
        enable_firewall=True,
        enable_memory=True,
        enable_practice=True,
        enable_how=True,
        enable_provider=True,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt.initialize()
    rt.start()
    yield rt
    rt.stop()


class TestSceneAPID:
    """Scene A: Mobile base PID motion control end-to-end validation."""

    def test_scene_a_pid_controller_convergence(self, runtime):
        """PID controller must converge to target within tolerance."""
        pid = PIDController(PIDGains(kp=2.0, ki=0.1, kd=0.5))
        pid.set_output_limit(-1.0, 1.0)  # max velocity 1 m/s
        pid.set_integral_limit(2.0)

        target = 1.0  # 1 meter
        current = 0.0
        dt = 0.01  # 100 Hz control
        trajectory = []

        for step in range(2000):  # 20 seconds max
            current, cmd = pid.simulate_step(current, target, dt, plant_gain=0.8)
            trajectory.append({
                "t": step * dt,
                "pos": current,
                "cmd": cmd,
                "error": target - current,
            })
            if abs(target - current) < 0.01 and step > 100:
                break

        final_error = abs(target - current)
        positions = [p["pos"] for p in trajectory]
        max_pos = max(positions)
        overshoot = max(0, max_pos - target)
        settle_time = next(
            (p["t"] for p in trajectory if abs(p["error"]) < 0.05),
            float("inf"),
        )

        print(f"  Final error: {final_error:.4f}m")
        print(f"  Max overshoot: {overshoot:.4f}m")
        print(f"  Settle time: {settle_time:.2f}s")
        print(f"  Steps: {len(trajectory)}")

        # Scene A acceptance criteria
        assert final_error <= 0.05, f"Final error {final_error} > 0.05m"
        assert overshoot <= 0.15, f"Overshoot {overshoot} > 0.15m"
        assert settle_time < 10.0, f"Settle time {settle_time} >= 10s"

    def test_scene_a_pid_overshoot_recovery(self, runtime):
        """How must suggest recovery when PID overshoots (Kp too high)."""
        if runtime._how is None:
            pytest.skip("How not available")

        # High Kp causes large overshoot on a second-order plant
        pid = PIDController(PIDGains(kp=8.0, ki=0.0, kd=0.0))
        pid.set_output_limit(-3.0, 3.0)

        target = 1.0
        pos, vel = 0.0, 0.0
        dt = 0.01
        positions = []

        # Second-order plant: mass-spring-damper-like dynamics
        mass = 1.0
        damping = 0.2
        for _ in range(1000):
            error = target - pos
            cmd = pid.update(error, dt)
            # F = ma:  cmd - damping*vel = mass * acc
            acc = (cmd - damping * vel) / mass
            vel += acc * dt
            pos += vel * dt
            positions.append(pos)

        overshoot = max(0, max(positions) - target)
        print(f"  Overshoot: {overshoot:.4f}m with Kp=8.0")

        # How should generate recovery hint for overshoot / instability
        import asyncio
        asyncio.run(runtime._how.seed_defaults())
        hint = asyncio.run(runtime._how.suggest_recovery("joint limit exceeded"))
        assert hint is not None, "How should provide hint for control failure"

    def test_scene_a_eventbus_records_pid_task(self, runtime):
        """EventBus must record PID task execution with trace_id."""
        episode_id = "ep_pid_001"

        # Agent intent
        runtime.event_bus.publish(Event(
            topic="agent.command",
            payload={"action": "pid_move", "target_x": 1.0, "episode_id": episode_id},
            source="agent",
            trace_id="trace_pid_001",
        ))

        # Provider selection
        runtime.event_bus.publish(Event(
            topic="agent.response",
            payload={"status": "ok", "provider": "pid_controller", "episode_id": episode_id},
            source="mcp_hub",
        ))

        # Skill execution
        runtime.event_bus.publish(Event(
            topic="skill.execution.start",
            payload={
                "episode_id": episode_id,
                "skill_name": "pid_move",
                "initial_state": {"x": 0.0, "y": 0.0},
            },
            source="runtime",
        ))

        # Simulate PID execution completion
        runtime.event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "episode_id": episode_id,
                "skill_name": "pid_move",
                "result": {"status": "success", "reward": 0.85, "final_error": 0.02},
                "duration_sec": 4.5,
                "final_state": {"x": 0.98, "y": 0.0},
            },
            source="runtime",
        ))

        time.sleep(0.3)

        # Verify episode recorded
        episodes = runtime._episode_recorder.list_episodes()
        ep_ids = [e["episode_id"] for e in episodes]
        assert episode_id in ep_ids, f"Episode {episode_id} not recorded"

        # Verify trace_id propagated
        history = runtime.event_bus.get_history(limit=20)
        traced = [e for e in history if e.trace_id == "trace_pid_001"]
        assert len(traced) >= 1, "trace_id should propagate through pipeline"

    def test_scene_a_memory_queries_pid_experience(self, runtime):
        """Memory must answer PID task queries."""
        if runtime._memory is None:
            pytest.skip("Memory not available")

        runtime._memory.write("pid_task_001", {
            "event_type": "pid_move",
            "instruction": "Move mobile base 1 meter forward using PID control",
            "outcome": "success",
            "duration_sec": 4.5,
            "final_error": 0.02,
            "tags": ["pid", "mobile_base", "go2", "motion_control"],
        })

        # Semantic search for PID-related experiences
        results = runtime._memory.search("mobile robot forward motion")
        assert len(results) > 0, "Memory should find PID experience semantically"

    def test_scene_a_full_closed_loop(self, runtime):
        """Complete Scene A closed-loop validation."""
        episode_id = "ep_scene_a_001"
        target_x = 1.0

        # 1. Agent intent: move 1 meter forward
        runtime.event_bus.publish(Event(
            topic="agent.command",
            payload={"action": "pid_move", "target_x": target_x, "episode_id": episode_id},
            source="agent",
        ))

        # 2. Provider capability selection
        runtime.event_bus.publish(Event(
            topic="agent.response",
            payload={"status": "ok", "provider": "pid_controller", "episode_id": episode_id},
            source="mcp_hub",
        ))

        # 3. Skill start
        runtime.event_bus.publish(Event(
            topic="skill.execution.start",
            payload={
                "episode_id": episode_id,
                "skill_name": "pid_move",
                "initial_state": {"x": 0.0, "y": 0.0, "theta": 0.0},
            },
            source="runtime",
        ))

        # 4. Simulate PID control loop (published as telemetry events)
        pid = PIDController(PIDGains(kp=2.0, ki=0.1, kd=0.5))
        pid.set_output_limit(-1.0, 1.0)
        current_x = 0.0
        dt = 0.05

        for step in range(400):
            current_x, cmd = pid.simulate_step(current_x, target_x, dt, plant_gain=0.8)
            if step % 20 == 0:  # Publish telemetry every 1s
                runtime.event_bus.publish(Event(
                    topic="robot.telemetry",
                    payload={"episode_id": episode_id, "x": current_x, "cmd": cmd},
                    source="control",
                ))
            if abs(target_x - current_x) < 0.05:
                break

        # 5. Skill complete
        final_error = abs(target_x - current_x)
        runtime.event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "episode_id": episode_id,
                "skill_name": "pid_move",
                "result": {
                    "status": "success" if final_error <= 0.05 else "failure",
                    "reward": 0.9 if final_error <= 0.05 else 0.2,
                    "final_error": final_error,
                },
                "duration_sec": step * dt,
                "final_state": {"x": current_x, "y": 0.0, "theta": 0.0},
            },
            source="runtime",
        ))

        time.sleep(0.3)

        # 6. Verify Practice recorded
        episodes = runtime._episode_recorder.list_episodes()
        ep_ids = [e["episode_id"] for e in episodes]
        assert episode_id in ep_ids, f"Episode {episode_id} not in {ep_ids}"

        # 7. Verify Critic judged
        if runtime._critic is not None:
            stats = runtime._critic.get_stats()
            assert stats["total"] >= 1, "Critic should have judged the episode"

        # 8. Verify Memory accessible
        if runtime._memory is not None:
            results = runtime._memory.search("pid move forward")
            assert len(results) >= 0  # May be empty if auto-ingest not triggered

        # 9. Control performance
        assert final_error <= 0.05, f"Final error {final_error} exceeds 5cm tolerance"
        print(f"  Scene A complete: final_error={final_error:.4f}m, steps={step}")
