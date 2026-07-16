"""Runtime.execute() full closed-loop validation.

Verifies the 11-step execution pipeline:
1. skill.execution.start
2. provider inference
3. sandbox firewall check
4. provider.inference.completed
5. trajectory generation
6. sandbox events
7. skill.execution.complete
8. critic evaluation
9. praxis.completed/failed
10. Memory auto-ingest
11. dashboard.trace.updated
"""


class TestRuntimeClosedLoop:
    """End-to-end closed loop via Runtime.execute()."""

    def test_execute_ok_creates_full_trace(self, tmp_path):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=True,
            enable_memory=True,
            enable_practice=True,
        )
        runtime = Runtime(config)
        runtime.initialize()

        action = {
            "request_id": "req_001",
            "skill_name": "reach",
            "instruction": "Reach to target point",
            "capability": "skill.reach",
            "parameters": {"target_pose": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]},
        }

        result = runtime.execute(action)

        assert result["status"] == "ok"
        assert "trajectory" in result
        assert "trajectory_data" in result
        assert len(result["trajectory_data"]) > 0
        assert "final_position" in result

        bus = runtime.event_bus
        # Full Trace emits start/completion records for grounding, provider,
        # sandbox, robot-state, critic, and write-back spans.
        history = bus.get_history(limit=200)
        topics = [e.topic for e in history]

        assert "skill.execution.start" in topics
        assert "rosclaw.provider.inference.completed" in topics
        assert "rosclaw.sandbox.episode.started" in topics
        assert "rosclaw.sandbox.action.allowed" in topics
        assert "skill.execution.complete" in topics
        assert "rosclaw.critic.success.detected" in topics
        assert "praxis.completed" in topics
        assert "rosclaw.dashboard.trace.updated" in topics

        mem = runtime.memory
        assert mem is not None
        stats = mem.get_statistics()
        assert stats["total_experiences"] >= 1

        similar = mem.find_similar_experiences("reach target point", limit=3)
        assert len(similar) >= 1

        runtime.stop()

    def test_execute_blocked_creates_failure_trace(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=True,
            enable_memory=True,
        )
        runtime = Runtime(config)
        runtime.initialize()

        # Use trajectory with joint position far beyond UR5e limits (±6.28)
        action = {
            "request_id": "req_002",
            "skill_name": "reach",
            "instruction": "Reach through table",
            "capability": "skill.reach",
            "parameters": {"target_pose": [0.5, 0.0, -0.1, 0.0, 0.0, 0.0]},
            "trajectory": [
                [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # way beyond joint limit
            ],
        }

        result = runtime.execute(action)

        # Accept either direct "blocked" from sandbox_check or "error" from validate_trajectory
        assert result["status"] in ("blocked", "error"), f"Unexpected status: {result['status']}"

        bus = runtime.event_bus
        history = bus.get_history(limit=200)
        topics = [e.topic for e in history]

        assert "skill.execution.start" in topics
        assert "firewall.action_blocked" in topics
        assert "praxis.failed" in topics
        assert "rosclaw.dashboard.trace.updated" in topics

        mem = runtime.memory
        failure = mem.explain_last_failure()
        assert failure is not None

        runtime.stop()

    def test_execute_memory_can_answer_questions(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=True,
            enable_memory=True,
        )
        runtime = Runtime(config)
        runtime.initialize()

        runtime.execute(
            {
                "request_id": "req_003",
                "skill_name": "pid_move",
                "instruction": "Move turtlebot 1 meter forward",
                "capability": "skill.pid_move",
                "parameters": {"target": 1.0},
            }
        )

        runtime.execute(
            {
                "request_id": "req_004",
                "skill_name": "reach",
                "instruction": "Reach to tabletop object",
                "capability": "skill.reach",
                "parameters": {"target_pose": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]},
            }
        )

        mem = runtime.memory
        stats = mem.get_statistics()
        assert stats["total_experiences"] >= 2

        # Query using keywords that match stored instructions
        similar = mem.find_similar_experiences("move forward turtlebot", limit=3)
        assert len(similar) >= 1, (
            f"Expected >=1 match for 'move forward turtlebot', got {len(similar)}"
        )

        similar = mem.find_similar_experiences("reach tabletop object", limit=3)
        assert len(similar) >= 1, (
            f"Expected >=1 match for 'reach tabletop object', got {len(similar)}"
        )

        runtime.stop()
