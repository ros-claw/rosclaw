"""Built-in robot procedures for ROSClaw V4.

Core procedures for robot operation: CONNECT, CALIBRATE, MOVE, DEBUG, RESET.
"""

from __future__ import annotations

from typing import Any

from rosclaw_core.runtime.executor import Procedure, ProcedureResult
from rosclaw_core.runtime.session import RuntimeSession, SessionState
from rosclaw_core.adapters.base import AdapterState, ControlMode


class ConnectProcedure(Procedure):
    """Connect to robot hardware.

    Parameters:
        port: Serial port path (optional, uses manifest default if not provided)
        auto_calibrate: Whether to auto-calibrate after connect (default: False)
    """

    @property
    def name(self) -> str:
        return "CONNECT"

    @property
    def description(self) -> str:
        return "Connect to robot hardware and initialize communication"

    @property
    def timeout_seconds(self) -> float:
        return 30.0

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        error = super().validate(session, params)
        if error:
            return error

        if not session.adapters:
            return "No adapters registered for session"

        return None

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Connect all adapters."""
        results = await session.connect_hardware()

        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        # Auto-calibrate if requested
        if params.get("auto_calibrate", False):
            cal_results = await session.calibrate_all()
            return {
                "connected": results,
                "calibrated": cal_results,
                "summary": f"{success_count}/{total_count} connected, "
                f"{sum(1 for v in cal_results.values() if v)}/{len(cal_results)} calibrated",
            }

        return {
            "connected": results,
            "summary": f"{success_count}/{total_count} connected",
        }


class CalibrateProcedure(Procedure):
    """Calibrate robot joints.

    Parameters:
        joints: List of joint names to calibrate (default: all)
        force: Force recalibration even if already calibrated (default: False)
    """

    @property
    def name(self) -> str:
        return "CALIBRATE"

    @property
    def description(self) -> str:
        return "Calibrate robot joint positions and limits"

    @property
    def timeout_seconds(self) -> float:
        return 300.0  # Calibration can take time

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        error = super().validate(session, params)
        if error:
            return error

        for adapter in session.adapters.values():
            if adapter.state not in (AdapterState.CONNECTED, AdapterState.READY):
                return f"Adapter {adapter.robot_id} not connected"

        return None

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Calibrate all adapters."""
        results = await session.calibrate_all()

        return {
            "calibrated": results,
            "summary": f"{sum(1 for v in results.values() if v)}/{len(results)} calibrated",
        }


class MoveProcedure(Procedure):
    """Move robot to target positions.

    Parameters:
        joint_positions: Dict of joint_name -> target_position
        velocity: Optional velocity limit (default: use manifest limits)
        acceleration: Optional acceleration limit
        wait: Whether to wait for motion complete (default: True)
        timeout: Motion timeout in seconds (default: 30)
    """

    @property
    def name(self) -> str:
        return "MOVE"

    @property
    def description(self) -> str:
        return "Move robot joints to target positions"

    @property
    def timeout_seconds(self) -> float:
        return 60.0

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        error = super().validate(session, params)
        if error:
            return error

        if "joint_positions" not in params:
            return "joint_positions required"

        # Validate positions against manifest limits
        manifest = session.manifest
        positions = params["joint_positions"]

        # Single robot validation
        if session.is_single_robot:
            errors = manifest.validate_state(positions)
            if errors:
                return f"; ".join(errors)

        return None

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute move on all adapters."""
        positions = params["joint_positions"]
        wait = params.get("wait", True)

        results = {}
        for robot_id, adapter in session.adapters.items():
            if not adapter.is_ready:
                results[robot_id] = {"error": "Adapter not ready"}
                continue

            # Write command
            success = await adapter.write_command(positions)
            results[robot_id] = {"success": success}

        return {
            "moved": results,
            "positions": positions,
            "summary": f"{sum(1 for r in results.values() if r.get('success'))}/{len(results)} succeeded",
        }


class DebugProcedure(Procedure):
    """Get debug information from robot.

    Parameters:
        level: Debug level ("basic", "full", "diagnostics")
        adapters: List of adapter IDs to debug (default: all)
    """

    @property
    def name(self) -> str:
        return "DEBUG"

    @property
    def description(self) -> str:
        return "Get diagnostic and debug information"

    @property
    def timeout_seconds(self) -> float:
        return 10.0

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect debug info from session and adapters."""
        level = params.get("level", "basic")
        adapter_ids = params.get("adapters", list(session.adapters.keys()))

        info = {
            "session": session.to_dict(),
            "adapters": {},
            "manifest": session.manifest.to_dict(),
        }

        if level in ("full", "diagnostics"):
            # Read current state from adapters
            for aid in adapter_ids:
                adapter = session.adapters.get(aid)
                if adapter:
                    try:
                        state = await adapter.read_state()
                        info["adapters"][aid] = {
                            "state": adapter.state.name,
                            "control_mode": adapter.control_mode.name,
                            "joint_states": {
                                name: {
                                    "position": js.position,
                                    "velocity": js.velocity,
                                    "effort": js.effort,
                                }
                                for name, js in state.joint_states.items()
                            },
                            "is_ready": state.is_ready,
                        }
                    except Exception as e:
                        info["adapters"][aid] = {"error": str(e)}

        if level == "diagnostics":
            info["diagnostics"] = {
                "history_count": len(session.history),
                "recent_procedures": [
                    {
                        "type": r.procedure_type,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in session.history[-5:]
                ],
            }

        return info


class ResetProcedure(Procedure):
    """Reset robot from error state.

    Parameters:
        force: Force reset even if not in error state (default: False)
        clear_history: Clear procedure history (default: False)
    """

    @property
    def name(self) -> str:
        return "RESET"

    @property
    def description(self) -> str:
        return "Reset robot from error state"

    @property
    def timeout_seconds(self) -> float:
        return 30.0

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Reset all adapters."""
        force = params.get("force", False)

        # Check if reset is needed
        if not force and session.state != SessionState.ERROR:
            return {"reset": False, "reason": "Session not in error state (use force=True)"}

        results = await session.reset_all()

        return {
            "reset": True,
            "adapters": results,
            "summary": f"{sum(1 for v in results.values() if v)}/{len(results)} reset",
        }


class TeleopProcedure(Procedure):
    """Enter/exit teleoperation mode.

    Parameters:
        action: "enter" or "exit"
        leader_mapping: Dict of follower_id -> leader_id (for assemblies)
    """

    @property
    def name(self) -> str:
        return "TELEOP"

    @property
    def description(self) -> str:
        return "Enter or exit teleoperation mode"

    @property
    def timeout_seconds(self) -> float:
        return 10.0

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        error = super().validate(session, params)
        if error:
            return error

        action = params.get("action")
        if action not in ("enter", "exit"):
            return "action must be 'enter' or 'exit'"

        return None

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle teleop mode."""
        action = params.get("action")

        results = {}
        for robot_id, adapter in session.adapters.items():
            if action == "enter":
                success = await adapter.enter_teleop()
            else:
                success = await adapter.exit_teleop()
            results[robot_id] = success

        return {
            "action": action,
            "results": results,
            "summary": f"{sum(1 for v in results.values() if v)}/{len(results)} {action}ed",
        }


class PolicyProcedure(Procedure):
    """Execute AI policy control.

    Parameters:
        policy_path: Path to trained policy checkpoint
        duration: Duration in seconds (0 = indefinite)
        frequency: Control frequency in Hz (default: 30)
    """

    @property
    def name(self) -> str:
        return "POLICY"

    @property
    def description(self) -> str:
        return "Execute AI policy for autonomous control"

    @property
    def timeout_seconds(self) -> float:
        return 3600.0  # 1 hour max

    def validate(self, session: RuntimeSession, params: dict[str, Any]) -> str | None:
        error = super().validate(session, params)
        if error:
            return error

        if "policy_path" not in params:
            return "policy_path required"

        return None

    async def execute(
        self,
        session: RuntimeSession,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Enter policy mode."""
        policy_path = params.get("policy_path")
        duration = params.get("duration", 0)
        frequency = params.get("frequency", 30)

        # Enter policy mode on all adapters
        results = {}
        for robot_id, adapter in session.adapters.items():
            success = await adapter.enter_policy()
            results[robot_id] = success

        # Note: Actual policy execution would be handled by a separate
        # policy inference service that publishes to the adapters

        return {
            "policy_path": policy_path,
            "duration": duration,
            "frequency": frequency,
            "adapters_ready": results,
            "note": "Policy execution requires external inference service",
        }
