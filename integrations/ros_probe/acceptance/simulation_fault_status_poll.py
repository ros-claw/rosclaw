"""Pure original socket-fault context review for bounded read-only status polls."""

from rosclaw.mcp.schemas.common import MCPError


def permits_unavailable_status_retry(effect, policy, ready, *, policy_sha256, error, now_ns):
    """An unavailable read is UNKNOWN; this never creates a terminal receipt."""
    if (
        not isinstance(error, MCPError)
        or error.code != "ROSCLAWD_UNAVAILABLE"
        or error.details.get("operation") != "get_action_status"
        or "rosclawd is unavailable" not in str(error)
        or "timed out" not in str(error)
        or policy.get("case") != "rosbridge_disconnect"
        or policy.get("phase") != "during_motion"
        or effect.get("schema_version") != "rosclaw.owned_rosbridge_disconnect.v1"
        or effect.get("case") != policy["case"]
        or effect.get("policy_sha256") != policy_sha256
        or effect.get("run_id") != policy["run_id"]
        or effect.get("body_snapshot_hash") != policy["body_snapshot_hash"]
        or effect.get("daemon_instance_id") != ready["daemon_instance_id"]
        or effect.get("daemon_pid") != ready["pid"]
        or effect.get("new_connections_opened") != 0
        or effect.get("physics_server_signals_sent") != 0
    ):
        return False
    before, after = effect.get("before_monotonic_ns"), effect.get("after_monotonic_ns")
    if not (type(before) is type(after) is int and 0 < before <= after <= now_ns):
        return False
    original = effect.get("original_navigation_sockets_before", [])
    closed = effect.get("original_navigation_sockets_after", [])
    return (
        len(original) == len(closed) == 4
        and {r.get("index") for r in original} == {0, 1, 2, 3}
        and {r.get("index") for r in closed} == {0, 1, 2, 3}
        and all(r.get("connected") is True for r in original)
        and all(r.get("closed") is True and r.get("websocket_released") is True for r in closed)
    )
