"""Read static connector bindings from the already compiled, matching Body."""

from rosclaw.body.resolver import BodyResolver


def configured_ros_body(robot_id: str, expected_hash: str | None = None) -> dict:
    resolver = BodyResolver(body_id=robot_id)
    if not resolver.effective_body_path.exists():
        raise ValueError("compiled Body is missing")
    effective = resolver.get_effective_body(recompile_if_stale=False)
    actual_hash = effective.compute_hash()
    if actual_hash != effective.effective_body_hash or (
        expected_hash is not None and actual_hash != expected_hash
    ):
        raise ValueError("compiled Body hash does not match the mission binding")
    return {
        "effective_body_hash": actual_hash,
        "frames": effective.frames,
        "ros_capability_bindings": effective.provider_interfaces.get("ros_capability_bindings", {}),
    }
