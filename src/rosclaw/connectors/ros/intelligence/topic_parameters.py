"""Consume ROS-host topic expansion evidence; never guess a relative namespace."""

from datetime import UTC, datetime


def observed_topic_parameter(model, node_name, key, *, now=None):
    raw = model.observations.get("node_parameters", {}).get(node_name, {}).get(key)
    if type(raw) is not str or not raw:
        return None
    if raw.startswith("/"):
        return raw
    record = model.observations.get("resolved_topic_parameters", {}).get(node_name, {}).get(key, {})
    stamp = model.observations.get("parameter_captured_at", {}).get(node_name)
    if (
        record.get("complete") is not True
        or record.get("raw_value") != raw
        or record.get("source") != node_name + "/get_parameters"
        or record.get("node_name") != node_name
        or record.get("captured_at") != stamp
        or record.get("method") != "rclpy.expand_topic_name_and_validate_full_topic_name"
        or record.get("remapping_applied") is not False
        or type(record.get("expanded_name")) is not str
        or not record["expanded_name"].startswith("/")
    ):
        return None
    try:
        age = ((now or datetime.now(UTC)) - datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return None
    return record["expanded_name"] if -0.1 <= age <= 5 else None
