"""Bounded DDS port preflight; no sockets, ROS participant or action permission."""

from pathlib import Path


def validate_ros_domain(domain, *, ephemeral_range=None, participant_reserve=120):
    if type(domain) is not int or not 0 <= domain <= 232:
        raise ValueError("bounded explicit ROS domain required")
    if type(participant_reserve) is not int or not 1 <= participant_reserve <= 120:
        raise ValueError("bounded preregistered DDS participant reserve required")
    if ephemeral_range is None:
        path = Path("/proc/sys/net/ipv4/ip_local_port_range")
        with path.open("rb") as source:
            raw = source.read(65)
        if not 0 < len(raw) <= 64:
            raise ValueError("bounded actual Linux ephemeral range required")
        try:
            ephemeral_range = tuple(int(s) for s in raw.split())
        except ValueError as error:
            raise ValueError("actual Linux ephemeral range cannot be decoded") from error
    if (
        type(ephemeral_range) is not tuple
        or len(ephemeral_range) != 2
        or any(type(v) is not int for v in ephemeral_range)
        or not 1 <= ephemeral_range[0] <= ephemeral_range[1] <= 65535
    ):
        raise ValueError("typed bounded actual ephemeral port range required")
    first = 7400 + 250 * domain
    last = first + 11 + 2 * (participant_reserve - 1)
    if last > 65535 or not (last < ephemeral_range[0] or first > ephemeral_range[1]):
        raise ValueError("DDS domain/participant ports overlap ephemeral range or exceed UDP limit")
    return {
        "domain_id": domain,
        "participant_reserve": participant_reserve,
        "DDS_port_span": [first, last],
        "observed_host_ephemeral_range": list(ephemeral_range),
        "actual_container_network_namespace_verified": False,
        "transport_ready": False,
        "physical_acceptance": "NOT_MEASURED",
        "authorization": False,
    }
