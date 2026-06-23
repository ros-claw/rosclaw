"""MCAP topic allowlist/denylist redactor."""

from __future__ import annotations


class McapRedactor:
    """Filter MCAP/rosbag topics according to the configured allow/deny lists."""

    DEFAULT_ALLOW_TOPICS = frozenset([
        "/joint_states",
        "/imu",
        "/odom",
        "/sandbox/decision",
        "/rosclaw/runtime/event",
        "/rosclaw/provider/perf",
        "/rosclaw/skill/outcome",
    ])

    DEFAULT_DENY_TOPICS = frozenset([
        "/camera",
        "/rgb",
        "/depth",
        "/audio",
        "/microphone",
        "/pointcloud",
    ])

    def __init__(
        self,
        allow_topics: set[str] | None = None,
        deny_topics: set[str] | None = None,
    ) -> None:
        self.allow_topics = set(allow_topics or self.DEFAULT_ALLOW_TOPICS)
        self.deny_topics = set(deny_topics or self.DEFAULT_DENY_TOPICS)

    def filter_topics(self, topics: list[str]) -> list[str]:
        """Return topics allowed for upload."""
        result = []
        for topic in topics:
            if any(topic.startswith(deny) for deny in self.deny_topics):
                continue
            if any(topic.startswith(allow) for allow in self.allow_topics):
                result.append(topic)
        return result

    def is_allowed(self, topic: str) -> bool:
        if any(topic.startswith(deny) for deny in self.deny_topics):
            return False
        return any(topic.startswith(allow) for allow in self.allow_topics)
