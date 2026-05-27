"""ROS2 Runtime Adapter.

For backends exposed as ROS2 actions or services (e.g., MoveIt, grasp pipeline).
"""

from typing import Any

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.base import RuntimeAdapter


class ROS2Runtime(RuntimeAdapter):
    """ROS2 action / service runtime adapter.

    This is a skeleton. Full implementation requires rclpy.
    """

    def __init__(
        self,
        name: str,
        action_name: str = "",
        service_name: str = "",
        timeout_sec: float = 30.0,
    ):
        super().__init__(name, config={
            "action_name": action_name,
            "service_name": service_name,
            "timeout": timeout_sec,
        })
        self.action_name = action_name
        self.service_name = service_name
        self.timeout_sec = timeout_sec
        self._node = None
        self._client = None

    async def start(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError:
            raise RuntimeError("rclpy is required for ROS2Runtime. Install ROS2.")
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(f"rosclaw_provider_{self.name}")
        # Action or service client would be created here
        self._started = True

    async def stop(self) -> None:
        if self._node:
            self._node.destroy_node()
            self._node = None
        self._started = False

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        # Skeleton: in production this sends a ROS2 action goal or service request
        raise RuntimeAdapterError(
            "ROS2Runtime.invoke() is not yet implemented",
            provider=self.name,
        )
