"""ROS 2 Action client over rosbridge（P1-B3，0824 总纲 §12/P1-B）。

不 import rclpy——rosbridge action 协议（send_goal/cancel_goal +
action_feedback/action_result 推送）。listener 线程收推送，回调经
调用方提供的 marshaller 投递（OperationManager 用
loop.call_soon_threadsafe 回事件循环——sqlite 连接不跨线程）。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from rosclaw.connectors.ros.transport.base import RosTransport

logger = logging.getLogger("rosclaw.connectors.ros.action_client")

#: action_msgs/GoalStatus 状态码。
STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6

FeedbackCallback = Callable[[dict], None]
ResultCallback = Callable[[int, dict], None]


class Ros2ActionClient:
    """rosbridge Action client（单 transport 多 goal 复用）。"""

    def __init__(self, transport: RosTransport) -> None:
        """单一 transport（rosbridge 语义：feedback/result 回到 goal
        发起连接——独立 listener 连接收不到推送）。死锁防护在
        transport 层：receive 空闲超时按有界切片返回（不持锁不死）。"
        """
        self._transport = transport
        self._goals: dict[str, str] = {}  # goal_id → action name
        self._feedback_cbs: dict[str, FeedbackCallback] = {}
        self._result_cbs: dict[str, ResultCallback] = {}
        self._lock = threading.RLock()
        self._listener: threading.Thread | None = None
        self._closed = False

    # ------------------------------------------------------------------
    def send_goal(
        self,
        *,
        action: str,
        action_type: str,
        args: dict[str, Any],
        goal_id: str,
        on_feedback: FeedbackCallback,
        on_result: ResultCallback,
    ) -> None:
        """发送 goal（异步——feedback/result 经回调推送）。"""
        with self._lock:
            self._goals[goal_id] = action
            self._feedback_cbs[goal_id] = on_feedback
            self._result_cbs[goal_id] = on_result
        result = self._transport.send({
            "op": "send_goal",
            "action": action,
            "action_type": action_type,
            "args": args,
            "feedback": True,
            "id": goal_id,
        })
        if not result.is_ok:
            with self._lock:
                self._goals.pop(goal_id, None)
                self._feedback_cbs.pop(goal_id, None)
                self._result_cbs.pop(goal_id, None)
            raise RuntimeError(f"send_goal failed: {result.error}")
        self._ensure_listener()

    def cancel_goal(self, goal_id: str) -> None:
        """请求取消（终态由 action_result(CANCELED) 确认）。"""
        with self._lock:
            action = self._goals.get(goal_id, "")
        self._transport.send({
            "op": "cancel_goal",
            "action": action,
            "id": goal_id,
        })

    def close(self) -> None:
        self._closed = True
        self._transport.close()

    # ------------------------------------------------------------------
    def _ensure_listener(self) -> None:
        with self._lock:
            if self._listener is not None and self._listener.is_alive():
                return
            self._listener = threading.Thread(
                target=self._listen_loop, daemon=True,
                name="ros2-action-listener",
            )
            self._listener.start()

    def _listen_loop(self) -> None:
        while not self._closed:
            try:
                result = self._transport.receive(timeout_sec=1.0)
            except Exception:  # noqa: BLE001 - listener 不死
                logger.debug("action listener receive error", exc_info=True)
                continue
            if result is None or not getattr(result, "is_ok", False):
                continue
            data = result.data or {}
            op = data.get("op", "")
            goal_id = str(data.get("id", ""))
            if op == "action_feedback":
                with self._lock:
                    callback = self._feedback_cbs.get(goal_id)
                if callback is not None:
                    try:
                        callback(dict(data.get("values") or {}))
                    except Exception:  # noqa: BLE001
                        logger.debug("feedback callback error", exc_info=True)
            elif op == "action_result":
                with self._lock:
                    callback = self._result_cbs.pop(goal_id, None)
                    self._feedback_cbs.pop(goal_id, None)
                    self._goals.pop(goal_id, None)
                if callback is not None:
                    values = data.get("values") or {}
                    status = int(values.get("status", 0))
                    try:
                        callback(status, dict(values.get("result") or values))
                    except Exception:  # noqa: BLE001
                        logger.debug("result callback error", exc_info=True)


__all__ = [
    "Ros2ActionClient",
    "STATUS_ABORTED",
    "STATUS_CANCELED",
    "STATUS_SUCCEEDED",
]
