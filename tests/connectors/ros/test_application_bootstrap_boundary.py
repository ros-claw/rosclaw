"""The bootstrap compiler must stay an offline proposal, outside runtime I/O."""

import ast
from pathlib import Path


def test_bootstrap_has_no_ros_runtime_process_network_or_file_io():
    path = (
        Path(__file__).resolve().parents[3]
        / "src/rosclaw/connectors/ros/engineering/application_bootstrap.py"
    )
    tree = ast.parse(path.read_text())
    forbidden = {"rclpy", "rospy", "subprocess", "socket", "requests", "httpx", "os", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] not in forbidden for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "__import__", "Runtime"}
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "publish",
                    "send_goal",
                    "request_action",
                    "write_text",
                    "write_bytes",
                    "register_executor",
                    "register_driver",
                }
