"""Verify that the ROS connector core never imports ROS Python libraries.

The whole point of the rosbridge transport is to avoid depending on rclpy
or rospy so that ROSClaw can be installed and run on machines that do not
have ROS installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROS_CONNECTOR_ROOT = Path(__file__).parent.parent.parent.parent / "src" / "rosclaw" / "connectors" / "ros"

# ROS client libraries that must never be imported in the core connector.
_FORBIDDEN_IMPORTS = {
    "rclpy",
    "rospy",
    "rospkg",
    "catkin_pkg",
    "roslib",
    "rosgraph",
    "roslaunch",
}


def _iter_python_files():
    for path in ROS_CONNECTOR_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        yield path


def _extract_imports(source: str):
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


@pytest.mark.parametrize("path", list(_iter_python_files()), ids=lambda p: str(p.relative_to(ROS_CONNECTOR_ROOT)))
def test_ros_connector_file_does_not_import_ros_python_libraries(path: Path):
    source = path.read_text(encoding="utf-8")
    imports = _extract_imports(source)
    forbidden = imports & _FORBIDDEN_IMPORTS
    assert not forbidden, f"{path} imports forbidden ROS library: {forbidden}"


def test_ros_connector_imports_core_modules_without_ros():
    """Importing the public ROS connector packages must not require rclpy/rospy."""
    from rosclaw.connectors.ros import compiler, discovery, provider, transport

    assert compiler is not None
    assert discovery is not None
    assert provider is not None
    assert transport is not None
