"""Tests for RealSense perception-only sandbox checks."""
from __future__ import annotations

import argparse

import pytest

from rosclaw.cli import cmd_sandbox_check


@pytest.mark.parametrize("robot_id", ["realsense-d405", "realsense-d435i", "realsense-dual"])
@pytest.mark.parametrize("action_type", ["move_base", "grasp", "reach", "joint_position", "trajectory"])
def test_actuator_actions_blocked(robot_id: str, action_type: str) -> None:
    args = argparse.Namespace(robot=robot_id, action=f'{{"type":"{action_type}"}}', trace_id=None)
    rc = cmd_sandbox_check(args)
    assert rc == 1


@pytest.mark.parametrize("robot_id", ["realsense-d405", "realsense-d435i", "realsense-dual"])
@pytest.mark.parametrize("action_type", ["capture_rgb", "capture_rgbd", "depth_health", "imu_read", "pointcloud"])
def test_sensor_actions_allowed(robot_id: str, action_type: str) -> None:
    args = argparse.Namespace(robot=robot_id, action=f'{{"type":"{action_type}"}}', trace_id=None)
    rc = cmd_sandbox_check(args)
    assert rc == 0


def test_plain_action_name_blocked() -> None:
    args = argparse.Namespace(robot="realsense-d405", action="move_base", trace_id=None)
    rc = cmd_sandbox_check(args)
    assert rc == 1


def test_plain_action_name_allowed() -> None:
    args = argparse.Namespace(robot="realsense-d405", action="capture_rgb", trace_id=None)
    rc = cmd_sandbox_check(args)
    assert rc == 0
