"""Tests for the D435iCapture.start escalation ladder (2026-08-04).

The lifecycle changed from "hardware_reset FIRST, always" (the 2026-07
wedge lesson) to the ladder (plain start -> ONE reset only on failure)
after reset churn killed the xHCI host controller twice (2026-08-03/04).
These tests pin the ladder: a healthy start must not reset at all, a
failed start earns exactly one reset + one retry, and a still-wedged
retry raises CameraWedgeError without further resets.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from rosclaw.evolution.hardware.camera import (
    CameraWedgeError,
    D435iCapture,
)


class _FakeDevice:
    def __init__(self, serial: str = "231122070092"):
        self._serial = serial
        self.resets = 0

    def get_info(self, info):
        name = getattr(info, "name", "")
        if "serial" in str(info).lower() or "SERIAL" in name:
            return self._serial
        if "firmware" in str(info).lower():
            return "5.17.0.10"
        return "Intel RealSense D435I"

    def hardware_reset(self):
        self.resets += 1


class _FakePipeline:
    """Fails to start (or first-frame) until `failures_left` is exhausted."""

    def __init__(self, rs_mod):
        self._rs = rs_mod

    def start(self, config):
        if self._rs.failures_left > 0:
            self._rs.failures_left -= 1
            raise RuntimeError("get_xu(...). xioctl(UVCIOC_CTRL_QUERY) failed ... timed out")

    def wait_for_frames(self, timeout_ms):
        if self._rs.frame_failures_left > 0:
            self._rs.frame_failures_left -= 1
            raise RuntimeError("Frame didn't arrive")
        return object()

    def stop(self):
        pass


class _FakeContext:
    def __init__(self, rs_mod):
        self._rs = rs_mod

    def query_devices(self):
        return [self._rs.device]


def _fake_rs(*, start_failures: int = 0, frame_failures: int = 0) -> types.ModuleType:
    rs = types.ModuleType("pyrealsense2")
    rs.device = _FakeDevice()
    rs.failures_left = start_failures
    rs.frame_failures_left = frame_failures
    rs.context = lambda: _FakeContext(rs)
    rs.pipeline = lambda: _FakePipeline(rs)
    rs.config = lambda: types.SimpleNamespace(
        enable_device=lambda s: None, enable_stream=lambda *a: None
    )
    rs.camera_info = types.SimpleNamespace(
        name="name", serial_number="serial", firmware_version="fw"
    )
    rs.stream = types.SimpleNamespace(color="color", depth="depth")
    rs.format = types.SimpleNamespace(bgr8="bgr8", z16="z16")
    return rs


def test_plain_start_success_never_resets():
    rs = _fake_rs()
    with patch.object(D435iCapture, "_rs", staticmethod(lambda: rs)):
        cap = D435iCapture()
        out = cap.start(serial="231122070092")
    assert out["serial"] == "231122070092"
    assert rs.device.resets == 0  # the whole point: no reset on a healthy start


def test_failed_start_earns_exactly_one_reset_and_retry():
    rs = _fake_rs(start_failures=1)
    with (
        patch.object(D435iCapture, "_rs", staticmethod(lambda: rs)),
        patch("rosclaw.evolution.hardware.camera.time.sleep", lambda s: None),
    ):
        cap = D435iCapture()
        cap.start(serial="231122070092")
    assert rs.device.resets == 1  # one reset, then the retry succeeded


def test_wedged_after_reset_raises_with_single_reset():
    rs = _fake_rs(start_failures=5)
    with (
        patch.object(D435iCapture, "_rs", staticmethod(lambda: rs)),
        patch("rosclaw.evolution.hardware.camera.time.sleep", lambda s: None),
    ):
        cap = D435iCapture()
        with pytest.raises(CameraWedgeError):
            cap.start(serial="231122070092")
    assert rs.device.resets == 1  # never a second reset for the same incident


def test_no_frame_after_start_counts_as_failure_and_uses_ladder():
    rs = _fake_rs(frame_failures=1)
    with (
        patch.object(D435iCapture, "_rs", staticmethod(lambda: rs)),
        patch("rosclaw.evolution.hardware.camera.time.sleep", lambda s: None),
    ):
        cap = D435iCapture()
        out = cap.start(serial="231122070092")
    assert rs.device.resets == 1
    assert out["serial"] == "231122070092"
