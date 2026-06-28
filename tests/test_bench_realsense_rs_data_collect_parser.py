"""Tests for ``rosclaw bench realsense`` and the rs-data-collect parser."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rosclaw.bench.realsense import bench_realsense, parse_rs_data_collect
from rosclaw.cli import cmd_bench_realsense


class TestBenchRealSenseRsDataCollectParser:
    """Phase I rs-data-collect parser tests."""

    def test_parse_json_lines(self):
        stdout = json.dumps(
            {
                "color_frames": 120,
                "depth_frames": 120,
                "fps": 30.0,
                "drops": 0,
                "usb_mode": "USB3",
            }
        )
        result = parse_rs_data_collect(stdout)
        assert result["color_frames"] == 120
        assert result["depth_frames"] == 120
        assert result["fps"] == 30.0
        assert result["drops"] == 0
        assert result["usb_mode"] == "USB3"
        assert result["degraded"] is False

    def test_parse_human_readable_output(self):
        stdout = """
        color frames: 95
        depth frames: 95
        FPS: 23.75
        Drops: 5
        USB mode: USB2
        """
        result = parse_rs_data_collect(stdout)
        assert result["color_frames"] == 95
        assert result["depth_frames"] == 95
        assert result["fps"] == 23.75
        assert result["drops"] == 5
        assert result["usb_mode"] == "USB2"
        assert result["degraded"] is True

    def test_parse_empty_output_marks_degraded_false(self):
        result = parse_rs_data_collect("")
        assert result["color_frames"] == 0
        assert result["depth_frames"] == 0
        assert result["degraded"] is False


class TestBenchRealSenseNoStub:
    """Phase I RealSense benchmark without hardware stubs."""

    def test_bench_realsense_falls_back_without_hardware(self, tmp_path):
        """Without pyrealsense2 or rs-data-collect, bench still writes a report."""
        output_dir = tmp_path / "bench"
        report = bench_realsense(duration_sec=0.1, output_dir=str(output_dir))

        assert report["schema_version"] == "rosclaw.bench.realsense.v1"
        assert report["duration_requested_sec"] == 0.1
        assert report["backend"] is None or report["backend"] == "pyrealsense2"
        assert report["errors"]
        assert (output_dir / "report.json").exists()

        saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        assert saved["schema_version"] == report["schema_version"]

    def test_cli_bench_realsense(self, tmp_path, capsys):
        output_dir = tmp_path / "bench_cli"
        args = SimpleNamespace(duration=0.1, output=str(output_dir), json=False)
        assert cmd_bench_realsense(args) == 0

        captured = capsys.readouterr().out
        assert "Bench — RealSense Capture" in captured
        assert (output_dir / "report.json").exists()

    def test_cli_bench_realsense_json(self, tmp_path, capsys):
        output_dir = tmp_path / "bench_cli_json"
        args = SimpleNamespace(duration=0.1, output=str(output_dir), json=True)
        assert cmd_bench_realsense(args) == 0

        captured = capsys.readouterr().out
        report = json.loads(captured)
        assert report["schema_version"] == "rosclaw.bench.realsense.v1"
        assert report["duration_requested_sec"] == 0.1
