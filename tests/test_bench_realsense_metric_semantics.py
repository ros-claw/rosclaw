"""Verify RealSense bench report exposes stream-level metrics."""

from __future__ import annotations

import json

from rosclaw.bench.realsense import bench_realsense, parse_rs_data_collect


class TestBenchRealSenseMetricSemantics:
    """P2 bench metric semantics verification."""

    def test_parser_includes_stream_and_aggregate_fields(self):
        stdout = json.dumps(
            {
                "color_frames": 120,
                "depth_frames": 130,
                "fps": 25.0,
                "drops": 2,
                "usb_mode": "USB3",
            }
        )
        result = parse_rs_data_collect(stdout)
        assert "streams" in result
        assert "aggregate" in result
        assert result["streams"]["color"]["frame_count"] == 120
        assert result["streams"]["depth"]["frame_count"] == 130
        assert result["aggregate"]["total_frame_count"] == 250
        assert result["aggregate"]["drop_count"] == 2

    def test_report_refines_stream_fps_with_duration(self, tmp_path):
        """When frame counts come from a backend without per-stream fps, bench
        should compute per-stream fps using the requested duration.
        """

        # Simulate a backend that only gives counts.
        def _fake_capture(_duration):
            return {
                "backend": "fake",
                "color_frames": 50,
                "depth_frames": 60,
                "fps": 0.0,
                "drops": 0,
                "usb_mode": "USB2",
                "degraded": True,
            }

        from rosclaw.bench import realsense as bench_mod

        original_capture = bench_mod._capture_with_pyrealsense2
        original_fallback = bench_mod._capture_with_rs_data_collect
        bench_mod._capture_with_pyrealsense2 = _fake_capture
        bench_mod._capture_with_rs_data_collect = _fake_capture
        try:
            report = bench_realsense(duration_sec=2.0, output_dir=str(tmp_path))
        finally:
            bench_mod._capture_with_pyrealsense2 = original_capture
            bench_mod._capture_with_rs_data_collect = original_fallback

        assert report["streams"]["color"]["fps"] == 25.0
        assert report["streams"]["depth"]["fps"] == 30.0
        assert report["aggregate"]["average_fps"] == 27.5
        assert report["aggregate"]["total_frame_count"] == 110

        saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "streams" in saved
        assert "aggregate" in saved

    def test_report_no_data_still_has_stream_schema(self, tmp_path):
        # Force the no-data path deterministically: a healthy camera on the
        # test host would otherwise deliver frames within the 50 ms window
        # and break the (unrelated) schema assertions.
        from rosclaw.bench import realsense as bench_mod

        def _no_hardware(_duration):
            raise RuntimeError("no camera")

        original_capture = bench_mod._capture_with_pyrealsense2
        original_fallback = bench_mod._capture_with_rs_data_collect
        bench_mod._capture_with_pyrealsense2 = _no_hardware
        bench_mod._capture_with_rs_data_collect = _no_hardware
        try:
            report = bench_realsense(duration_sec=0.05, output_dir=str(tmp_path))
        finally:
            bench_mod._capture_with_pyrealsense2 = original_capture
            bench_mod._capture_with_rs_data_collect = original_fallback
        assert "streams" in report
        assert report["streams"]["color"]["frame_count"] == 0
        assert report["streams"]["depth"]["frame_count"] == 0
        assert "aggregate" in report
        assert report["aggregate"]["total_frame_count"] == 0


def test_interval_stats_reports_acceptance_fields() -> None:
    """RS-01 acceptance fields: drop rate, P99, max gap, jitter."""
    from rosclaw.bench.realsense import _interval_stats

    # 10 frames at perfect 30 fps over 0.3 s -> no drops, ~33.3 ms intervals.
    arrivals = [i / 30.0 for i in range(10)]
    stats = _interval_stats(arrivals, elapsed=0.3)
    assert stats["frame_count"] == 10
    assert stats["drop_rate"] == 0.0
    assert 33.0 <= stats["inter_frame_p99_ms"] <= 34.0
    assert 33.0 <= stats["max_gap_ms"] <= 34.0
    assert stats["jitter_ms"] < 1.0

    # One 1.2 s hole: max_gap surfaces the >1 s no-frame violation and the
    # missing frames count into the drop rate.
    gapped = [0.0, 0.033, 0.066, 1.266, 1.3, 1.333]
    stats = _interval_stats(gapped, elapsed=1.333)
    assert stats["max_gap_ms"] >= 1200.0
    assert stats["drop_rate"] > 0.5


def test_interval_stats_degenerate_inputs() -> None:
    from rosclaw.bench.realsense import _interval_stats

    assert _interval_stats([], 1.0)["max_gap_ms"] is None
    assert _interval_stats([0.5], 1.0)["drop_rate"] is None
