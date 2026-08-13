"""十二审 PR-12.6 红测试：仿真结果产品闭环。

1. render_trace_preview：确定性 GIF（魔数 GIF89a、≥30 帧、可解码）、
   诚实 COMMAND_REPLAY 标注；空 trace fail closed；
2. SimRunReceiptV1 合约存在且 evidence_level 默认 COMMAND_REPLAY。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTracePreview:
    def test_render_produces_valid_gif(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        # 合成一个五角星状 trace。
        import math

        from rosclaw.sim import ur5e_mcp

        trace = [
            {
                "x": 0.35 + 0.1 * math.cos(t * 4 * math.pi / 50),
                "y": 0.25 + 0.1 * math.sin(t * 4 * math.pi / 50),
                "z": 0.30,
            }
            for t in range(51)
        ]
        ur5e_mcp._state["trace"] = trace
        result = json.loads(ur5e_mcp.render_trace_preview())
        assert result["ok"]
        assert result["evidence_level"] == "COMMAND_REPLAY"
        assert "路径预演" in result["label"]
        artifact = result["artifact"]
        path = Path(artifact["path"])
        assert path.exists()
        data = path.read_bytes()
        assert data.startswith(b"GIF89a") or data.startswith(b"GIF87a")
        assert artifact["frames"] >= 30
        assert artifact["bytes"] > 1000
        # 可解码验证（PIL 真解码）。
        from PIL import Image

        img = Image.open(path)
        frames = getattr(img, "n_frames", 1)
        assert frames >= 30
        img.close()

    def test_empty_trace_fail_closed(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        from rosclaw.sim import ur5e_mcp

        ur5e_mcp._state["trace"] = []
        with pytest.raises(ValueError, match="fail closed"):
            ur5e_mcp.render_trace_preview()


class TestSimReceipt:
    def test_receipt_schema_defaults(self) -> None:
        from rosclaw.contracts.sim_run_receipt import SimRunReceiptV1

        receipt = SimRunReceiptV1(body_id="sim/ur5e", backend="command-replay")
        assert receipt.evidence_level == "COMMAND_REPLAY"
        assert receipt.schema_version == "rosclaw.sim_run_receipt.v1"
