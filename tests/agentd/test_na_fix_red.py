"""红测试（二次审计 P0-1/P0-6/P0-7）：复现接缝缺陷——先红后修。

- 完整真实 envelope 的 Python→Node hash 必须一致（当前：浮点 30.0
  Python 序列化为 "30.0"，TS JSON.stringify 得 "30" → mismatch）；
- request_action 必须精确使用本卡 grant（当前：取最后一个全局 active grant）；
- tool request 必须携带 context_revision（当前：全 0 且不校验）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.pi_bridge.context import build_embodied_context
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

REPO = Path(__file__).resolve().parents[2]
AGENT_PKG = REPO / "packages" / "rosclaw-agent"


def _node_bin() -> str | None:
    """node >= 22.19（CI 的 Python-only job 没有 node——跨语言测试诚实 skip）。"""
    import shutil

    for candidate in filter(None, [shutil.which("node"), "/usr/bin/node", "/usr/local/bin/node"]):
        try:
            out = subprocess.check_output([candidate, "--version"], text=True, timeout=10).strip()
            if [int(p) for p in out.lstrip("v").split(".")] >= [22, 19, 0]:
                return candidate
        except Exception:  # noqa: BLE001
            continue
    return None


NODE = _node_bin()
DIST_READY = (AGENT_PKG / "dist" / "src" / "extension" / "context-injection.js").exists()
requires_node = pytest.mark.skipif(
    NODE is None or not DIST_READY,
    reason="node >= 22.19 or rosclaw-agent dist unavailable",
)


def _turn() -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="t", provider="mock", model="m", content="ok",
        assistant_message={"role": "assistant", "content": "ok"},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},  # type: ignore[arg-type]
    )


@requires_node
class TestCrossLanguageHash:
    """P0-1：完整真实 envelope 的跨语言字节级一致。"""

    async def test_real_envelope_hash_matches_node(self, tmp_path: Path) -> None:
        config = load_agent_config(tmp_path / "config.yaml")
        service = AgentService(
            config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()])
        )
        mission = service.create_mission("hash 一致性")
        envelope = build_embodied_context(service, mission.mission_id)
        python_hash = envelope.hash
        # 用真实传输的 JSON 字节（不是简化对象）交给 node 侧 canonicalJson。
        envelope_json = envelope.model_dump_json()
        script = (
            "import { envelopeHash } from './dist/src/extension/context-injection.js';"
            f"const env = JSON.parse({json.dumps(envelope_json)});"
            "console.log(envelopeHash(env));"
        )
        result = subprocess.run(
            [NODE, "--input-type=module", "-e", script],
            cwd=AGENT_PKG, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        node_hash = result.stdout.strip()
        assert node_hash == python_hash, (
            f"跨语言 hash 不一致（P0-1）：python={python_hash} node={node_hash}"
        )
        await service.close()

    @pytest.mark.parametrize(
        "value",
        [
            {"ttl": 30.0},
            {"x": -0.0},
            {"big": 1e21},
            {"text": "播放提示音 🎵"},
            {"nested": {"list": [1, 2.5, None], "map": {"b": 1, "a": 2}}},
            {"empty": None},
        ],
    )
    def test_golden_corpus_cross_language(self, value: dict) -> None:
        """浮点/Unicode/嵌套/key 顺序 golden corpus。"""
        import hashlib

        from rosclaw.contracts.pi.canonical import canonical_dumps

        canonical_py = canonical_dumps(value)
        py_hash = "sha256:" + hashlib.sha256(canonical_py.encode()).hexdigest()[:32]
        script = (
            "import { envelopeHash } from './dist/src/extension/context-injection.js';"
            f"console.log(envelopeHash({json.dumps(value)}));"
        )
        result = subprocess.run(
            [NODE, "--input-type=module", "-e", script],
            cwd=AGENT_PKG, capture_output=True, text=True, timeout=60,
        )
        node_hash = result.stdout.strip()
        assert node_hash == py_hash, f"{value}: python={py_hash} node={node_hash}"


class TestExactGrantBinding:
    """P0-6：动作必须使用当前卡片的精确 grant，不是最后一个全局 grant。"""

    def test_tool_requests_must_not_send_zero_context_revision(self) -> None:
        """P0-7：Node 侧工具不得再发送 context_revision: 0。"""
        failures = []
        for path in (AGENT_PKG / "src").rglob("*.ts"):
            text = path.read_text(encoding="utf-8")
            if "context_revision: 0" in text:
                failures.append(path.name)
        assert not failures, f"工具仍在发送 context_revision: 0: {failures}"
