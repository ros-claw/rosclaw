"""Test helper: stdio ACP server over a Mock-gateway AgentService.

Usage: ROSCLAW_ACP_TEST_HOME=<dir> python -m tests.agentd.acp_test_server
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def main() -> None:
    home = Path(os.environ["ROSCLAW_ACP_TEST_HOME"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from rosclaw.adapters.acp.server import serve_stdio
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService
    from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

    def _answer(request) -> ModelTurnResultV1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "d",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "ANSWER",
            "summary": "ok",
            "evidence_refs": [],
        }
        return ModelTurnResultV1(
            turn_id="t",
            provider="mock",
            model="m",
            content=f"```json\n{json.dumps(decision)}\n```",
            assistant_message={"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
        )

    config = load_agent_config(home / "config.yaml")
    service = AgentService(
        config, home, gateway=MockModelGateway(mock_profile(), [_answer] * 50)
    )
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_stdio(service))


if __name__ == "__main__":
    main()
