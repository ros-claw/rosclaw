"""`rosclaw acp serve` — ACP stdio server entry（批次 G）。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def dispatch_acp_argv(argv: list[str]) -> int | None:
    if argv[:1] != ["acp"]:
        return None
    if len(argv) < 2 or argv[1] != "serve":
        print("用法: rosclaw acp serve [--home DIR]", file=sys.stderr)
        return 2
    home = Path(
        argv[argv.index("--home") + 1]
        if "--home" in argv
        else os.environ.get("ROSCLAW_HOME", Path.home() / ".rosclaw")
    )
    from rosclaw.adapters.acp.server import serve_stdio
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.pi_config import pi_model_configured
    from rosclaw.agentd.service import AgentService

    # P1-A3：credential 单源——不再 inject legacy agentd/credentials.json；
    # 凭据只来自进程 env 与 Pi auth.json（chat 引擎自解析）。
    if not pi_model_configured(home):
        print("未配置模型。先运行 `rosclaw setup model`。", file=sys.stderr)
        return 2
    config = load_agent_config(home / "config.yaml")
    service = AgentService(config, home)
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_stdio(service))
    return 0
