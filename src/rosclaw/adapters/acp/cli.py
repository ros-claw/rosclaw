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
    from rosclaw.agentd.credentials import AgentCredentialStore, CredentialStoreError
    from rosclaw.agentd.service import AgentService

    try:
        AgentCredentialStore(home).inject()
    except CredentialStoreError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    config = load_agent_config(home / "config.yaml")
    if not config.profiles:
        print("未配置模型。先运行 `rosclaw agent init`。", file=sys.stderr)
        return 2
    service = AgentService(config, home)
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_stdio(service))
    return 0
