"""Root CLI 产品化（NA-FIX-8，规格 §8）：精简 root help + 机器可读注册表。

原则（审计 §12）：不做大规模搬家——dispatch 顺序与旧别名行为不变；
本模块只负责：精简 help、help --all 透传、commands --json、topic 指引。
"""

from __future__ import annotations

import json

#: 产品级入口（root help 默认只显示这些）。
SLIM_HELP = """ROSClaw — Embodied Agent Runtime

Get started
  chat        Start the Native Agent
  setup       Configure model, body and integrations
  status      Show runtime and embodiment status
  doctor      Diagnose installation and safety readiness

Runtime
  start       Start ROSClaw services
  stop        Stop ROSClaw services
  restart     Restart services
  dashboard   Open the dashboard

Domains
  robot       Body, sensing, ROS, MCP and capabilities
  worker      External Agent Workers and WorkOrders
  safety      Operator, daemon, permits, firewall and evidence
  evolve      Practice, memory, Darwin, Dream and SimForge
  hub         Skills, apps, assets and deployable providers
  dev         Logs, traces, sandbox, tests and benchmarks

  help <topic>       Topic help
  help --all         All compatibility commands
  commands --json    Machine-readable command registry
"""

#: 领域分组 → 兼容命令映射（指引用；旧命令全部继续可执行）。
DOMAIN_GROUPS: dict[str, list[str]] = {
    "robot": ["body", "eurdf", "robot", "ros", "sense", "capability", "mcp", "fleet"],
    "worker": ["worker", "collective"],
    "safety": [
        "daemon", "operatord", "firewall", "regime", "acceptance", "evidence",
        "release",
    ],
    "evolve": [
        "memory", "practice", "learning", "darwin", "dream", "simforge", "auto",
        "eval", "bench", "forge",
    ],
    "hub": ["skill", "app", "hub", "provider", "lerobot"],
    "dev": [
        "logs", "events", "trace", "db", "sandbox", "runtime", "test", "demo",
        "feedback",
    ],
    "setup": ["init", "firstboot", "setup", "config", "profile"],
}

#: 机器可读注册表（name → (group, summary)）。dispatch 实现见 entrypoint.py。
COMMAND_REGISTRY: dict[str, tuple[str, str]] = {
    "chat": ("core", "Start the Native Agent"),
    "setup": ("setup", "Configure model, body and integrations"),
    "status": ("core", "Show runtime and embodiment status"),
    "doctor": ("core", "Diagnose installation and safety readiness"),
    "start": ("runtime", "Start ROSClaw services"),
    "stop": ("runtime", "Stop ROSClaw services"),
    "restart": ("runtime", "Restart services"),
    "dashboard": ("runtime", "Open the dashboard"),
    "agent": ("core", "Native Agent management (agentd)"),
    "operatord": ("safety", "Independent operator authorization process"),
    "daemon": ("safety", "rosclawd control plane inspection"),
    "release verify": ("safety", "Verify a release bundle offline"),
    "evidence verify": ("safety", "Verify an acceptance evidence pack"),
    "commands --json": ("core", "Machine-readable command registry"),
}


def _topics_help(topic: str) -> str:
    lines = [f"# {topic}\n"]
    members = DOMAIN_GROUPS.get(topic, [])
    if members:
        lines.append("Commands in this domain (all fully executable):")
        lines.extend(f"  {name}" for name in members)
    else:
        lines.append(f"unknown topic {topic!r}; topics: {', '.join(DOMAIN_GROUPS)}")
    lines.append("\n`rosclaw help --all` for the full compatibility command list.")
    return "\n".join(lines)


def dispatch_root_cli(argv: list[str]) -> int | None:
    """精简 root help / commands --json / topic help。未命中返回 None。"""
    if not argv or argv == ["help"] or argv == ["-h"] or argv == ["--help"]:
        print(SLIM_HELP)
        return 0
    if argv[:2] == ["help", "--all"]:
        return None  # 透传 legacy 全量 help
    if argv[:2] == ["commands", "--json"]:
        print(
            json.dumps(
                {
                    "schema_version": "rosclaw.commands.v1",
                    "product_entries": [
                        line.strip().split()[0]
                        for line in SLIM_HELP.splitlines()
                        if line.startswith(("  chat", "  setup", "  status", "  doctor"))
                    ],
                    "registry": {
                        name: {"group": group, "summary": summary}
                        for name, (group, summary) in COMMAND_REGISTRY.items()
                    },
                    "domain_groups": DOMAIN_GROUPS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if argv[:1] == ["help"] and len(argv) == 2:
        print(_topics_help(argv[1]))
        return 0
    if argv[:1] == ["commands"]:
        if len(argv) == 2 and argv[1] == "--help-all":
            return None
        print("用法: rosclaw commands --json")
        return 2
    return None
