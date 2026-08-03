"""External harness WorkerPacks (PR-WF-054, 总纲 §9.10).

Codex / Claude Code 以 **Official WorkerPack** 方式接入：不 vendoring
第三方源码、不默认安装重型框架；二进制由用户显式安装、版本锁 +
conformance tests 分级（T0 Discovered / T1 Compatible）。

P0/P1 范围：repo analysis / test analysis / docs 类认知任务；不做物理
任务；side_effect_class=none（提示词与 data scope 双重约束）。
"""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.contracts.worker.card import (
    CapabilityDecl,
    WorkerCardV1,
    WorkerConstraints,
    WorkerHealth,
    WorkerImplementation,
    WorkerKind,
    WorkerProvenance,
    WorkerSecurity,
    WorkerTrust,
)

ADAPTER_EXTERNAL_CLI = "external_cli"


@dataclass(frozen=True)
class WorkerPackManifest:
    """Official WorkerPack 描述（不随 wheel 安装任何二进制）。"""

    pack_id: str
    worker_id: str
    product: str
    display_name: str
    executable: str
    min_version: str
    install_hint: str
    license: str
    capabilities: tuple[tuple[str, str], ...]  # (name, input/output schema)
    env_passthrough: tuple[str, ...] = ()
    max_turns: int = 8
    default_timeout_sec: float = 300.0


CLAUDE_CODE_PACK = WorkerPackManifest(
    pack_id="claude-code",
    worker_id="worker:claude-code:local",
    product="claude-code",
    display_name="Claude Code Worker",
    executable="claude",
    min_version="2.0.0",
    install_hint=(
        "安装 Claude Code: https://claude.com/claude-code — 安装后运行 "
        "`rosclaw worker probe worker:claude-code:local` 与 `rosclaw worker enable`。"
    ),
    license="proprietary",
    capabilities=(
        ("code.repository_analysis", "rosclaw://schemas/text-task.v1"),
        ("code.test_analysis", "rosclaw://schemas/text-task.v1"),
        ("docs.write", "rosclaw://schemas/text-task.v1"),
    ),
    env_passthrough=(
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ),
)

CODEX_CLI_PACK = WorkerPackManifest(
    pack_id="codex-cli",
    worker_id="worker:codex:local",
    product="codex-cli",
    display_name="Codex CLI Worker",
    executable="codex",
    min_version="0.20.0",
    install_hint=(
        "安装 Codex CLI: https://developers.openai.com/codex/cli — 安装后运行 "
        "`rosclaw worker probe worker:codex:local` 与 `rosclaw worker enable`。"
    ),
    license="proprietary",
    capabilities=(
        ("code.repository_analysis", "rosclaw://schemas/text-task.v1"),
        ("code.test_analysis", "rosclaw://schemas/text-task.v1"),
    ),
    env_passthrough=("OPENAI_API_KEY", "OPENAI_BASE_URL"),
)

ALL_PACKS = (CLAUDE_CODE_PACK, CODEX_CLI_PACK)


def card_for_pack(pack: WorkerPackManifest, *, trust: str = "T1") -> WorkerCardV1:
    return WorkerCardV1(
        worker_id=pack.worker_id,
        display_name=pack.display_name,
        kind=WorkerKind.HARNESS,
        adapter_type=ADAPTER_EXTERNAL_CLI,
        adapter_version="1.0.0",
        implementation=WorkerImplementation(
            product=pack.product,
            version=f">={pack.min_version}",
            executable_ref=f"path:{pack.executable}",
        ),
        capabilities=[
            CapabilityDecl(
                name=name, input_schema=schema, output_schema=schema, side_effect_class="none"
            )
            for name, schema in pack.capabilities
        ],
        constraints=WorkerConstraints(
            supported_platforms=["linux", "darwin"],
            requires_network=True,
            max_concurrency=1,
        ),
        security=WorkerSecurity(
            isolation="process",
            default_data_scopes=["mission_artifacts"],
        ),
        health=WorkerHealth(probe="adapter:ping", heartbeat_interval_sec=30, lease_ttl_sec=360),
        provenance=WorkerProvenance(source="official_worker_pack", license=pack.license),
        trust=WorkerTrust(initial_level=trust, evidence_count=0),
    )


def version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def version_ok(found: str, minimum: str) -> bool:
    """found 需 >= minimum（如 "2.1.220" >= "2.0.0"）。"""
    f, m = version_tuple(found), version_tuple(minimum)
    width = max(len(f), len(m))
    return f + (0,) * (width - len(f)) >= m + (0,) * (width - len(m))
