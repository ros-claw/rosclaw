"""Console entrypoint with a lightweight product-command fast path."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Dispatch product workflows without importing the full legacy CLI."""

    # 0902 R3-a（§6.2）：Ctrl-C 安静退出返回 130——traceback 只进
    # debug 日志（ROSCLAW_DEBUG=1 时原样抛出便于诊断），不打终端。
    try:
        return _dispatch(sys.argv[1:])
    except KeyboardInterrupt:
        if os.environ.get("ROSCLAW_DEBUG"):
            raise
        print("\n已取消（Ctrl-C）", file=sys.stderr)
        return 130


def _dispatch(argv: list[str]) -> int:
    """命令分派（main 的 Ctrl-C 防护外壳之下的真实路由）。"""
    # 0902 R3-a（§6.2）：`rosclaw open <artifact-id>` 最短主入口——
    # 重写为 artifact open（同一 handler，不允许第二条实现）。
    if argv and argv[0] == "open":
        argv = ["artifact", "open", *argv[1:]]

    # NA-FIX-8：root CLI 产品化——精简 help / commands --json / topic 指引
    # （命中即返回；未命中走既有 dispatcher，行为不变）。
    from rosclaw.root_cli import dispatch_root_cli

    _help_all = argv == ["help", "--all"]
    root = dispatch_root_cli(argv if not _help_all else ["commands", "--help-all"])
    if root is not None:
        return root

    # PR-N3：inspect self/robot/capability/asset（生态索引自检）。
    from rosclaw.cognition.inspect_cli import dispatch_inspect_argv

    inspected = dispatch_inspect_argv(argv)
    if inspected is not None:
        return inspected

    from rosclaw.operator.cli import dispatch_operator_argv

    result = dispatch_operator_argv(argv)
    if result is not None:
        return result

    # 0901 P0-1：artifact 交付面挂进 `rosclaw` 入口（实证事故：
    # TerminalPresenter 给的 `rosclaw artifact open <id>` 此前落到
    # legacy parser 打顶层帮助——dispatch 链缺失，不是版本漂移）。
    from rosclaw.agentd.cli import dispatch_artifact_argv

    result = dispatch_artifact_argv(argv)
    if result is not None:
        return result

    # P0-CLI-01：统一 setup 向导（model/body/operator/worker/integration）
    # 先于 legacy parser——legacy 的 setup 只有 lerobot，契约失真。
    from rosclaw.setup_cli import dispatch_setup_argv

    result = dispatch_setup_argv(argv)
    if result is not None:
        return result

    from rosclaw.daemon.cli import dispatch_daemon_argv

    result = dispatch_daemon_argv(argv)
    if result is not None:
        return result

    from rosclaw.adapters.acp.cli import dispatch_acp_argv

    result = dispatch_acp_argv(argv)
    if result is not None:
        return result

    from rosclaw.operatord.cli import dispatch_operatord_argv

    result = dispatch_operatord_argv(argv)
    if result is not None:
        return result

    from rosclaw.release_verify import dispatch_release_argv

    result = dispatch_release_argv(argv)
    if result is not None:
        return result

    from rosclaw.knowledge.cli import dispatch_knowledge_argv

    result = dispatch_knowledge_argv(argv)
    if result is not None:
        return result

    from rosclaw.evidence_verify import dispatch_evidence_argv

    result = dispatch_evidence_argv(argv)
    if result is not None:
        return result

    from rosclaw.robot_pack.cli import dispatch_robot_pack_argv

    result = dispatch_robot_pack_argv(argv)
    if result is not None:
        return result

    from rosclaw.app.cli import dispatch_app_argv

    result = dispatch_app_argv(argv)
    if result is not None:
        return result

    from rosclaw.collective.cli import dispatch_collective_argv

    result = dispatch_collective_argv(argv)
    if result is not None:
        return result

    from rosclaw.dream.cli import dispatch_dream_argv

    result = dispatch_dream_argv(argv)
    if result is not None:
        return result

    from rosclaw.continual.cli import dispatch_continual_argv

    result = dispatch_continual_argv(argv)
    if result is not None:
        return result

    from rosclaw.simforge.g1_muscle_memory_cli import dispatch_muscle_memory_argv

    result = dispatch_muscle_memory_argv(argv)
    if result is not None:
        return result

    from rosclaw.simforge.g1_hat_trick_cli import dispatch_hat_trick_argv

    result = dispatch_hat_trick_argv(argv)
    if result is not None:
        return result

    from rosclaw.simforge.phase4_cli import dispatch_phase4_argv

    result = dispatch_phase4_argv(argv)
    if result is not None:
        return result

    from rosclaw.simforge.phase3_cli import dispatch_phase3_argv

    result = dispatch_phase3_argv(argv)
    if result is not None:
        return result

    from rosclaw.product.cli import dispatch_product_argv

    result = dispatch_product_argv(argv)
    if result is not None:
        return result

    from rosclaw.simforge.cli import dispatch_simforge_argv

    result = dispatch_simforge_argv(argv)
    if result is not None:
        return result

    if _help_all:
        sys.argv = [sys.argv[0], "--help"]
    from rosclaw.cli import main as legacy_main

    return legacy_main()


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
