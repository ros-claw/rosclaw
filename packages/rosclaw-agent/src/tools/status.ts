// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** `rosclaw_status` 自定义工具（PNA-0 + 六审 PR-SIX-1）：只读状态，
 * 与 /status、Header、Footer 共享同一个 KernelSnapshotV1——数据只经
 * UDS pi.status（Native Agent 路径不再访问旧 HTTP 8765 面；
 * chat 的 agentd 用 port=0，HTTP 必然误报 UNREACHABLE）。
 * 永不伪造可达性。 */

import { compactRenderResult } from "../ui/compact-result.js";
import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { ProductStateCenter } from "../session/state-center.js";

export function buildStatusTool(center: ProductStateCenter) {
	return defineTool({
		name: "rosclaw_status",
		label: "ROSClaw Status",
		description:
			"Read ROSClaw embodied-kernel status (agentd/missions/body/mode). " +
			"Read-only. Returns honest unreachable errors when agentd is down — " +
			"never invent robot state.",
		parameters: Type.Object({}),
		// 0902 R3-c（§6.1）：用户面单行摘要——模型上下文保留完整
		// JSON（诚实性不降级），TUI 不再整段打原始 JSON。
		renderResult: compactRenderResult,
		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			try {
				const report = await center.statusReport();
				const snap = report.snapshot;
				const payload = {
					agentd: report.agentd || (report.ok ? "READY" : "DEGRADED"),
					authorization_profile: report.authorization_profile,
					mission: report.mission,
					// 与 Header/Footer 同一快照——模型看到的状态与人一致。
					kernel: snap.kernel,
					context_state: snap.context_state,
					context_revision: snap.context_revision,
					lease_state: snap.lease_state,
					operator: snap.operator,
					action_readiness: snap.action_readiness.state,
					action_block_reasons: snap.action_readiness.reason_codes,
					snapshot_seq: snap.snapshot_seq,
				};
				return {
					content: [{ type: "text" as const, text: JSON.stringify(payload, null, 1) }],
					details: {
						reachable: true,
						kernel: snap.kernel,
						snapshot_seq: snap.snapshot_seq,
					},
				};
			} catch (err) {
				return {
					content: [
						{
							type: "text" as const,
							text: JSON.stringify({
								agentd: "UNREACHABLE",
								error: (err as Error).message,
								note: "agentd UDS 不可达——不要编造机器人状态。",
							}),
						},
					],
					details: { reachable: false, kernel: "UNREACHABLE" },
					isError: true,
				};
			}
		},
	});
}
