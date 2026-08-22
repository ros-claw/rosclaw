// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** Embodiment Pack 执行工具（PR-H5，总纲 v2 §10.2）。
 *
 * PR-N5D：rosclaw_execute 已退出模型面（能力物化为精确工具）；
 * wire 验证链保留在 bridge。
 * rosclaw_wait_operation / rosclaw_stop_operation：长 operation 的
 * 有界等待与账本先行取消。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildEmbodimentExecTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		// PR-N5D：rosclaw_execute 退出模型面——能力经 snapshot 物化为
		// 精确工具/direct+propose_（materialize.ts）；wire 入口保留在
		// bridge（物化工具内部走它）。
		defineTool({
			name: "rosclaw_wait_operation",
			label: "ROSClaw Wait Operation",
			description:
				"Wait for a background operation to reach a terminal state " +
				"(bounded; returns final state + output tail). Prefer doing other " +
				"work and relying on the completion notification for long runs.",
			parameters: Type.Object({
				operation_id: Type.String(),
				timeout_sec: Type.Optional(Type.Number()),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_wait_operation", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_stop_operation",
			label: "ROSClaw Stop Operation",
			description: "Stop a running operation (ledger-first cancel, audited).",
			parameters: Type.Object({
				operation_id: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_stop_operation", params as Record<string, unknown>);
			},
		}),
	];
}
