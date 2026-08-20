/** Embodiment Pack 执行工具（PR-H5，总纲 v2 §10.2）。
 *
 * rosclaw_execute：统一能力执行入口——OBSERVE 只读 / COMPUTE 免审批
 * 内联 / PHYSICAL_ACTION 走同一 admission 链（SIM 安全自动、REAL
 * 永远 rosclawd+审批）。未知 ID 诚实拒绝。
 * rosclaw_wait_operation / rosclaw_stop_operation：长 operation 的
 * 有界等待与账本先行取消。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildEmbodimentExecTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "rosclaw_execute",
			label: "ROSClaw Execute",
			description:
				"Execute a capability by exact ID from rosclaw_capabilities. " +
				"Routing: observation = read-only, compute = inline no-approval, " +
				"physical = the same admission chain (SIM safe actions auto-execute " +
				"with audit; REAL always requires rosclawd + operator). Unknown IDs " +
				"are honestly rejected — never invent capability names.",
			parameters: Type.Object({
				capability_id: Type.String(),
				arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
				expected_effect: Type.Optional(Type.String()),
				risk_tier: Type.Optional(Type.String()),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_execute", params as Record<string, unknown>);
			},
		}),
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
