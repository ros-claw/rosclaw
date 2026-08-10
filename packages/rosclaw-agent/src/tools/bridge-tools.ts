/** ROSClaw 工具集（PNA-3）：经 pi-bridge 调用 agentd，全部带验证链。
 *
 * Pi 主 Agent 只有这些工具（noTools:"all"）；每个调用都携带
 * session/mission/lease/idempotency 语义，由 agentd 强制。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import type { ActiveSessionContext } from "../session/active-context.js";
import type { ProductStateCenter } from "../session/state-center.js";

export interface BridgeToolContext {
	rosclawHome: string;
	/** NA-FIX-1/2：动态 session 上下文（执行时读取，不捕获启动值）。 */
	active: ActiveSessionContext;
	/** PR-SIX-1：唯一状态中心——工具的桥调用经 center.call（UDS 失败
	 *  原子降级 kernel/context，不允许局部报错而 Header 保持 FRESH）。 */
	center: ProductStateCenter;
}

let requestCounter = 0;

async function executeVia(
	ctx: BridgeToolContext,
	toolName: string,
	arguments_: Record<string, unknown>,
) {
	requestCounter += 1;
	const state = ctx.active.current;
	if (!state.missionId) {
		return {
			content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
			details: { ok: false, error_code: "NO_MISSION" },
			isError: true,
		};
	}
	const request = {
		schema_version: "rosclaw.pi_tool_request.v1",
		request_id: `ptr_${Date.now()}_${requestCounter}`,
		pi_session_id: state.sessionId,
		mission_id: state.missionId,
		// P0-7：携带已验证 envelope 的精确 revision/body/mode。
		context_revision: state.contextRevision,
		body_hash: state.bodyHash ?? "",
		mode: state.mode,
		tool_name: toolName,
		arguments: arguments_,
		requested_at: new Date().toISOString(),
		idempotency_key: `idem_${state.sessionId}_${Date.now()}_${requestCounter}`,
		actor: { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 },
	};
	const response = await ctx.center.call("pi.tools.execute", { request });
	const result = (response.result ?? {}) as { ok?: boolean; summary?: string; error_code?: string };
	const ok = response.ok === true;
	return {
		content: [
			{
				type: "text" as const,
				text: ok
					? (result.summary ?? "ok")
					: `REJECTED [${result.error_code ?? response.code ?? "?"}]: ${String(response.error ?? result.summary ?? "")}`,
			},
		],
		details: { ok, error_code: result.error_code ?? null },
		isError: !ok,
	};
}

export function buildBridgeTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "rosclaw_observe",
			label: "ROSClaw Observe",
			description:
				"Read-only observation via agentd (MCP capability / body / self state). " +
				"Action-class capabilities are refused — they need the approval chain.",
			parameters: Type.Object({
				capability_id: Type.String({ description: "catalog capability id, e.g. limo.get_pose" }),
				arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
				reason: Type.Optional(Type.String()),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_observe", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_verify",
			label: "ROSClaw Verify",
			description:
				"Read execution receipts and check success criteria. A submitted command " +
				"is never proof of a completed task.",
			parameters: Type.Object({
				receipt_id: Type.Optional(Type.String()),
				success_criteria: Type.Optional(Type.String()),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_verify", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_memory_query",
			label: "ROSClaw Memory Query",
			description: "Query memory/practice/how with evidence references.",
			parameters: Type.Object({
				query: Type.String(),
				scope: Type.Optional(Type.String()),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_memory_query", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_fail_safe",
			label: "ROSClaw Fail Safe",
			description:
				"Pause the current turn and request operator attention. NOT an emergency " +
				"stop — E-Stop uses the independent operator path (/estop).",
			parameters: Type.Object({
				reason: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_fail_safe", params as Record<string, unknown>);
			},
		}),
	];
}
