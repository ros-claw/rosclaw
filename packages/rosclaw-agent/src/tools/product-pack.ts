// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** Product Pack 工具（PR-H4，总纲 v2 §10.2/§12）——交付登记/收尾/阻塞。
 *
 * 任务不能靠一句话完成：action-oriented 任务必须 task_finish（带登记
 * artifact）或 task_blocked；终态由 Verifier 决定，模型自述不算数。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildProductPackTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "rosclaw_artifact_register",
			label: "ROSClaw Artifact Register",
			description:
				"Register a produced deliverable (file is really read and hashed — " +
				"mentioning a file is NOT registering). Only registered artifacts " +
				"count at finish time.",
			parameters: Type.Object({
				path: Type.String({ description: "交付物路径（相对任务工作区或绝对）" }),
				media_type: Type.Optional(Type.String()),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_artifact_register", {
					...(params as Record<string, unknown>),
					cwd: ctx.workspaceRoot ?? "",
				});
			},
		}),
		defineTool({
			name: "rosclaw_task_finish",
			label: "ROSClaw Task Finish",
			description:
				"Finish the current task: the verifier really runs (registered " +
				"artifacts checked by content hash; the acceptance frozen at " +
				"task creation is checked — you CANNOT pass new acceptance " +
				"rules here). REPAIR_REQUIRED returns failures — fix in the " +
				"SAME task, then finish again. Zero evidence never succeeds.",
			parameters: Type.Object({
				summary: Type.String({ description: "完成摘要（用户可见）" }),
				artifact_ids: Type.Optional(Type.Array(Type.String())),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_task_finish", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_blocked",
			label: "ROSClaw Task Blocked",
			description:
				"Honestly mark the current task BLOCKED with a stable reason code " +
				"and recovery actions — when the goal cannot be achieved with " +
				"current capabilities. Never fake success instead.",
			parameters: Type.Object({
				reason_code: Type.String({ description: "稳定原因码（如 MISSING_CAPABILITY）" }),
				detail: Type.String(),
				recovery: Type.Optional(Type.Array(Type.String())),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_task_blocked", params as Record<string, unknown>);
			},
		}),
	];
}
