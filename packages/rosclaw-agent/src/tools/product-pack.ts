// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** Product Pack 工具（P0-D，0824 总纲 §8.1）——模型面只剩幂等
 * rosclaw_deliver：普通文件工具创建的交付物用它交付（实读+hash+
 * 登记，幂等——同一文件重复交付返回同一 ArtifactRef）。task_finish/
 * task_blocked/artifact_register 已退出模型面：capability 产物自动
 * 登记，Task Coordinator 自动收集/验证/出 Outcome——模型不需要
 * 也不能手动收尾。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

// Explicit intent survives PI primitive coercion: "null", "1", "false"
// cannot become final roles. Namespace suffixes retain their existing semantics.
const insensitive = (word: string) => [...word].map(c => `[${c}${c.toUpperCase()}]`).join("");
const DELIVERY_ROLE_PATTERN = `^\\s*(?:${["report", "plot", "image", "video", "data"].map(insensitive).join("|")}|(?:${["progress", "diagnostic"].map(insensitive).join("|")})(?:_[\\s\\S]*)?)\\s*$`;

export function buildProductPackTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "rosclaw_deliver",
			label: "ROSClaw Deliver",
			description:
				"Deliver a file you created with normal file tools (really read + " +
				"hashed + registered — idempotent: re-delivering the same file " +
				"returns the same ArtifactRef). Capability-produced artifacts " +
				"register automatically. Choose an explicit role: progress_/diagnostic_ " +
				"register evidence and keep the task open (including SOURCE_ONLY work); " +
				"The explicit final roles report/plot/image/video/data trigger Task Coordinator acceptance. " +
				"— you do NOT call any finish/close tool.",
			parameters: Type.Object({
				path: Type.String({ description: "交付物路径（相对任务工作区或绝对）" }),
				media_type: Type.Optional(Type.String()),
				role: Type.String({ minLength: 1, pattern: DELIVERY_ROLE_PATTERN, description: "必须明确交付意图：progress/diagnostic 及其 progress_/diagnostic_ 后缀仅登记证据并保持任务（source-only 用 progress_report）。最终角色仅 report/plot/image/video/data，会触发任务验收；未知角色拒绝。" }),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_deliver", {
					...(params as Record<string, unknown>),
					cwd: ctx.workspaceRoot ?? "",
				});
			},
		}),
	];
}
