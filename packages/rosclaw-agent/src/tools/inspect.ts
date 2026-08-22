// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** rosclaw_inspect 工具（PR-N3，N 总纲 §4.3）——生态索引自检。
 *
 * Agent 不再从 / 全盘搜索：inspect robot 一次调用返回权威资产链
 * （eurdf/mjcf/urdf/safety/capabilities + digest + source）；inspect
 * self 返回版本/根目录/索引健康；capability/asset 走全文搜索。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildInspectTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_inspect",
		label: "ROSClaw Inspect",
		description:
			"Inspect ROSClaw itself (program-probed, never guessed): " +
			"kind='self' returns version/roots/index health; " +
			"kind='robot' with query='<robot_id>' returns the canonical asset " +
			"chain (e-URDF profile, MJCF, URDF, safety, capabilities, digest); " +
			"kind='capability'/'asset' with a query searches the ecosystem index. " +
			"Use this FIRST before searching the filesystem for robot assets.",
		parameters: Type.Object({
			kind: Type.Union(
				[
					Type.Literal("self"),
					Type.Literal("robot"),
					Type.Literal("capability"),
					Type.Literal("asset"),
				],
				{ description: "自检主题" },
			),
			query: Type.Optional(Type.String({ description: "robot_id 或搜索词" })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const args = params as { kind: string; query?: string };
			return await executeVia(ctx, "rosclaw_inspect", {
				kind: args.kind,
				query: args.query ?? "",
			});
		},
	});
}
