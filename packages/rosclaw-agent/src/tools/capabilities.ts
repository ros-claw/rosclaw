/** `rosclaw_capabilities` 工具（六审 §6.2.1/PR-SIX-3）：当前 body 的
 * 可信能力面。
 *
 * 模型不再靠猜 capability ID——只读、按 bound body 过滤：
 * action_capabilities 只含 body 兼容且未隔离的 PHYSICAL_ACTION；
 * 不兼容项进 excluded 并附机器原因码（BODY_CAPABILITY_MISMATCH 等）。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

export function buildCapabilitiesTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_capabilities",
		label: "ROSClaw Capabilities",
		description:
			"List the capabilities available on the CURRENT bound body in three " +
			"buckets (observation/compute/action) plus excluded capabilities with " +
			"machine reason codes. PR-N5D: observation/compute capabilities are " +
			"ALSO materialized as exact strongly-typed tools in your tool surface " +
			"(e.g. ur5e__plan_cartesian_path) — call those directly instead of " +
			"hand-building generic calls; physical capabilities appear as " +
			"propose_<name> tools that enter the admission chain (REAL is always " +
			"gated by rosclawd/operator). Never invent capability names.",
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			const state = ctx.active.current;
			if (!state.missionId) {
				return {
					content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
					details: { ok: false, error_code: "NO_MISSION" },
					isError: true,
				};
			}
			const result = await ctx.center.call("pi.capabilities", {
				mission_id: state.missionId,
			});
			if (!result.ok) {
				return {
					content: [
						{
							type: "text" as const,
							text: `能力面查询失败 [${String(result.code ?? "")}]: ${String(result.error ?? "")}`,
						},
					],
					details: { ok: false, error_code: String(result.code ?? "") },
					isError: true,
				};
			}
			const actions = (result.action_capabilities ?? []) as Array<Record<string, unknown>>;
			const observation = (result.observation_capabilities ?? []) as Array<Record<string, unknown>>;
			// 八审 §1.2/P0-2：compute 桶必须透出——七审期间它被静默
			// 丢弃，真实模型看不到 plan/verify 能力。
			const compute = (result.compute_capabilities ?? []) as Array<Record<string, unknown>>;
			const excluded = (result.excluded ?? []) as Array<Record<string, unknown>>;
			const lines = [
				`body: ${String(result.body_id)}  mode: ${String(result.mode)}`,
				"",
				"action_capabilities（可经 rosclaw_request_action 提案的精确 ID）:",
				...(actions.length
					? actions.map(
						(c) =>
							`  - ${String(c.capability_id)} [${String(c.risk_tier)}/${String(c.side_effect_class)}] ${String(c.description ?? "")}`,
					)
					: ["  （当前 body 没有兼容的动作能力——不要编造动作名）"]),
				"",
				"compute_capabilities（规划/验证等纯计算——已物化为精确工具，直接调用，免审批）:",
				...(compute.length
					? compute.map(
						(c) => `  - ${String(c.capability_id)} ${String(c.description ?? "")}`,
					)
					: ["  （无）"]),
				"",
				"observation_capabilities（只读观测）:",
				...(observation.length
					? observation.map((c) => `  - ${String(c.capability_id)}`)
					: ["  （无）"]),
			];
			if (excluded.length) {
				lines.push("", "excluded（不兼容/被隔离，禁止提案）:");
				for (const e of excluded) {
					lines.push(`  - ${String(e.capability_id)} [${String(e.reason)}]`);
				}
			}
			return {
				content: [{ type: "text" as const, text: lines.join("\n") }],
				details: {
					ok: true,
					action_ids: actions.map((c) => String(c.capability_id)),
					observation_ids: observation.map((c) => String(c.capability_id)),
					compute_ids: compute.map((c) => String(c.capability_id)),
					excluded,
				},
			};
		},
	});
}
