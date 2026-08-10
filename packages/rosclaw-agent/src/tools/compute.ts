/** rosclaw_compute 工具（七审 §2.2/PR-SEVEN-2）：COMPUTE 能力免审批
 * 调用（纯计算，无物理副作用——不需要授权链，但仍是内核验证链的
 * 一次真实调用）。 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

export function buildComputeTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_compute",
		label: "ROSClaw Compute",
		description:
			"Run a COMPUTE-class capability (pure calculation/verification, no " +
			"physical side effect, no approval needed) — e.g. sim_reach physics " +
			"evaluation or path planning. Only IDs from compute_capabilities.",
		parameters: Type.Object({
			capability_id: Type.String({ description: "exact COMPUTE capability id" }),
			arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx2) {
			const state = ctx.active.current;
			if (!state.missionId) {
				return {
					content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
					details: { ok: false, error_code: "NO_MISSION" },
					isError: true,
				};
			}
			const response = await ctx.center.call("pi.tools.execute", {
				request: {
					schema_version: "rosclaw.pi_tool_request.v1",
					request_id: `ptr_compute_${Date.now()}`,
					pi_session_id: state.sessionId,
					mission_id: state.missionId,
					context_revision: state.contextRevision,
					tool_name: "rosclaw_compute",
					arguments: {
						capability_id: String(params.capability_id),
						arguments: params.arguments ?? {},
					},
					requested_at: new Date().toISOString(),
					idempotency_key: `idem_compute_${state.sessionId}_${Date.now()}`,
					actor: { engine: "pi" },
				},
			});
			const result = (response.result ?? {}) as { summary?: string; error_code?: string };
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
		},
	});
}
