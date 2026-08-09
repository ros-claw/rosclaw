/** rosclaw_delegate 工具（PNA-4，规格 §19）：后台委派 + 原位进度更新。 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

let counter = 0;

export function buildDelegateTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_delegate",
		label: "ROSClaw Delegate",
		description:
			"Hire a bounded worker for a self-contained subtask (WorkOrder with " +
			"budget + verification). Worker output enters the main context only " +
			"after ROSClaw verification passes.",
		parameters: Type.Object({
			goal: Type.String({ description: "self-contained subtask goal" }),
			worker_id: Type.Optional(
				Type.String({ description: "auto | worker:native:local | worker:claude-code:local | ..." }),
			),
			capability: Type.Optional(Type.String()),
			instructions: Type.Optional(Type.String()),
			budget: Type.Optional(
				Type.Object({
					wall_time_sec: Type.Optional(Type.Number()),
					model_tokens: Type.Optional(Type.Number()),
				}),
			),
		}),
		async execute(_id, params, signal, onUpdate, _ctx) {
			counter += 1;
			const requestId = `ptr_delegate_${Date.now()}_${counter}`;
			const state = ctx.active.current;
			const request = {
				schema_version: "rosclaw.pi_tool_request.v1",
				request_id: requestId,
				pi_session_id: state.sessionId,
				mission_id: state.missionId,
				context_revision: state.contextRevision,
				body_hash: state.bodyHash ?? "",
				mode: state.mode,
				tool_name: "rosclaw_delegate",
				arguments: params,
				requested_at: new Date().toISOString(),
				idempotency_key: `idem_${requestId}`,
				actor: { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 },
			};
			// 后台委派 + 轮询 worker 状态原位更新（规格 §19.3）。
			const done = ctx.center.call("pi.tools.execute", { request });
			const poll = async () => {
				while (true) {
					await new Promise((resolve) => setTimeout(resolve, 2000));
					if (signal?.aborted) return;
					try {
						const status = await ctx.center.call("pi.worker.status", {
							mission_id: state.missionId,
						});
						const orders = (status.orders ?? []) as Array<Record<string, unknown>>;
						const latest = orders[orders.length - 1];
						if (latest) {
							onUpdate?.({
								content: [
									{
										type: "text",
										text: `Worker ${String(latest.assigned_to ?? "?")}: ${String(latest.status ?? "")}`,
									},
								],
								details: { status: latest.status },
							});
							if (["ACCEPTED", "FAILED", "VERIFY_FAILED"].includes(String(latest.status))) return;
						}
					} catch {
						return;
					}
				}
			};
			const polling = poll();
			try {
				const response = await done;
				const result = (response.result ?? {}) as {
					ok?: boolean;
					summary?: string;
					error_code?: string;
				};
				const ok = response.ok === true;
				return {
					content: [
						{
							type: "text" as const,
							text: ok
								? (result.summary ?? "worker completed")
								: `Worker 未通过验证或被拒 [${result.error_code ?? "?"}]: ${result.summary ?? response.error ?? ""}`,
						},
					],
					details: { ok, error_code: result.error_code ?? null },
					isError: !ok,
				};
			} finally {
				await polling.catch(() => undefined);
			}
		},
	});
}
