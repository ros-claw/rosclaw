/** rosclaw_delegate 工具（十审 W0 重写）：真正异步委派。
 *
 * 废止的旧行为（P0-WORKER-BLOCK / P0-ORDER-CORRELATION）：
 * - await 整个 run_to_completion——父会话被"Working…"阻塞数小时；
 * - 轮询 mission 全部 orders 取最后一条——并发/残留时等错任务；
 * - finally 等 polling 结束——第二种无限等待。
 *
 * 新语义：
 * - agentd 端 hire + 后台驱动，工具在 grace（≤3s）内返回；
 * - 快任务 grace 内完成 → 直接返回已验证结果（旧体验保持）；
 * - 慢任务返回 STARTED + 精确 WorkOrder ID/worker/预算/deadline
 *   （第一屏信息是审计硬要求）；
 * - work_order_id 由本工具预生成并随请求下发——signal abort 在响应
 *   返回前也能按精确 ID cancel（闭环到进程组 kill）。
 */

import { randomBytes } from "node:crypto";

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

let counter = 0;

function buildRequest(
	ctx: BridgeToolContext,
	toolName: string,
	arguments_: Record<string, unknown>,
) {
	counter += 1;
	const state = ctx.active.current;
	return {
		schema_version: "rosclaw.pi_tool_request.v1",
		request_id: `ptr_${toolName}_${Date.now()}_${counter}`,
		pi_session_id: state.sessionId,
		mission_id: state.missionId,
		context_revision: state.contextRevision,
		body_hash: state.bodyHash ?? "",
		mode: state.mode,
		tool_name: toolName,
		arguments: arguments_,
		requested_at: new Date().toISOString(),
		idempotency_key: `idem_${toolName}_${Date.now()}_${counter}`,
		actor: { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 },
	};
}

/** abort → 精确 ID cancel（尽力而为，不阻塞 abort 路径）。 */
function cancelOnAbort(ctx: BridgeToolContext, signal: AbortSignal | undefined, workOrderId: string) {
	if (!signal) return;
	signal.addEventListener(
		"abort",
		() => {
			const request = buildRequest(ctx, "rosclaw_cancel_work", {
				work_order_id: workOrderId,
				reason: "tool_abort",
			});
			void ctx.center.call("pi.tools.execute", { request }).catch(() => undefined);
		},
		{ once: true },
	);
}

export function buildDelegateTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_delegate",
		label: "ROSClaw Delegate",
		description:
			"Start a background worker for a self-contained subtask. Returns " +
			"immediately with a precise WorkOrder ID (worker, budget, soft target). " +
			"Poll with rosclaw_check_work; cancel with rosclaw_cancel_work. " +
			"IMPORTANT: never invent a hard deadline — a worker that is making " +
			"progress keeps running; only the user can set hard limits. Worker " +
			"output enters the main context only after ROSClaw verification passes.",
		parameters: Type.Object({
			goal: Type.String({ description: "self-contained subtask goal" }),
			worker_id: Type.Optional(
				Type.String({ description: "auto | worker:rosclaw:pi | worker:native:basic | ..." }),
			),
			worker_profile: Type.Optional(
				Type.String({ description: "built-in worker profile: scout | analyst | developer | sim-builder" }),
			),
			workspace: Type.Optional(
				Type.String({
					description:
						"target workspace path (developer/sim-builder only; a git repo gets an isolated worktree)",
				}),
			),
			base_ref: Type.Optional(
				Type.String({ description: "git base ref for the worktree (default HEAD)" }),
			),
			// 十二审 PR-12.4：WorkSpecV2——任务类型 + 交付物（验收按此，
			// 不再 profile 硬编码）。
			task_type: Type.Optional(
				Type.String({ description: "analyze | code_change | artifact_build | simulation_run" }),
			),
			deliverables: Type.Optional(
				Type.Array(
					Type.Object({
						id: Type.Optional(Type.String()),
						media_types: Type.Array(Type.String()),
						required: Type.Optional(Type.Boolean()),
					}),
					{ description: "required deliverables, e.g. [{media_types:['image/gif']}]" },
				),
			),
			capability: Type.Optional(Type.String()),
			instructions: Type.Optional(Type.String()),
			budget: Type.Optional(
				Type.Object({
					// 十三审：这是 soft target（提醒用），不是处决时间——
					// Worker 有进度就让它继续。
					wall_time_sec: Type.Optional(Type.Number()),
					model_tokens: Type.Optional(Type.Number()),
				}),
			),
			execution_policy: Type.Optional(
				Type.Object({
					soft_target_sec: Type.Optional(Type.Number()),
					hard_deadline_sec: Type.Optional(Type.Number()),
					hard_deadline_source: Type.Optional(Type.String()),
					token_soft_limit: Type.Optional(Type.Number()),
				}),
			),
		}),
		async execute(_id, params, signal, _onUpdate, toolCtx) {
			const workOrderId = `wo_${randomBytes(8).toString("hex")}`;
			// 十审 W1：无 secret 模型快照——Worker 与主 Agent 同一模型配置
			// （凭据由子进程从同一 agentDir/auth.json 读取，绝不经 WorkOrder）。
			const model = (toolCtx as { model?: { provider: string; id: string } } | undefined)?.model;
			const thinking = (toolCtx as { thinkingLevel?: string } | undefined)?.thinkingLevel;
			const boundWorkspace = ctx.workspace?.() ?? undefined;
			const request = buildRequest(ctx, "rosclaw_delegate", {
				...(params as Record<string, unknown>),
				// 十一审 PR-D：模型未指定时默认绑定 workspace（一等状态，
				// 不靠自然语言路径）。
				...(!(params as Record<string, unknown>).workspace && boundWorkspace
					? { workspace: boundWorkspace }
					: {}),
				work_order_id: workOrderId,
				...(model
					? { model_snapshot: { provider: model.provider, model: model.id, ...(thinking ? { thinking } : {}) } }
					: {}),
			});
			cancelOnAbort(ctx, signal, workOrderId);
			const response = await ctx.center.call("pi.tools.execute", { request });
			const result = (response.result ?? {}) as {
				ok?: boolean;
				status?: string;
				summary?: string;
				error_code?: string;
				artifact_refs?: string[];
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
				details: {
					ok,
					status: result.status ?? null,
					error_code: result.error_code ?? null,
					work_order_id: workOrderId,
				},
				isError: !ok,
			};
		},
	});
}

export function buildRetryWorkTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_retry_work",
		label: "ROSClaw Retry Work",
		description:
			"Retry a terminal WorkOrder as a new attempt. Carries recorded steer " +
			"notes into the new attempt and preserves parent/root lineage.",
		parameters: Type.Object({
			work_order_id: Type.String({ description: "exact wo_... id of the TERMINAL order" }),
		}),
		async execute(_id, params, _signal, _onUpdate, _toolCtx) {
			const request = buildRequest(ctx, "rosclaw_retry_work", params as Record<string, unknown>);
			const response = await ctx.center.call("pi.tools.execute", { request });
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
							? (result.summary ?? "")
							: `retry 被拒 [${result.error_code ?? response.code ?? "?"}]: ${result.summary ?? response.error ?? ""}`,
					},
				],
				details: { ok, error_code: result.error_code ?? null },
				isError: !ok,
			};
		},
	});
}

export function buildListWorkTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_list_work",
		label: "ROSClaw List Work",
		description: "List background WorkOrders of the current mission with status.",
		parameters: Type.Object({}),
		async execute(_id, _params, _signal, _onUpdate, _toolCtx) {
			const request = buildRequest(ctx, "rosclaw_list_work", {});
			const response = await ctx.center.call("pi.tools.execute", { request });
			const result = (response.result ?? {}) as { ok?: boolean; summary?: string };
			const ok = response.ok === true;
			return {
				content: [{ type: "text" as const, text: result.summary ?? "" }],
				details: { ok },
				isError: !ok,
			};
		},
	});
}

export function buildUpdateWorkTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_update_work",
		label: "ROSClaw Update Work",
		description:
			"Append a constraint/steer note to a running WorkOrder. NOTE: takes " +
			"effect on retry/next attempt; to redirect immediately, cancel and " +
			"re-delegate.",
		parameters: Type.Object({
			work_order_id: Type.String({ description: "exact wo_... id" }),
			note: Type.String({ description: "the additional constraint or steering note" }),
		}),
		async execute(_id, params, _signal, _onUpdate, _toolCtx) {
			const request = buildRequest(ctx, "rosclaw_update_work", params as Record<string, unknown>);
			const response = await ctx.center.call("pi.tools.execute", { request });
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
							? (result.summary ?? "")
							: `更新被拒 [${result.error_code ?? response.code ?? "?"}]: ${result.summary ?? response.error ?? ""}`,
					},
				],
				details: { ok, error_code: result.error_code ?? null },
				isError: !ok,
			};
		},
	});
}

export function buildCheckWorkTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_check_work",
		label: "ROSClaw Check Work",
		description:
			"Check a background WorkOrder by its exact ID: current phase, terminal " +
			"verdict, verified summary and artifacts.",
		parameters: Type.Object({
			work_order_id: Type.String({ description: "exact wo_... id returned by rosclaw_delegate" }),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const request = buildRequest(ctx, "rosclaw_check_work", params as Record<string, unknown>);
			const response = await ctx.center.call("pi.tools.execute", { request });
			const result = (response.result ?? {}) as {
				ok?: boolean;
				status?: string;
				summary?: string;
				error_code?: string;
			};
			const ok = response.ok === true;
			return {
				content: [
					{
						type: "text" as const,
						text: ok
							? (result.summary ?? "")
							: `查询被拒 [${result.error_code ?? response.code ?? "?"}]: ${result.summary ?? response.error ?? ""}`,
					},
				],
				details: {
					ok,
					status: result.status ?? null,
					error_code: result.error_code ?? null,
					work_order_id: (params as { work_order_id?: string }).work_order_id ?? null,
				},
				isError: !ok,
			};
		},
	});
}

export function buildCancelWorkTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_cancel_work",
		label: "ROSClaw Cancel Work",
		description:
			"Cancel a running background WorkOrder by exact ID. Kills the worker " +
			"process tree; the order transitions to CANCELLED.",
		parameters: Type.Object({
			work_order_id: Type.String({ description: "exact wo_... id" }),
			reason: Type.Optional(Type.String()),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const request = buildRequest(ctx, "rosclaw_cancel_work", params as Record<string, unknown>);
			const response = await ctx.center.call("pi.tools.execute", { request });
			const result = (response.result ?? {}) as {
				ok?: boolean;
				status?: string;
				summary?: string;
				error_code?: string;
			};
			const ok = response.ok === true;
			return {
				content: [
					{
						type: "text" as const,
						text: ok
							? (result.summary ?? "")
							: `取消失败 [${result.error_code ?? response.code ?? "?"}]: ${result.summary ?? response.error ?? ""}`,
					},
				],
				details: {
					ok,
					status: result.status ?? null,
					error_code: result.error_code ?? null,
					work_order_id: (params as { work_order_id?: string }).work_order_id ?? null,
				},
				isError: !ok,
			};
		},
	});
}
