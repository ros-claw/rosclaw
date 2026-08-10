/** `rosclaw_task` 工具（八审 §2.2/P0-5）：任务级入口。
 *
 * 已知任务（draw_shape 等）走确定性 Task Compiler/Runner——模型只交
 * TaskSpec（goal + 业务参数），内核完成规划、策略判定、单动作执行与
 * 自动验证；模型不搬轨迹/hash/lease/grant，不逐点控制。
 *
 * ASK 策略下两阶段：submit 返回 WAITING_APPROVAL（展卡）→ 人工决定
 * → resume（task_id）执行+验证。与 rosclaw_request_action 共用同一
 * overlay/决定通道。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

const AWAIT_TIMEOUT_MS = 330_000;

export function buildTaskTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_task",
		label: "ROSClaw Task",
		description:
			"Run a known task end-to-end via the deterministic task compiler: " +
			"planning, safety policy, ONE task-level action, and automatic " +
			"verification. Prefer this over hand-chaining capabilities for known " +
			"goals. goal='draw_shape' with parameters {shape:'star5', " +
			"center_m:[x,y,z], radius_m} draws a closed 5-point star on the " +
			"current body. Safe first-party SIM runs under POLICY_AUTO (no human " +
			"card); REAL is always gated by rosclawd/operator.",
		parameters: Type.Object({
			goal: Type.String({ description: "task goal id (e.g. draw_shape)" }),
			parameters: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
		}),
		async execute(_id, params, signal, onUpdate, _ctx2) {
			const state = ctx.active.current;
			if (!state.missionId) {
				return {
					content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
					details: { ok: false, error_code: "NO_MISSION" },
					isError: true,
				};
			}
			const callTask = (args: Record<string, unknown>, idemSuffix: string) =>
				ctx.center.call("pi.tools.execute", {
					request: {
						schema_version: "rosclaw.pi_tool_request.v1",
						request_id: `ptr_task_${Date.now()}`,
						pi_session_id: state.sessionId,
						mission_id: state.missionId,
						context_revision: state.contextRevision,
						context_lease_id: state.contextLeaseId ?? "",
						tool_name: "rosclaw_task",
						arguments: args,
						requested_at: new Date().toISOString(),
						idempotency_key: `idem_task_${state.sessionId}_${idemSuffix}`,
						actor: { engine: "pi" },
					},
				});
			const parse = (response: Record<string, unknown>) => {
				const result = (response.result ?? {}) as { summary?: string };
				try {
					return JSON.parse(String(result.summary ?? "{}")) as Record<string, unknown>;
				} catch {
					return { state: "FAILED", error: String(result.summary ?? "") };
				}
			};
			const notifyView = (payload: Record<string, unknown>) => {
				const view = String(payload.user_view ?? "");
				if (view) {
					onUpdate?.({
						content: [{ type: "text" as const, text: view }],
						details: { phase: "TASK_PROGRESS", task_id: payload.task_id ?? "" },
					});
				}
			};
			const submitted = parse(
				await callTask(
					{
						goal: String(params.goal),
						parameters: (params.parameters ?? {}) as Record<string, unknown>,
					},
					`${Date.now()}`,
				),
			);
			const taskState = String(submitted.state ?? "FAILED");
			notifyView(submitted);
			if (taskState === "VERIFIED") {
				// POLICY_AUTO——安全 SIM 政策自动执行，只通知不弹卡。
				onUpdate?.({
					content: [{ type: "text", text: "安全仿真自动执行（POLICY_AUTO，全链审计）" }],
					details: { phase: "POLICY_AUTO", approval_id: submitted.approval_id ?? "" },
				});
				// 模型面向结构化结果（state/verification 是完成判定的
				// 权威——不让模型从自然语言猜）。
				return {
					content: [{ type: "text" as const, text: JSON.stringify(submitted) }],
					details: { ok: true, ...submitted },
				};
			}
			if (taskState !== "WAITING_APPROVAL") {
				return {
					content: [
						{
							type: "text" as const,
							text: `任务${taskState}：${String(submitted.error ?? "")}`,
						},
					],
					details: { ok: false, error_code: taskState, ...submitted },
					isError: true,
				};
			}
			// ASK：展卡 → 等人工决定 → resume。
			const approvalId = String(submitted.approval_id ?? "");
			const taskId = String(submitted.task_id ?? "");
			onUpdate?.({
				content: [
					{
						type: "text" as const,
						text: `等待 Operator 决定（approval ${approvalId}）…默认拒绝`,
					},
				],
				details: {
					phase: "AWAITING_OPERATOR",
					approval_id: approvalId,
					display_hash: String(submitted.display_hash ?? ""),
				},
			});
			const deadline = Date.now() + AWAIT_TIMEOUT_MS;
			let status = "PENDING";
			while (Date.now() < deadline) {
				if (signal?.aborted) {
					return {
						content: [{ type: "text" as const, text: "已取消（未执行）" }],
						details: { ok: false, status: "CANCELLED", task_id: taskId },
						isError: true,
					};
				}
				const current = await ctx.center.call("pi.action.status", {
					pi_session_id: state.sessionId,
					approval_id: approvalId,
				});
				status = String(current.status ?? "PENDING");
				if (status !== "PENDING") break;
				await new Promise((resolve) => setTimeout(resolve, 1500));
			}
			if (status !== "APPROVED") {
				return {
					content: [
						{
							type: "text" as const,
							text: `Operator 未批准（${status}）——任务未执行，无 grant`,
						},
					],
					details: { ok: false, status: "DENIED", task_id: taskId, error_code: "OPERATOR_DECLINED" },
					isError: true,
				};
			}
			const resumed = parse(await callTask({ task_id: taskId }, `resume_${taskId}`));
			notifyView(resumed);
			const verified = String(resumed.state ?? "") === "VERIFIED";
			return {
				content: [{ type: "text" as const, text: JSON.stringify(resumed) }],
				details: { ok: verified, ...resumed },
				isError: !verified,
			};
		},
	});
}
