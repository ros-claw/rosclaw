/** rosclaw_request_action 工具（NA-FIX-4/5，二次审计 P0-4/P0-5/P0-6）。
 *
 * 两阶段协调（P0-5 修复——不再在 tool_execution_start 盲等卡片）：
 *   execute → pi.action.propose → approval_id + 卡片
 *   → onUpdate({phase: AWAITING_OPERATOR, approval_id, display_hash})
 *   → 轮询 pi.action.status（TUI 按 approval_id 精确展卡收集 Y/N）
 *   → APPROVED 后 pi.action.execute → 结构化 ExecutionReceipt。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { bridgeCall } from "../bridge/bridge-client.js";
import type { BridgeToolContext } from "./bridge-tools.js";

const AWAIT_TIMEOUT_MS = 330_000;

export function buildRequestActionTool(ctx: BridgeToolContext) {
	return defineTool({
		name: "rosclaw_request_action",
		label: "ROSClaw Request Action",
		description:
			"Propose one bounded physical action. Creates an approval card; a human " +
			"operator decides (Y/N). You cannot approve. On approval the action is " +
			"executed via rosclawd and a structured receipt is returned.",
		parameters: Type.Object({
			capability_id: Type.String({ description: "exact catalog capability id" }),
			arguments: Type.Record(Type.String(), Type.Unknown()),
			expected_effect: Type.Optional(Type.String()),
			risk_tier: Type.Optional(Type.String({ description: "LOW|MEDIUM|HIGH|CRITICAL" })),
			verification_plan: Type.Optional(Type.Array(Type.String())),
		}),
		async execute(_id, params, signal, onUpdate, _ctx) {
			const state = ctx.active.current;
			if (!state.missionId) {
				return {
					content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
					details: { ok: false },
					isError: true,
				};
			}
			// phase 1: propose（卡片存在后才通知 UI——P0-5 顺序修复）。
			// P0-NA-10：完整请求上下文——session/revision/body/mode/idempotency
			// 一个都不能少（admission 硬校验，缺即拒）。
			const requestContext = {
				pi_session_id: state.sessionId,
				mission_id: state.missionId,
				context_revision: state.contextRevision,
				body_hash: state.bodyHash ?? "",
				mode: state.mode,
				idempotency_key: `idem_reqact_${state.sessionId}_${Date.now()}`,
				// HOTFIX-1：agentd 签发的 ValidatedContextLease——无 lease
				// 即 CONTEXT_LEASE_REQUIRED（fail closed）。
				context_lease_id: state.contextLeaseId ?? "",
			};
			const proposed = await bridgeCall(ctx.rosclawHome, "pi.action.propose", {
				...requestContext,
				capability_id: String(params.capability_id),
				arguments: params.arguments ?? {},
				expected_effect: String(params.expected_effect ?? params.capability_id),
				risk_tier: String(params.risk_tier ?? "LOW"),
			});
			if (!proposed.ok) {
				return {
					content: [
						{
							type: "text" as const,
							text: `动作提案被拒 [${String(proposed.code ?? "")}]: ${String(proposed.error ?? "")}`,
						},
					],
					details: { ok: false },
					isError: true,
				};
			}
			const card = proposed.card as {
				approval_id: string;
				display_hash: string;
				expires_at: string;
			};
			onUpdate?.({
				content: [
					{
						type: "text",
						text: `等待 Operator 决定（approval ${card.approval_id}）…默认拒绝`,
					},
				],
				details: {
					phase: "AWAITING_OPERATOR",
					approval_id: card.approval_id,
					display_hash: card.display_hash,
					expires_at: card.expires_at,
				},
			});
			// phase 2a: 等 operator 决定（超时/中断 = 拒绝语义）。
			const deadline = Date.now() + AWAIT_TIMEOUT_MS;
			let status = "PENDING";
			while (Date.now() < deadline) {
				if (signal?.aborted) {
					return {
						content: [
							{
								type: "text" as const,
								text: `已中断——approval ${card.approval_id} 按取消语义处理（未执行）`,
							},
						],
						details: { ok: false, approval_id: card.approval_id, cancelled: true },
						isError: true,
					};
				}
				const current = await bridgeCall(ctx.rosclawHome, "pi.action.status", {
					// HOTFIX-1：status 也做卡主校验——必须带 session。
					pi_session_id: state.sessionId,
					approval_id: card.approval_id,
				});
				status = String(current.status ?? "PENDING");
				if (status !== "PENDING") break;
				await new Promise((resolve) => setTimeout(resolve, 1500));
			}
			if (status === "PENDING") {
				return {
					content: [
						{
							type: "text" as const,
							text: `Operator 未在期限内决定（默认拒绝）——approval ${card.approval_id} 未执行`,
						},
					],
					details: { ok: false, approval_id: card.approval_id, error_code: "APPROVAL_TIMEOUT" },
					isError: true,
				};
			}
			if (status !== "APPROVED") {
				return {
					content: [
						{
							type: "text" as const,
							text: `Operator 拒绝了该动作（${status}）——未执行，无 grant`,
						},
					],
					details: { ok: false, approval_id: card.approval_id, error_code: "OPERATOR_DECLINED" },
					isError: true,
				};
			}
			onUpdate?.({
				content: [{ type: "text", text: "已批准——执行中…" }],
				details: { phase: "EXECUTING", approval_id: card.approval_id },
			});
			// phase 2b: 精确 grant 执行 → 结构化回执。
			// P0-NA-10：execute 带同一请求上下文做 TOCTOU 复验——批准后
			// revision/body/lease 任一变化都必须拒绝。
			const executed = await bridgeCall(ctx.rosclawHome, "pi.action.execute", {
				...requestContext,
				approval_id: card.approval_id,
			});
			const result = (executed.result ?? {}) as Record<string, unknown>;
			return {
				content: [
					{
						type: "text" as const,
						text: String(result.summary ?? result.status ?? ""),
					},
				],
				details: {
					ok: result.executed === true,
					status: result.status,
					approval_id: card.approval_id,
					grant_id: result.grant_id ?? null,
					terminal_receipt: result.terminal_receipt ?? false,
					// P0-NA-13：结构化证据引用（receipt://action_id）随结果
					// 返回——/evidence 按本回合 action 精确展示，不是摘要。
					evidence_ref: result.evidence_ref ?? null,
					error_code: result.error_code ?? null,
				},
				isError: result.executed !== true && result.status !== "DECLINED",
			};
		},
	});
}
