/** State reducer (批次 C)：AgentEventV2 → UiState + 渲染效果。
 *
 * 纯函数，可单测；TUI 只负责把 effects 画出来。
 * agent.settled 是停止 spinner 的唯一可靠信号（§5.4）。
 */

import type { AgentEvent } from "../client/sse.js";
import type { UiState } from "./types.js";

export type Effect =
	| { kind: "append_markdown"; text: string }
	| { kind: "append_delta"; text: string }
	| { kind: "flush_delta" }
	| { kind: "append_card"; card: CardModel }
	| { kind: "spinner"; label: string }
	| { kind: "spinner_stop" }
	| { kind: "status_refresh" };

export interface CardModel {
	cardType: "tool" | "worker" | "approval" | "receipt" | "info";
	title: string;
	lines: string[];
	tone: "ok" | "warn" | "error" | "info";
}

const PHASE_BY_EVENT: Record<string, string> = {
	"agent.started": "正在理解",
	"context.compilation.started": "正在编译上下文",
	"context.compilation.completed": "正在编译上下文",
	"model.request.started": "正在调用模型",
	"model.retry.scheduled": "正在重试",
	"model.failover": "正在切换模型",
	"tool.started": "正在调用工具",
	"worker.started": "正在等待 Worker",
	"approval.requested": "正在等待批准",
	"action.dispatched": "正在执行/验证",
	"verification.started": "正在执行/验证",
	"compaction.started": "正在压缩上下文",
};

export function reduce(state: UiState, event: AgentEvent): Effect[] {
	const effects: Effect[] = [];
	const payload = event.payload ?? {};
	state.lastSeq = Math.max(state.lastSeq, event.sequence);

	const phase = PHASE_BY_EVENT[event.type];
	if (phase) {
		state.phase = phase;
		effects.push({ kind: "spinner", label: phase });
	}

	switch (event.type) {
		case "agent.started":
			state.turnInFlight = true;
			break;
		case "agent.settled":
			state.turnInFlight = false;
			state.phase = "";
			effects.push({ kind: "spinner_stop" });
			effects.push({ kind: "flush_delta" });
			break;
		case "agent.failed":
			state.turnInFlight = false;
			effects.push({
				kind: "append_card",
				card: {
					cardType: "info",
					title: "Agent 失败",
					lines: [String(payload.error ?? "unknown")],
					tone: "error",
				},
			});
			break;
		case "model.text.delta":
			effects.push({ kind: "append_delta", text: String(payload.text ?? "") });
			break;
		case "model.request.ended":
			effects.push({ kind: "flush_delta" });
			break;
		case "mission.state.changed":
			state.missionState = String(payload.to ?? state.missionState);
			effects.push({ kind: "status_refresh" });
			break;
		case "mission.renamed":
			state.missionName = String(payload.name ?? state.missionName);
			effects.push({ kind: "status_refresh" });
			break;
		case "tool.started": {
			const name = String(payload.name ?? payload.tool ?? "tool");
			state.tools.push({ name, status: "running" });
			effects.push({
				kind: "append_card",
				card: {
					cardType: "tool",
					title: `⚙ ${name}`,
					lines: ["running…"],
					tone: "info",
				},
			});
			break;
		}
		case "tool.completed": {
			const name = String(payload.name ?? payload.tool ?? "tool");
			const ok = payload.ok !== false;
			const run = state.tools.findLast((t) => t.name === name);
			if (run) run.status = ok ? "completed" : "failed";
			const lines = [ok ? "completed" : "failed"];
			if (payload.artifact_ref) lines.push(`artifact: ${String(payload.artifact_ref)}`);
			effects.push({
				kind: "append_card",
				card: {
					cardType: "tool",
					title: `⚙ ${name}`,
					lines,
					tone: ok ? "ok" : "error",
				},
			});
			break;
		}
		case "worker.offered":
		case "worker.claimed":
		case "worker.started":
		case "worker.submitted":
		case "worker.verifying":
		case "worker.accepted":
		case "worker.failed":
		case "worker.expired": {
			const statusText = event.type.split(".")[1];
			const orderId = String(payload.work_order_id ?? "");
			const workerId = String(payload.worker_id ?? "");
			const existing = state.workers.find((w) => w.workOrderId === orderId);
			if (existing) existing.status = statusText;
			else state.workers.push({ workOrderId: orderId, workerId, status: statusText });
			const terminal = ["accepted", "failed", "expired"].includes(statusText);
			effects.push({
				kind: "append_card",
				card: {
					cardType: "worker",
					title: `⛏ ${workerId || "worker"}`,
					lines: [`order: ${orderId}`, `status: ${statusText}`],
					tone: statusText === "accepted" ? "ok" : statusText === "failed" || statusText === "expired" ? "error" : "info",
				},
			});
			if (terminal) effects.push({ kind: "status_refresh" });
			break;
		}
		case "approval.requested": {
			const item = {
				requestId: String(payload.request_id ?? ""),
				title: String(payload.title ?? "动作授权"),
				riskTier: String(payload.risk_tier ?? "LOW"),
				expiresAt: payload.expires_at ? String(payload.expires_at) : undefined,
			};
			state.pendingApprovals.push(item);
			effects.push({
				kind: "append_card",
				card: {
					cardType: "approval",
					title: `🔐 授权请求 ${shortId(item.requestId)}`,
					lines: [
						item.title,
						`风险等级: ${item.riskTier}`,
						"输入 /approve " + shortId(item.requestId) + " 批准，/deny " + shortId(item.requestId) + " 拒绝",
					],
					tone: "warn",
				},
			});
			effects.push({ kind: "status_refresh" });
			break;
		}
		case "approval.decided": {
			const requestId = String(payload.request_id ?? "");
			state.pendingApprovals = state.pendingApprovals.filter((a) => a.requestId !== requestId);
			effects.push({
				kind: "append_card",
				card: {
					cardType: "approval",
					title: `🔐 授权${payload.approved ? "已批准" : "已拒绝"}`,
					lines: [shortId(requestId)],
					tone: payload.approved ? "ok" : "error",
				},
			});
			effects.push({ kind: "status_refresh" });
			break;
		}
		case "action.receipt":
		case "receipt.received":
			effects.push({
				kind: "append_card",
				card: {
					cardType: "receipt",
					title: "🧾 执行回执",
					lines: Object.entries(payload)
						.slice(0, 6)
						.map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`),
					tone: "ok",
				},
			});
			break;
		case "compaction.completed":
			state.compactions += 1;
			effects.push({
				kind: "append_card",
				card: {
					cardType: "info",
					title: "🗜 上下文已压缩",
					lines: [
						`reason: ${String(payload.reason ?? "")}`,
						`tokens: ${String(payload.tokens_before ?? "?")} → ${String(payload.tokens_after ?? "?")}`,
						"canonical journal 完整保留",
					],
					tone: "info",
				},
			});
			break;
		case "warning":
			effects.push({
				kind: "append_card",
				card: { cardType: "info", title: "⚠ 警告", lines: [String(payload.message ?? "")], tone: "warn" },
			});
			break;
		case "error":
			effects.push({
				kind: "append_card",
				card: { cardType: "info", title: "✖ 错误", lines: [String(payload.error ?? "")], tone: "error" },
			});
			break;
		default:
			break;
	}
	return effects;
}

export function shortId(id: string): string {
	return id.length > 14 ? id.slice(0, 14) : id;
}
