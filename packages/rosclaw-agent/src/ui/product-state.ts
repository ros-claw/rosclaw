/** Product chrome 渲染（三审 P0-NA-16 + 六审 PR-SIX-1）：
 * Header/Footer 都从同一个 KernelSnapshotV1 渲染——纯函数，不读缓存。
 *
 * 红线：
 * - 快照只能来自 ProductStateCenter（字段全部权威源）；
 * - 未完成的 bootstrap 显示 LOADING/UNKNOWN——绝不乐观默认；
 * - 模型不能决定或改写 UI 安全状态；
 * - Action 状态显示首要受阻原因（不是干巴巴的 LOCKED）。
 */

import type {
	ActionReadinessV1,
	KernelSnapshotV1,
} from "../session/state-center.js";

/** 受阻原因码 → 一行可读说明（机器合约保持英文 code）。 */
const REASON_LABEL: Record<string, string> = {
	NO_MISSION: "未绑定 Mission",
	KERNEL_UNREACHABLE: "内核不可达",
	NO_WRITER_LEASE: "无写租约",
	CONTEXT_STALE: "上下文过期",
	NO_CONTEXT_LEASE: "无上下文租约",
	OPERATOR_OFFLINE: "操作员离线",
};

export function renderActionState(readiness: ActionReadinessV1): string {
	if (readiness.state === "READY") return "Action READY";
	const primary = readiness.reason_codes[0] ?? "UNKNOWN";
	const label = REASON_LABEL[primary];
	const reason = label ? `${primary}（${label}）` : primary;
	return `Action BLOCKED (${reason})`;
}

/** 推荐头部（P0-NA-16 规格 + P0-5F Action 状态 + 六审首要受阻原因）：
 *   ROSClaw 1.2.0 · SIMULATION · Kimi K3
 *   Mission mis_... · Body sim/ur5e · Context FRESH r12 · Operator OFFLINE · Action BLOCKED (NO_WRITER_LEASE)
 */
export function renderHeader(state: KernelSnapshotV1): string {
	const line1Parts = [`ROSClaw ${state.product_version}`, state.mode];
	if (state.model) line1Parts.push(state.model);
	const line1 = line1Parts.join(" · ");
	if (!state.mission_id) {
		return `${line1}\n未绑定 Mission · /help 查看命令`;
	}
	const body = state.body_id ?? "LOADING";
	const context =
		state.context_state === "FRESH"
			? `Context FRESH r${state.context_revision}`
			: `Context ${state.context_state}`;
	const kernel = state.kernel === "READY" ? "" : ` · Kernel ${state.kernel}`;
	const line2 =
		`Mission ${state.mission_id.slice(0, 24)} · Body ${body} · ` +
		`${context} · Operator ${state.operator}${kernel} · ` +
		renderActionState(state.action_readiness);
	return `${line1}\n${line2}`;
}

/** Footer：与 Header 同一快照——model/mode/operator 绝不分叉。 */
export function renderFooter(state: KernelSnapshotV1): string {
	const model = state.model || "未选模型";
	const parts = [model, state.mode, `Operator ${state.operator}`];
	if (state.kernel !== "READY") parts.push(`Kernel ${state.kernel}`);
	return parts.join(" · ");
}
