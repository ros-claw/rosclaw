/** 结构化活动区（PR-N9，调整方案 §八）——"Working…"替换。
 *
 * 展示可审计事件（当前工具/Operation 阶段）与诚实的 Provider 等待
 * 阶段/耗时，不是思维链，也不是静态 spinner 文案。thinking 等模型
 * 内容文本绝不上屏——这里只有阶段名与计时数字。
 */

import type { ProviderWaitPhase } from "../native/provider-watchdog.js";

export interface ActivityPhase {
	currentTool: string | null;
	operation: { id: string; label: string; detail?: string } | null;
	/** 真实 Provider 等待阶段与累计等待毫秒（暂停期不计入）。
	 *  缺省 = 无 Provider 计时上下文（保持旧 "Working…" 兜底）。 */
	provider?: { phase: ProviderWaitPhase; elapsedMs: number } | null;
}

function elapsedSeconds(ms: number): string {
	return `${Number((Math.max(0, ms) / 1_000).toFixed(1))}s`;
}

export function phaseWorkingMessage(phase: ActivityPhase): string {
	if (phase.operation) {
		const detail = phase.operation.detail ? ` · ${phase.operation.detail}` : "";
		return `运行 ${phase.operation.label}（${phase.operation.id}）${detail}`;
	}
	if (phase.currentTool) {
		return `调用 ${phase.currentTool}`;
	}
	if (phase.provider) {
		const waited = elapsedSeconds(phase.provider.elapsedMs);
		switch (phase.provider.phase) {
			case "waiting":
				return `等待模型首个响应（已等待 ${waited}）`;
			case "streaming":
				return `模型输出中（累计等待 Provider ${waited}）`;
			case "user_decision":
				return `等待用户确认（Provider 计时已暂停，暂停前等待 ${waited}）`;
			case "tool":
				return `工具执行中（Provider 计时暂停，暂停前等待 ${waited}）`;
			case "idle":
				break;
		}
	}
	return "Working…";
}
