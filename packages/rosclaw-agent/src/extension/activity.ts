/** 结构化活动区（PR-N9，调整方案 §八）——"Working…"替换。
 *
 * 展示可审计事件（当前工具/Operation 阶段），不是思维链，也不是
 * 静态 spinner 文案。
 */

export interface ActivityPhase {
	currentTool: string | null;
	operation: { id: string; label: string; detail?: string } | null;
}

export function phaseWorkingMessage(phase: ActivityPhase): string {
	if (phase.operation) {
		const detail = phase.operation.detail ? ` · ${phase.operation.detail}` : "";
		return `运行 ${phase.operation.label}（${phase.operation.id}）${detail}`;
	}
	if (phase.currentTool) {
		return `调用 ${phase.currentTool}`;
	}
	return "Working…";
}
