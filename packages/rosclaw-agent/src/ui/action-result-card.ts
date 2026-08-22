// HP2-COMPAT: pi-tui 组件原语（Component/Text）——TUI 渲染原语，HP3 前保持；不新增会话装配引用。
/** ActionResultCard（五审 P0-5F）：kernel 渲染的不可变动作结果卡。
 *
 * 动作终态只能来自结构化 ExecutionOutcome——模型自然语言不得
 * 宣布/改写完成状态。本卡由 ROSClaw runtime 在 tool_execution_end
 * 渲染，内容是 outcome 的结构化字段，与模型叙述无关。
 */

import type { Component } from "@earendil-works/pi-tui";

export interface ActionResultData {
	status: string; // COMPLETED | FAILED | DECLINED | REJECTED | CANCELLED
	capabilityId: string;
	approvalId?: string;
	grantId?: string;
	txnId?: string;
	actionId?: string;
	receiptId?: string;
	errorCode?: string;
	verified?: boolean;
}

const STATUS_LABEL: Record<string, string> = {
	COMPLETED: "✓ 动作已完成（receipt 已验证）",
	FAILED: "✗ 动作失败（fail closed）",
	DECLINED: "⊘ Operator 拒绝——动作未执行",
	REJECTED: "⊘ 动作被准入拒绝——未执行",
	CANCELLED: "⊘ 动作已取消——未执行",
};

export class ActionResultCardComponent implements Component {
	constructor(private readonly data: ActionResultData) {}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(10, Math.min(width - 4, 60)));
		const label = STATUS_LABEL[this.data.status] ?? `? ${this.data.status}`;
		const lines = [
			`┌${border}┐`,
			`│ ROSClaw 动作结果（内核权威）`,
			`│ ${label}`,
			`│ capability: ${this.data.capabilityId || "—"}`,
		];
		// 只显示真实存在的 ID 链——不伪造。
		const chain: Array<[string, string | undefined]> = [
			["approval", this.data.approvalId],
			["grant", this.data.grantId],
			["txn", this.data.txnId],
			["action", this.data.actionId],
			["receipt", this.data.receiptId],
		];
		for (const [name, value] of chain) {
			if (value) lines.push(`│ ${name}: ${value.slice(0, 40)}`);
		}
		if (this.data.errorCode) {
			lines.push(`│ error: ${this.data.errorCode}`);
		}
		if (this.data.status === "COMPLETED") {
			lines.push("│ 证据域: simulation（不可用作 REAL 证明）");
		}
		lines.push(`└${border}┘`);
		return lines;
	}

	invalidate(): void {}
}
