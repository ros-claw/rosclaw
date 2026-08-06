/** ApprovalCard 组件（PNA-5，规格 §20.2/§20.3）：不可变卡片 + 显式 Y/N。
 *
 * - 显示 mission/mode/capability/完整参数/风险/TTL/display_hash；
 * - Y=批准 N=拒绝 Esc=拒绝；超时不操作=拒绝（由调用方超时兜底）；
 * - 决定经 operatord（display_hash 绑定 + 签名链）——模型文本永远无法
 *   触发本组件的按键。
 */

import type { Component } from "@earendil-works/pi-tui";

export interface ApprovalCardData {
	requestId: string;
	title: string;
	summary: string;
	riskTier: string;
	mode: string;
	capability: string;
	parameters: Record<string, unknown>;
	expiresAt: string;
	displayHash: string;
}

export class ApprovalCardComponent implements Component {
	private decided: boolean | null = null;

	constructor(
		private readonly card: ApprovalCardData,
		private readonly onDecision: (approve: boolean) => void,
	) {}

	handleInput(data: string): void {
		if (this.decided !== null) return;
		const key = data.toLowerCase();
		if (key === "y") {
			this.decided = true;
			this.onDecision(true);
		} else if (key === "n" || data === "\x1b") {
			this.decided = false;
			this.onDecision(false);
		}
	}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(10, Math.min(width - 4, 60)));
		const lines = [
			`┌${border}┐`,
			`│ ⚠ ROSCLAW 授权请求 [${this.card.mode}] ${this.card.riskTier}`,
			`│ ${this.card.title}`,
			`│ ${this.card.summary}`,
			`│ capability: ${this.card.capability}`,
		];
		for (const [key, value] of Object.entries(this.card.parameters)) {
			lines.push(`│   ${key} = ${JSON.stringify(value)}`);
		}
		lines.push(`│ display_hash: ${this.card.displayHash}`);
		lines.push(`│ 过期: ${this.card.expiresAt}`);
		lines.push(`│ [Y] 批准   [N/Esc] 拒绝（默认拒绝）`);
		if (this.decided !== null) {
			lines.push(`│ → 已${this.decided ? "批准（等待回执）" : "拒绝"}`);
		}
		lines.push(`└${border}┘`);
		return lines;
	}

	invalidate(): void {}
}
