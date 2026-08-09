/** OperatorBootstrap 卡（六审 §7/PR-SIX-5）：未初始化/未运行时的一键
 * 初始化——I=初始化，Esc=取消。只在 SIMULATION developer 出现；
 * 决定仍由 operatord 独立进程持有（本卡只触发产品 supervisor）。
 * 所有可见字符串由调用方从 i18n catalog 传入（卡片不硬编码）。
 */

import type { Component } from "@earendil-works/pi-tui";

export interface OperatorBootstrapLabels {
	title: string;
	state: string;
	offer: string;
	hint: string;
}

export class OperatorBootstrapComponent implements Component {
	private decided: boolean | null = null;

	constructor(
		private readonly labels: OperatorBootstrapLabels,
		private readonly onDecision: (init: boolean) => void,
	) {}

	handleInput(data: string): void {
		if (this.decided !== null) return;
		const key = data.toLowerCase();
		if (key === "i") {
			this.decided = true;
			this.onDecision(true);
		} else if (data === "\x1b" || key === "n") {
			this.decided = false;
			this.onDecision(false);
		}
	}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(10, Math.min(width - 4, 60)));
		return [
			`┌${border}┐`,
			`│ ${this.labels.title}`,
			`│ ${this.labels.state}`,
			`│ ${this.labels.offer}`,
			`│ ${this.labels.hint}`,
			`└${border}┘`,
		];
	}

	invalidate(): void {}
}
