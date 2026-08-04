/** Masked secret input（审计 P0-03.3）：自绘 masked 输入组件。
 *
 * secret 只进内存闭包，不进 transcript/journal/history——渲染永远是 •。
 */

import type { Component, Focusable, TUI } from "@earendil-works/pi-tui";
import { Key, matchesKey } from "@earendil-works/pi-tui";

export class MaskedInput implements Component, Focusable {
	focused = false;
	onSubmit?: (secret: string) => void;
	onCancel?: () => void;
	private value = "";
	private cursorPos = 0;

	constructor(
		private readonly tui: TUI,
		private readonly prompt: string,
	) {}

	invalidate(): void {
		/* stateless render */
	}

	render(width: number): string[] {
		const bullets = "•".repeat(this.value.length);
		const line = `${this.prompt} ${bullets}`;
		const hint = "（输入不显示原文；Enter 确认，Esc 取消）";
		return [line.slice(0, width), hint];
	}

	handleInput(data: string): void {
		if (matchesKey(data, "return")) {
			const secret = this.value;
			this.value = "";
			this.onSubmit?.(secret);
			return;
		}
		if (matchesKey(data, "escape")) {
			this.value = "";
			this.onCancel?.();
			return;
		}
		if (matchesKey(data, "backspace")) {
			this.value = this.value.slice(0, -1);
			return;
		}
		if (data.length === 1 && data >= " ") {
			this.value += data;
			this.cursorPos = this.value.length;
		}
	}
}
