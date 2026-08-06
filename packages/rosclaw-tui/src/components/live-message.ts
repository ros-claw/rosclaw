/** Live assistant message（审计 P0-02）：可变增量消息组件。
 *
 * delta 不再攒到 flush 才出现——每个 turn 一个可变 Markdown 组件，
 * 20–40ms 节流刷新；终止/错误/取消/agent.settled 时强制 flush。
 */

import { Markdown, type MarkdownTheme } from "@earendil-works/pi-tui";

export class LiveAssistantMessage {
	readonly component: Markdown;
	private buffer = "";
	private pending = "";
	private timer: ReturnType<typeof setInterval> | null = null;
	private onFlush: () => void;
	private flushed = false;

	constructor(theme: MarkdownTheme, onFlush: () => void) {
		this.component = new Markdown("", 0, 0, theme);
		this.onFlush = onFlush;
	}

	append(text: string): void {
		this.pending += text;
		if (this.timer === null) {
			// 首个 delta 立即显示（不做无谓等待），随后节流。
			this.flush();
			this.timer = setInterval(() => this.flush(), 30);
			this.timer.unref?.();
		}
	}

	/** 节流刷新；force 用于 turn 结束/错误/取消/settled。 */
	flush(force = false): void {
		if (this.pending) {
			this.buffer += this.pending;
			this.pending = "";
			this.component.setText(this.buffer);
			this.onFlush();
		} else if (force && !this.flushed && !this.buffer) {
			this.stop();
			return;
		}
		if (force) this.stop();
	}

	stop(): void {
		if (this.timer !== null) {
			clearInterval(this.timer);
			this.timer = null;
		}
		this.flushed = true;
	}

	get text(): string {
		return this.buffer + this.pending;
	}
}
