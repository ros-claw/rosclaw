/** /job 实时会话查看器（十三审 PR-13.3/13.4，总纲 §3）。
 *
 * 不是一次性 notify——overlay 组件持续订阅 WorkerEventStore（cursor
 * 增量），展示公开可验证的工作过程：
 * assistant 公开文本、工具名+脱敏参数、输出增量/退出码、文件改动、
 * 产物、stall/预算/失联状态。
 *
 * 不展示：隐藏思维链、API key、系统提示、敏感全文。
 *
 * 键位：q/Esc 关闭 · f follow/pause · s steer · x cancel · r retry/resume
 */

import type { Component } from "@earendil-works/pi-tui";

export interface ViewerDeps {
	workOrderId: string;
	/** 增量拉取（cursor 后最早 N 条）。 */
	fetchEvents: (
		afterSeq: number,
		limit: number,
	) => Promise<{ events: Array<Record<string, unknown>>; status: string }>;
	/** steer 输入（ui.input 由扩展注入）。 */
	onSteer: () => Promise<string | undefined>;
	sendSteer: (text: string) => Promise<string>;
	onCancel: () => Promise<string>;
	notify: (text: string, kind: "info" | "warning" | "error") => void;
	/** overlay 关闭回调（ctx.ui.custom 的 done）。 */
	onClose: () => void;
}

const POLL_MS = 1500;
const VIEWPORT = 22;

function fmtEvent(event: Record<string, unknown>): string | null {
	const kind = String(event.kind ?? "");
	switch (kind) {
		case "liveness":
		case "usage":
			return null; // 心跳/计量不刷屏（header 有）
		case "attempt_started":
			return "● Worker 启动";
		case "model_started":
			return "◌ model turn";
		case "message_delta":
			return `… ${String(event.preview ?? "")}`;
		case "tool_started":
			return `▶ ${String(event.tool ?? "?")}  ${String(event.args_preview ?? "")}`;
		case "tool_progress":
			return `  … ${String(event.message ?? event.tool ?? "")}`;
		case "tool_finished": {
			const err = event.is_error === true ? " (error)" : "";
			const out = String(event.output_preview ?? "").split("\n")[0].slice(0, 80);
			return `✓ ${String(event.tool ?? "?")}${err}${out ? `  ⎿ ${out}` : ""}`;
		}
		case "stall_warning":
			return "⚠ 长时间无公开进度（进程仍存活）";
		case "unreachable":
			return "⚠ 事件管道失联——进程探测中";
		case "budget_warning":
			return "⚠ token 预算 80%";
		case "budget_paused":
			return "⏸ 预算暂停——/job extend 追加后继续";
		case "waiting_input":
			return `? Worker 提问：${String(event.question ?? "")}（/job answer 回答）`;
		case "steer_ack":
			return "⇢ steer 已送达";
		case "session_resumed":
			return "⟲ 已从检查点恢复会话";
		case "attempt_finished":
			return "■ 完成";
		case "attempt_failed":
			return `■ 失败 [${String(event.error_code ?? "?")}] ${String(event.message ?? "").slice(0, 100)}`;
		case "attempt_cancelled":
			return "■ 已取消";
		default:
			return `· ${kind}`;
	}
}

export class JobViewerComponent implements Component {
	private lines: string[] = [];
	private cursor = 0;
	private follow = true;
	private scrollOffset = 0;
	private status = "STARTING";
	private timer: ReturnType<typeof setInterval> | undefined;
	private disposed = false;

	constructor(private readonly deps: ViewerDeps) {
		this.timer = setInterval(() => {
			void this.poll();
		}, POLL_MS);
		if (typeof this.timer === "object" && "unref" in this.timer) this.timer.unref();
		void this.poll();
	}

	private async poll(): Promise<void> {
		if (this.disposed) return;
		try {
			const page = await this.deps.fetchEvents(this.cursor, 200);
			for (const event of page.events) {
				const seq = Number(event.seq ?? 0);
				this.cursor = Math.max(this.cursor, seq);
				const line = fmtEvent(event);
				if (line) this.lines.push(line);
			}
			this.status = page.status;
		} catch {
			// 下一轮再试
		}
	}

	handleInput(data: string): void {
		const key = data.toLowerCase();
		if (key === "q" || data === "\x1b") {
			this.dispose();
			this.deps.onClose();
			return;
		}
		if (key === "f") {
			this.follow = !this.follow;
			return;
		}
		if (key === "s") {
			void this.deps.onSteer().then(async (text) => {
				if (text) {
					const reply = await this.deps.sendSteer(text);
					this.deps.notify(reply, "info");
				}
			});
			return;
		}
		if (key === "x") {
			void this.deps.onCancel().then((msg) => this.deps.notify(msg, "info"));
			return;
		}
		// 滚动（暂停 follow 后查看历史）。
		if (data === "\x1b[A") {
			this.follow = false;
			this.scrollOffset += 1;
		} else if (data === "\x1b[B") {
			this.scrollOffset = Math.max(0, this.scrollOffset - 1);
		}
	}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(20, Math.min(width - 4, 72)));
		const total = this.lines.length;
		const end = this.follow ? total : Math.max(0, total - this.scrollOffset);
		const start = Math.max(0, end - VIEWPORT);
		const body = this.lines.slice(start, end);
		const out = [
			`┌${border}┐`,
			`│ Job ${this.deps.workOrderId} · ${this.status} · ${total} 事件${this.follow ? " · follow" : " · paused"}`,
			`├${border}┤`,
			...body.map((l) => `│ ${l.slice(0, Math.min(width - 6, 72))}`),
			`├${border}┤`,
			"│ q 关闭 · f follow · s steer · x cancel · ↑↓ 滚动",
			`└${border}┘`,
		];
		return out;
	}

	dispose(): void {
		this.disposed = true;
		if (this.timer) clearInterval(this.timer);
		this.timer = undefined;
	}

	invalidate(): void {
		// 无缓存渲染态——无需动作。
	}
}
