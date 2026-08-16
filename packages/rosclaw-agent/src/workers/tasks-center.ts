/** F2 Tasks Center（十四审 PR-14.4，总纲 §4）。
 *
 * 不用 /job <长 ID> 也能看、切、控：
 * - 一个用户任务一张卡（root job 聚合 attempts——retry/resume 是内部
 *   attempt，绝不显示三张失败卡）；
 * - ↑↓ 选择、Enter 展开 attempts、Tab 切 Live/Transcript/Files/
 *   Artifacts/Metrics；
 * - p 暂停/恢复、x 二次确认取消、s steer、f follow、r 可恢复终态
 *   retry/resume——全部走服务端 ACK/协调器，不乐观更新。
 *
 * 参考 pi-subagents（MIT，上游 0e39260）的 agent-widget/conversation-viewer
 * 信息架构移植——见 THIRD_PARTY_NOTICES.md；未安装其扩展（权限边界）。
 */

import type { Component } from "@earendil-works/pi-tui";

import { fmtEvent } from "./job-viewer.js";

export interface AttemptInfo {
	work_order_id: string;
	seq: number;
	actor: string;
	status: string;
	termination_cause: string;
	started_at?: string | null;
	finished_at?: string | null;
	duration_ms?: number | null;
}

export interface JobCard {
	root_job_id: string;
	goal: string;
	state: string;
	attempts: AttemptInfo[];
	/** 十五审 PR-RF-8：execution 级卡片的执行主体（runtime/executor）。 */
	runtime?: string;
}

export interface TasksCenterDeps {
	fetchJobs: () => Promise<JobCard[]>;
	fetchEvents: (
		wo: string,
		afterSeq: number,
		limit: number,
	) => Promise<{ events: Array<Record<string, unknown>>; status: string }>;
	fetchTranscript: (
		wo: string,
		afterSeq: number,
		limit: number,
		channel: string,
	) => Promise<{
		records: Array<Record<string, unknown>>;
		has_more: boolean;
		next_cursor: number;
		total: number;
	}>;
	onSteer: () => Promise<string | undefined>;
	sendSteer: (wo: string, text: string) => Promise<string>;
	sendControl: (wo: string, action: string) => Promise<{ ok: boolean; state?: string; error?: string }>;
	sendRetry: (wo: string) => Promise<string>;
	notify: (text: string, kind: "info" | "warning" | "error") => void;
	onClose: () => void;
}

const POLL_MS = 1500;
const VIEWPORT = 14;
const TABS = ["Live", "Transcript", "Files", "Artifacts", "Metrics"] as const;
type Tab = (typeof TABS)[number];
const TAB_CHANNEL: Record<Tab, string> = {
	Live: "",
	Transcript: "conversation",
	Files: "files",
	Artifacts: "artifacts",
	Metrics: "usage",
};
const TERMINAL = new Set(["ACCEPTED", "FAILED", "EXPIRED", "CANCELLED", "SUCCEEDED", "BLOCKED"]);
const RETRYABLE = new Set(["INTERRUPTED_RESUMABLE", "FAILED", "CANCELLED", "EXPIRED"]);

function icon(state: string): string {
	if (state === "ACCEPTED" || state === "SUCCEEDED") return "✓";
	if (TERMINAL.has(state)) return "✗";
	if (state === "BLOCKED" || state === "PAUSED" || state === "BUDGET_PAUSED") return "⚠";
	return "●";
}

function fmtElapsedMs(ms: number | null | undefined): string {
	if (ms == null) return "";
	const sec = Math.floor(ms / 1000);
	if (sec < 60) return `${sec}s`;
	return `${Math.floor(sec / 60)}m${sec % 60 ? `${sec % 60}s` : ""}`;
}

export class TasksCenterComponent implements Component {
	private jobs: JobCard[] = [];
	private selected = 0;
	private expanded = false;
	private tab: Tab = "Live";
	private follow = true;
	private confirmCancel = false;
	private showHelp = false;
	private lines: string[] = [];
	private cursor = 0;
	private scrollOffset = 0;
	private timer: ReturnType<typeof setInterval> | undefined;
	private disposed = false;

	constructor(private readonly deps: TasksCenterDeps) {
		this.timer = setInterval(() => {
			void this.poll();
		}, POLL_MS);
		if (typeof this.timer === "object" && "unref" in this.timer) this.timer.unref();
		void this.poll();
	}

	private current(): JobCard | undefined {
		return this.jobs[Math.min(this.selected, this.jobs.length - 1)];
	}

	/** 控制动作的目标 attempt：活跃优先，否则最新。 */
	private targetAttempt(card: JobCard | undefined): AttemptInfo | undefined {
		if (!card || card.attempts.length === 0) return undefined;
		return (
			[...card.attempts].reverse().find((a) => !TERMINAL.has(a.status))
			?? card.attempts[card.attempts.length - 1]
		);
	}

	private async poll(): Promise<void> {
		if (this.disposed) return;
		try {
			this.jobs = await this.deps.fetchJobs();
			if (this.selected >= this.jobs.length) {
				this.selected = Math.max(0, this.jobs.length - 1);
			}
			await this.pollContent();
		} catch {
			// 下一轮再试
		}
	}

	private async pollContent(): Promise<void> {
		const card = this.current();
		const attempt = this.targetAttempt(card);
		if (!attempt) {
			this.lines = [];
			return;
		}
		if (this.tab === "Live") {
			const page = await this.deps.fetchEvents(attempt.work_order_id, this.cursor, 200);
			for (const event of page.events) {
				const seq = Number(event.seq ?? 0);
				this.cursor = Math.max(this.cursor, seq);
				const line = fmtEvent(event);
				if (line) this.lines.push(line);
			}
			return;
		}
		const page = await this.deps.fetchTranscript(
			attempt.work_order_id,
			this.cursor,
			100,
			TAB_CHANNEL[this.tab],
		);
		for (const record of page.records) {
			const tseq = Number(record.tseq ?? 0);
			this.cursor = Math.max(this.cursor, tseq);
			this.lines.push(this.fmtRecord(record));
		}
	}

	private fmtRecord(record: Record<string, unknown>): string {
		const channel = String(record.channel ?? "");
		if (channel === "conversation") {
			const text = String(record.text ?? "");
			const first = text.split("\n")[0] ?? "";
			return `💬 ${first.slice(0, 72)}${text.includes("\n") ? " …" : ""}`;
		}
		if (channel === "tools") {
			return record.phase === "start"
				? `▶ ${String(record.tool ?? "?")} ${String(record.args ?? "").slice(0, 60)}`
				: `${record.is_error ? "✗" : "✓"} ${String(record.tool ?? "?")} ${String(record.output ?? "").split("\n")[0].slice(0, 60)}`;
		}
		if (channel === "files") {
			return `📄 ${String(record.op ?? record.kind ?? "?")} ${String(record.path ?? "")}`;
		}
		if (channel === "artifacts") {
			const files = (record.files ?? []) as Array<{ name?: string; bytes?: number }>;
			return `📦 ${files.map((f) => `${f.name}(${f.bytes}B)`).join(", ")}`;
		}
		if (channel === "usage") {
			return `⚡ in=${record.input} out=${record.output} turns=${record.turns}`;
		}
		return `· ${channel}`;
	}

	private switchTab(tab: Tab): void {
		this.tab = tab;
		this.lines = [];
		this.cursor = 0;
		this.scrollOffset = 0;
		void this.pollContent();
	}

	handleInput(data: string): void {
		// 全串匹配优先（方向键是 ESC 序列，不能被 Esc 关闭吞掉）。
		if (data === "\x1b[A") {
			this.follow = false;
			this.scrollOffset += 1;
			this.confirmCancel = false;
			return;
		}
		if (data === "\x1b[B") {
			if (this.scrollOffset > 0) {
				this.scrollOffset -= 1;
			} else {
				this.selected = Math.min(this.selected + 1, Math.max(0, this.jobs.length - 1));
			}
			this.confirmCancel = false;
			return;
		}
		if (data === "\x1b" || data === "q") {
			this.dispose();
			this.deps.onClose();
			return;
		}
		const key = data.toLowerCase();
		if (key !== "x") this.confirmCancel = false;
		if (key === "\t") {
			const idx = TABS.indexOf(this.tab);
			this.switchTab(TABS[(idx + 1) % TABS.length]);
			return;
		}
		if (key === "\r" || key === "\n") {
			this.expanded = !this.expanded;
			return;
		}
		if (key === "f") {
			this.follow = !this.follow;
			return;
		}
		if (key === "?") {
			this.showHelp = !this.showHelp;
			return;
		}
		if (key === "a") {
			this.switchTab("Artifacts");
			return;
		}
		const card = this.current();
		const attempt = this.targetAttempt(card);
		if (key === "s" && attempt) {
			void this.deps.onSteer().then(async (text) => {
				if (text) {
					const reply = await this.deps.sendSteer(attempt.work_order_id, text);
					this.deps.notify(reply, "info");
				}
			});
			return;
		}
		if (key === "p" && card && attempt) {
			const action =
				card.state === "PAUSED" || card.state === "BUDGET_PAUSED" ? "resume" : "pause";
			void this.deps.sendControl(attempt.work_order_id, action).then((result) => {
				this.deps.notify(
					result.ok
						? `${action === "pause" ? "已暂停" : "已恢复"}（ACK ${result.state}）`
						: `控制失败：${result.error ?? "未知"}`,
					result.ok ? "info" : "error",
				);
				void this.poll();
			});
			return;
		}
		if (key === "x" && attempt) {
			if (TERMINAL.has(card?.state ?? "")) {
				this.deps.notify("任务已终态——无需取消", "warning");
				return;
			}
			if (!this.confirmCancel) {
				this.confirmCancel = true; // 二次确认（运行工具时解释影响）
				return;
			}
			this.confirmCancel = false;
			void this.deps.sendControl(attempt.work_order_id, "cancel").then((result) => {
				this.deps.notify(
					result.ok ? "已取消（Worker 收到控制取消）" : `取消失败：${result.error ?? "未知"}`,
					result.ok ? "info" : "error",
				);
				void this.poll();
			});
			return;
		}
		if (key === "r" && card && attempt) {
			if (!RETRYABLE.has(card.state)) {
				this.deps.notify("运行中的任务不能 retry——可先 p 暂停或 x 取消", "warning");
				return;
			}
			void this.deps.sendRetry(attempt.work_order_id).then((reply) => {
				this.deps.notify(reply, "info");
				void this.poll();
			});
			return;
		}
	}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(20, Math.min(width - 4, 76)));
		const out: string[] = [`┌ Tasks ${border.slice(8)}┐`];
		if (this.jobs.length === 0) {
			out.push("│ （无任务——对话中委派的 Worker 任务会出现在这里）");
		}
		this.jobs.forEach((card, idx) => {
			const mark = idx === this.selected ? ">" : " ";
			const latest = this.targetAttempt(card);
			const elapsed = fmtElapsedMs(latest?.duration_ms);
			const who = card.runtime ? ` · ${card.runtime}` : "";
			const attempts = card.attempts.length
				? ` · attempt ${card.attempts.length}/${card.attempts.length}`
				: "";
			out.push(
				`│ ${mark} ${icon(card.state)} ${card.goal.slice(0, 34)}${who} · ${card.state}${elapsed ? ` · ${elapsed}` : ""}${attempts}`,
			);
			if (this.expanded && idx === this.selected) {
				for (const a of card.attempts) {
					out.push(
						`│     └ attempt ${a.seq} · ${a.actor} · ${a.status}${a.termination_cause ? ` · ${a.termination_cause}` : ""} · ${a.work_order_id.slice(0, 14)}…`,
					);
				}
			}
		});
		const tabsRow = TABS.map((t) => (t === this.tab ? `[${t}]` : t)).join(" | ");
		out.push(`├ ${tabsRow} ${border.slice(tabsRow.length + 3)}┤`);
		const end = this.follow ? this.lines.length : Math.max(0, this.lines.length - this.scrollOffset);
		const start = Math.max(0, end - VIEWPORT);
		for (const line of this.lines.slice(start, end)) {
			out.push(`│ ${line.slice(0, Math.min(width - 6, 74))}`);
		}
		if (this.lines.length === 0) out.push("│ （等待事件…）");
		out.push(`├${border}┤`);
		if (this.confirmCancel) {
			out.push("│ ⚠ 再按 x 确认取消（正在运行的工具会被中断；partial 成果保留）");
		}
		if (this.showHelp) {
			out.push("│ F2/Esc 关闭 · ↑↓ 选择/滚动 · Enter 展开 attempts · Tab 切页");
			out.push("│ f follow · s steer · p 暂停/恢复 · x 取消(二次确认) · r 恢复/重试 · a 产物");
		} else {
			out.push("│ ↑↓选择 Enter详情 Esc关闭  f跟随 s转向 p暂停 x取消 ?帮助");
		}
		out.push(`└${border}┘`);
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
