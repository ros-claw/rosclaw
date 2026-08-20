/** F2 Task Panel（PR-H9 重写，总纲 v2 §18.3 task/task-panel.ts）。
 *
 * 数据源全面切换到 TaskKernel（pi.kernel.list/events/artifacts）——
 * WorkOrder/execution 旧链已删除（H9）。渲染复用 native/task-activity
 * （与 /activity /artifacts 命令同一渲染器——UI 不再维护第二套映射）。
 *
 * 交互：↑↓ 选卡 · Tab 切 Activity/Artifacts · r 刷新 · ? 帮助 ·
 * Esc/q 关闭。kernel 任务由对话驱动——面板只读（修复/取消走对话
 * 与 /done，不做第二控制面）。
 */

import type { Component } from "@earendil-works/pi-tui";

import {
	renderArtifactList,
	renderTaskActivity,
	type KernelEvent,
} from "../native/task-activity.js";

export interface TasksCenterDeps {
	/** pi.kernel.list 行。 */
	fetchTasks: () => Promise<Array<Record<string, unknown>>>;
	/** pi.kernel.events（全量，组件内排序）。 */
	fetchEvents: (taskId: string) => Promise<KernelEvent[]>;
	/** pi.kernel.artifacts。 */
	fetchArtifacts: (taskId: string) => Promise<Array<Record<string, unknown>>>;
	notify: (text: string, kind: "info" | "warning" | "error") => void;
	onClose: () => void;
}

const POLL_MS = 2000;
const VIEWPORT = 14;
const TABS = ["Activity", "Artifacts"] as const;
type Tab = (typeof TABS)[number];

const TERMINAL = new Set(["SUCCEEDED", "FAILED", "CANCELLED", "BLOCKED"]);

function icon(state: string): string {
	if (state === "SUCCEEDED") return "✓";
	if (state === "BLOCKED") return "⚠";
	if (TERMINAL.has(state)) return "✗";
	return "●";
}

/** kernel task 行 → 卡片行（纯函数——测试面）。 */
export function taskCardLine(task: Record<string, unknown>, mark = " "): string {
	const goal = String(task.root_goal ?? "").split("\n")[0].slice(0, 34);
	const state = String(task.state ?? "?");
	const rev = Number(task.active_revision ?? 1);
	return `${mark} ${icon(state)} ${goal} · ${state}${rev > 1 ? ` · r${rev}` : ""}`;
}

export class TasksCenterComponent implements Component {
	private tasks: Array<Record<string, unknown>> = [];
	private selected = 0;
	private tab: Tab = "Activity";
	private showHelp = false;
	private lines: string[] = [];
	private timer: ReturnType<typeof setInterval> | undefined;
	private disposed = false;

	constructor(private readonly deps: TasksCenterDeps) {
		this.timer = setInterval(() => {
			void this.poll();
		}, POLL_MS);
		if (typeof this.timer === "object" && "unref" in this.timer) this.timer.unref();
		void this.poll();
	}

	private current(): Record<string, unknown> | undefined {
		return this.tasks[Math.min(this.selected, this.tasks.length - 1)];
	}

	private async poll(): Promise<void> {
		if (this.disposed) return;
		try {
			this.tasks = await this.deps.fetchTasks();
			if (this.selected >= this.tasks.length) {
				this.selected = Math.max(0, this.tasks.length - 1);
			}
			await this.pollContent();
		} catch {
			// 桥暂不可用——下周期再试。
		}
	}

	private async pollContent(): Promise<void> {
		const task = this.current();
		if (!task) {
			this.lines = [];
			return;
		}
		const taskId = String(task.task_id ?? "");
		if (this.tab === "Activity") {
			this.lines = renderTaskActivity(await this.deps.fetchEvents(taskId));
			return;
		}
		this.lines = renderArtifactList(await this.deps.fetchArtifacts(taskId));
	}

	handleInput(data: string): void {
		if (data === "\x1b[A") {
			this.selected = Math.max(0, this.selected - 1);
			void this.pollContent();
			return;
		}
		if (data === "\x1b[B") {
			this.selected = Math.min(this.selected + 1, Math.max(0, this.tasks.length - 1));
			void this.pollContent();
			return;
		}
		if (data === "\x1b" || data === "q") {
			this.dispose();
			this.deps.onClose();
			return;
		}
		const key = data.toLowerCase();
		if (key === "\t") {
			this.tab = TABS[(TABS.indexOf(this.tab) + 1) % TABS.length];
			void this.pollContent();
			return;
		}
		if (key === "r") {
			void this.poll();
			return;
		}
		if (key === "a") {
			this.tab = "Artifacts";
			void this.pollContent();
			return;
		}
		if (key === "?") {
			this.showHelp = !this.showHelp;
			return;
		}
	}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(20, Math.min(width - 4, 76)));
		const out: string[] = [`┌ Tasks ${border.slice(8)}┐`];
		if (this.tasks.length === 0) {
			out.push("│ （无任务——对话中的任务会出现在这里）");
		}
		this.tasks.forEach((task, idx) => {
			out.push(`│ ${taskCardLine(task, idx === this.selected ? ">" : " ")}`);
		});
		const tabsRow = TABS.map((t) => (t === this.tab ? `[${t}]` : t)).join(" | ");
		out.push(`├ ${tabsRow} ${border.slice(tabsRow.length + 3)}┤`);
		for (const line of this.lines.slice(-VIEWPORT)) {
			out.push(`│ ${line.slice(0, Math.min(width - 6, 74))}`);
		}
		if (this.lines.length === 0) out.push("│ （等待事件…）");
		out.push(`├${border}┤`);
		if (this.showHelp) {
			out.push("│ F2/Esc 关闭 · ↑↓ 选择 · Tab 切 Activity/Artifacts · r 刷新 · a 产物");
			out.push("│ 任务由对话驱动：修改目标直接说，验收用 /done——面板只读");
		} else {
			out.push("│ ↑↓选择 Tab切页 r刷新 ?帮助 Esc关闭");
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
