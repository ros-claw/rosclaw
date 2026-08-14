/** TUI 实时 Job 卡（十一审 PR-C，总纲 §P0-4）。
 *
 * 常驻 widget：每个 WorkOrder 一行（phase/当前工具/耗时/turn/token），
 * 数据源是 WorkerEventStore tail（pi.worker.events RPC）——与
 * `/job log` 同一事件源。liveness 事件驱动 ≤3s 刷新（只证明活着，
 * 不冒充进度）。
 *
 * 渲染样例：
 *   ● Jobs
 *   ⠹ Pi Developer · 实现 rollout · 2m17s
 *      ⎿ bash: pytest tests/sim · 12 tools · 38.4k tok
 */

import type { ActiveSessionContext } from "../session/active-context.js";
import type { ProductStateCenter } from "../session/state-center.js";

const POLL_MS = 2000;
const TERMINAL_KEEP_MS = 10 * 60 * 1000;
const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

interface OrderProjection {
	work_order_id: string;
	assigned_to?: string;
	status: string;
	goal?: string;
	// 十三审 HOTFIX-13.1：服务端权威时间。
	created_at?: string | null;
	started_at?: string | null;
	finished_at?: string | null;
	duration_ms?: number | null;
	paused_at?: string | null;
	settled?: boolean;
}

interface WorkerEvent {
	seq: number;
	kind: string;
	phase?: string;
	tool?: string;
	message?: string;
	is_error?: boolean;
	args_preview?: string;
	output_preview?: string;
	chars?: number;
	preview?: string;
	input_tokens?: number;
	output_tokens?: number;
	turns?: number;
}

interface JobView {
	order: OrderProjection;
	lastTool: string;
	lastPhase: string;
	toolCount: number;
	inputTokens: number;
	outputTokens: number;
	turns: number;
	stall: boolean;
	lastSeq: number;
}

function fmtElapsed(ms: number): string {
	const s = Math.floor(ms / 1000);
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	return `${m}m${String(s % 60).padStart(2, "0")}`;
}

function fmtTokens(n: number): string {
	if (n < 1000) return `${n}`;
	if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
	return `${Math.round(n / 1000)}k`;
}

function workerLabel(assigned: string | undefined): string {
	switch (assigned) {
		case "worker:rosclaw:pi":
			return "Pi Worker";
		case "worker:native:basic":
			return "Native Basic";
		case "worker:claude-code:local":
			return "Claude Code";
		case "worker:codex:local":
			return "Codex";
		default:
			return assigned ?? "?";
	}
}

export class JobsWidget {
	private timer: ReturnType<typeof setInterval> | undefined;
	private readonly views = new Map<string, JobView>();
	private readonly terminalAt = new Map<string, number>();
	private tickCount = 0;
	private lastRendered = "";
	/** /job log <id> follow 钉住的展开 job。 */
	pinned: string | undefined;

	constructor(
		private readonly deps: {
			active: ActiveSessionContext;
			center: ProductStateCenter;
			setWidget: (lines: string[] | undefined) => void;
		},
	) {}

	start(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			void this.tick().catch(() => undefined);
		}, POLL_MS);
		if (typeof this.timer === "object" && "unref" in this.timer) this.timer.unref();
	}

	stop(): void {
		if (this.timer) clearInterval(this.timer);
		this.timer = undefined;
		this.views.clear();
		this.terminalAt.clear();
		this.lastRendered = "";
	}

	private async tailJob(view: JobView): Promise<void> {
		const response = await this.deps.center.call("pi.worker.events", {
			work_order_id: view.order.work_order_id,
			after_seq: view.lastSeq,
			limit: 50,
		});
		const events = (response.events ?? []) as WorkerEvent[];
		for (const event of events) {
			view.lastSeq = Math.max(view.lastSeq, event.seq);
			if (event.kind === "liveness") {
				if (event.phase) view.lastPhase = event.phase;
				continue;
			}
			if (event.kind === "tool_started" && event.tool) {
				view.lastTool = event.tool;
				view.toolCount += 1;
				view.lastPhase = "RUNNING_TOOL";
			} else if (event.kind === "tool_finished") {
				view.lastPhase = "RUNNING_MODEL";
			} else if (event.kind === "model_started") {
				view.lastPhase = "RUNNING_MODEL";
			} else if (event.kind === "tool_progress" && event.message) {
				view.lastTool = event.message.slice(0, 40);
			} else if (event.kind === "usage") {
				view.inputTokens = event.input_tokens ?? view.inputTokens;
				view.outputTokens = event.output_tokens ?? view.outputTokens;
				view.turns = event.turns ?? view.turns;
			} else if (event.kind === "stall_warning") {
				view.stall = true;
			}
		}
	}

	async tick(): Promise<void> {
		const missionId = this.deps.active.current.missionId;
		if (!missionId) {
			this.render([]);
			return;
		}
		const status = await this.deps.center.call("pi.worker.status", {
			mission_id: missionId,
		});
		const orders = (status.orders ?? []) as OrderProjection[];
		const now = Date.now();
		for (const order of orders) {
			const terminal = ["ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"].includes(order.status);
			let view = this.views.get(order.work_order_id);
			if (!view) {
				view = {
					order,
					lastTool: "",
					lastPhase: "STARTING",
					toolCount: 0,
					inputTokens: 0,
					outputTokens: 0,
					turns: 0,
					stall: false,
					lastSeq: 0,
				};
				this.views.set(order.work_order_id, view);
			}
			view.order = order;
			if (!terminal) {
				await this.tailJob(view);
				this.terminalAt.delete(order.work_order_id);
			} else if (!this.terminalAt.has(order.work_order_id)) {
				this.terminalAt.set(order.work_order_id, now);
				await this.tailJob(view);
			}
		}
		// 清理：终态超过保留期 / 不属于当前 mission 的视图。
		for (const [id] of this.views) {
			const at = this.terminalAt.get(id);
			if (at !== undefined && now - at > TERMINAL_KEEP_MS) {
				this.views.delete(id);
				this.terminalAt.delete(id);
			}
		}
		this.render([...this.views.values()]);
	}

	private render(views: JobView[]): void {
		this.tickCount += 1;
		// 十四审 PR-14.4：footer 常驻可发现——无任务也显示 `F2 Tasks`
		// （内容不变不重绘，idle CPU 红线不变）。
		if (views.length === 0) {
			const idle = "F2 Tasks";
			if (this.lastRendered !== idle) {
				this.lastRendered = idle;
				this.deps.setWidget([idle]);
			}
			return;
		}
		const frame = SPINNER[this.tickCount % SPINNER.length];
		const now = Date.now();
		const terminalSet = ["ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"];
		const running = views.filter((v) => !terminalSet.includes(v.order.status)).length;
		const waiting = views.filter((v) => v.order.status === "BLOCKED").length;
		const lines: string[] = [
			`F2 Tasks · ${running} running${waiting ? ` · ${waiting} waiting` : ""}`,
		];
		for (const view of views) {
			const id = view.order.work_order_id;
			const status = view.order.status;
			const terminal = ["ACCEPTED", "FAILED", "EXPIRED", "CANCELLED"].includes(status);
			// 十三审：中断/暂停/失联的可读标记。
			// 十三审 HOTFIX-13.1：权威计时——终态冻结（finished-started），
			// 运行态用服务端 started_at；缺 finished_at 的终态显示数据
			// 错误而非继续计时；TUI 重启显示一致。
			let elapsed: string;
			if (terminal) {
				elapsed = view.order.duration_ms != null
					? fmtElapsed(view.order.duration_ms)
					: "时长数据不完整";
			} else {
				const startedMs = view.order.started_at
					? Date.parse(view.order.started_at)
					: NaN;
				// 中断/预算暂停：计时冻结在 paused_at（Worker 不在工作）。
				const endMs = view.order.paused_at ? Date.parse(view.order.paused_at) : now;
				elapsed = Number.isNaN(startedMs)
					? "起始时间未知"
					: fmtElapsed(endMs - startedMs);
			}
			const goal = (view.order.goal ?? "").slice(0, 32);
			const icon = terminal ? (status === "ACCEPTED" ? "✓" : "✗") : status === "BLOCKED" ? "⚠" : frame;
			const stallMark = view.stall && !terminal ? " · 静默>90s（仍存活）" : "";
			const waitMark = status === "BLOCKED" ? " · 等待用户输入（/job answer 回答）" : "";
			const stateMark =
				status === "UNREACHABLE" ? " · 失联探测中（进程仍活）"
				: status === "INTERRUPTED_RESUMABLE" ? " · 中断可恢复（/job resume）"
				: status === "BUDGET_PAUSED" ? " · 预算暂停（/job extend 追加）"
				: "";
			lines.push(
				`${icon} ${workerLabel(view.order.assigned_to)} · ${goal} · ${elapsed}${stallMark}${waitMark}${stateMark}`,
			);
			const detail = terminal
				? `${status}${view.turns ? ` · ${view.turns} turns` : ""}`
				: `${view.lastTool || view.lastPhase}${view.toolCount ? ` · ${view.toolCount} tools` : ""} · ${fmtTokens(view.inputTokens + view.outputTokens)} tok`;
			lines.push(`   ⎿ ${detail}   [${id.slice(0, 12)}… /job ${id} 查看]`);
		}
		const rendered = lines.join("\n");
		if (rendered !== this.lastRendered) {
			this.lastRendered = rendered;
			this.deps.setWidget(lines);
		}
	}
}

/** /job log 渲染（与 widget 同一事件源）。 */
export function renderJobLog(events: WorkerEvent[], workOrderId: string): string {
	if (events.length === 0) return `(${workOrderId} 暂无事件)`;
	const lines: string[] = [];
	for (const event of events) {
		switch (event.kind) {
			case "liveness":
				continue; // 默认不刷屏——liveness 只供 widget
			case "tool_started":
				lines.push(
					`#${event.seq} ▶ tool ${event.tool}${event.args_preview ? ` ${event.args_preview}` : ""}`,
				);
				break;
			case "tool_finished":
				lines.push(`#${event.seq} ✓ tool ${event.tool}${event.is_error ? " (error)" : ""}`);
				if (event.output_preview) {
					lines.push(`     ⎿ ${event.output_preview.slice(0, 120)}`);
				}
				break;
			case "message_delta":
				lines.push(`#${event.seq} … ${event.preview ?? ""}`);
				break;
			case "model_started":
				lines.push(`#${event.seq} ◌ model turn`);
				break;
			case "usage":
				lines.push(
					`#${event.seq}   tokens ${fmtTokens(event.input_tokens ?? 0)}↑ ${fmtTokens(event.output_tokens ?? 0)}↓`,
				);
				break;
			case "stall_warning":
				lines.push(`#${event.seq} ⚠ 语义静默（进程仍存活）`);
				break;
			case "attempt_finished":
				lines.push(`#${event.seq} ■ finished`);
				break;
			case "attempt_failed":
				lines.push(`#${event.seq} ■ failed`);
				break;
			default:
				lines.push(`#${event.seq} ${event.kind}`);
		}
	}
	return lines.join("\n") || "(无可见事件——仅 liveness)";
}
