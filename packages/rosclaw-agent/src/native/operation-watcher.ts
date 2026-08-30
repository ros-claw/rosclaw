/** OperationWatcher V2（P1-B2，0824 总纲 §12.3）——operation 事件流
 *  驱动：progress 流式进 TUI + 终态一次性 followUp。
 *
 * - 每个 tick 只发 pi.kernel.events（task_id + last_seq 增量游标，
 *   不重不漏）——不再逐 op 轮询 pi.op.get（注册时一次性取 task_id
 *   除外）；
 * - operation.output/progress 事件 → sink.setWidget 按 operation_id
 *   原位更新（单活动区，同 key 覆盖）；终态 → widget 清除；
 * - progress 绝不进模型上下文（setWidget ≠ sendMessage，零 LLM
 *   token 开销）；
 * - 终态一次性 followUp（WP-1 语义不变：owning task 终态/旧
 *   revision 只存档不触发回合）。
 *
 * P0-3（0827 审计）：trackTask（输入路由任务）只投影——终态回复由
 * Coordinator 经 TerminalPresenter 确定性呈现（display:true,
 * triggerTurn:false），绝不 followUp 唤醒 Agent（双控制者根治）。
 */

import { renderTerminalReply, type TerminalOutcome } from "./terminal-presenter.js";

interface SendSink {
	sendMessage(
		message: {
			customType: string;
			content: string;
			display: boolean;
			details: Record<string, unknown>;
		},
		options: { triggerTurn: boolean; deliverAs?: "nextTurn" | "followUp" },
	): void;
}

interface WatcherSink {
	api: SendSink;
	isIdle: boolean;
	notify?: (text: string) => void;
	setWidget?: (key: string, lines: string[] | undefined) => void;
}

interface OperationWatcherDeps {
	call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>;
	sink: () => WatcherSink | undefined;
}

interface KernelEvent {
	seq: number;
	event_type: string;
	operation_id?: string;
	payload?: Record<string, unknown>;
}

const POLL_MS = 2000;
const TERMINAL_STATES = new Set(["SUCCEEDED", "FAILED", "CANCELLED", "LOST"]);
const TERMINAL_EVENTS = new Set([
	"operation.completed", "operation.failed", "operation.cancelled", "operation.lost",
]);

export class OperationWatcher {
	private timer: ReturnType<typeof setInterval> | undefined;
	private readonly tracked = new Map<string, string>(); // operation_id → task_id（注册后解析）
	private readonly pending = new Set<string>(); // task_id 未解析的 operation
	private readonly delivered = new Set<string>();
	private readonly seqByTask = new Map<string, number>();
	private readonly lastLineByOp = new Map<string, string>();
	/** R0-1.5：自动路由任务跟踪（task_id 集合——plan 进度 +
	 *  终态一次 followUp）。 */
	private readonly trackedTasks = new Set<string>();
	private readonly deliveredTasks = new Set<string>();
	private readonly completedNodesByTask = new Map<string, Set<string>>();

	constructor(private readonly deps: OperationWatcherDeps) {}

	/** 模型启动 operation 时登记（tool_execution_end: process_start）。 */
	track(operationId: string): void {
		if (!operationId || this.tracked.has(operationId)) return;
		this.pending.add(operationId);
	}

	/** R0-1.5：自动路由任务登记（输入路由执行——无 operation，
	 *  跟踪 task 事件流：plan.node 进度 widget + 终态确定性呈现）。
	 *  P0-3（0827 审计）：只投影不唤醒——终态回复由 Presenter
	 *  确定性发布（triggerTurn:false），不再是模型回合。
	 *  修订重跑（"改成画圆形"→ 同 task 新 revision 再执行）必须
	 *  重新武装终态呈现——deliveredTasks 按执行周期清理（旧事件由
	 *  seq 游标去重，不靠终态集合挡新一轮）。 */
	trackTask(taskId: string): void {
		if (!taskId) return;
		this.deliveredTasks.delete(taskId);
		if (this.trackedTasks.has(taskId)) return;
		this.trackedTasks.add(taskId);
	}

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
	}

	/** 注册解析（每个 op 仅一次）：task_id 是事件流订阅键。 */
	private async resolvePending(): Promise<void> {
		for (const operationId of [...this.pending]) {
			try {
				const result = await this.deps.call("pi.op.get", {
					operation_id: operationId,
				});
				const op = (result.operation ?? {}) as Record<string, unknown>;
				const taskId = String(op.task_id ?? "");
				if (!taskId) continue; // 桥暂不可知——下周期再试（不报假死）
				this.pending.delete(operationId);
				this.tracked.set(operationId, taskId);
				if (TERMINAL_STATES.has(String(op.state ?? ""))) {
					await this.handleTerminal(operationId, op);
				}
			} catch {
				// 桥暂不可用——下周期再试。
			}
		}
	}

	private async tick(): Promise<void> {
		await this.resolvePending();
		if (!this.tracked.size && !this.trackedTasks.size) return;
		const sink = this.deps.sink();
		// R0-1.5：op 任务与自动路由任务同一增量轮询（不重不漏）。
		const taskIds = [
			...new Set([...this.tracked.values(), ...this.trackedTasks]),
		];
		for (const taskId of taskIds) {
			let events: KernelEvent[] = [];
			try {
				const result = await this.deps.call("pi.kernel.events", {
					task_id: taskId,
					last_seq: this.seqByTask.get(taskId) ?? 0,
				});
				events = (result.events ?? []) as KernelEvent[];
			} catch {
				continue; // 桥暂不可用——下周期从同游标重放（不重不漏）
			}
			for (const event of events) {
				this.seqByTask.set(taskId, Math.max(
					this.seqByTask.get(taskId) ?? 0, Number(event.seq) || 0,
				));
				const operationId = String(event.operation_id ?? "");
				if (this.trackedTasks.has(taskId)) {
					await this.handleTaskEvent(taskId, event);
				}
				if (!operationId || !this.tracked.has(operationId)) continue;
				if (event.event_type === "operation.output") {
					const text = String(event.payload?.text ?? "").trim();
					if (text) this.upsertWidget(sink, operationId, text);
				} else if (event.event_type === "operation.progress") {
					const progress = (event.payload?.progress ?? {}) as Record<string, unknown>;
					const label = [
						progress.pct !== undefined ? `${progress.pct}%` : "",
						String(progress.stage ?? ""),
					].filter(Boolean).join(" ");
					if (label) this.upsertWidget(sink, operationId, label);
				} else if (TERMINAL_EVENTS.has(event.event_type)) {
					await this.handleTerminal(operationId, {
						operation_id: operationId,
						task_id: taskId,
						state: String(event.payload?.state ?? ""),
					});
				}
			}
		}
	}

	private upsertWidget(sink: WatcherSink | undefined, operationId: string, line: string): void {
		if (!sink?.setWidget) return;
		this.lastLineByOp.set(operationId, line);
		sink.setWidget(`op:${operationId}`, [
			`⠋ Operation ${operationId.slice(0, 18)}… ${line}`,
		]);
	}

	private clearWidget(sink: WatcherSink | undefined, operationId: string): void {
		if (!sink?.setWidget || !this.lastLineByOp.has(operationId)) return;
		this.lastLineByOp.delete(operationId);
		sink.setWidget(`op:${operationId}`, undefined);
	}

	/** R0-1.5：自动路由任务事件——plan.node 进度原位 widget +
	 *  终态（verification.completed）一次 followUp（不重复、
	 *  progress 不进模型上下文）。 */
	private static readonly NODE_LABELS: Record<string, string> = {
		resolve_robot: "资源",
		make_path: "规划",
		simulate: "仿真",
		render: "渲染",
		render_scene: "场景视频",
		verify: "验证",
	};

	private async handleTaskEvent(taskId: string, event: KernelEvent): Promise<void> {
		if (this.deliveredTasks.has(taskId)) return;
		const sink = this.deps.sink();
		if (event.event_type === "plan.node_completed") {
			const nodeId = String(event.payload?.node_id ?? "");
			const done = this.completedNodesByTask.get(taskId) ?? new Set<string>();
			done.add(nodeId);
			this.completedNodesByTask.set(taskId, done);
			if (sink?.setWidget) {
				const labels = [...done].map(
					(n) => `✓ ${OperationWatcher.NODE_LABELS[n] ?? n}`,
				);
				sink.setWidget(`task:${taskId}`, [
					`⠋ 任务执行中（确定性链）：${labels.join(" ")}`,
				]);
			}
			return;
		}
		if (event.event_type !== "verification.completed") return;
		// P0-3（0827 审计）：Coordinator 是唯一终态发布者——终态回复由
		// TaskOutcome 确定性生成、display 直接呈现；绝不 followUp 唤醒
		// Agent（0827 实证：followUp 触发模型回合与确定性链互相矛盾
		// =双控制者）。trackTask 只投影，不唤醒。
		this.deliveredTasks.add(taskId);
		this.trackedTasks.delete(taskId);
		this.completedNodesByTask.delete(taskId);
		if (sink?.setWidget) sink.setWidget(`task:${taskId}`, undefined);
		let outcomeText = "";
		try {
			const result = await this.deps.call("pi.coordinator.consider", {
				task_id: taskId,
			});
			const outcome = (result.outcome ?? {}) as Record<string, unknown>;
			outcomeText = renderTerminalReply(outcome as TerminalOutcome);
		} catch {
			outcomeText = "任务已终态（outcome 拉取失败——/activity 查看账本）";
		}
		sink?.api.sendMessage(
			{
				customType: "rosclaw.task_terminal",
				content: outcomeText,
				display: true,
				details: { task_id: taskId },
			},
			{ triggerTurn: false },
		);
	}

	private async handleTerminal(
		operationId: string, op: Record<string, unknown>,
	): Promise<void> {
		if (this.delivered.has(operationId)) return;
		this.delivered.add(operationId);
		this.pending.delete(operationId);
		this.tracked.delete(operationId);
		const sink = this.deps.sink();
		this.clearWidget(sink, operationId);
		const state = String(op.state ?? "");
		// WP-1（0823 审计 P0-3）：终态一致性——owning task 已终态
		// （或 operation 属旧 revision）时，终态事件只更新账本和
		// TUI，绝不触发模型回合。
		const taskId = String(op.task_id ?? "");
		let taskTerminal = false;
		let staleRevision = false;
		if (taskId) {
			try {
				const taskResult = await this.deps.call("pi.kernel.get", {
					task_id: taskId,
				});
				const task = (taskResult.task ?? null) as Record<string, unknown> | null;
				const taskState = String(task?.state ?? "");
				taskTerminal = task !== null && taskState !== "RUNNING"
					&& taskState !== "CREATED" && taskState !== "WAITING_APPROVAL";
				const opRevision = Number(op.revision ?? 0);
				const activeRevision = Number(task?.active_revision ?? 0);
				staleRevision = opRevision > 0 && activeRevision > 0
					&& opRevision !== activeRevision;
			} catch {
				taskTerminal = true; // 查询失败不赌——按终态处理（不触发回合）
			}
		}
		if (taskTerminal || staleRevision) {
			sink?.notify?.(
				`Operation ${state}（任务已${staleRevision ? "换 revision" : "终态"}——已存档，不再打扰）：${operationId.slice(0, 18)}…`,
			);
			return;
		}
		const content =
			`后台 Operation ${operationId} 已终止：${state}`
			+ (op.failure_code ? `（${String(op.failure_code)}）` : "")
			+ "。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。";
		if (sink?.api) {
			sink.api.sendMessage(
				{
					customType: "rosclaw.operation.result",
					content,
					display: false,
					details: { operation_id: operationId, state },
				},
				sink.isIdle
					? { triggerTurn: true }
					: { triggerTurn: true, deliverAs: "followUp" },
			);
		}
		sink?.notify?.(`Operation ${state}：${operationId.slice(0, 18)}…`);
	}
}
