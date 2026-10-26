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
	attempt_id?: string;
	payload?: Record<string, unknown>;
}

/** UI observation only: never a heartbeat, deadline or process supervisor.
 * Entries exist only for explicitly tracked, nonterminal operations. Session
 * continuation does not discover/reattach arbitrary external processes. */
interface OperationObservation {
	since: number;
	unavailable: boolean;
	ledgerSource: string;
	outputAt?: number;
	output?: string;
	progressAt?: number;
	progress?: string;
}

const STATUS_TEXT_LIMIT = 240;
function statusText(text: string): string {
	// One bounded, inert UI line; never retain an accumulating output log.
	const tail = text.slice(-STATUS_TEXT_LIMIT).replace(/^[\uDC00-\uDFFF]/, "")
		.replace(/[\u0000-\u001f\u007f-\u009f]/g, " ").replace(/\s+/g, " ").trim();
	return (text.length > STATUS_TEXT_LIMIT ? "…" : "") + tail;
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
	private readonly operations = new Map<string, Record<string, unknown>>();
	/** Busy notifications stay retractable here, never in PI's follow-up queue. */
	private readonly pendingTerminalOps = new Map<string, Record<string, unknown>>();
	private readonly pendingProofs = new Map<string, Record<string, unknown>>();
	private readonly seqByTask = new Map<string, number>();
	private readonly lastLineByOp = new Map<string, string>();
	private readonly observations = new Map<string, OperationObservation>();
	private ticking = false;
	private stopped = false;
	private generation = 0;
	/** R0-1.5：自动路由任务跟踪（task_id 集合——plan 进度 +
	 *  终态一次 followUp）。 */
	private readonly trackedTasks = new Set<string>();
	private readonly deliveredTasks = new Set<string>();
	private readonly completedNodesByTask = new Map<string, Set<string>>();
	/** 空闲门控挂起：终态事件已到但 Agent 流式中——延迟到空闲呈现
	 *  （seq 游标已前进，靠本集合重试，不是重放事件）。 */
	private readonly pendingTerminalTasks = new Set<string>();

	constructor(private readonly deps: OperationWatcherDeps) {}

	/** 模型启动 operation 时登记（tool_execution_end: process_start）。 */
	track(operationId: string): void {
		if (!operationId || this.pending.has(operationId) || this.tracked.has(operationId)
			|| this.delivered.has(operationId) || this.pendingTerminalOps.has(operationId)) return;
		this.pending.add(operationId);
		this.observations.set(operationId, {
			since: Date.now(), unavailable: false, ledgerSource: "registration pending; no public result yet",
		});
		this.upsertWidget(this.deps.sink(), operationId);
	}

	/** ACK only finalized toolResult messages retained by the public PI agent. */
	observeToolMessage(message: Record<string, unknown>): void {
		if (message.role !== "toolResult" || message.isError === true
			|| !["process_status", "process_output", "process_stop"].includes(String(message.toolName))) return;
		this.observeToolResult((message.details ?? {}) as Record<string, unknown>);
	}

	/** A successful terminal status/output result has already reached this agent.
	 * Match the authoritative operation identity, owning task and revision; a
	 * partial output or failed read must not suppress a later terminal result. */
	observeToolResult(result: Record<string, unknown>): void {
		const op = result.operation as Record<string, unknown> | undefined;
		if (result.ok !== true || !op || !TERMINAL_STATES.has(String(result.status))
			|| op.state !== result.status) return;
		const id = String(op.operation_id ?? "");
		if (!id || (!this.pending.has(id) && !this.tracked.has(id)
			&& !this.pendingTerminalOps.has(id))) return;
		if (!this.operations.has(id)) {
			this.pendingProofs.set(id, op);
			return;
		}
		this.consumeTerminalProof(id, op);
	}

	private consumeTerminalProof(id: string, proof: Record<string, unknown>): void {
		const known = this.operations.get(id);
		const terminal = this.pendingTerminalOps.get(id);
		if (!known || !proof.task_id || proof.task_id !== known.task_id
			|| !Number.isInteger(proof.revision) || Number(proof.revision) < 0
			|| Number(proof.revision) !== Number(known.revision ?? 0)
			|| (terminal && terminal.state !== proof.state)) return;
		this.delivered.add(id);
		this.pending.delete(id);
		this.tracked.delete(id);
		this.pendingTerminalOps.delete(id);
		this.clearWidget(this.deps.sink(), id);
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
		this.stopped = false;
		if (this.timer) return;
		this.timer = setInterval(() => {
			void this.tick().catch(() => undefined);
		}, POLL_MS);
		if (typeof this.timer === "object" && "unref" in this.timer) this.timer.unref();
	}

	stop(): void {
		this.stopped = true;
		if (this.timer) clearInterval(this.timer);
		this.timer = undefined;
		this.generation++; // An in-flight read cannot recreate a stopped UI.
		const sink = this.deps.sink();
		for (const id of this.lastLineByOp.keys()) sink?.setWidget?.(`op:${id}`, undefined);
		this.lastLineByOp.clear();
		for (const view of this.observations.values()) view.unavailable = true;
	}

	/** 注册解析（每个 op 仅一次）：task_id 是事件流订阅键。 */
	private async resolvePending(generation: number): Promise<void> {
		for (const operationId of [...this.pending]) {
			try {
				const result = await this.deps.call("pi.op.get", {
					operation_id: operationId,
				});
				if (generation !== this.generation) return;
				const op = (result.operation ?? {}) as Record<string, unknown>;
				const taskId = String(op.task_id ?? "");
				if (!taskId || (op.operation_id && op.operation_id !== operationId)) {
					throw new Error("operation identity unavailable");
				}
				this.pending.delete(operationId);
				this.tracked.set(operationId, taskId);
				this.operations.set(operationId, op);
				const view = this.observations.get(operationId);
				if (view) {
					view.unavailable = false;
					view.ledgerSource = "registration snapshot, not a fresh liveness observation";
				}
				const proof = this.pendingProofs.get(operationId);
				if (proof) {
					this.pendingProofs.delete(operationId);
					this.consumeTerminalProof(operationId, proof);
				}
				if (TERMINAL_STATES.has(String(op.state ?? ""))) {
					await this.handleTerminal(operationId, op);
				} else {
					// Visible even if the first event read is slow or there is no output.
					this.upsertWidget(this.deps.sink(), operationId);
				}
			} catch {
				if (generation !== this.generation) return;
				const view = this.observations.get(operationId);
				if (view) view.unavailable = true;
				this.upsertWidget(this.deps.sink(), operationId);
			}
			if (generation !== this.generation) return;
		}
	}

	private async tick(): Promise<void> {
		// A slow bridge must not create overlapping 2s requests/cursor races.
		if (this.ticking) {
			if (this.stopped) return;
			// Keep observation ages moving without treating the pending read as
			// fresh evidence, issuing another request or inventing a deadline.
			for (const id of this.observations.keys()) this.upsertWidget(this.deps.sink(), id);
			return;
		}
		// A new observation cycle (also used without a timer) resumes tracking;
		// an overlap with a stopped in-flight cycle must not resume/redraw it.
		this.stopped = false;
		this.ticking = true;
		try {
			await this.poll(this.generation);
		} finally {
			this.ticking = false;
		}
	}

	private async poll(generation: number): Promise<void> {
		await this.resolvePending(generation);
		if (generation !== this.generation) return;
		if (!this.tracked.size && !this.trackedTasks.size
			&& !this.pendingTerminalTasks.size && !this.pendingTerminalOps.size) return;
		// R0-1.5：op 任务与自动路由任务同一增量轮询（不重不漏）。
		const taskIds = [
			...new Set([...this.tracked.values(), ...this.trackedTasks]),
		];
		for (const taskId of taskIds) {
			let events: KernelEvent[] = [];
			let unavailable = false;
			try {
				const result = await this.deps.call("pi.kernel.events", {
					task_id: taskId,
					last_seq: this.seqByTask.get(taskId) ?? 0,
				});
				if (!Array.isArray(result.events)) throw new Error("event feed unavailable");
				events = result.events as KernelEvent[];
			} catch {
				unavailable = true; // Keep cursor, ledger and output ages unchanged.
			}
			if (generation !== this.generation) return;
			for (const event of events) {
				if (!Number.isSafeInteger(event.seq)
					|| event.seq <= (this.seqByTask.get(taskId) ?? 0)) continue;
				this.seqByTask.set(taskId, event.seq);
				const operationId = String(event.operation_id ?? "");
				if (this.trackedTasks.has(taskId)) {
					await this.handleTaskEvent(taskId, event);
					if (generation !== this.generation) return;
				}
				if (!operationId || this.tracked.get(operationId) !== taskId) continue;
				const op = this.operations.get(operationId)!;
				if (event.attempt_id && op.attempt_id && event.attempt_id !== op.attempt_id) continue;
				const view = this.observations.get(operationId)!;
				if (event.event_type === "operation.output") {
					const text = String(event.payload?.text ?? "");
					if (text) {
						view.output = statusText(text) || "(whitespace)";
						view.outputAt = Date.now(); // UI receipt time, not producer time.
					}
				} else if (event.event_type === "operation.progress") {
					const progress = (event.payload?.progress ?? {}) as Record<string, unknown>;
					const label = [
						typeof progress.pct === "number" && Number.isFinite(progress.pct)
							&& progress.pct >= 0 && progress.pct <= 100 ? `${progress.pct}%` : "",
						typeof progress.stage === "string" ? statusText(progress.stage) : "",
					].filter(Boolean).join(" ");
					if (label) {
						view.progress = statusText(label);
						view.progressAt = Date.now();
					}
				} else if (event.event_type === "operation.canceling"
					|| ((event.event_type === "operation.degraded" || event.event_type === "operation.resumed")
						&& op.state !== "CANCELING")) {
					op.state = event.event_type === "operation.canceling" ? "CANCELING"
						: event.event_type === "operation.degraded" ? "DEGRADED" : "RUNNING";
					view.ledgerSource = event.event_type;
				} else if (TERMINAL_EVENTS.has(event.event_type)) {
					const state = String(event.payload?.state ?? "");
					if (!TERMINAL_STATES.has(state) || (op.state === "CANCELING" && state !== "CANCELLED")) continue;
					await this.handleTerminal(operationId, {
						...op, operation_id: operationId, task_id: taskId, state,
					});
					if (generation !== this.generation) return;
				}
			}
			// One bounded replacement per operation per tick, not per output chunk.
			for (const [id, owner] of this.tracked) {
				if (owner !== taskId) continue;
				this.observations.get(id)!.unavailable = unavailable;
				this.upsertWidget(this.deps.sink(), id);
			}
		}
		for (const [operationId, op] of [...this.pendingTerminalOps]) {
			await this.handleTerminal(operationId, op);
			if (generation !== this.generation) return;
		}
		// 空闲门控挂起 drain：流式中延迟的终态在空闲后独立呈现。
		for (const taskId of [...this.pendingTerminalTasks]) {
			await this.presentTerminal(taskId);
			if (generation !== this.generation) return;
		}
	}

	private upsertWidget(sink: WatcherSink | undefined, operationId: string): void {
		const view = this.observations.get(operationId);
		if (this.stopped || !sink?.setWidget || !view) return;
		const now = Date.now();
		const age = (at: number) => `${Math.floor(Math.max(0, now - at) / 1000)}s`;
		const state = String(this.operations.get(operationId)?.state ?? "UNKNOWN");
		const lines = [
			`⠋ Operation ${statusText(operationId.slice(0, 18))}… last known ledger: ${statusText(state)} (${view.ledgerSource})`,
			`UI elapsed since tracking: ${age(view.since)}; output: ` + (view.outputAt === undefined
				? `none observed in this UI (quiet ≥ ${age(view.since)} since tracking)`
				: `${age(view.outputAt)} since UI-observed output (replay may be old)`),
			view.progressAt === undefined ? "Progress: UNKNOWN (no structured progress observed)"
				: `Last structured progress: ${view.progress} (${age(view.progressAt)} since UI observation; may be old)`,
			view.unavailable ? "Bridge: status unavailable; showing last known ledger only"
				: this.pending.has(operationId) ? "Bridge: registration pending; no public result yet"
					: "Bridge: last public read succeeded; not a process liveness check",
			"Current liveness/completion: UNKNOWN; silence does not prove deadlock or death",
		];
		if (state === "DEGRADED") lines.push("Ledger DEGRADED warning; not proof of deadlock/death");
		if (state === "CANCELING") lines.push("Cancellation requested; process/physical stop not confirmed");
		if (view.output !== undefined) lines.push(`Last output (bounded UI tail): ${view.output}`);
		const rendered = lines.join("\n");
		if (this.lastLineByOp.get(operationId) === rendered) return;
		sink.setWidget(`op:${operationId}`, lines);
		this.lastLineByOp.set(operationId, rendered);
	}

	private clearWidget(sink: WatcherSink | undefined, operationId: string): void {
		this.observations.delete(operationId);
		const shown = this.lastLineByOp.delete(operationId);
		if (shown) sink?.setWidget?.(`op:${operationId}`, undefined);
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
		if (event.event_type !== "task.terminal") return;
		// 0827 真实 K3 复验实证：verification.completed 会在中间态
		// （REPAIR_REQUIRED/FAIL）触发——把它当终态呈现 = 把瞬态失败
		// 钉成用户可见终态并毒化 deliveredTasks（链恢复 PASS 后回复
		// 缺席，内核 SUCCEEDED 与屏幕 FAIL 矛盾——正是 0827 审计的
		// 双真相）。终态发布只认 task.terminal（内核权威终态）。
		// 空闲门控：Agent 还在流式回答时，pi 会把 triggerTurn:false 的
		// custom message steer 进正在运行的回合——终态回复消失在流
		// 里且违反"终态后零模型回合"。不空闲 → 挂起
		// （pendingTerminalTasks），空闲后由 tick 呈现。
		const sinkNow = this.deps.sink();
		if (sinkNow && !sinkNow.isIdle) {
			this.pendingTerminalTasks.add(taskId);
			return;
		}
		await this.presentTerminal(taskId);
	}

	/** P0-3（0827 审计）：Coordinator 是唯一终态发布者——终态回复由
	 *  TaskOutcome 确定性生成、display 直接呈现；绝不 followUp 唤醒
	 *  Agent（0827 实证：followUp 触发模型回合与确定性链互相矛盾
	 *  =双控制者）。trackTask 只投影，不唤醒。 */
	private async presentTerminal(taskId: string): Promise<void> {
		if (this.deliveredTasks.has(taskId)) return;
		const sink = this.deps.sink();
		if (sink && !sink.isIdle) {
			this.pendingTerminalTasks.add(taskId);
			return;
		}
		this.pendingTerminalTasks.delete(taskId);
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
		const generation = this.generation;
		this.pendingTerminalOps.set(operationId, op);
		this.pending.delete(operationId);
		this.tracked.delete(operationId);
		const sink = this.deps.sink();
		this.clearWidget(sink, operationId);
		if (!sink?.isIdle) return;
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
				if (!task || (task.task_id && task.task_id !== taskId)) return;
				const taskState = String(task.state ?? "");
				taskTerminal = task !== null && taskState !== "RUNNING"
					&& taskState !== "CREATED" && taskState !== "WAITING_APPROVAL";
				const opRevision = Number(op.revision ?? 0);
				const activeRevision = Number(task?.active_revision ?? 0);
				staleRevision = opRevision > 0 && activeRevision > 0
					&& opRevision !== activeRevision;
			} catch {
				return; // Preserve the reminder until ownership can be verified.
			}
		}
		if (generation !== this.generation || this.delivered.has(operationId)) return;
		// A terminal tool proof or session stop may have arrived during the await.
		if (!this.deps.sink()?.isIdle) return;
		this.delivered.add(operationId);
		this.pendingTerminalOps.delete(operationId);
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
			try {
				sink.api.sendMessage(
					{
						customType: "rosclaw.operation.result",
						content,
						display: false,
						details: { operation_id: operationId, state },
					},
					{ triggerTurn: true },
				);
			} catch {
				this.delivered.delete(operationId);
				this.pendingTerminalOps.set(operationId, op);
				return;
			}
		}
		sink?.notify?.(`Operation ${state}：${operationId.slice(0, 18)}…`);
	}
}
