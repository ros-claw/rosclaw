/** OperationWatcher（PR-H3，总纲 v2 §11.4）——operation 终态一次性
 *  followUp 注入同一 session。
 *
 * 不是 completion watcher 轮询链：只跟踪模型显式启动的 operation；
 * 终态只触发一次 followUp（紧凑结构化结果），heartbeat/progress 绝不
 * 进模型上下文（零 LLM token 开销——进度展示走 UI 事件流）。
 */

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

interface OperationWatcherDeps {
	call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>;
	sink: () => { api: SendSink; isIdle: boolean; notify?: (text: string) => void } | undefined;
}

const POLL_MS = 2000;

export class OperationWatcher {
	private timer: ReturnType<typeof setInterval> | undefined;
	private readonly tracked = new Map<string, string>(); // operation_id → state(已见终态则删)
	private readonly delivered = new Set<string>();

	constructor(private readonly deps: OperationWatcherDeps) {}

	/** 模型启动 operation 时登记（tool_execution_end: process_start）。 */
	track(operationId: string): void {
		if (operationId) this.tracked.set(operationId, "RUNNING");
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

	private async tick(): Promise<void> {
		if (!this.tracked.size) return;
		const sink = this.deps.sink();
		for (const operationId of [...this.tracked.keys()]) {
			if (this.delivered.has(operationId)) {
				this.tracked.delete(operationId);
				continue;
			}
			let op: Record<string, unknown> | undefined;
			try {
				const result = await this.deps.call("pi.op.get", {
					operation_id: operationId,
				});
				op = result.operation as Record<string, unknown> | undefined;
			} catch {
				continue; // 桥暂不可用——下周期再试（不报假死）
			}
			const state = String(op?.state ?? "");
			if (!op || !(state === "SUCCEEDED" || state === "FAILED" || state === "CANCELLED")) {
				continue;
			}
			this.delivered.add(operationId);
			this.tracked.delete(operationId);
			// WP-1（0823 审计 P0-3）：终态一致性——owning task 已终态
			// （或 operation 属旧 revision）时，终态事件只更新账本和
			// TUI，绝不触发模型回合。规则：Task 终态后、下一条用户
			// 输入前，模型调用次数恒等于 0。
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
					// 查询失败不赌——按终态处理（不触发回合），下周期
					// 由 delivered 集合保证不重复。
					taskTerminal = true;
				}
			}
			if (taskTerminal || staleRevision) {
				sink?.notify?.(
					`Operation ${state}（任务已${staleRevision ? "换 revision" : "终态"}——已存档，不再打扰）：${operationId.slice(0, 18)}…`,
				);
				continue;
			}
			// 终态一次性 followUp（compact 结构化结果——同一 session 继续
			// 验证/修复，不新建 Worker/任务）。
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
					// idle 时立即触发回合；忙时 followUp 排队。
					sink.isIdle
						? { triggerTurn: true }
						: { triggerTurn: true, deliverAs: "followUp" },
				);
			}
			sink?.notify?.(`Operation ${state}：${operationId.slice(0, 18)}…`);
		}
	}
}
