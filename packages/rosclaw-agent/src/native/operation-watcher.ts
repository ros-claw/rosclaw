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
