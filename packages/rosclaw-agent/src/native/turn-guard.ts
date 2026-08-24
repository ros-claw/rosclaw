/** TurnGuard（PR-H4，总纲 v2 §12.2）。
 *
 * action-oriented 回合（模型用了 write/edit/bash/process 工具）结束
 * 而未调用 rosclaw_task_finish / rosclaw_task_blocked → 注入一次结构
 * 化 follow-up（同一逻辑回合继续，提醒缺的验收证据）。同一
 * (task, revision) 只注入一次；模型仍不收尾则由用户在 UI 决定
 * （不无限递归）。
 */

interface GuardSink {
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

export interface TurnGuardDeps {
	call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>;
	missionId: () => string;
	sessionRef: () => string;
	sink: () => { api: GuardSink; isIdle: boolean } | undefined;
	notify?: (text: string) => void;
}

const WORK_TOOLS = new Set(["write", "edit", "bash"]);
// 注意：process_start 不算"该收尾的同步工作"——长 operation 在飞是合法
// 挂起点（终态由 OperationWatcher followUp 接管），此时催收尾会诱发
// 重复 start（实证：H3 回归双开 operation）。
const FINISH_TOOLS = new Set(["rosclaw_task_finish", "rosclaw_task_blocked"]);

// P0-B（0824 总纲 §19.P0-B）：Terminal Fence——task 已终态时
// TurnGuard 不得注入 follow-up（终态后触发模型回合=幽灵执行）。
// 与 task_kernel.TASK_ACTIVE 同集合（ ACTIVE 子态才可被催收尾）。
const TASK_ACTIVE = new Set([
	"RUNNING", "WAITING_OPERATION", "WAITING_INPUT",
	"WAITING_PERMISSION", "PAUSED", "VERIFYING", "RECOVERING",
]);

export class TurnGuard {
	private usedWorkTools = false;
	private finished = false;
	private readonly nudged = new Set<string>(); // `${taskId}:r${revision}`

	constructor(private readonly deps: TurnGuardDeps) {}

	/** tool_execution_end 钩子：记录本回合的工具使用。 */
	noteTool(toolName: string): void {
		if (WORK_TOOLS.has(toolName)) this.usedWorkTools = true;
		if (FINISH_TOOLS.has(toolName)) this.finished = true;
	}

	/** turn_end 钩子：需要时注入一次验收提醒。 */
	async onTurnEnd(): Promise<void> {
		const usedWork = this.usedWorkTools;
		const finished = this.finished;
		this.usedWorkTools = false;
		this.finished = false;
		if (!usedWork || finished) return;
		let task: Record<string, unknown> | null = null;
		try {
			const result = await this.deps.call("pi.kernel.active", {
				mission_id: this.deps.missionId(),
				session_ref: this.deps.sessionRef(),
			});
			task = (result.task as Record<string, unknown> | null) ?? null;
		} catch {
			return; // 桥不可用——下回合再说（不阻塞回合收尾）
		}
		if (!task) return;
		// P0-B：终态栅栏——非 ACTIVE 子态一律不触发（延迟事件只
		// 归档，不变成新的模型请求）。
		if (!TASK_ACTIVE.has(String(task.state ?? ""))) return;
		const key = `${String(task.task_id)}:r${String(task.active_revision)}`;
		if (this.nudged.has(key)) return; // 每 revision 只提醒一次
		this.nudged.add(key);
		const sink = this.deps.sink();
		if (!sink?.api) return;
		const content =
			"你刚才用工作工具改动了内容，但没有收尾：任务只有经过验收才算完成。" +
			"请调用 rosclaw_artifact_register 登记交付物，然后 rosclaw_task_finish" +
			"（带 artifact_ids）让验收器真实检查；做不到就 rosclaw_task_blocked" +
			"（带原因码）。零证据不算完成。";
		sink.api.sendMessage(
			{
				customType: "rosclaw.turn_guard",
				content,
				display: false,
				details: {
					task_id: String(task.task_id),
					revision: Number(task.active_revision),
					// P0-B：因果续接——同一 (task, revision) 的催促因果
					// 唯一，内核/账本按 causation_id 幂等去重。
					causation_id: `turn_guard:${String(task.task_id)}:r${String(task.active_revision)}`,
				},
			},
			sink.isIdle ? { triggerTurn: true } : { triggerTurn: true, deliverAs: "followUp" },
		);
		this.deps.notify?.("任务未验收——已提醒收尾（TurnGuard）");
	}
}
