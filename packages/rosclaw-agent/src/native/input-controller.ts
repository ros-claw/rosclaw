/** InputController（PR-H2，ADR-0012，总纲 v2 §9.3）。
 *
 * Root Task 绑定算法（模型不能创建 root task）：
 * 1. TUI 生成稳定 message_id；
 * 2. 消息先落账（pi.task.bind——tasks/revisions/bindings 事务）；
 * 3. 事务提交后才投递 Harness（input handler 返回 continue）；
 * 4. 绑定失败 → handled + 通知（不投递=无幽灵执行；用户重发即可，
 *    message_id 由调用方持有一次提交内不变——重发用新 id，重放同 id
 *    幂等）；
 * 5. /new 显式开新任务（force_new）；其余消息进活跃 task 的 revision。
 */

import { randomUUID } from "node:crypto";

export interface TaskBindResult {
	task_id: string;
	revision: number;
	created_task: boolean;
	replayed: boolean;
	workspace_path: string;
	state: string;
}

export interface InputControllerDeps {
	call: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>;
	missionId: () => string;
	sessionRef: () => string;
	backendNativeId: () => string;
	cwd: () => string;
	/** 当前绑定 body（PR-N0 熔断的执行面：机器人行为任务的受信
	 *  证据要求由 body_id 驱动——不传则熔断在 chat 路径惰性失效）。 */
	bodyId?: () => string;
	notify: (text: string, kind: "info" | "warning" | "error") => void;
}

export class InputController {
	/** /new 设置——下一条输入强制新 root task。 */
	forceNewNext = false;
	/** 当前绑定的 task（TUI 卡/命令展示用）。 */
	currentTaskId = "";
	currentRevision = 0;

	constructor(private readonly deps: InputControllerDeps) {}

	/** P0-C（0824 总纲 §6.1）：输入先落会话（pi.input.persist——
	 *  不立即创建 Task；persist 失败不投递，HP1 防线语义不变）。
	 *  返回 null = 不投递。 */
	async persist(text: string): Promise<{ input_id?: string } | null> {
		const messageId = `msg_${randomUUID()}`;
		try {
			const result = (await this.deps.call("pi.input.persist", {
				mission_id: this.deps.missionId(),
				session_ref: this.deps.sessionRef(),
				backend_native_id: this.deps.backendNativeId(),
				message_id: messageId,
				text,
				force_new: this.forceNewNext,
			})) as unknown as { ok?: boolean; input?: { input_id?: string } };
			this.forceNewNext = false;
			this.deps.call("pi.input.dispatched", {
				mission_id: this.deps.missionId(),
				message_id: messageId,
			}).catch(() => undefined);
			return result.input ?? {};
		} catch (err) {
			// 持久化失败不投递（幽灵执行防线）：消息不消失于沉默——
			// 明确通知用户重发。
			this.deps.notify(
				`输入持久化失败（消息未发送，请重试）：${(err as Error).message}`.slice(0, 200),
				"error",
			);
			return null;
		}
	}

	/** P0-C：活跃 task 来自内核查询（同一事实源）——不再依赖
	 *  bind 返回值（输入不再逐条建 task）。 */
	async activeTaskId(): Promise<string> {
		try {
			const result = await this.deps.call("pi.kernel.active", {
				mission_id: this.deps.missionId(),
				session_ref: this.deps.sessionRef(),
			});
			const task = result.task as { task_id?: string } | null | undefined;
			return String(task?.task_id ?? "");
		} catch {
			return "";
		}
	}

	/** 最近 task（含刚终态）——/activity /logs /artifacts 的展示
	 *  目标（终态不抹掉刚完成的任务账本）。 */
	async latestTaskId(): Promise<string> {
		try {
			const result = await this.deps.call("pi.kernel.latest", {
				mission_id: this.deps.missionId(),
				session_ref: this.deps.sessionRef(),
			});
			const task = result.task as { task_id?: string } | null | undefined;
			return String(task?.task_id ?? "");
		} catch {
			return "";
		}
	}
}
