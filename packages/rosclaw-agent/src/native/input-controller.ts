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
	notify: (text: string, kind: "info" | "warning" | "error") => void;
}

export class InputController {
	/** /new 设置——下一条输入强制新 root task。 */
	forceNewNext = false;
	/** 当前绑定的 task（TUI 卡/命令展示用）。 */
	currentTaskId = "";
	currentRevision = 0;

	constructor(private readonly deps: InputControllerDeps) {}

	/** 输入事务：先落账绑定，再决定是否投递。返回 null = 不投递。 */
	async bind(text: string): Promise<TaskBindResult | null> {
		const messageId = `msg_${randomUUID()}`;
		try {
			const result = (await this.deps.call("pi.task.bind", {
				mission_id: this.deps.missionId(),
				session_ref: this.deps.sessionRef(),
				backend_native_id: this.deps.backendNativeId(),
				message_id: messageId,
				text,
				cwd: this.deps.cwd(),
				force_new: this.forceNewNext,
			})) as unknown as TaskBindResult & { ok?: boolean };
			this.forceNewNext = false;
			this.currentTaskId = result.task_id;
			this.currentRevision = result.revision;
			return result;
		} catch (err) {
			// 绑定失败不投递（幽灵执行防线）：消息不消失于沉默——
			// 明确通知用户重发。
			this.deps.notify(
				`任务绑定失败（消息未发送，请重试）：${(err as Error).message}`.slice(0, 200),
				"error",
			);
			return null;
		}
	}
}
