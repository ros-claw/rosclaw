/** ActiveSessionContext（NA-FIX-1/2）：所有工具/注入/镜像/审批在执行时
 * 动态读取的单一状态源——绝不捕获启动时的字符串。
 *
 * NA-FIX-2 的切换事务会原子替换本对象的字段。
 */

import type { EmbodiedContextEnvelope } from "../extension/context-injection.js";

export interface ActiveSessionState {
	sessionId: string;
	sessionFile?: string;
	missionId?: string;
	bindingId?: string;
	bindingRevision?: number;
	leaseToken?: string;
	contextRevision: number;
	bodyId?: string;
	bodyHash?: string;
	mode: string;
	profile: "developer" | "robot" | "worker";
	/** HOTFIX-1：agentd 签发的 ValidatedContextLease（action 准入凭证）。 */
	contextLeaseId?: string;
}

export class ActiveSessionContext {
	private state: ActiveSessionState;

	constructor(initial: ActiveSessionState) {
		this.state = { ...initial };
	}

	get current(): Readonly<ActiveSessionState> {
		return this.state;
	}

	/** 原子替换（NA-FIX-2 切换事务的唯一写入点）。 */
	replace(next: ActiveSessionState): void {
		this.state = { ...next };
	}

	patch(partial: Partial<ActiveSessionState>): void {
		this.state = { ...this.state, ...partial };
	}

	/** 每轮注入验证通过后写入 revision/body/mode（P0-7 的精确数据来源）。
	 *  HOTFIX-1：agentd 签发的 context lease 一并记录——action 工具必须
	 *  出示它（lease 只存 id；它本身不是执行权）。 */
	applyEnvelope(envelope: EmbodiedContextEnvelope, contextLeaseId?: string): void {
		const body = envelope.body as { body_id?: string; effective_body_hash?: string };
		const safety = envelope.safety as { mode?: string };
		this.patch({
			contextRevision: envelope.context_revision,
			bodyId: body.body_id,
			bodyHash: body.effective_body_hash,
			...(safety.mode ? { mode: safety.mode } : {}),
			...(contextLeaseId ? { contextLeaseId } : {}),
		});
	}
}
