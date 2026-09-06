/** ActiveSessionContext（NA-FIX-1/2）：所有工具/注入/镜像/审批在执行时
 * 动态读取的单一状态源——绝不捕获启动时的字符串。
 *
 * NA-FIX-2 的切换事务会原子替换本对象的字段。
 */

import type { EmbodiedContextEnvelope } from "../extension/context-injection.js";

export type ContextState = "LOADING" | "FRESH" | "STALE" | "UNAVAILABLE";
export type LeaseState = "ACTIVE" | "LOST" | "NONE";

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
	/** HOTFIX-3（P0-4E）：显式安全状态——不再是"revision>0 就猜 FRESH"。
	 *  context 未验证时 LOADING/UNAVAILABLE，验证过才 FRESH，过期即 STALE。 */
	contextState: ContextState;
	/** writer lease 状态——heartbeat 连续失败即 LOST（动作禁行）。 */
	leaseState: LeaseState;
	/** 动作准入的 UI/tool 层单一判据——只有 FRESH + ACTIVE 才 true。
	 *  admission 的内核校验仍是最终权威；本字段让 UI/工具提前诚实拒绝。 */
	actionsAllowed: boolean;
	/** 大道至简 R1-2a：活跃任务运行目录（pi.context active_run——
	 *  任务沙箱 scratch 区= runDir/scratch，write/edit/bash cwd 的
	 *  额外允许根）。无活跃任务时缺省。 */
	taskRunDir?: string;
}

export class ActiveSessionContext {
	private state: ActiveSessionState;
	private readonly listeners = new Set<() => void>();

	constructor(initial: ActiveSessionState) {
		this.state = { ...initial };
	}

	get current(): Readonly<ActiveSessionState> {
		return this.state;
	}

	/** PR-SIX-1：状态变化订阅——patch/replace/leaseLost/contextStale 后
	 *  统一 fan-out（chrome 刷新不再依赖各调用点手动触发）。 */
	subscribe(listener: () => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	private notify(): void {
		for (const listener of this.listeners) listener();
	}

	/** 原子替换（NA-FIX-2 切换事务的唯一写入点）。
	 *  HOTFIX-3：replace 也必须重算派生字段——否则 A 的 actionsAllowed=true
	 *  会随切换存活（commitState 设了 UNAVAILABLE 但 actionsAllowed 没变）。 */
	replace(next: ActiveSessionState): void {
		this.state = { ...next };
		this.state = { ...this.state, actionsAllowed: this.computeActionsAllowed() };
		this.notify();
	}

	patch(partial: Partial<ActiveSessionState>): void {
		this.state = { ...this.state, ...partial };
		// actionsAllowed 是派生字段——任何 patch 后重算，不留陈旧 true。
		this.state = { ...this.state, actionsAllowed: this.computeActionsAllowed() };
		this.notify();
	}

	private computeActionsAllowed(): boolean {
		return (
			this.state.missionId !== undefined
			&& this.state.contextState === "FRESH"
			&& this.state.leaseState === "ACTIVE"
			&& this.state.contextLeaseId !== undefined
		);
	}

	/** 每轮注入验证通过后写入 revision/body/mode（P0-7 的精确数据来源）。
	 *  HOTFIX-1：agentd 签发的 context lease 一并记录——action 工具必须
	 *  出示它（lease 只存 id；它本身不是执行权）。
	 *  HOTFIX-3：验证通过即 FRESH + 重算动作准入。 */
	applyEnvelope(envelope: EmbodiedContextEnvelope, contextLeaseId?: string): void {
		const body = envelope.body as { body_id?: string; effective_body_hash?: string };
		const safety = envelope.safety as { mode?: string };
		this.patch({
			contextRevision: envelope.context_revision,
			bodyId: body.body_id,
			bodyHash: body.effective_body_hash,
			...(safety.mode ? { mode: safety.mode } : {}),
			contextState: "FRESH",
			...(contextLeaseId ? { contextLeaseId } : {}),
		});
	}

	/** context 拉取失败/过期（P0-4E）：清空 freshness 与动作准入——
	 *  绝不保留旧 revision/body 当作"还能动作"。 */
	markContextStale(note?: string): void {
		this.patch({
			contextState: "STALE",
			contextLeaseId: undefined,
		});
	}

	/** heartbeat 失败/lease 过期（P0-4E）：动作禁行。 */
	markLeaseLost(): void {
		this.patch({ leaseState: "LOST", contextLeaseId: undefined });
	}
}
