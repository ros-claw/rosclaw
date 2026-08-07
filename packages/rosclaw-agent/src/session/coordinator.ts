/** AgentSessionCoordinator（三审 P0-NA-12）：session/mission/lease 的
 * 唯一事务协调器。
 *
 * 此前的问题：
 * - lifecycle 直接 bridgeCall("pi.session.bind")，丢弃 lease_token、
 *   不起 heartbeat；
 * - ActiveSessionContext 只 patch missionId——session/binding/lease/
 *   revision/body 可能分裂；
 * - session_before_switch 用当前 sessionId 查绑定，忽略 event.target；
 * - --resume/--continue 启动不做初始绑定。
 *
 * 本协调器把一次切换做成一个事务：
 *
 *   解析 target session（明确 id，不猜）
 *   → 查询 target binding 与 Mission 状态
 *   → 释放旧 lease（停旧 heartbeat）
 *   → bind target（保存新 lease token）
 *   → 启动唯一 heartbeat（leaseManager 内）
 *   → 拉取并验证 fresh embodied context
 *   → 原子 replace ActiveSessionContext
 *
 * 任一步失败 → NEEDS_BINDING：missionId 清空、动作禁行，绝不半切换、
 * 绝不继续用旧 revision。
 */

import { bridgeCall } from "../bridge/bridge-client.js";
import {
	fetchEmbodiedContext,
	type EmbodiedContextEnvelope,
} from "../extension/context-injection.js";
import type { ActiveSessionContext, ActiveSessionState } from "./active-context.js";
import type { SessionLeaseManager } from "./lease-manager.js";

export interface CoordinatorDeps {
	rosclawHome: string;
	active: ActiveSessionContext;
	leaseManager: SessionLeaseManager;
	notify: (message: string, type?: "info" | "warning" | "error") => void;
	call?: typeof bridgeCall;
}

export type SwitchOutcome =
	| { ok: true; missionId: string; rebound: boolean }
	| { ok: false; reason: string };

export class AgentSessionCoordinator {
	private readonly call: typeof bridgeCall;

	constructor(private readonly deps: CoordinatorDeps) {
		this.call = deps.call ?? bridgeCall;
	}

	/** UI 上下文在 hook 里才可用——事件触发时注入真实 notify。 */
	setNotify(notify: CoordinatorDeps["notify"]): void {
		this.deps.notify = notify;
	}

	/** 切换事务失败时的安全落点——不半切换。 */
	private enterNeedsBinding(reason: string): SwitchOutcome {
		const current = this.deps.active.current;
		this.deps.active.replace({
			...current,
			missionId: undefined,
			bindingId: undefined,
			leaseToken: undefined,
		});
		return { ok: false, reason };
	}

	/** 绑定查询：target session 当前绑定的 mission（若有）。 */
	private async lookupBinding(
		targetSessionId: string,
	): Promise<{ missionId: string; archived: boolean } | null> {
		const looked = await this.call(this.deps.rosclawHome, "pi.session.binding.get", {
			pi_session_id: targetSessionId,
		});
		if (!looked.ok) return null;
		const binding = looked.binding as { mission_id?: string } | null;
		if (!binding) return null;
		return {
			missionId: String(binding.mission_id ?? ""),
			archived: Boolean(looked.mission_archived),
		};
	}

	/** fresh context 拉取 + 验证 + lease（失败即 stale——不落盘 revision）。 */
	private async freshContext(
		missionId: string,
		sessionId: string,
	): Promise<{ envelope: EmbodiedContextEnvelope; leaseId?: string } | null> {
		const fetched = await fetchEmbodiedContext(
			this.deps.rosclawHome,
			missionId,
			sessionId,
		);
		if (fetched.stale || !fetched.envelope) return null;
		return { envelope: fetched.envelope, leaseId: fetched.contextLeaseId };
	}

	/** 原子落盘（P0-4E）：构造完整候选状态后一次性替换。
	 *  context 失败 = 绑定成功但 NOT_READY——清空 revision/body/lease/
	 *  动作准入，绝不继承旧 Mission 的任何数据（此前 `?? current.x`
	 *  会把 A 的 revision/body 带进 B）。 */
	private commitState(
		targetSessionId: string,
		missionId: string,
		fresh: { envelope: EmbodiedContextEnvelope; leaseId?: string } | null,
	): void {
		const current = this.deps.active.current;
		const body = (fresh?.envelope.body ?? {}) as {
			body_id?: string;
			effective_body_hash?: string;
		};
		const safety = (fresh?.envelope.safety ?? {}) as { mode?: string };
		const next: ActiveSessionState = {
			...current,
			sessionId: targetSessionId,
			missionId,
			bindingId: this.deps.leaseManager.active?.bindingId,
			leaseToken: undefined, // token 只存 leaseManager，不进共享状态
			leaseState: "ACTIVE",
			// context：fresh 才有值；失败一律清空（NOT_READY，不继承旧值）。
			contextRevision: fresh?.envelope.context_revision ?? 0,
			bodyId: fresh ? body.body_id : undefined,
			bodyHash: fresh ? body.effective_body_hash : undefined,
			mode: fresh ? (safety.mode ?? current.mode) : current.mode,
			contextState: fresh ? "FRESH" : "UNAVAILABLE",
			contextLeaseId: fresh?.leaseId,
		};
		this.deps.active.replace(next);
	}

	/** 核心事务：绑定/重绑 target session 到其既有或新建 SIM Mission。 */
	async switchTo(
		targetSessionId: string,
		opts: { goal: string; forkOf?: string; createIfUnbound: boolean },
	): Promise<SwitchOutcome> {
		if (!targetSessionId) {
			return this.enterNeedsBinding("target session id 为空——拒绝切换");
		}
		// 1. 查询 target binding 与 mission 状态。
		const existing = await this.lookupBinding(targetSessionId);
		let missionId: string;
		let rebound = false;
		if (existing && !existing.archived) {
			missionId = existing.missionId;
			rebound = true;
		} else {
			if (!opts.createIfUnbound) {
				return this.enterNeedsBinding(
					"target session 无有效 Mission 绑定——拒绝猜测性切换",
				);
			}
			// 2. 无绑定/已归档 → 新建 SIM Mission（NEEDS_BINDING 安全默认；
			//    authority 永不复制——grant/permit 只在 agentd）。
			const created = await this.call(this.deps.rosclawHome, "pi.mission.create", {
				goal: opts.goal,
				mode: "SIMULATION",
			});
			if (!created.ok) {
				return this.enterNeedsBinding(
					`新建 Mission 失败：${String(created.error ?? "")}`,
				);
			}
			missionId = String(created.mission_id);
		}
		// 3. 释放旧 lease → bind target（leaseManager 原子停旧/起新
		//    heartbeat，保存 lease_token）。
		try {
			await this.deps.leaseManager.release();
			await this.deps.leaseManager.bind(targetSessionId, missionId);
		} catch (err) {
			return this.enterNeedsBinding(`绑定失败：${(err as Error).message}`);
		}
		// 4. fresh context（P0-4E：失败 = 绑定成功但 NOT_READY——
		//    清空 revision/body/动作准入，绝不继承旧 Mission 数据）。
		const fresh = await this.freshContext(missionId, targetSessionId);
		// 5. 原子替换（完整候选状态，一次性）。
		this.commitState(targetSessionId, missionId, fresh);
		if (!fresh) {
			this.deps.notify(
				"已绑定新 Mission，但具身上下文不可用——动作已禁止（context 恢复后自动解除）",
				"warning",
			);
		}
		if (existing?.archived) {
			this.deps.notify(
				"RESUME_REBOUND_TO_NEW_SIM：原 Mission 已归档——已新建 SIM Mission 绑定（旧记录只读保留，这不是原 Mission 的恢复）",
				"warning",
			);
		}
		return { ok: true, missionId, rebound };
	}

	/** /new 或 /fork 的新 session：新建 SIM Mission 并绑定（authority
	 * 结构性不复制——grant/permit 只在 agentd，Pi session 里没有可
	 * 复制的授权材料）。 */
	async beginNew(
		sessionId: string,
		reason: "new" | "fork",
		sourceMissionId?: string,
	): Promise<SwitchOutcome> {
		if (!sessionId) {
			return this.enterNeedsBinding("session id 为空——拒绝新建绑定");
		}
		const created = await this.call(this.deps.rosclawHome, "pi.mission.create", {
			goal:
				reason === "fork" ? `fork of ${sourceMissionId ?? "session"}` : "pi session",
			mode: "SIMULATION",
		});
		if (!created.ok) {
			return this.enterNeedsBinding(
				`新建 Mission 失败：${String(created.error ?? "")}`,
			);
		}
		const missionId = String(created.mission_id);
		try {
			await this.deps.leaseManager.release();
			await this.deps.leaseManager.bind(sessionId, missionId);
		} catch (err) {
			return this.enterNeedsBinding(`绑定失败：${(err as Error).message}`);
		}
		const fresh = await this.freshContext(missionId, sessionId);
		this.commitState(sessionId, missionId, fresh);
		if (!fresh) {
			this.deps.notify(
				"已绑定新 Mission，但具身上下文不可用——动作已禁止（context 恢复后自动解除）",
				"warning",
			);
		}
		this.deps.notify(
			reason === "fork"
				? `已 fork：新 SIM Mission ${missionId}（authority 不复制，来源 ${sourceMissionId ?? "—"} 仅作只读历史）`
				: `新 Mission ${missionId}（SIMULATION）`,
			"info",
		);
		return { ok: true, missionId, rebound: false };
	}

	/** 启动恢复（--resume/--continue）：把已打开的 session 重新接入
	 * binding + heartbeat + fresh context——此前启动只显示 header，
	 * 绑定从未恢复（P0-NA-12）。 */
	async resumeInitial(targetSessionId: string): Promise<SwitchOutcome> {
		return await this.switchTo(targetSessionId, {
			goal: `resume rebind (${targetSessionId.slice(0, 8)})`,
			createIfUnbound: true,
		});
	}
}
