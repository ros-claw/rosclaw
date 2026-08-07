/** Session 生命周期映射（PNA-6 + 三审 P0-NA-12，规格 §13）。
 *
 * 所有绑定变更都经 AgentSessionCoordinator 的单一事务：
 * - session_start(reason=new|fork)：coordinator.beginNew（新 SIM
 *   Mission + bind + heartbeat + fresh context + 原子状态替换）；
 * - session_start(reason=resume)：coordinator.resumeInitial（既有绑定
 *   重接；丢失/已归档 → 新建 SIM 绑定并明确告知）；
 * - session_before_switch：只读预检（target 文件/头可读）——绑定动作
 *   移到 session_start（此时 target 已是活动 session，id 无歧义；
 *   此前在 before_switch 用当前 sessionId 查绑定是错的：event 里的
 *   target 才是目标）；
 * - session_before_tree：有进行中动作/待决授权 → veto（fail closed，
 *   规格 §13.5：tree 不得回滚物理 lane）。
 */

import type { AgentSessionCoordinator } from "./coordinator.js";

export interface LifecycleDeps {
	coordinator: AgentSessionCoordinator;
	/** 切换前记录来源 mission（fork 只读历史用）。 */
	coordinatorMissionId: () => string | undefined;
	notify: (message: string, type?: "info" | "warning" | "error") => void;
}

export function sessionIdOf(ctx: {
	sessionManager?: { getSessionId?: () => string };
}): string {
	return ctx.sessionManager?.getSessionId?.() ?? "";
}

export async function handleSessionStart(
	deps: LifecycleDeps,
	reason: string,
	sessionId: string,
): Promise<void> {
	if (reason === "new" || reason === "fork") {
		const source = deps.coordinatorMissionId();
		const outcome = await deps.coordinator.beginNew(
			sessionId,
			reason as "new" | "fork",
			source,
		);
		if (!outcome.ok) {
			deps.notify(outcome.reason, "error");
		}
		return;
	}
	if (reason === "resume") {
		const outcome = await deps.coordinator.resumeInitial(sessionId);
		if (!outcome.ok) {
			deps.notify(outcome.reason, "error");
		}
		return;
	}
	// startup/reload：初始绑定由 main.ts 显式完成（--mission / --resume /
	// --continue），此处不重复。
}

export async function shouldCancelSwitch(
	targetSessionFile: string | undefined,
	readHeaderId: (file: string) => string | null,
): Promise<string | null> {
	// 只读预检：target 必须存在且头可解析——绑定动作在 session_start。
	if (!targetSessionFile) return null; // new：无 target 文件，放行
	const headerId = readHeaderId(targetSessionFile);
	if (headerId === null) {
		return "目标 session 文件缺失或头损坏——拒绝切换（fail closed）";
	}
	return null;
}

export async function shouldCancelTree(deps: {
	rosclawHome: string;
	missionId?: string;
	call?: (
		home: string,
		method: string,
		params?: Record<string, unknown>,
	) => Promise<Record<string, unknown>>;
}): Promise<string | null> {
	// tree navigation 前置：进行中动作/待决授权 → veto（§13.5 fail closed）。
	if (!deps.missionId) return null;
	const { bridgeCall } = await import("../bridge/bridge-client.js");
	const call = deps.call ?? bridgeCall;
	try {
		const ctxResponse = await call(deps.rosclawHome, "pi.context", {
			mission_id: deps.missionId,
		});
		if (!ctxResponse.ok) {
			// HOTFIX-3（P0-4E）：context 不可达时 tree 不得放行——
			// 无法证明没有进行中动作，fail closed。
			return "具身上下文不可达——无法验证 tree 安全性，已阻止（fail closed）";
		}
		const context = ctxResponse.context as {
			pending_approvals?: unknown[];
			active_actions?: unknown[];
		};
		if ((context.pending_approvals ?? []).length > 0) {
			return "有待决授权——先完成 reconciliation 再切换认知路径";
		}
		if ((context.active_actions ?? []).length > 0) {
			return "有进行中的真实动作——tree navigation fail closed（§13.5）";
		}
	} catch {
		// 查询异常同样不能证明安全——fail closed。
		return "具身上下文查询异常——无法验证 tree 安全性，已阻止（fail closed）";
	}
	return null;
}
