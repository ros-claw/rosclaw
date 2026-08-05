/** Session 生命周期映射（PNA-6，规格 §13）。
 *
 * - session_start(reason=new|fork)：创建新 SIM Mission 并绑定（fork 记录
 *   来源 mission，authority 永不复制——authority 只存 agentd，Pi session
 *   里没有可复制的授权材料，结构性保证）；
 * - session_before_switch(resume)：目标 session 无绑定 → 自动新建
 *   SIM Mission 绑定（NEEDS_BINDING 的安全默认）；绑定 mission 已归档
 *   → 同样新建绑定；旧 binding 保持只读历史；
 * - session_before_tree：有进行中动作/待决授权 → veto（fail closed，
 *   规格 §13.5：tree 不得回滚物理 lane）。
 */

import { bridgeCall } from "../bridge/bridge-client.js";

export type BridgeCallFn = (
	home: string,
	method: string,
	params?: Record<string, unknown>,
) => Promise<Record<string, unknown>>;

export interface LifecycleDeps {
	rosclawHome: string;
	/** 启动时绑定的 mission（--mission）；hook 内新绑定会更新它。 */
	getMissionId: () => string | undefined;
	setMissionId: (missionId: string) => void;
	notify: (message: string, type?: "info" | "warning" | "error") => void;
	call?: BridgeCallFn;
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
	const call = deps.call ?? bridgeCall;
	if (reason !== "new" && reason !== "fork") return;
	const sourceMissionId = deps.getMissionId();
	const created = await call(deps.rosclawHome, "pi.mission.create", {
		goal: reason === "fork" ? `fork of ${sourceMissionId ?? "session"}` : "pi session",
		mode: "SIMULATION",
	});
	if (!created.ok) {
		deps.notify(`新建 Mission 失败：${String(created.error ?? "")}`, "error");
		return;
	}
	const missionId = String(created.mission_id);
	const bound = await call(deps.rosclawHome, "pi.session.bind", {
		pi_session_id: sessionId,
		mission_id: missionId,
	});
	if (!bound.ok) {
		deps.notify(
			`绑定失败 [${String(bound.code ?? "")}]：${String(bound.error ?? "")}`,
			"error",
		);
		return;
	}
	deps.setMissionId(missionId);
	deps.notify(
		reason === "fork"
			? `已 fork：新 SIM Mission ${missionId}（authority 不复制，来源 ${sourceMissionId ?? "—"} 仅作只读历史）`
			: `新 Mission ${missionId}（SIMULATION）`,
		"info",
	);
}

export async function shouldCancelSwitch(
	deps: LifecycleDeps,
	targetSessionId: string,
): Promise<string | null> {
	// resume/switch 前置检查。返回 veto 原因；null=放行（可能已完成新绑定）。
	const call = deps.call ?? bridgeCall;
	const looked = await call(deps.rosclawHome, "pi.session.binding.get", {
		pi_session_id: targetSessionId,
	});
	if (!looked.ok) return `绑定查询失败：${String(looked.error ?? "")}`;
	const binding = looked.binding as { mission_id?: string } | null;
	if (binding && !looked.mission_archived) {
		deps.setMissionId(String(binding.mission_id));
		return null;
	}
	// 绑定丢失或 mission 已归档：不猜——新建 SIM Mission 绑定（NEEDS_BINDING
	// 的安全默认；规格 §13.3）。
	const created = await call(deps.rosclawHome, "pi.mission.create", {
		goal: `resume rebind (${targetSessionId.slice(0, 8)})`,
		mode: "SIMULATION",
	});
	if (!created.ok) return `绑定丢失且新建 Mission 失败：${String(created.error ?? "")}`;
	const bound = await call(deps.rosclawHome, "pi.session.bind", {
		pi_session_id: targetSessionId,
		mission_id: String(created.mission_id),
	});
	if (!bound.ok) return `rebind 失败：${String(bound.error ?? "")}`;
	deps.setMissionId(String(created.mission_id));
	deps.notify(
		binding
			? "原 Mission 已归档——已新建 SIM Mission 绑定（旧记录只读保留）"
			: "该 session 无 Mission 绑定——已新建 SIM Mission 绑定",
		"warning",
	);
	return null;
}

export async function shouldCancelTree(deps: LifecycleDeps): Promise<string | null> {
	// tree navigation 前置：进行中动作/待决授权 → veto（§13.5 fail closed）。
	const missionId = deps.getMissionId();
	if (!missionId) return null;
	const call = deps.call ?? bridgeCall;
	try {
		const ctxResponse = await call(deps.rosclawHome, "pi.context", {
			mission_id: missionId,
		});
		if (!ctxResponse.ok) return null; // context 拉不到由注入层标 stale；tree 不背锅
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
		return null;
	}
	return null;
}
