/** ProductUiStateV1（三审 P0-NA-16）：header/footer 只读权威快照。
 *
 * 红线：
 * - 每个字段只能来自权威组件——operatord 健康来自真实 socket 探测，
 *   body/context 来自验证过的 envelope，版本来自 Python launcher
 *   显式传入的产品版本（禁止内部 npm 子包版本）；
 * - 未完成的 bootstrap 显示 LOADING/UNKNOWN——绝不乐观默认
 *   （此前 `Operator ready` 是硬编码字符串，`Body —` 创建后从不刷新）；
 * - 模型不能决定或改写 UI 安全状态（本对象不接受模型事件输入）。
 */

import { operatorCall } from "../bridge/operatord-client.js";
import type { ActiveSessionContext } from "../session/active-context.js";

export type ContextState = "LOADING" | "FRESH" | "STALE" | "UNAVAILABLE";
export type OperatorState = "READY" | "OFFLINE" | "UNKNOWN";

export interface ProductUiStateV1 {
	productVersion: string;
	missionId?: string;
	mode: string;
	bodyId?: string;
	contextState: ContextState;
	contextRevision: number;
	operatorState: OperatorState;
	/** 状态快照序列号——header/status/context 同源断言用。 */
	snapshotSeq: number;
}

export class ProductUiState {
	private seq = 0;
	private operatorState: OperatorState = "UNKNOWN";
	private lastOperatorProbe = 0;

	constructor(
		private readonly active: ActiveSessionContext,
		private readonly operatorSocket: string,
		private readonly productVersion: string,
	) {}

	/** 当前只读快照（字段全部来自权威源）。
	 *  HOTFIX-3：contextState 用 ActiveSessionContext 的显式状态——
	 *  合法 revision 0 的 FRESH 不再误显 LOADING，过期 revision 12
	 *  不再误显 FRESH（此前是"revision>0 猜 FRESH"的伪 freshness）。 */
	snapshot(): ProductUiStateV1 {
		const state = this.active.current;
		return {
			productVersion: this.productVersion,
			missionId: state.missionId,
			mode: state.mode,
			bodyId: state.bodyId,
			contextState: state.missionId ? state.contextState : "UNAVAILABLE",
			contextRevision: state.contextRevision,
			operatorState: this.operatorState,
			snapshotSeq: this.seq,
		};
	}

	/** operatord readiness 真实探测（30s 缓存——header 是热路径）。
	 * READY 仅表示授权服务可达，不表示任何动作已获批。 */
	async probeOperator(): Promise<OperatorState> {
		const now = Date.now();
		if (now - this.lastOperatorProbe < 30_000) return this.operatorState;
		this.lastOperatorProbe = now;
		try {
			const result = (await operatorCall(this.operatorSocket, "approvals.list", {
				mission_id: this.active.current.missionId ?? "",
			})) as { ok?: boolean };
			this.operatorState = result.ok ? "READY" : "OFFLINE";
		} catch {
			this.operatorState = "OFFLINE";
		}
		this.seq += 1;
		return this.operatorState;
	}

	/** context/envelope 变化时由扩展调用（刷新 header 的触发点）。 */
	noteContextChanged(): void {
		this.seq += 1;
	}
}

/** 推荐头部（P0-NA-16 规格）：
 *   ROSClaw 1.2.0 · SIMULATION · Kimi K3
 *   Mission mis_... · Body sim/ur5e · Context FRESH r12 · Operator OFFLINE
 */
export function renderHeader(state: ProductUiStateV1, modelName: string): string {
	const line1Parts = [`ROSClaw ${state.productVersion}`, state.mode];
	if (modelName) line1Parts.push(modelName);
	const line1 = line1Parts.join(" · ");
	if (!state.missionId) {
		return `${line1}\n未绑定 Mission · /help 查看命令`;
	}
	const body = state.bodyId ?? "LOADING";
	const context =
		state.contextState === "FRESH"
			? `Context FRESH r${state.contextRevision}`
			: `Context ${state.contextState}`;
	const line2 =
		`Mission ${state.missionId.slice(0, 24)} · Body ${body} · ` +
		`${context} · Operator ${state.operatorState}`;
	return `${line1}\n${line2}`;
}
