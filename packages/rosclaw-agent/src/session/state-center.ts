/** ProductStateCenter（六审 PR-SIX-1）：Single Control Plane 的唯一
 * 产品状态快照。
 *
 * 此前的问题（六审 §2/§3 实证）：
 * - rosclaw_status 走旧 HTTP 8765（chat 的 agentd 用 port=0——必然
 *   误报 UNREACHABLE），/status 与 Header 走 UDS——两套状态通道；
 * - Header/Footer 各自读不同时间/不同缓存的字段（顶部 Kimi K3/
 *   OFFLINE 与底部未选模型/UNKNOWN 长期共存）；
 * - 显式 --mission 路径绕过 coordinator，leaseState 不写回——
 *   Action LOCKED 是假锁，工具侧根本没有准入检查。
 *
 * 本中心是 Header/Footer/status tool/context 的唯一状态源：
 * - 所有字段来自权威组件（UDS pi.status/pi.context、operatord 探测、
 *   ActiveSessionContext 显式状态、launcher 版本、当前 model）；
 * - 任何变化（patch/replace/lease lost/context stale/operator probe/
 *   model select/kernel 不可达）触发同一批 subscriber——extension 在
 *   一次 refreshChrome() 里用同一 snapshot 同时重绘 Header 与 Footer；
 * - UDS 失败原子降级：kernel=UNREACHABLE + context=STALE +
 *   action=BLOCKED(KERNEL_UNREACHABLE)，不允许局部报错而 Header
 *   保持 FRESH。
 */

import type { BridgeToolContext } from "../tools/bridge-tools.js";
import { bridgeCall } from "../bridge/bridge-client.js";
import { operatorCall } from "../bridge/operatord-client.js";
import type { ActiveSessionContext } from "./active-context.js";

export type KernelState = "READY" | "UNREACHABLE";
export type ContextState = "LOADING" | "FRESH" | "STALE" | "UNAVAILABLE";
export type LeaseState = "ACTIVE" | "LOST" | "NONE";
export type OperatorState = "READY" | "OFFLINE" | "UNKNOWN";
export type ReadinessState = "READY" | "BLOCKED" | "DEGRADED";

/** ActionReadinessV1（六审 §3.3）：工具侧硬门 + UI 首要受阻原因。 */
export interface ActionReadinessV1 {
	state: ReadinessState;
	stage?: "PROPOSE" | "APPROVE" | "EXECUTE";
	reason_codes: string[];
	snapshot_seq: number;
}

/** KernelSnapshotV1（ProductSnapshotV2）：一次不可变读取——Header、
 *  Footer、/status、rosclaw_status 全部渲染自同一个对象。 */
export interface KernelSnapshotV1 {
	snapshot_seq: number;
	product_version: string;
	kernel: KernelState;
	model: string;
	mode: string;
	mission_id?: string;
	body_id?: string;
	/** 七审 PR-SEVEN-5：机器人友好名（kit display_name）——UI 默认
	 *  显示它而不是内部 body_id。 */
	body_display?: string;
	/** 七审 PR-SEVEN-5：Robot Kit 摘要（BROKEN 时 UI 给一键修复）。 */
	robot_kit?: RobotKitSummary;
	context_state: ContextState;
	context_revision: number;
	lease_state: LeaseState;
	operator: OperatorState;
	action_readiness: ActionReadinessV1;
}

export interface RobotKitSummary {
	state: string;
	reason?: string;
	remediation?: {
		kind: string;
		kit_id: string;
		command?: string;
		idempotent?: boolean;
		cancellable?: boolean;
		real_authorization?: boolean;
	} | null;
}

export interface StateCenterDeps {
	rosclawHome: string;
	active: ActiveSessionContext;
	operatorSocket: string;
	productVersion: string;
	/** 可注入的桥调用（测试用）；默认真实 UDS bridgeCall。 */
	call?: typeof bridgeCall;
	/** 可注入的 operatord 调用（测试用）。 */
	operatorCallFn?: typeof operatorCall;
}

type Listener = () => void;

export class ProductStateCenter {
	private seq = 0;
	private kernelState: KernelState = "READY";
	private operatorState: OperatorState = "UNKNOWN";
	private lastOperatorProbe = 0;
	private modelDisplay = "";
	private capabilityBlocker: string | null = null;
	private lastCapabilityProbe = 0;
	private lastRobotProbe = 0;
	/** 七审 §2.5：SIM 审批策略（auto=安全仿真自动执行——operator
	 *  离线不是 blocker；ask=每次人工确认）。 */
	private simPolicy: "auto" | "ask" = "auto";
	private bodyDisplay = "";
	private robotKit: RobotKitSummary | null = null;
	private readonly listeners = new Set<Listener>();
	private readonly callFn: typeof bridgeCall;
	private readonly operatorCallFn: typeof operatorCall;

	constructor(private readonly deps: StateCenterDeps) {
		this.callFn = deps.call ?? bridgeCall;
		this.operatorCallFn = deps.operatorCallFn ?? operatorCall;
		// ActiveSessionContext 任何 patch/replace（context stale、lease
		// lost、envelope 到达、切换事务）都统一 fan-out。
		this.deps.active.subscribe(() => this.changed());
	}

	/** 订阅统一刷新（extension 的 refreshChrome）。 */
	subscribe(listener: Listener): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	private lastProbeMissionId: string | undefined;

	private changed(): void {
		this.seq += 1;
		// PR-SEVEN-2（perf 修复）：mission 一绑定就在 setup 阶段做能力
		// 探测——首次 MCP discovery 的进程启动 CPU burst 落在绑定时刻，
		// 不能懒到 30s 定时器在 idle 测量窗口内首次触发（Perf Gate
		// idle CPU 0.22s/5s 超 0.15 上限的回归根因）。
		const missionId = this.deps.active.current.missionId;
		if (missionId && missionId !== this.lastProbeMissionId) {
			this.lastProbeMissionId = missionId;
			void this.refreshCapabilities(true);
		}
		for (const listener of this.listeners) listener();
	}

	/** 当前不可变快照（一次读取，所有 chrome 共享）。 */
	snapshot(): KernelSnapshotV1 {
		const state = this.deps.active.current;
		return {
			snapshot_seq: this.seq,
			product_version: this.deps.productVersion,
			kernel: this.kernelState,
			model: this.modelDisplay,
			mode: state.mode,
			mission_id: state.missionId,
			body_id: state.bodyId,
			body_display: this.bodyDisplay || undefined,
			robot_kit: this.robotKit ?? undefined,
			context_state: state.missionId ? state.contextState : "UNAVAILABLE",
			context_revision: state.contextRevision,
			lease_state: state.leaseState,
			operator: this.operatorState,
			action_readiness: this.computeReadiness(),
		};
	}

	/** ActionReadinessV1：UI 提前诚实拒绝；内核 admission 仍是最终权威。 */
	private computeReadiness(): ActionReadinessV1 {
		const state = this.deps.active.current;
		const codes: string[] = [];
		if (!state.missionId) codes.push("NO_MISSION");
		if (this.kernelState !== "READY") codes.push("KERNEL_UNREACHABLE");
		if (state.leaseState !== "ACTIVE") codes.push("NO_WRITER_LEASE");
		if (state.missionId && state.contextState !== "FRESH") codes.push("CONTEXT_STALE");
		if (state.missionId && !state.contextLeaseId) codes.push("NO_CONTEXT_LEASE");
		// 七审 §2.1：kit/能力 blocker 先于 operator——action count=0
		// 时只怪 Operator 是误导。
		if (state.missionId && this.capabilityBlocker) {
			codes.push(this.capabilityBlocker);
		}
		if (
			this.operatorState === "OFFLINE"
			&& !(this.simPolicy === "auto" && state.mode === "SIMULATION")
		) {
			codes.push("OPERATOR_OFFLINE");
		}
		return {
			state: codes.length === 0 ? "READY" : "BLOCKED",
			...(codes.length ? { stage: "PROPOSE" as const } : {}),
			reason_codes: codes,
			snapshot_seq: this.seq,
		};
	}

	/** 动作工具入口的实时 readiness：强制新鲜 operator/kernel 探测后再判。 */
	async actionReadiness(): Promise<ActionReadinessV1> {
		await this.probeOperator(true);
		return this.computeReadiness();
	}

	get isSimAutoPolicy(): boolean {
		return this.simPolicy === "auto";
	}

	noteModel(display: string): void {
		if (display && display !== this.modelDisplay) {
			this.modelDisplay = display;
			this.changed();
		}
	}

	/** UDS 桥调用包装：失败即原子降级（kernel UNREACHABLE + context
	 *  STALE——不允许局部报错而 Header 保持 FRESH）。 */
	async call(
		method: string,
		params: Record<string, unknown> = {},
	): Promise<Record<string, unknown>> {
		try {
			const result = await this.callFn(this.deps.rosclawHome, method, params);
			if (this.kernelState !== "READY") {
				this.kernelState = "READY";
				this.changed();
			}
			return result;
		} catch (err) {
			if (this.kernelState !== "UNREACHABLE") {
				this.kernelState = "UNREACHABLE";
				// 原子降级：context 一并 STALE（不再信任何缓存 freshness）。
				this.deps.active.markContextStale("kernel unreachable");
			}
			this.changed();
			throw err;
		}
	}

	/** operatord readiness 真实探测（30s 缓存；force 用于动作门前）。 */
	async probeOperator(force = false): Promise<OperatorState> {
		const now = Date.now();
		if (!force && now - this.lastOperatorProbe < 60_000) return this.operatorState;
		this.lastOperatorProbe = now;
		let next: OperatorState;
		try {
			const result = (await this.operatorCallFn(this.deps.operatorSocket, "approvals.list", {
				mission_id: this.deps.active.current.missionId ?? "",
			})) as { ok?: boolean };
			next = result.ok ? "READY" : "OFFLINE";
		} catch {
			next = "OFFLINE";
		}
		// 六审 PR-SIX-5（perf 红线）：只在状态变化时 fan-out——30s 周期
		// 探测不再引发无谓的 chrome 重绘/widget 刷新。
		if (next !== this.operatorState) {
			this.operatorState = next;
			this.changed();
		}
		return this.operatorState;
	}

	/** 七审 PR-SEVEN-2.3：能力面探测（60s 缓存）——action count=0
	 *  或 executor 缺失时 readiness 含 ROBOT_KIT_INCOMPLETE。 */
	async refreshCapabilities(force = false): Promise<void> {
		const missionId = this.deps.active.current.missionId;
		if (!missionId) return;
		const now = Date.now();
		if (!force && now - this.lastCapabilityProbe < 60_000) return;
		this.lastCapabilityProbe = now;
		try {
			const result = await this.call("pi.capabilities", { mission_id: missionId });
			if (!result.ok) return;
			const actions = (result.action_capabilities ?? []) as unknown[];
			const next = actions.length === 0 ? "ROBOT_KIT_INCOMPLETE" : null;
			if (next !== this.capabilityBlocker) {
				this.capabilityBlocker = next;
				this.changed();
			}
		} catch {
			// 桥失败已由 call() 原子降级——capability blocker 保持现状。
		}
	}

	/** 七审 PR-SEVEN-5：机器人信息探测（友好名 + kit 状态）——60s
	 *  缓存，变化才 fan-out。 */
	async refreshRobotInfo(force = false): Promise<void> {
		const now = Date.now();
		if (!force && now - this.lastRobotProbe < 60_000) return;
		this.lastRobotProbe = now;
		try {
			const status = await this.call("pi.status", {});
			if (!status.ok) return;
			// 七审 PR-SEVEN-7 Journey B：ask 策略必须在启动探测即生效
			// （此前 simPolicy 只在 /status 的 statusReport 里刷新——ask
			// 会话的 operator widget 永远不出现）。
			const policy = String(status.sim_policy ?? "");
			if ((policy === "auto" || policy === "ask") && policy !== this.simPolicy) {
				this.simPolicy = policy;
				this.changed();
			}
			const nextDisplay = String(status.body_display ?? "");
			const nextKit = (status.robot_kit as RobotKitSummary | undefined) ?? null;
			if (nextDisplay !== this.bodyDisplay ||
				JSON.stringify(nextKit) !== JSON.stringify(this.robotKit)) {
				this.bodyDisplay = nextDisplay;
				this.robotKit = nextKit;
				this.changed();
			}
		} catch {
			// 桥失败已由 call() 原子降级——机器人信息保持现状。
		}
	}

	/** /status 与 rosclaw_status 共享的新鲜报告：UDS pi.status + 快照。 */
	async statusReport(): Promise<{
		ok: boolean;
		snapshot: KernelSnapshotV1;
		agentd?: string;
		authorization_profile?: string;
		mission?: Record<string, unknown> | null;
		error?: string;
	}> {
		const status = await this.call("pi.status", {
			...(this.deps.active.current.missionId
				? { mission_id: this.deps.active.current.missionId }
				: {}),
		});
		// 七审 §2.5：SIM 审批策略随 status 刷新。
		const policy = String(status.sim_policy ?? "");
		if ((policy === "auto" || policy === "ask") && policy !== this.simPolicy) {
			this.simPolicy = policy;
			this.changed();
		}
		const nextDisplay = String(status.body_display ?? "");
		const nextKit = (status.robot_kit as RobotKitSummary | undefined) ?? null;
		if (nextDisplay !== this.bodyDisplay ||
			JSON.stringify(nextKit) !== JSON.stringify(this.robotKit)) {
			this.bodyDisplay = nextDisplay;
			this.robotKit = nextKit;
			this.changed();
		}
		return {
			ok: status.ok === true,
			snapshot: this.snapshot(),
			agentd: String(status.agentd ?? ""),
			authorization_profile: String(status.authorization_profile ?? ""),
			mission: (status.mission as Record<string, unknown> | null) ?? null,
			...(status.ok ? {} : { error: String(status.error ?? "unknown") }),
		};
	}
}

/** BridgeToolContext + center（六审 PR-SIX-1：工具共享状态源）。 */
export interface CenteredToolContext extends BridgeToolContext {
	center: ProductStateCenter;
}
