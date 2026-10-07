/** ProviderStallWatchdog（0902 审计 R1-b，§7）——Provider 停滞的
 *  分阶段看门狗。
 *
 * 0902 实证：Provider 是根因不等于 ROSClaw 没责任——静默 300 秒
 * 是产品缺陷（超时提示/取消/恢复属于 ROSClaw 产品责任）。
 *
 * 分阶段（§7）：
 * - 首 token 迟滞：10s 提示，30s 取消（pi abort——取消传播到模型
 *   请求）；
 * - 流式 idle：15s 状态更新，45s 恢复；
 * - 有字节流动的长任务不杀——每次内容流动都重置 idle 计时（不
 *   违反"不杀 turn"红线：取消的是无声停滞，不是活跃生成）；
 * - 回合终态即解除（无 stray 触发）。
 */

export interface ProviderStallWatchdogOptions {
	/** 阶段提示（用户可见）。 */
	notice: (text: string) => void;
	/** 停滞取消（调用方接 pi ctx.abort——传播到模型请求）。 */
	stallAbort: () => void;
	firstTokenNoticeMs?: number;
	firstTokenAbortMs?: number;
	streamIdleStatusMs?: number;
	streamIdleAbortMs?: number;
}

const DEFAULTS = {
	firstTokenNoticeMs: 10_000,
	firstTokenAbortMs: 30_000,
	streamIdleStatusMs: 15_000,
	streamIdleAbortMs: 45_000,
};

/** Optional longer waits for slow providers. Values are milliseconds,
 * bounded to 1s..1h; invalid values retain the existing defaults. */
export function providerWatchdogTimingFromEnv(
	env: Record<string, string | undefined> = process.env,
): Pick<Required<ProviderStallWatchdogOptions>,
	"firstTokenNoticeMs" | "firstTokenAbortMs" | "streamIdleStatusMs" | "streamIdleAbortMs"> {
	const read = (name: string, fallback: number): number => {
		const raw = env[name];
		if (!raw || !/^\d+$/.test(raw)) return fallback;
		const value = Number(raw);
		return Number.isSafeInteger(value) && value >= 1_000 && value <= 3_600_000
			? value : fallback;
	};
	const firstTokenAbortMs = read("ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS", DEFAULTS.firstTokenAbortMs);
	const streamIdleAbortMs = read("ROSCLAW_PROVIDER_STREAM_IDLE_TIMEOUT_MS", DEFAULTS.streamIdleAbortMs);
	return {
		firstTokenNoticeMs: Math.min(DEFAULTS.firstTokenNoticeMs, firstTokenAbortMs / 3),
		firstTokenAbortMs,
		streamIdleStatusMs: Math.min(DEFAULTS.streamIdleStatusMs, streamIdleAbortMs / 3),
		streamIdleAbortMs,
	};
}

function seconds(ms: number): string {
	return `${Number((ms / 1_000).toFixed(3))}s`;
}

/** PI1.0.4 assistantMessageEvent 分类——只有"有意义的增量"才算
 *  Provider 真实进展（PI104 实证反例：重复的 text_start 边界事件与
 *  空 text_delta 会持续重置看门狗，首 token/流式 idle 取消永不触发）。
 *  - text_delta / thinking_delta / toolcall_delta 且 delta 非空 → 真实进展；
 *  - text_start / thinking_start / toolcall_start 等边界事件与空 delta
 *    → 不是内容，不续期；
 *  - 未知类型/无结构事件 → 保持旧兼容（视为流动）。
 *  注意：thinking 增量只作为活性信号，其文本绝不上屏（不外泄思考内容）。 */
export function isMeaningfulAssistantEvent(assistantEvent: unknown): boolean {
	if (assistantEvent == null || typeof assistantEvent !== "object") return true; // 旧 pi 兼容
	const ev = assistantEvent as { type?: unknown; delta?: unknown };
	const type = typeof ev.type === "string" ? ev.type : "";
	if (type === "text_delta" || type === "thinking_delta" || type === "toolcall_delta") {
		return typeof ev.delta === "string" && ev.delta.length > 0;
	}
	if (
		type === "text_start" || type === "text_end"
		|| type === "thinking_start" || type === "thinking_end"
		|| type === "toolcall_start" || type === "toolcall_end"
	) {
		return false;
	}
	return true; // 未知类型保持旧语义
}

/** 活动区诚实的 Provider 等待阶段（不含任何思考/内容文本）。 */
export type ProviderWaitPhase = "waiting" | "streaming" | "tool" | "user_decision" | "idle";

export class ProviderStallWatchdog {
	private readonly opts: Required<ProviderStallWatchdogOptions>;
	private firstTokenTimers: ReturnType<typeof setTimeout>[] = [];
	private streamIdleTimers: ReturnType<typeof setTimeout>[] = [];
	private active = false;
	private sawContent = false;
	private abortedOnce = false;
	private userBusy = false;
	/** 0914 PR-3：工具执行阶段计数（嵌套/连发工具安全）——
	 *  >0 时 Provider 时钟暂停：工具运行不是 Provider 停滞。 */
	private toolBusyCount = 0;
	private lastStreamIdleNoticeAt = -Infinity;
	/** 诚实 Provider 等待耗时：暂停（工具/用户确认）期不计入等待，
	 *  也不把工具时间误标为 Provider 停滞。 */
	private waitAccumMs = 0;
	private waitResumedAt: number | null = null;

	constructor(options: ProviderStallWatchdogOptions) {
		this.opts = { ...DEFAULTS, ...options } as Required<ProviderStallWatchdogOptions>;
	}

	/** 回合开始（turn_start）——进入"等首 token"阶段。 */
	turnStarted(): void {
		this._disarm();
		this.userBusy = false;
		this.toolBusyCount = 0;
		this.active = true;
		this.sawContent = false;
		this.abortedOnce = false;
		this.lastStreamIdleNoticeAt = -Infinity;
		this.waitAccumMs = 0;
		this.waitResumedAt = performance.now();
		this._armFirstToken();
	}

	private _armFirstToken(): void {
		if (this.userBusy || this.toolBusyCount > 0) return;
		this.firstTokenTimers.push(
			setTimeout(() => {
				if (!this.active || this.sawContent || this.userBusy || this.toolBusyCount > 0) return;
				try {
					this.opts.notice(
						`模型响应迟滞（${seconds(this.opts.firstTokenNoticeMs)} 无首个内容）——可能是 Provider 排队或网络慢；`
						+ `${seconds(this.opts.firstTokenAbortMs)} 仍无响应将自动取消本次请求（可重发）`,
					);
				} catch {
					// 通知失败不崩宿主（M8）。
				}
			}, this.opts.firstTokenNoticeMs),
			setTimeout(() => this._stall(`首 token ${seconds(this.opts.firstTokenAbortMs)} 无响应`), this.opts.firstTokenAbortMs),
		);
	}

	/** 内容流动——首个内容结束首 token 阶段；之后每次流动都重置
	 *  流式 idle 计时（流动即续命——长任务不杀）。 */
	contentProgress(): void {
		if (!this.active || this.abortedOnce || this.userBusy || this.toolBusyCount > 0) return;
		if (!this.sawContent) {
			this.sawContent = true;
			for (const t of this.firstTokenTimers) clearTimeout(t);
			this.firstTokenTimers = [];
		}
		this._resetStreamIdle();
	}

	/** 回合终态/空闲——全部解除。 */
	turnEnded(): void {
		this._disarm();
	}

	/** PI104 message_update 入口：只有有意义的增量才续期（空边界/
	 *  空 delta 不推迟首 token 与流式 idle 取消）。返回是否计为进展。 */
	assistantEventProgress(assistantEvent: unknown): boolean {
		if (!isMeaningfulAssistantEvent(assistantEvent)) return false;
		this.contentProgress();
		return true;
	}

	/** 当前 Provider 等待阶段（活动区诚实文案用；不含内容文本）。 */
	currentPhase(): ProviderWaitPhase {
		if (this.userBusy) return "user_decision";
		if (this.toolBusyCount > 0) return "tool";
		if (!this.active) return "idle";
		return this.sawContent ? "streaming" : "waiting";
	}

	/** 本回合真实等待 Provider 的累计毫秒（工具/用户确认暂停期不计）。 */
	providerWaitElapsedMs(): number {
		const running = this.waitResumedAt != null ? performance.now() - this.waitResumedAt : 0;
		return this.waitAccumMs + running;
	}

	private _pauseWait(): void {
		if (this.waitResumedAt != null) {
			this.waitAccumMs += performance.now() - this.waitResumedAt;
			this.waitResumedAt = null;
		}
	}

	private _resumeWait(): void {
		// 重叠暂停修复：任一类暂停（用户确认/工具执行，含嵌套工具）
		// 仍活跃时不得恢复 Provider 等待计时——只有全部暂停清空才恢复。
		if (this.active && !this.userBusy && this.toolBusyCount === 0 && this.waitResumedAt == null) {
			this.waitResumedAt = performance.now();
		}
	}

	/** 模态对话框打开（确认卡等用户决定中）= 用户在场——暂停计时
	 *  （等用户回答不是 Provider 停滞；journey 实证：委派腿的确认卡
	 *  等待被 45s idle 误判取消）。 */
	pauseForUser(): void {
		this.userBusy = true;
		this._disarmTimers();
		this._pauseWait();
	}

	/** 对话框关闭——恢复计时（从当前状态重新武装）。 */
	resumeFromUser(): void {
		this.userBusy = false;
		this._resumeWait();
		if (this.active && !this.abortedOnce && this.toolBusyCount === 0) {
			if (this.sawContent) this._resetStreamIdle();
			else this._armFirstToken();
		}
	}

	/** 0914 PR-3（审计 §5）：tool_execution_start——进入工具执行
	 *  阶段。Provider 时钟暂停：工具运行期（90s 渲染/长 bash/sleep）
	 *  没有任何 pi 事件，但那不是 Provider 停滞（0914 实证：后台
	 *  Operation 已 SUCCEEDED，界面却流式 idle 45s 取消模型请求）。
	 *  工具自有进度与可取消机制监督，不共用聊天时钟。 */
	pauseForTool(): void {
		this.toolBusyCount += 1;
		this._disarmTimers();
		this._pauseWait();
	}

	/** tool_execution_end——离开工具阶段；计数归零才恢复 Provider
	 *  时钟（嵌套/连发工具不提前武装）。 */
	resumeFromTool(): void {
		if (this.toolBusyCount > 0) this.toolBusyCount -= 1;
		if (this.toolBusyCount > 0) return;
		this._resumeWait();
		if (this.active && !this.abortedOnce && !this.userBusy) {
			if (this.sawContent) this._resetStreamIdle();
			else this._armFirstToken();
		}
	}


	private _resetStreamIdle(): void {
		for (const t of this.streamIdleTimers) clearTimeout(t);
		this.streamIdleTimers = [
			setTimeout(() => {
				if (!this.active || this.abortedOnce) return;
				// Slow but live chunks can repeatedly cross the status threshold.
				// Bound duplicate warnings; the independent abort timer stays armed.
				const now = performance.now();
				if (now - this.lastStreamIdleNoticeAt < 60_000) return;
				this.lastStreamIdleNoticeAt = now;
				try {
					this.opts.notice(`模型暂未输出新内容（${seconds(this.opts.streamIdleStatusMs)}）——仍在等待 Provider，尚未取消…`);
				} catch {
					// M8。
				}
			}, this.opts.streamIdleStatusMs),
			setTimeout(() => this._stall(`流式 idle ${seconds(this.opts.streamIdleAbortMs)}`), this.opts.streamIdleAbortMs),
		];
	}

	private _stall(reason: string): void {
		if (!this.active || this.abortedOnce || this.userBusy || this.toolBusyCount > 0) return;
		this.abortedOnce = true;
		// 0902 复核 M8：回调异常不得崩扩展宿主（setTimeout 回调里
		// 裸调 = uncaught exception）。
		try {
			this.opts.notice(`Provider 无响应（${reason}）——请求取消本次模型请求；取消是否完成以实际回合状态为准`);
		} catch {
			// 通知失败不阻断取消。
		}
		try {
			this.opts.stallAbort();
		} catch {
			// abort 失败同理。
		}
		this._disarm();
	}

	private _disarmTimers(): void {
		for (const t of this.firstTokenTimers) clearTimeout(t);
		for (const t of this.streamIdleTimers) clearTimeout(t);
		this.firstTokenTimers = [];
		this.streamIdleTimers = [];
	}

	private _disarm(): void {
		this._disarmTimers();
		// 终态冻结修复：先把仍在累计的等待段并入 waitAccumMs 再停表——
		// 终态后 providerWaitElapsedMs 保留实测耗时（不再继续累加空闲
		// 时间，也不回退为 0）；新回合由 turnStarted 自行重置。
		this._pauseWait();
		this.active = false;
	}
}
