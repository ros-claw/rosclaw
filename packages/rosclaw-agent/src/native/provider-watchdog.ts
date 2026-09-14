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
		this._armFirstToken();
	}

	private _armFirstToken(): void {
		if (this.userBusy || this.toolBusyCount > 0) return;
		this.firstTokenTimers.push(
			setTimeout(() => {
				if (!this.active || this.sawContent || this.userBusy || this.toolBusyCount > 0) return;
				try {
					this.opts.notice(
						"模型响应迟滞（10s 无首个内容）——可能是 Provider 排队或网络慢；"
						+ "30s 仍无响应将自动取消本次请求（可重发）",
					);
				} catch {
					// 通知失败不崩宿主（M8）。
				}
			}, this.opts.firstTokenNoticeMs),
			setTimeout(() => this._stall("首 token 30s 无响应"), this.opts.firstTokenAbortMs),
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

	/** 模态对话框打开（确认卡等用户决定中）= 用户在场——暂停计时
	 *  （等用户回答不是 Provider 停滞；journey 实证：委派腿的确认卡
	 *  等待被 45s idle 误判取消）。 */
	pauseForUser(): void {
		this.userBusy = true;
		this._disarmTimers();
	}

	/** 对话框关闭——恢复计时（从当前状态重新武装）。 */
	resumeFromUser(): void {
		this.userBusy = false;
		if (this.active && !this.abortedOnce) {
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
	}

	/** tool_execution_end——离开工具阶段；计数归零才恢复 Provider
	 *  时钟（嵌套/连发工具不提前武装）。 */
	resumeFromTool(): void {
		if (this.toolBusyCount > 0) this.toolBusyCount -= 1;
		if (this.toolBusyCount > 0) return;
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
				try {
					this.opts.notice("模型生成中断流（15s 无新内容）——仍在等待 Provider…");
				} catch {
					// M8。
				}
			}, this.opts.streamIdleStatusMs),
			setTimeout(() => this._stall("流式 idle 45s"), this.opts.streamIdleAbortMs),
		];
	}

	private _stall(reason: string): void {
		if (!this.active || this.abortedOnce || this.userBusy || this.toolBusyCount > 0) return;
		this.abortedOnce = true;
		// 0902 复核 M8：回调异常不得崩扩展宿主（setTimeout 回调里
		// 裸调 = uncaught exception）。
		try {
			this.opts.notice(`Provider 无响应（${reason}）——已取消本次请求，可重发`);
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
		this.active = false;
	}
}
