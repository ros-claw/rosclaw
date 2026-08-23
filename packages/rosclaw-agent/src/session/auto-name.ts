/** 会话自动命名（WP-7，0823 审计 §四.WP-7）。
 *
 * 0823 实测：会话被第一条消息 "hello" 命名——会话列表/退出提示
 * 展示的是闲聊而非任务。规则：首个**驱动工具活动**的输入用来
 * 命名（真实任务的确定性信号——工具执行开始）；只闲聊的会话
 * 不命名（会话列表回退展示首条消息，不谎报任务）。命名后冻结。
 */

export class AutoNamer {
	private pending: string | null = null;
	private named: string | null = null;

	/** 每条自然语言输入登记（命名前最后一条驱动活动的输入胜出）。 */
	noteInput(text: string): void {
		if (this.named !== null) return;
		const trimmed = text.trim();
		if (trimmed) this.pending = trimmed;
	}

	/** 工具执行开始 = 当前任务有真实工作（确定性信号，不是内容猜测）。 */
	noteToolActivity(): void {
		if (this.named !== null || this.pending === null) return;
		this.named = this.pending.slice(0, 30);
	}

	/** 当前应命名的标题（null = 还没到命名时机）。 */
	name(): string | null {
		return this.named;
	}
}
