/** 三层信息密度（P0-H，0824 总纲 §16.1/§19.P0-H）。
 *
 * Conversation：用户必须看到的（任务结果、错误、操作确认）——
 * 永远可见；Activity：任务活动/进度（POLICY_AUTO 自动放行等
 * 治理事件的用户可读形态）——默认可见；Debug：治理/审计机制
 * 细节（approval/grant id、lease、内部状态码）——默认隐藏，
 * 用户切换后可见（内部状态不甩给用户）。
 */

export type NoticeLevel = "conversation" | "activity" | "debug";

const GOVERNANCE_PATTERN =
	/\b(approval|grant|lease|permit|capability_snapshot|context_revision)\b|apr_[a-z0-9]+|grt_[a-z0-9]+/i;

/** 通知文本 → 信息层（治理/审计机制细节 = debug）。 */
export function classifyNotice(text: string): NoticeLevel {
	if (GOVERNANCE_PATTERN.test(text)) return "debug";
	if (/^(任务完成|已批准|已拒绝|绑定失败|操作失败|不能接受|查询失败)/.test(text)) {
		return "conversation";
	}
	return "activity";
}

/** Debug 层开关（默认隐藏；用户切换后可见）。 */
export class NotificationLevelFilter {
	private debugVisible = false;

	visible(level: NoticeLevel): boolean {
		if (level === "debug") return this.debugVisible;
		return true;
	}

	toggle(): void {
		this.debugVisible = !this.debugVisible;
	}
}
