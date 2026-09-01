// HP2-COMPAT: pi-tui 组件原语（Component）——TUI 渲染原语，HP3 前保持；不新增会话装配引用。
/** 内核消息卡（0901 体验审计 P0-6）：内部 customType 消息以卡片
 *  呈现——[rosclaw.user_directive]/[rosclaw.task_terminal] 这类
 *  协议标签绝不上屏（0901 实证：用户看到内部标签，不知道那是
 *  什么）。
 *
 * 三类卡：
 * - user_directive：用户指令回声（确定性链已接管——输入被
 *   suppress 不进模型回合，但必须可见、可审计）；
 * - task_terminal：任务终态卡（Coordinator 权威 outcome 的确定性
 *   呈现——唯一终态发布者）；
 * - task_explain：解释卡（只读账本回答——零新任务/零新仿真）。
 */

import type { Component } from "@earendil-works/pi-tui";

/** 内核自定义消息的最小视图（pi CustomMessage 的结构子集——卡片
 *  只用 customType/content，不与 details 的具体类型耦合）。 */
export interface KernelMessageView {
	customType: string;
	content: unknown;
}

class KernelMessageCard implements Component {
	constructor(
		private readonly title: string,
		private readonly content: string,
	) {}

	render(width: number): string[] {
		const border = "─".repeat(Math.max(10, Math.min(width - 4, 76)));
		const lines = [`┌${border}┐`, `│ ${this.title}`];
		for (const raw of this.content.split("\n")) {
			lines.push(raw ? `│ ${raw}` : "│");
		}
		lines.push(`└${border}┘`);
		return lines;
	}

	invalidate(): void {}
}

/** content 可能是 string 或 TextContent[]（pi CustomMessage 联合
 *  类型）——卡片只呈现文本。 */
function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.map((c) =>
				c && typeof c === "object" && "text" in c
					? String((c as { text?: unknown }).text ?? "")
					: String(c),
			)
			.filter(Boolean)
			.join("\n");
	}
	return String(content ?? "");
}

const CARD_TITLES: Record<string, string> = {
	"rosclaw.user_directive": "📌 任务指令（确定性链接管执行——不进模型回合）",
	"rosclaw.task_terminal": "任务终态（内核权威）",
	"rosclaw.task_explain": "任务解释（只读账本回答——未起新任务）",
};

/** customType → 渲染卡。注册进 pi.registerMessageRenderer。 */
export const kernelMessageRenderers: Record<
	string,
	(message: KernelMessageView) => Component | undefined
> = Object.fromEntries(
	Object.entries(CARD_TITLES).map(([customType, title]) => [
		customType,
		(message: KernelMessageView) =>
			new KernelMessageCard(title, contentText(message.content)),
	]),
);
