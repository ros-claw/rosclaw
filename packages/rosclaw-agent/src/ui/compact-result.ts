// HP2-COMPAT: pi-tui Text 渲染原语——HP3 投影层落地前保持；不新增会话装配引用。
/** 0902 R3-c（§6.1 三层界面）：静态只读工具的 renderResult 折叠钩。
 *
 * 模型上下文保留完整 JSON（诚实性不降级）；TUI 只渲染单行摘要——
 * 0902 实证：rosclaw_status 整段 JSON 打进 scrollback。
 */

import { Text } from "@earendil-works/pi-tui";

import { summarizeStatusText } from "./tool-display.js";

/** renderResult 折叠钩（pi 工具定义直接挂）：单行摘要 Text。 */
export function compactRenderResult(
	result: { content?: Array<{ type: string; text?: string }> },
): Text {
	const text = (result.content ?? [])
		.map((b) => (b.type === "text" ? String(b.text ?? "") : ""))
		.join("\n");
	return new Text(summarizeStatusText(text), 1, 0);
}
