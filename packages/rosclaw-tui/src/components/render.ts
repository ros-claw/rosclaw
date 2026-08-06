/** Presentational helpers (pure, testable)：status line 与卡片文本。 */

import type { CardModel } from "../state/reducer.js";
import type { UiState } from "../state/types.js";

export function statusLine(state: UiState): string {
	const parts = [
		`ROSClaw`,
		state.missionName || state.missionId,
		state.mode,
	];
	if (state.bodyId) parts.push(state.bodyId);
	if (state.model) parts.push(`model=${state.model}`);
	parts.push(`state=${state.missionState}`);
	if (state.pendingApprovals.length > 0) parts.push(`待批准=${state.pendingApprovals.length}`);
	if (state.workers.some((w) => !["accepted", "failed", "expired"].includes(w.status))) {
		parts.push(`workers=${state.workers.filter((w) => !["accepted", "failed", "expired"].includes(w.status)).length}`);
	}
	if (state.compactions > 0) parts.push(`compactions=${state.compactions}`);
	if (state.reconnecting) parts.push("reconnecting…");
	if (state.degraded) parts.push(`degraded=${state.degraded}`);
	return parts.join(" | ");
}

export function renderCard(card: CardModel): string {
	const border = "─".repeat(Math.max(8, Math.min(60, visibleLen(card.title) + 4)));
	const lines = [
		`┌${border}┐`,
		`│ ${card.title}`,
		...card.lines.map((l) => `│ ${l}`),
		`└${border}┘`,
	];
	return lines.join("\n");
}

function visibleLen(text: string): number {
	let width = 0;
	for (const ch of text) {
		width += ch.charCodeAt(0) > 0xff ? 2 : 1;
	}
	return width;
}

export const HOTKEYS_TEXT = [
	"Enter        发送 / 执行命令",
	"Shift+Enter  换行（多行编辑）",
	"Ctrl+C       turn 运行中：取消 turn；否则退出提示",
	"Ctrl+D       输入为空时退出 TUI",
	"↑ / ↓        历史输入",
	"Tab          命令补全",
].join("\n");
