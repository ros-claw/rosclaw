/** 会话查询解析（总纲 WP-P0-1）：精确 ID → 唯一前缀 → 标题。
 * 纯函数（测试直接驱动）；歧义报候选不猜。 */

import type { SessionInfo } from "@earendil-works/pi-coding-agent";

export type SessionResolution =
	| { ok: true; path: string; info: SessionInfo }
	| { ok: false; error: "NOT_FOUND" | "AMBIGUOUS"; candidates: SessionInfo[] };

export function resolveSessionQuery(
	query: string,
	sessions: SessionInfo[],
): SessionResolution {
	const q = query.trim();
	if (!q) return { ok: false, error: "NOT_FOUND", candidates: [] };
	const exact = sessions.find((s) => s.id === q);
	if (exact) return { ok: true, path: exact.path, info: exact };
	const prefix = sessions.filter((s) => s.id.startsWith(q));
	if (prefix.length === 1) return { ok: true, path: prefix[0].path, info: prefix[0] };
	if (prefix.length > 1) return { ok: false, error: "AMBIGUOUS", candidates: prefix };
	const titled = sessions.filter(
		(s) => (s.name ?? "").includes(q) || s.firstMessage.includes(q),
	);
	if (titled.length === 1) return { ok: true, path: titled[0].path, info: titled[0] };
	if (titled.length > 1) return { ok: false, error: "AMBIGUOUS", candidates: titled };
	return { ok: false, error: "NOT_FOUND", candidates: [] };
}
