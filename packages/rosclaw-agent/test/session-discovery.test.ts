/** WP-P0-1（总纲 §5.1）：会话查询解析——精确 ID → 唯一前缀 →
 * 标题/首条消息；歧义报候选不猜。 */

import assert from "node:assert/strict";
import test from "node:test";
import type { SessionInfo } from "@earendil-works/pi-coding-agent";

function info(id: string, name: string, firstMessage: string): SessionInfo {
	return {
		path: `/sessions/2026-08-10T10-00-00_${id}.jsonl`,
		id,
		cwd: "/x",
		name,
		created: new Date("2026-08-10T10:00:00Z"),
		modified: new Date("2026-08-10T11:00:00Z"),
		messageCount: 3,
		firstMessage,
		allMessagesText: "",
	} as SessionInfo;
}

test("resolveSessionQuery：ID/前缀/标题/歧义", async () => {
	const { resolveSessionQuery } = await import("../src/harness/pi/pi-resolve.js");
	const sessions = [
		info("abc123", "五角星轨迹仿真", "画五角星"),
		info("abd999", "五角星复测", "再画一次"),
	];
	const byId = resolveSessionQuery("abc123", sessions);
	assert.ok(byId.ok && byId.path.includes("abc123"));
	const byTitle = resolveSessionQuery("五角星轨迹仿真", sessions);
	assert.ok(byTitle.ok && byTitle.info.id === "abc123");
	const ambiguous = resolveSessionQuery("ab", sessions);
	assert.ok(!ambiguous.ok && ambiguous.error === "AMBIGUOUS");
	assert.equal(ambiguous.ok ? "" : ambiguous.candidates.length, 2);
	const missing = resolveSessionQuery("zzz", sessions);
	assert.ok(!missing.ok && missing.error === "NOT_FOUND");
});

test("picker 是公开 API 薄组装（不复制上游 picker 内核）", async () => {
	const { readFileSync } = await import("node:fs");
	const source = readFileSync("src/harness/pi/pi-picker.ts", "utf-8");
	assert.ok(source.includes("SessionSelectorComponent"), "必须用 Pi 公开组件");
	assert.ok(!source.includes("class SessionList"), "不得复制上游列表内核");
	assert.ok(source.split("\n").length < 80, "薄组装层应保持极小");
});
