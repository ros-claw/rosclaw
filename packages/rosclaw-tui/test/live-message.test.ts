import assert from "node:assert/strict";
import test from "node:test";
import { Chalk } from "chalk";
import { LiveAssistantMessage } from "../src/components/live-message.js";

const chalk = new Chalk({ level: 0 });
const theme = {
	heading: (t: string) => t,
	link: (t: string) => t,
	linkUrl: (t: string) => t,
	code: (t: string) => t,
	codeBlock: (t: string) => t,
	codeBlockBorder: (t: string) => t,
	quote: (t: string) => t,
	quoteBorder: (t: string) => t,
	hr: (t: string) => t,
	listBullet: (t: string) => t,
	bold: (t: string) => t,
	italic: (t: string) => t,
	strikethrough: (t: string) => t,
	underline: (t: string) => t,
};

test("first delta renders immediately, not on flush", () => {
	let flushes = 0;
	const live = new LiveAssistantMessage(theme, () => (flushes += 1));
	live.append("你好");
	// 首个 delta 立即显示（审计 P0-02.2：不允许傻等到 flush_delta）。
	assert.equal(live.text, "你好");
	assert.ok(flushes >= 1);
	live.flush(true);
});

test("throttled accumulation, force flush on settle", async () => {
	const live = new LiveAssistantMessage(theme, () => undefined);
	live.append("a");
	live.append("b");
	live.append("c");
	assert.equal(live.text, "abc");
	live.flush(true);
	assert.equal(live.text, "abc");
});

test("stop is idempotent", () => {
	const live = new LiveAssistantMessage(theme, () => undefined);
	live.append("x");
	live.stop();
	live.stop();
	live.flush(true);
	assert.equal(live.text, "x");
});
