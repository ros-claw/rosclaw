import assert from "node:assert/strict";
import test from "node:test";

import { ApprovalCardComponent } from "../src/ui/approval-card.js";

const CARD = {
	requestId: "appr_1",
	title: "播放提示音",
	summary: "limo.speaker.play_tone",
	riskTier: "LOW",
	mode: "SHADOW",
	capability: "limo.speaker.play_tone",
	parameters: { frequency_hz: 660 },
	expiresAt: "2026-08-05T00:00:00Z",
	displayHash: "abcd1234",
};

test("Y approves, N denies, Esc denies, unknown key ignored", () => {
	const decisions: boolean[] = [];
	const card = new ApprovalCardComponent(CARD, (approve) => decisions.push(approve));
	card.handleInput("x"); // 未知键：无决定
	assert.equal(decisions.length, 0);
	card.handleInput("y");
	assert.deepEqual(decisions, [true]);
	card.handleInput("n"); // 已决定后忽略
	assert.deepEqual(decisions, [true]);

	const d2: boolean[] = [];
	const card2 = new ApprovalCardComponent(CARD, (approve) => d2.push(approve));
	card2.handleInput("\x1b");
	assert.deepEqual(d2, [false]);
});

test("card renders immutable fields (display_hash bound)", () => {
	const card = new ApprovalCardComponent(CARD, () => undefined);
	const text = card.render(80).join("\n");
	assert.ok(text.includes("abcd1234"));
	assert.ok(text.includes("limo.speaker.play_tone"));
	assert.ok(text.includes("frequency_hz"));
	assert.ok(text.includes("SHADOW"));
});
