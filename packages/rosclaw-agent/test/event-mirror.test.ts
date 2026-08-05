import assert from "node:assert/strict";
import test from "node:test";

import { contentHash, EventMirror } from "../src/extension/event-mirror.js";

test("mirror stores hashes only — never assistant text", async () => {
	const sent: Array<Record<string, unknown>> = [];
	const call = async (_home: string, method: string, params: Record<string, unknown> = {}) => {
		if (method === "pi.events.batch") {
			sent.push(...(params.events as Array<Record<string, unknown>>));
			return { ok: true, stored: (params.events as unknown[]).length };
		}
		return { ok: false };
	};
	const mirror = new EventMirror("/tmp/rh", "pi_1", "mis_1", call);
	mirror.push("message_end", { text: "助手完整回答文本", model: "k3" });
	mirror.push("turn_end", { text: "助手完整回答文本" });
	await mirror.flush();
	assert.equal(sent.length, 2);
	for (const event of sent) {
		assert.ok(String(event.content_hash).startsWith("sha256:"));
		const serialized = JSON.stringify(event);
		assert.ok(!serialized.includes("助手完整回答文本"), "全文绝不进镜像");
	}
});

test("content hash is stable and text-sensitive", () => {
	assert.equal(contentHash("abc"), contentHash("abc"));
	assert.notEqual(contentHash("abc"), contentHash("abd"));
});

test("flush failure keeps events queued (bounded)", async () => {
	const call = async () => ({ ok: false, error: "down" });
	const mirror = new EventMirror("/tmp/rh", "pi_1", "mis_1", call as never);
	mirror.push("turn_end", { text: "x" });
	await mirror.flush();
	assert.equal(mirror.pending, 1);
});
