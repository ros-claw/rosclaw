// NATIVE-TOKENS：native message_end 镜像用量入账回归（编译后运行，
// 注入 inert bridge——零 provider/网络/ROS）。
import assert from "node:assert/strict";
import test from "node:test";

import { EventMirror } from "../src/extension/event-mirror.js";
import { buildCommandHandlers } from "../src/extension/commands.js";

function recordingMirror() {
	const rows: Array<Record<string, unknown>> = [];
	const mirror = new EventMirror(
		"/tmp/rh", "pi_1", "mis_1",
		async (_home: string, _method: string, params: Record<string, unknown> = {}) => {
			rows.push(...(params.events as Array<Record<string, unknown>>));
			return { ok: true, stored: (params.events as unknown[]).length };
		},
	);
	return { mirror, rows };
}

test("hash-only metadata: assistant text never enters the mirror", async () => {
	const { mirror, rows } = recordingMirror();
	mirror.push("message_end", { text: "private assistant sentence", model: "k3", usage: { input: 1 } });
	await mirror.flush();
	assert.equal(rows.length, 1);
	assert.ok(String(rows[0].content_hash).startsWith("sha256:"));
	assert.ok(!JSON.stringify(rows).includes("private assistant sentence"));
	assert.deepEqual(rows[0].usage, { input: 1 });
});

test("two distinct messages with identical content and no entry id are NOT deduplicated", async () => {
	const { mirror, rows } = recordingMirror();
	// 历史空身份消息：内容/usage 相同仍是两笔独立付费用量。
	mirror.push("message_end", { text: "same", model: "k3", usage: { input: 1 } });
	mirror.push("message_end", { text: "same", model: "k3", usage: { input: 1 } });
	await mirror.flush();
	assert.equal(rows.length, 2);
	assert.notEqual(rows[0].mirror_id, rows[1].mirror_id);
	assert.equal(rows[0].pi_entry_id, "");
});

test("genuine provider entry id replays are deduplicated exactly once", async () => {
	const { mirror, rows } = recordingMirror();
	// provider retry/重连重放同一 responseId——只镜像一次。
	for (let i = 0; i < 3; i++) {
		mirror.push("message_end", { entryId: "real-response-id", text: "same", usage: { input: 1 } });
	}
	await mirror.flush();
	assert.equal(rows.length, 1);
	assert.equal(rows[0].pi_entry_id, "real-response-id");
});

test("distinct entry ids are distinct messages", async () => {
	const { mirror, rows } = recordingMirror();
	mirror.push("message_end", { entryId: "resp-a", text: "same", usage: { input: 1 } });
	mirror.push("message_end", { entryId: "resp-b", text: "same", usage: { input: 1 } });
	await mirror.flush();
	assert.equal(rows.length, 2);
});

interface NativeUsage {
	known_message_count: number; input_uncached: number | string | null; cache_read: number | string | null;
	cache_write: number | string | null; input_including_cache: number | string | null; output: number | string | null;
	reasoning_subset_output: number | string | null; total_tokens: number | string | null;
	unknown_message_count: number; cost_usd_estimate: number | null;
	cost_unknown_message_count: number; identity_scope: string;
	reasoning_unknown_message_count?: number;
	billing_authority_unknown_message_count?: number;
	cost_usd_known_subtotal?: number | null;
}

async function renderTokens(native: NativeUsage): Promise<string> {
	const notices: string[] = [];
	const handlers = buildCommandHandlers({
		rosclawHome: "/tmp/rh",
		active: { current: { missionId: "mis_1" } },
		locale: { effective: "zh-CN" },
		registeredToolNames: () => [],
		center: {
			call: async () => ({
				ok: true,
				usage: {
					model_turns: 2, prompt_tokens: 300, completion_tokens: 100,
					total_tokens: 400, cost_microunits: 7000000, native_usage: native,
				},
			}),
		},
	} as never);
	await handlers.tokens.handler("", {
		ui: { notify: (message: string) => notices.push(message) },
	} as never);
	return notices.join("\n");
}

test("/tokens renders native cache breakdown and USD estimate separate from legacy yuan", async () => {
	const text = await renderTokens({
		known_message_count: 1, input_uncached: 10, cache_read: 20, cache_write: 3,
		input_including_cache: 33, output: 7, reasoning_subset_output: 2,
		total_tokens: 40, unknown_message_count: 0, cost_usd_estimate: 0.33,
		cost_unknown_message_count: 0, identity_scope: "message_end",
	});
	assert.match(text, /当前任务原生会话累计/); // mission 聚合口径，不是“本次会话”
	assert.ok(!/本次原生会话/.test(text));
	assert.match(text, /33/); // 含缓存输入合计 = input+cacheRead+cacheWrite
	assert.match(text, /40/); // total
	assert.match(text, /缓存读入\/写入 20\/3/);
	assert.match(text, /USD/);
	assert.match(text, /0\.33/);
	// 旧账（人民币）分开呈现，无虚假总计。
	assert.match(text, /其他模型请求/);
	assert.match(text, /元/);
	// 绝不向用户暴露数据库表名。
	assert.ok(!/model_usage|pi_event_mirrors/.test(text));
});

test("/tokens shows missing native cost as unknown — never as free", async () => {
	const text = await renderTokens({
		known_message_count: 0, input_uncached: 0, cache_read: 0, cache_write: 0,
		input_including_cache: 0, output: 0, reasoning_subset_output: 0,
		total_tokens: 0, unknown_message_count: 1, cost_usd_estimate: null,
		cost_unknown_message_count: 1, identity_scope: "message_end",
	});
	assert.match(text, /未知/);
	assert.match(text, /尚未确认 1 条/);
	assert.match(text, /不是 live pending request proof/);
	assert.ok(!/免费|free/i.test(text));
	assert.ok(!/model_usage|pi_event_mirrors/.test(text));
});

test("stable identity with conflicting payload throws typed — never silently dropped before RPC", async () => {
	const { mirror, rows } = recordingMirror();
	mirror.push("message_end", { entryId: "stable", usage: { input: 10, output: 1 } });
	let typed = false;
	try {
		mirror.push("message_end", { entryId: "stable", usage: { input: 10, output: 2 } });
	} catch (err) {
		typed = Boolean((err as { code?: string }).code) || /conflict/i.test(String(err));
	}
	assert.ok(typed, "conflicting payload must be observable/typed");
	await mirror.flush();
	assert.equal(rows.length, 1); // 原始载荷保留
	assert.deepEqual(rows[0].usage, { input: 10, output: 1 });
});

test("same stable identity with identical payload stays idempotent", async () => {
	const { mirror, rows } = recordingMirror();
	const options = { entryId: "stable", usage: { input: 10, output: 1 } };
	mirror.push("message_end", options);
	mirror.push("message_end", options);
	assert.equal(mirror.pending, 1);
	await mirror.flush();
	assert.equal(rows.length, 1);
});

test("throwing bridge preserves bounded pending queue and later replay succeeds", async () => {
	let calls = 0;
	const mirror = new EventMirror(
		"/tmp/rh", "pi_1", "mis_1",
		async (_home: string, _method: string, params: Record<string, unknown> = {}) => {
			calls++;
			if (calls === 1) throw new Error("INERT_TRANSIENT_BRIDGE_FAILURE");
			return { ok: true, stored: (params.events as unknown[]).length };
		},
	);
	mirror.push("message_end", { entryId: "stable", usage: { input: 10, output: 1 } });
	assert.equal(await mirror.flush(), 0);
	assert.equal(mirror.pending, 1, "thrown bridge must not erase pending usage");
	assert.equal(await mirror.flush(), 1);
	assert.equal(calls, 2);
	assert.equal(mirror.pending, 0);
});

// BACKLOG-FIX：有界 drain——每个 pi.events.batch 请求 ≤256 条，
// <=1000 的有限 pending 在 bridge 恢复后完整 drain。
test("backlog of 1000 drains in <=256-event requests after bridge recovery", async () => {
	const batches: number[] = [];
	const seen: string[] = [];
	let offline = true;
	const mirror = new EventMirror(
		"/tmp/rh", "pi_1", "mis_1",
		async (_home: string, _method: string, params: Record<string, unknown> = {}) => {
			const events = params.events as Array<Record<string, unknown>>;
			assert.ok(events.length <= 256, `batch exceeds server boundary: ${events.length}`);
			batches.push(events.length);
			if (offline) throw new Error("INERT_BRIDGE_OFFLINE");
			seen.push(...events.map((e) => String(e.pi_entry_id)));
			return { ok: true, stored: events.length };
		},
	);
	for (let i = 0; i < 1000; i++) mirror.push("message_end", { entryId: `e_${i}`, usage: { input: 1 } });
	assert.equal(await mirror.flush(), 0);
	assert.equal(mirror.pending, 1000, "failed request must not drop accepted pending");
	offline = false;
	await mirror.flush();
	assert.equal(mirror.pending, 0);
	assert.deepEqual(seen, Array.from({ length: 1000 }, (_, i) => `e_${i}`));
	assert.ok(batches.every((n) => n <= 256));
});

test("failed middle batch preserves failed batch and later suffix in order", async () => {
	let calls = 0;
	const seen: string[] = [];
	const mirror = new EventMirror(
		"/tmp/rh", "pi_1", "mis_1",
		async (_home: string, _method: string, params: Record<string, unknown> = {}) => {
			const events = params.events as Array<Record<string, unknown>>;
			calls++;
			if (calls === 2) throw new Error("INERT_SECOND_BATCH_FAILURE");
			seen.push(...events.map((e) => String(e.pi_entry_id)));
			return { ok: true, stored: events.length };
		},
	);
	for (let i = 0; i < 600; i++) mirror.push("message_end", { entryId: `s_${i}`, usage: { input: 1 } });
	assert.equal(await mirror.flush(), 256, "first batch commits before the failure");
	assert.equal(mirror.pending, 344, "failed batch + later suffix retained");
	await mirror.flush();
	assert.equal(mirror.pending, 0);
	assert.deepEqual(seen, Array.from({ length: 600 }, (_, i) => `s_${i}`));
});

test("concurrent push during in-flight flush neither duplicates nor drops", async () => {
	const seen: string[] = [];
	let unblock: () => void = () => {};
	let started: () => void = () => {};
	const gate = new Promise<void>((resolve) => (unblock = resolve));
	const startedP = new Promise<void>((resolve) => (started = resolve));
	let first = true;
	const mirror = new EventMirror(
		"/tmp/rh", "pi_1", "mis_1",
		async (_home: string, _method: string, params: Record<string, unknown> = {}) => {
			const events = params.events as Array<Record<string, unknown>>;
			assert.ok(events.length <= 256);
			if (first) {
				first = false;
				started();
				await gate;
			}
			seen.push(...events.map((e) => String(e.pi_entry_id)));
			return { ok: true, stored: events.length };
		},
	);
	for (let i = 0; i < 256; i++) mirror.push("message_end", { entryId: `c_${i}`, usage: { input: 1 } });
	const active = mirror.flush();
	await startedP;
	for (let i = 256; i < 600; i++) mirror.push("message_end", { entryId: `c_${i}`, usage: { input: 1 } });
	const overlap = mirror.flush(); // 在飞期间的重叠 flush 立即返回，不重复发送
	unblock();
	await Promise.all([active, overlap]);
	assert.equal(mirror.pending, 0, "同一 drain 循环拾起 flush 期间 push 的 suffix");
	assert.deepEqual(seen, Array.from({ length: 600 }, (_, i) => `c_${i}`));
});

test("/tokens renders decimal-string aggregates beyond JS exact-integer range verbatim", async () => {
	const text = await renderTokens({
		known_message_count: 2, input_uncached: "9007199254740993", cache_read: 0, cache_write: 0,
		input_including_cache: "9007199254740993", output: 0, reasoning_subset_output: 0,
		total_tokens: "9007199254740993", unknown_message_count: 0, cost_usd_estimate: null,
		cost_unknown_message_count: 2, identity_scope: "message_end",
	});
	assert.match(text, /9007199254740993/);
	assert.match(text, /未知/);
	assert.ok(!/model_usage|pi_event_mirrors/.test(text));
});

// NATIVE-TOKENS-TERMINAL：终端出处信封原样穿过 mirror（hash-only，
// 绝不带原始错误/文本），responseId 不是完成权威。
test("terminal provenance envelope passes through the mirror hash-only", async () => {
	const { mirror, rows } = recordingMirror();
	const usage = {
		input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
		cost: { total: 0 },
		_rosclaw_terminal: { stopReason: "aborted", responseId: "msg_real" },
	};
	mirror.push("message_end", { entryId: "msg_real", text: "aborted draft", usage });
	await mirror.flush();
	assert.equal(rows.length, 1);
	assert.deepEqual(rows[0].usage, usage);
	assert.ok(!JSON.stringify(rows).includes("aborted draft"));
});

test("/tokens shows reported cost subtotal as a lower bound — never complete billing", async () => {
	const text = await renderTokens({
		known_message_count: 0, input_uncached: 10, cache_read: 20, cache_write: 3,
		input_including_cache: 33, output: 7, reasoning_subset_output: null,
		total_tokens: 40, unknown_message_count: 34, cost_usd_estimate: null,
		cost_unknown_message_count: 34, identity_scope: "message_end",
		reasoning_unknown_message_count: 34,
		billing_authority_unknown_message_count: 34,
		cost_usd_known_subtotal: 0.18126421,
	});
	// 已上报部分成本是已知下限事实——不被未知记录清零。
	assert.match(text, /0\.18126421/);
	assert.match(text, /USD/);
	// 完整费用与完成状态保持未知——绝不说成免费或完整账单。
	assert.match(text, /未知/);
	assert.match(text, /尚未确认 34 条/);
	assert.match(text, /缺少终端完成凭证 34 条/);
	assert.match(text, /reasoning 细分未知 34 条/);
	assert.ok(!/免费|free/i.test(text));
	assert.ok(!/model_usage|pi_event_mirrors/.test(text));
});
