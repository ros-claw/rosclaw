/** P1-A4 红测试（0824 总纲 P1-A）：/compact 保留 TaskRefs。
 *
 * 复现的真实事故：Pi compaction（manual /compact、threshold、overflow）
 * 把会话历史压成摘要——task_id/root_goal/artifact refs（TraceRef、
 * GIF/MP4 路径）随被压掉的消息一起丢失，模型 compact 后"失忆"：
 * 不知道自己在哪个 task、交付物在哪。
 *
 * 断言：任何 reason 的 compaction 完成后，扩展从 TaskKernel 权威账本
 * （pi.kernel.latest + pi.kernel.artifacts——不是模型记忆）取最近 task
 * 与产物 refs，经 pi.sendMessage(deliverAs:"nextTurn") 把 TaskRefs 锚
 * 定回 LLM 上下文；无 task 时不发锚（诚实 no-op）；同 key 锚去重。
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
	buildTaskAnchor,
	registerCompactAnchor,
} from "../src/extension/compact-anchor.js";

const TASK = {
	task_id: "task_abc123",
	root_goal: "画五角星并交付 GIF+MP4",
	state: "ACTIVE",
	active_revision: 3,
};

const ARTIFACTS = [
	{
		artifact_id: "art_1",
		path: "/run/traces/trace_1/trace_1-scene.gif",
		media_type: "image/gif",
	},
	{
		artifact_id: "art_2",
		path: "/run/traces/trace_1/trace_1-scene.mp4",
		media_type: "video/mp4",
	},
];

function fakePi() {
	const sent: Array<{ message: unknown; options: unknown }> = [];
	const handlers = new Map<string, (event: unknown) => Promise<void>>();
	return {
		sent,
		handlers,
		on(event: string, handler: (event: unknown) => Promise<void>) {
			handlers.set(event, handler);
		},
		sendMessage(message: unknown, options: unknown) {
			sent.push({ message, options });
		},
	};
}

function fakeCall(task: unknown, artifacts: unknown[]) {
	const calls: string[] = [];
	const call = async (method: string, _params?: unknown) => {
		calls.push(method);
		if (method === "pi.kernel.latest") return { ok: true, task };
		if (method === "pi.kernel.artifacts") return { ok: true, artifacts };
		throw new Error(`unexpected ${method}`);
	};
	return { call, calls };
}

const COMPACT_EVENT = {
	type: "session_compact",
	reason: "manual",
	willRetry: false,
	compactionEntry: { id: "entry_1" },
};

test("buildTaskAnchor 含 task/goal/state/revision 与全部产物 refs", () => {
	const anchor = buildTaskAnchor(TASK, ARTIFACTS, "manual");
	assert.match(anchor, /task_abc123/);
	assert.match(anchor, /画五角星并交付 GIF\+MP4/);
	assert.match(anchor, /ACTIVE/);
	assert.match(anchor, /rev.*3|revision.*3/);
	assert.match(anchor, /trace_1-scene\.gif/);
	assert.match(anchor, /trace_1-scene\.mp4/);
	assert.match(anchor, /image\/gif/);
	assert.match(anchor, /video\/mp4/);
});

test("compaction 后锚定 TaskRefs（deliverAs nextTurn，不触发回合）", async () => {
	const pi = fakePi();
	const { call } = fakeCall(TASK, ARTIFACTS);
	registerCompactAnchor(pi as never, {
		call: call as never,
		missionId: () => "m1",
		sessionRef: () => "s1",
	});
	const handler = pi.handlers.get("session_compact");
	assert.ok(handler, "session_compact 未注册");
	await handler(COMPACT_EVENT);
	assert.equal(pi.sent.length, 1);
	const { message, options } = pi.sent[0] as {
		message: { customType: string; content: string; display: boolean };
		options: { triggerTurn?: boolean; deliverAs?: string };
	};
	assert.equal(message.customType, "rosclaw.task_anchor");
	assert.match(message.content, /task_abc123/);
	assert.match(message.content, /trace_1-scene\.mp4/);
	assert.equal(options.triggerTurn ?? false, false);
	assert.equal(options.deliverAs, "nextTurn");
});

test("无 task → 诚实 no-op（不发空锚）", async () => {
	const pi = fakePi();
	const { call } = fakeCall(null, []);
	registerCompactAnchor(pi as never, {
		call: call as never,
		missionId: () => "m1",
		sessionRef: () => "s1",
	});
	await pi.handlers.get("session_compact")!(COMPACT_EVENT);
	assert.equal(pi.sent.length, 0);
});

test("同 key 锚去重（task+revision 未变不重复发）", async () => {
	const pi = fakePi();
	const { call } = fakeCall(TASK, ARTIFACTS);
	registerCompactAnchor(pi as never, {
		call: call as never,
		missionId: () => "m1",
		sessionRef: () => "s1",
	});
	const handler = pi.handlers.get("session_compact")!;
	await handler(COMPACT_EVENT);
	await handler({ ...COMPACT_EVENT, compactionEntry: { id: "entry_2" } });
	assert.equal(pi.sent.length, 1);
	// revision 前进 → 新锚。
	const { call: call2 } = fakeCall({ ...TASK, active_revision: 4 }, ARTIFACTS);
	const pi2 = fakePi();
	registerCompactAnchor(pi2 as never, {
		call: call2 as never,
		missionId: () => "m1",
		sessionRef: () => "s1",
	});
	await pi2.handlers.get("session_compact")!(COMPACT_EVENT);
	await pi2.handlers.get("session_compact")!({
		...COMPACT_EVENT,
		compactionEntry: { id: "entry_3" },
	});
	assert.equal(pi2.sent.length, 1);
});

test("kernel 拉取失败 → 不发锚（不编造 refs）", async () => {
	const pi = fakePi();
	registerCompactAnchor(pi as never, {
		call: (async () => ({ ok: false, error: "bridge down" })) as never,
		missionId: () => "m1",
		sessionRef: () => "s1",
	});
	await pi.handlers.get("session_compact")!(COMPACT_EVENT);
	assert.equal(pi.sent.length, 0);
});
