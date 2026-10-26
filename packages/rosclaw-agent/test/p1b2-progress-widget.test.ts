/** P1-B2 红测试（0824 总纲 §12.3）：operation progress 流式进 TUI。
 *
 * 真实缺口：progress 只有两条路——模型/用户主动 process_output 或
 * /logs（拉模式），TUI 没有按 operation_id 原位更新的活动区；watcher
 * 每 2s 对每个 op 发 pi.op.get（N 个 op N 次调用）。
 *
 * 断言 V2：
 * 1. 稳态 tick 只发 pi.kernel.events（task_id + last_seq 增量游标）——
 *    不再每个 op 一次 pi.op.get（注册时一次性取 task_id 除外）；
 * 2. operation.output/progress 事件 → sink.setWidget 以 operation_id
 *    为 key 原位更新（同 key 覆盖，不追加）；
 * 3. 终态事件 → widget 清除 + 既有 followUp 语义（WP-1 不变）；
 * 4. progress 绝不进模型上下文（setWidget 不是 sendMessage）。
 */

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { OperationWatcher } from "../src/native/operation-watcher.js";

interface WidgetCall { key: string; lines: string[] | undefined }

function fakeClock(t: TestContext) {
	let now = 10_000;
	t.mock.method(Date, "now", () => now);
	return { advance(ms: number) { now += ms; } };
}
function tick(w: OperationWatcher): Promise<void> {
	return (w as unknown as { tick(): Promise<void> }).tick();
}
function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: Error) => void;
	const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
	return { promise, resolve, reject };
}
function latest(widgets: WidgetCall[], id: string): string {
	return widgets.filter((w) => w.key === `op:${id}`).at(-1)?.lines?.join("\n") ?? "";
}

function harness(opts: {
	op?: Record<string, unknown>;
	opsById?: Record<string, Record<string, unknown>>;
	eventsByTask?: Record<string, Array<Record<string, unknown>>>;
	task?: Record<string, unknown> | null;
	idle?: boolean;
	bridgeDown?: boolean;
	getDown?: boolean;
	replay?: boolean;
	uiAvailable?: boolean;
	onCall?: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>> | undefined;
}) {
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	const widgets: WidgetCall[] = [];
	const sent: Array<{ message: unknown; options: unknown }> = [];
	const seqs = new Map<string, number>();
	const deps = {
		call: async (method: string, params: Record<string, unknown>) => {
			calls.push({ method, params });
			const intercepted = opts.onCall?.(method, params);
			if (intercepted) return intercepted;
			if (method === "pi.op.get") {
				if (opts.getDown) throw new Error("fake registration bridge unavailable");
				return { operation: opts.opsById?.[String(params.operation_id)] ?? opts.op ?? {} };
			}
			if (method === "pi.kernel.get") return { task: opts.task ?? null };
			if (method === "pi.kernel.events") {
				if (opts.bridgeDown) throw new Error("fake event bridge unavailable");
				const taskId = String(params.task_id);
				const since = Number(params.last_seq ?? 0);
				seqs.set(taskId, since);
				const events = (opts.eventsByTask?.[taskId] ?? []).filter(
					(e) => opts.replay || Number(e.seq) > since,
				);
				return { events };
			}
			throw new Error(`unexpected ${method}`);
		},
		sink: () => opts.uiAvailable === false ? undefined : ({
			isIdle: opts.idle ?? false,
			api: {
				sendMessage(message: unknown, options: unknown) {
					sent.push({ message, options });
				},
			},
			setWidget(key: string, lines: string[] | undefined) {
				widgets.push({ key, lines });
			},
			notify: () => undefined,
		}),
	};
	return { deps, calls, widgets, sent, seqs };
}

test("稳态 tick 只发 pi.kernel.events（无逐 op pi.op.get）", async () => {
	const h = harness({
		op: { operation_id: "op_aaa", task_id: "task_1", state: "RUNNING" },
		eventsByTask: { task_1: [] },
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("op_aaa");
	await (w as never as { tick(): Promise<void> }).tick();
	const eventCalls = h.calls.filter((c) => c.method === "pi.kernel.events");
	const getCalls = h.calls.filter((c) => c.method === "pi.op.get");
	assert.ok(eventCalls.length >= 1, "未走 pi.kernel.events 增量流");
	assert.equal(getCalls.length <= 1, true, "稳态仍逐 op 轮询");
	// 第二 tick：events 游标前进（last_seq 带上次位置）。
	await (w as never as { tick(): Promise<void> }).tick();
	const second = h.calls.filter((c) => c.method === "pi.kernel.events").pop();
	assert.ok(Number(second?.params.last_seq ?? 0) >= 0);
});

test("operation.output 事件 → setWidget 按 operation_id 原位更新", async () => {
	const h = harness({
		op: { operation_id: "op_bbb", task_id: "task_1", state: "RUNNING" },
		eventsByTask: {
			task_1: [
				{ seq: 1, event_type: "operation.output", operation_id: "op_bbb", payload: { text: "building 10%\n" } },
				{ seq: 2, event_type: "operation.output", operation_id: "op_bbb", payload: { text: "building 42%\n" } },
			],
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("op_bbb");
	await (w as never as { tick(): Promise<void> }).tick();
	const opWidgets = h.widgets.filter((wc) => wc.key.includes("op_bbb"));
	assert.ok(opWidgets.length >= 1, "无 widget upsert");
	const last = opWidgets[opWidgets.length - 1];
	assert.ok(last.lines?.some((l) => l.includes("building 42%")), `widget 未含最新输出: ${JSON.stringify(last.lines)}`);
	// key 含 operation_id（稳定原位更新——不新增行）。
	assert.ok(opWidgets.every((wc) => wc.key === opWidgets[0].key), "widget key 不稳定");
	// progress 不进模型上下文。
	assert.equal(h.sent.length, 0, "progress 进了模型上下文");
});

test("operation.progress 事件 → widget 展示结构化进度", async () => {
	const h = harness({
		op: { operation_id: "op_ccc", task_id: "task_1", state: "RUNNING" },
		eventsByTask: {
			task_1: [
				{ seq: 3, event_type: "operation.progress", operation_id: "op_ccc", payload: { progress: { pct: 55, stage: "render" } } },
			],
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("op_ccc");
	await (w as never as { tick(): Promise<void> }).tick();
	const opWidgets = h.widgets.filter((wc) => wc.key.includes("op_ccc"));
	assert.ok(opWidgets.length >= 1);
	assert.ok(
		opWidgets[opWidgets.length - 1].lines?.some((l) => l.includes("55") || l.includes("render")),
		`widget 未含结构化进度: ${JSON.stringify(opWidgets)}`,
	);
});

test("终态事件 → widget 清除 + followUp（WP-1 语义不变）", async () => {
	const h = harness({
		// 生产流：注册时 RUNNING，终态经事件流到达。
		op: { operation_id: "op_ddd", task_id: "task_1", state: "RUNNING", revision: 1 },
		task: { task_id: "task_1", state: "RUNNING", active_revision: 1 },
		idle: true,
		eventsByTask: {
			task_1: [
				{ seq: 4, event_type: "operation.output", operation_id: "op_ddd", payload: { text: "final line\n" } },
				{ seq: 5, event_type: "operation.completed", operation_id: "op_ddd", payload: { state: "SUCCEEDED" } },
			],
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("op_ddd");
	await (w as never as { tick(): Promise<void> }).tick();
	assert.equal(h.sent.length, 1, "终态未触发 followUp");
	const cleared = h.widgets.filter((wc) => wc.key.includes("op_ddd") && wc.lines === undefined);
	assert.ok(cleared.length >= 1, "终态后 widget 未清除");
});

test("task 已终态 → 不触发模型回合（WP-1），widget 清除", async () => {
	const h = harness({
		op: { operation_id: "op_eee", task_id: "task_1", state: "SUCCEEDED", revision: 1 },
		task: { task_id: "task_1", state: "SUCCEEDED", active_revision: 1 },
		eventsByTask: {
			task_1: [
				{ seq: 6, event_type: "operation.completed", operation_id: "op_eee", payload: { state: "SUCCEEDED" } },
			],
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("op_eee");
	await (w as never as { tick(): Promise<void> }).tick();
	assert.equal(h.sent.length, 0, "任务终态后仍触发模型回合");
});

// UI-only observation: these fakes have no provider, process or physical action.
test("track is visible with a never-settling first get; overlap ages pending cards without RPCs or success", async (t) => {
	const clock = fakeClock(t);
	const entered = deferred<void>();
	const never = new Promise<Record<string, unknown>>(() => undefined);
	const h = harness({
		onCall: (method) => {
			if (method === "pi.op.get") { entered.resolve(); return never; }
		},
	});
	const w = new OperationWatcher(h.deps as never);
	t.after(() => w.stop());
	w.track("unresolved");
	const initial = latest(h.widgets, "unresolved");
	assert.match(initial, /last known ledger: UNKNOWN \(registration pending; no public result yet\)/);
	assert.match(initial, /Bridge: registration pending; no public result yet/);
	assert.match(initial, /UI elapsed since tracking: 0s/);
	assert.match(initial, /Progress: UNKNOWN/);
	assert.match(initial, /Current liveness\/completion: UNKNOWN/);
	assert.equal(h.calls.length, 0, "track must render without waiting for or sending a read");
	w.track("unresolved");
	assert.equal(h.widgets.length, 1, "duplicate tracking replaced the pending observation");
	void tick(w); // Intentionally never settles: no sleeps or timers needed.
	await entered.promise;
	for (const elapsed of [2, 32]) {
		clock.advance(elapsed === 2 ? 2_000 : 30_000);
		await tick(w);
		const text = latest(h.widgets, "unresolved");
		assert.match(text, new RegExp(`UI elapsed since tracking: ${elapsed}s`));
		assert.match(text, new RegExp(`quiet ≥ ${elapsed}s since tracking`));
		assert.match(text, /Bridge: registration pending; no public result yet/);
		assert.match(text, /last known ledger: UNKNOWN/);
		assert.match(text, /Progress: UNKNOWN/);
		assert.match(text, /Current liveness\/completion: UNKNOWN/);
	}
	assert.ok(h.widgets.every((v) => v.key === "op:unresolved" && (v.lines?.length ?? 0) <= 7));
	assert.ok(h.widgets.every((v) => !/last public read succeeded|RUNNING|SUCCEEDED|\d+%|deadline|timeout remaining/i.test(v.lines?.join("\n") ?? "")));
	assert.deepEqual(h.calls, [{ method: "pi.op.get", params: { operation_id: "unresolved" } }]);
	assert.equal(h.sent.length, 0);
	w.stop();
	assert.equal(latest(h.widgets, "unresolved"), "");
	const count = h.widgets.length;
	clock.advance(60_000);
	await tick(w); // Stopped overlap must not reopen the UI or dispatch a read.
	w.track("while-stopped");
	await tick(w);
	assert.equal(h.widgets.length, count);
	assert.equal(h.calls.length, 1);
	assert.equal(h.sent.length, 0);
});

for (const outcome of ["resolve", "reject"] as const) {
	test(`stopped initial get ${outcome} cannot redraw; restart retries registration with fresh ownership`, async (t) => {
		t.mock.timers.enable({ apis: ["setInterval", "Date"], now: 10_000 });
		const drain = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
		const entered = deferred<void>();
		const registration = deferred<Record<string, unknown>>();
		let firstRead = true;
		const h = harness({
			op: { operation_id: "late-registration", task_id: "current-task", state: "RUNNING", revision: 2 },
			onCall: (method) => {
				if (method === "pi.op.get" && firstRead) {
					firstRead = false;
					entered.resolve();
					return registration.promise;
				}
			},
		});
		const w = new OperationWatcher(h.deps as never);
		t.after(() => w.stop());
		w.track("late-registration");
		w.start();
		const pending = tick(w);
		await entered.promise;
		t.mock.timers.tick(2_000);
		await drain();
		assert.match(latest(h.widgets, "late-registration"), /UI elapsed since tracking: 2s/);
		assert.doesNotMatch(latest(h.widgets, "late-registration"), /last public read succeeded|RUNNING|SUCCEEDED/);
		assert.equal(h.calls.length, 1, "timer overlapped the outstanding registration RPC");
		w.stop();
		assert.equal(latest(h.widgets, "late-registration"), "");
		const count = h.widgets.length;
		t.mock.timers.tick(60_000);
		await drain();
		await tick(w);
		assert.equal(h.widgets.length, count, "stopped pending overlap redrew a card");
		if (outcome === "resolve") registration.resolve({ operation: {
			operation_id: "late-registration", task_id: "stale-task", state: "SUCCEEDED", revision: 1,
		} });
		else registration.reject(new Error("late registration failure"));
		await pending;
		assert.equal(h.widgets.length, count, "stale registration completion redrew the stopped UI");
		assert.equal(h.calls.length, 1, "stale result subscribed to events or verified terminal ownership");
		assert.equal(h.sent.length, 0);
		w.start();
		t.mock.timers.tick(2_000);
		await drain();
		const resumed = latest(h.widgets, "late-registration");
		assert.match(resumed, /last known ledger: RUNNING \(registration snapshot, not a fresh liveness observation\)/);
		assert.match(resumed, /UI elapsed since tracking: 64s/);
		assert.match(resumed, /Bridge: last public read succeeded; not a process liveness check/);
		assert.match(resumed, /Current liveness\/completion: UNKNOWN/);
		assert.doesNotMatch(resumed, /SUCCEEDED|stale-task/);
		assert.deepEqual(h.calls.map((c) => [c.method, c.params]), [
			["pi.op.get", { operation_id: "late-registration" }],
			["pi.op.get", { operation_id: "late-registration" }],
			["pi.kernel.events", { task_id: "current-task", last_seq: 0 }],
		]);
		t.mock.timers.tick(2_000);
		await drain();
		assert.equal(h.calls.filter((c) => c.method === "pi.op.get").length, 2);
		assert.equal(h.calls.filter((c) => c.method === "pi.kernel.events").length, 2);
		assert.equal(h.sent.length, 0);
	});
}

test("no-output startup is visible before a slow event read; silence stays UNKNOWN", async (t) => {
	const clock = fakeClock(t);
	const entered = deferred<void>();
	const feed = deferred<Record<string, unknown>>();
	let slow = true;
	const h = harness({
		op: { operation_id: "quiet", task_id: "task_1", state: "RUNNING", heartbeat_at: "2000-01-01T00:00:00Z" },
		onCall: (method) => {
			if (method === "pi.kernel.events" && slow) {
				entered.resolve();
				return feed.promise;
			}
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("quiet");
	const pending = tick(w);
	await entered.promise;
	const initial = latest(h.widgets, "quiet");
	assert.match(initial, /last known ledger: RUNNING/);
	assert.match(initial, /registration snapshot, not a fresh liveness observation/);
	assert.match(initial, /UI elapsed since tracking: 0s/);
	assert.match(initial, /none observed in this UI/);
	assert.match(initial, /Progress: UNKNOWN/);
	assert.doesNotMatch(initial, /\d+%|deadline|timeout remaining|step \d+/i);
	clock.advance(30_000);
	slow = false;
	feed.resolve({ events: [] });
	await pending;
	clock.advance(120_000);
	await tick(w);
	const silent = latest(h.widgets, "quiet");
	assert.match(silent, /UI elapsed since tracking: 150s/);
	assert.match(silent, /quiet ≥ 150s since tracking/);
	assert.match(silent, /Current liveness\/completion: UNKNOWN/);
	assert.match(silent, /silence does not prove deadlock or death/);
	assert.ok(h.widgets.every((v) => v.key === "op:quiet" && (v.lines?.length ?? 0) <= 7));
	assert.equal(h.calls.filter((v) => v.method === "pi.op.get").length, 1);
	assert.equal(h.calls.filter((v) => v.method === "pi.kernel.events").length, 2);
	assert.equal(h.sent.length, 0);
});

test("degraded/resumed use real events; output and structured progress keep separate UI ages", async (t) => {
	const clock = fakeClock(t);
	const events: Array<Record<string, unknown>> = [];
	const h = harness({
		op: { operation_id: "live", task_id: "task_1", state: "RUNNING" },
		eventsByTask: { task_1: events },
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("live");
	await tick(w);
	clock.advance(31_000);
	events.push({ seq: 1, event_type: "operation.degraded", operation_id: "live", payload: { stale_after_s: 30 } });
	await tick(w);
	assert.match(latest(h.widgets, "live"), /last known ledger: DEGRADED \(operation.degraded\)/);
	assert.match(latest(h.widgets, "live"), /not proof of deadlock\/death/);
	clock.advance(10_000);
	events.push({ seq: 2, event_type: "operation.output", operation_id: "live", payload: { text: "discarded-prefix" + "x".repeat(6000) + "\n\u001b[31mnewest tail\n" } });
	await tick(w);
	let text = latest(h.widgets, "live");
	assert.match(text, /last known ledger: DEGRADED/, "output is not a synthetic ledger resume");
	assert.match(text, /0s since UI-observed output \(replay may be old\)/);
	assert.match(text, /newest tail/);
	assert.doesNotMatch(text, /discarded-prefix|\u001b/);
	assert.ok(text.length < 1800, "unbounded output retained in widget");
	clock.advance(5_000);
	events.push({ seq: 3, event_type: "operation.progress", operation_id: "live", payload: { progress: { pct: 45, stage: "render" } } });
	await tick(w);
	text = latest(h.widgets, "live");
	assert.match(text, /5s since UI-observed output/);
	assert.match(text, /Last structured progress: 45% render \(0s since UI observation/);
	assert.match(text, /newest tail/, "structured progress must not replace the last output");
	clock.advance(7_000);
	events.push({ seq: 4, event_type: "operation.resumed", operation_id: "live", payload: {} });
	await tick(w);
	text = latest(h.widgets, "live");
	assert.match(text, /last known ledger: RUNNING \(operation.resumed\)/);
	assert.doesNotMatch(text, /Ledger DEGRADED warning/);
	assert.match(text, /12s since UI-observed output/);
	assert.match(text, /7s since UI observation/);
	clock.advance(6_000);
	events.push({ seq: 5, event_type: "operation.output", operation_id: "live", payload: { text: "replacement" } });
	await tick(w);
	text = latest(h.widgets, "live");
	assert.match(text, /Last output \(bounded UI tail\): replacement/);
	assert.doesNotMatch(text, /newest tail/);
	assert.match(text, /13s since UI observation/);
	assert.equal(h.sent.length, 0);
	assert.ok(h.widgets.every((v) => v.key === "op:live" && (v.lines?.length ?? 0) <= 7));
	assert.ok(h.calls.every((v) => ["pi.op.get", "pi.kernel.events"].includes(v.method)));
});

test("bridge outage keeps ledger/cursor; reconnect rejects duplicate/late events and terminal resurrection", async (t) => {
	const clock = fakeClock(t);
	const opts = {
		op: { operation_id: "replay", task_id: "task_1", state: "RUNNING", revision: 1 },
		task: { task_id: "task_1", state: "RUNNING", active_revision: 1 },
		idle: true, bridgeDown: false, replay: true,
		eventsByTask: { task_1: [
			{ seq: 7, event_type: "operation.output", operation_id: "replay", payload: { text: "original" } },
		] as Array<Record<string, unknown>> },
	};
	const h = harness(opts);
	const w = new OperationWatcher(h.deps as never);
	w.track("replay");
	await tick(w);
	clock.advance(60_000);
	opts.bridgeDown = true;
	await tick(w);
	let text = latest(h.widgets, "replay");
	assert.match(text, /Bridge: status unavailable; showing last known ledger only/);
	assert.match(text, /last known ledger: RUNNING/);
	assert.match(text, /60s since UI-observed output/);
	assert.match(text, /original/);
	assert.equal(h.calls.at(-1)?.params.last_seq, 7);
	assert.equal(h.sent.length, 0);
	opts.bridgeDown = false;
	opts.eventsByTask.task_1 = [
		{ seq: 7, event_type: "operation.output", operation_id: "replay", payload: { text: "old replay" } },
		{ seq: 8, event_type: "operation.output", operation_id: "replay", payload: { text: "recovered" } },
		{ seq: 8, event_type: "operation.output", operation_id: "replay", payload: { text: "duplicate" } },
	];
	await tick(w);
	assert.equal(h.calls.at(-1)?.params.last_seq, 7, "outage advanced cursor");
	assert.match(latest(h.widgets, "replay"), /recovered/);
	clock.advance(10_000);
	await tick(w);
	text = latest(h.widgets, "replay");
	assert.match(text, /10s since UI-observed output/, "duplicates refreshed observation time");
	assert.doesNotMatch(text, /old replay|duplicate|status unavailable/);
	opts.eventsByTask.task_1 = [
		{ seq: 9, event_type: "operation.completed", operation_id: "replay", payload: { state: "SUCCEEDED" } } as typeof opts.eventsByTask.task_1[number],
		{ seq: 10, event_type: "operation.output", operation_id: "replay", payload: { text: "late output" } },
	];
	await tick(w);
	assert.equal(h.sent.length, 1);
	assert.equal(latest(h.widgets, "replay"), "");
	const count = h.widgets.length;
	w.track("replay");
	await tick(w);
	assert.equal(h.widgets.length, count, "terminal widget resurrected");
	assert.equal(h.sent.length, 1, "duplicate terminal follow-up");
	const internal = w as unknown as { observations: Map<string, unknown>; lastLineByOp: Map<string, unknown> };
	assert.equal(internal.observations.size, 0);
	assert.equal(internal.lastLineByOp.size, 0);
});

test("unavailable registration is UNKNOWN and retries only until identity resolves", async (t) => {
	const clock = fakeClock(t);
	const opts = { op: { operation_id: "pending", task_id: "task_1", state: "RUNNING" }, getDown: true };
	const h = harness(opts);
	const w = new OperationWatcher(h.deps as never);
	w.track("pending");
	w.track("pending");
	await tick(w);
	assert.match(latest(h.widgets, "pending"), /last known ledger: UNKNOWN/);
	assert.match(latest(h.widgets, "pending"), /status unavailable/);
	clock.advance(2_000);
	opts.getDown = false;
	await tick(w);
	assert.match(latest(h.widgets, "pending"), /last known ledger: RUNNING/);
	assert.match(latest(h.widgets, "pending"), /UI elapsed since tracking: 2s/);
	await tick(w);
	assert.equal(h.calls.filter((c) => c.method === "pi.op.get").length, 2);
	assert.equal(h.calls.filter((c) => c.method === "pi.kernel.events").length, 2);
	assert.equal(h.sent.length, 0);
});

test("owning task and attempt isolate output; old revision terminal is archived, not followed up", async (t) => {
	fakeClock(t);
	const eventsByTask: Record<string, Array<Record<string, unknown>>> = {
		task_a: [
			{ seq: 1, event_type: "operation.output", operation_id: "b", payload: { text: "wrong owner" } },
			{ seq: 2, event_type: "operation.output", operation_id: "a", payload: { text: "old revision output" } },
		],
		task_b: [
			{ seq: 3, event_type: "operation.output", operation_id: "b", attempt_id: "wrong", payload: { text: "wrong attempt" } },
			{ seq: 4, event_type: "operation.output", operation_id: "b", attempt_id: "attempt_b", payload: { text: "own output" } },
		],
	};
	const h = harness({
		opsById: {
			a: { operation_id: "a", task_id: "task_a", state: "RUNNING", revision: 1 },
			b: { operation_id: "b", task_id: "task_b", state: "RUNNING", revision: 2, attempt_id: "attempt_b" },
		}, eventsByTask, idle: true,
		onCall: (method, params) => method === "pi.kernel.get"
			? Promise.resolve({ task: { task_id: params.task_id, state: "RUNNING", active_revision: 2 } }) : undefined,
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("a");
	w.track("b");
	await tick(w);
	assert.match(latest(h.widgets, "a"), /old revision output/);
	assert.match(latest(h.widgets, "b"), /own output/);
	assert.doesNotMatch(latest(h.widgets, "b"), /wrong owner|wrong attempt/);
	eventsByTask.task_a.push(
		{ seq: 5, event_type: "operation.completed", operation_id: "b", payload: { state: "SUCCEEDED" } },
		{ seq: 6, event_type: "operation.completed", operation_id: "a", payload: { state: "SUCCEEDED" } },
	);
	eventsByTask.task_b.push({ seq: 7, event_type: "operation.completed", operation_id: "b", payload: { state: "SUCCEEDED" } });
	await tick(w);
	assert.equal(h.sent.length, 1);
	assert.equal((h.sent[0].message as { details: { operation_id: string } }).details.operation_id, "b");
	assert.equal(latest(h.widgets, "a"), "");
	assert.equal(latest(h.widgets, "b"), "");
	assert.deepEqual(h.calls.filter((c) => c.method === "pi.kernel.events").map((c) => c.params), [
		{ task_id: "task_a", last_seq: 0 }, { task_id: "task_b", last_seq: 0 },
		{ task_id: "task_a", last_seq: 2 }, { task_id: "task_b", last_seq: 4 },
	]);
});

test("cancel request is not stop; late output/resume/completion cannot undo CANCELING; terminal proof wins race", async (t) => {
	fakeClock(t);
	const entered = deferred<void>();
	const ownership = deferred<Record<string, unknown>>();
	const events: Array<Record<string, unknown>> = [
		{ seq: 1, event_type: "operation.canceling", operation_id: "cancel", payload: { reason: "user" } },
		{ seq: 2, event_type: "operation.output", operation_id: "cancel", payload: { text: "late output" } },
		{ seq: 3, event_type: "operation.resumed", operation_id: "cancel", payload: {} },
		{ seq: 4, event_type: "operation.completed", operation_id: "cancel", payload: { state: "SUCCEEDED" } },
		{ seq: 5, event_type: "operation.progress", operation_id: "cancel", payload: { progress: { stage: "finishing" } } },
	];
	const h = harness({
		op: { operation_id: "cancel", task_id: "task_1", state: "RUNNING", revision: 1 },
		eventsByTask: { task_1: events }, idle: true,
		onCall: (method) => {
			if (method === "pi.kernel.get") { entered.resolve(); return ownership.promise; }
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("cancel");
	await tick(w);
	assert.match(latest(h.widgets, "cancel"), /last known ledger: CANCELING/);
	assert.match(latest(h.widgets, "cancel"), /Cancellation requested; process\/physical stop not confirmed/);
	assert.match(latest(h.widgets, "cancel"), /late output/);
	assert.equal(h.sent.length, 0);
	w.observeToolResult({ ok: true, status: "CANCELLED", operation: {
		operation_id: "cancel", task_id: "task_1", revision: 2, state: "CANCELLED",
	} });
	assert.match(latest(h.widgets, "cancel"), /CANCELING/, "wrong revision consumed the operation");
	events.push({ seq: 6, event_type: "operation.cancelled", operation_id: "cancel", payload: { state: "CANCELLED" } });
	const pending = tick(w);
	await entered.promise;
	w.observeToolMessage({ role: "toolResult", toolName: "process_stop", details: {
		ok: true, status: "CANCELLED", operation: { operation_id: "cancel", task_id: "task_1", revision: 1, state: "CANCELLED" },
	} });
	ownership.resolve({ task: { task_id: "task_1", state: "RUNNING", active_revision: 1 } });
	await pending;
	await tick(w);
	assert.equal(h.sent.length, 0, "consumed proof raced into a duplicate follow-up");
	assert.equal(latest(h.widgets, "cancel"), "");
	assert.ok(h.calls.every((c) => ["pi.op.get", "pi.kernel.events", "pi.kernel.get"].includes(c.method)), "UI acquired action authority");
});

test("busy terminal drains exactly once; failed/partial/wrong-owner proof cannot consume it", async (t) => {
	fakeClock(t);
	const opts = {
		op: { operation_id: "busy", task_id: "task_1", state: "RUNNING", revision: 1 },
		task: { task_id: "task_1", state: "RUNNING", active_revision: 1 }, idle: false,
		eventsByTask: { task_1: [{ seq: 1, event_type: "operation.completed", operation_id: "busy", payload: { state: "SUCCEEDED" } }] },
	};
	const h = harness(opts);
	const w = new OperationWatcher(h.deps as never);
	w.track("busy");
	await tick(w);
	const proof = { operation_id: "busy", task_id: "task_1", revision: 1, state: "SUCCEEDED" };
	w.observeToolResult({ ok: false, status: "SUCCEEDED", operation: proof });
	w.observeToolResult({ ok: true, status: "RUNNING", operation: proof });
	w.observeToolResult({ ok: true, status: "SUCCEEDED", operation: { ...proof, task_id: "wrong" } });
	w.observeToolMessage({ role: "toolResult", toolName: "process_status", isError: true, details: { ok: true, status: "SUCCEEDED", operation: proof } });
	assert.equal(h.sent.length, 0);
	assert.equal(latest(h.widgets, "busy"), "");
	opts.idle = true;
	await tick(w);
	await tick(w);
	assert.equal(h.sent.length, 1);
	assert.equal(h.calls.filter((c) => c.method === "pi.kernel.events").length, 1);
});

test("terminal owning task suppresses model turns even when the sink is idle", async (t) => {
	fakeClock(t);
	const h = harness({
		op: { operation_id: "archived", task_id: "task_1", state: "SUCCEEDED", revision: 1 },
		task: { task_id: "task_1", state: "SUCCEEDED", active_revision: 1 }, idle: true,
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("archived");
	await tick(w);
	await tick(w);
	assert.equal(h.sent.length, 0);
	assert.equal(h.calls.filter((c) => c.method === "pi.kernel.get").length, 1);
});

test("single 2s timer, per-task polling, stop cleanup and explicit-only continuation", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval", "Date"], now: 10_000 });
	// Only promise jobs are drained: no real-time sleeps or process execution.
	const drain = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
	const h = harness({
		opsById: {
			a: { operation_id: "a", task_id: "shared", state: "RUNNING" },
			b: { operation_id: "b", task_id: "shared", state: "RUNNING" },
		},
	});
	const w = new OperationWatcher(h.deps as never);
	t.after(() => w.stop());
	w.start();
	w.start();
	t.mock.timers.tick(2_000);
	await drain();
	assert.equal(h.calls.length, 0, "continuation invented external process tracking");
	w.track("a");
	w.track("b");
	for (let i = 0; i < 4; i++) { t.mock.timers.tick(2_000); await drain(); }
	assert.equal(h.calls.filter((c) => c.method === "pi.op.get").length, 2);
	assert.equal(h.calls.filter((c) => c.method === "pi.kernel.events").length, 4);
	assert.ok(h.widgets.every((v) => ["op:a", "op:b"].includes(v.key)));
	assert.equal(h.sent.length, 0);
	w.stop();
	w.stop();
	assert.equal(latest(h.widgets, "a"), "");
	assert.equal(latest(h.widgets, "b"), "");
	const calls = h.calls.length;
	const widgets = h.widgets.length;
	t.mock.timers.tick(60_000);
	await drain();
	assert.equal(h.calls.length, calls);
	assert.equal(h.widgets.length, widgets);
	w.start();
	t.mock.timers.tick(2_000);
	await drain();
	assert.equal(h.calls.filter((c) => c.method === "pi.op.get").length, 2);
	assert.equal(h.calls.filter((c) => c.method === "pi.kernel.events").length, 5);
	assert.match(latest(h.widgets, "a"), /last known ledger: RUNNING/);
	assert.match(latest(h.widgets, "a"), /not a fresh liveness observation/);
});

test("slow bridge cannot overlap polls or redraw after stop; continuation preserves cursor", async (t) => {
	const clock = fakeClock(t);
	const entered = deferred<void>();
	const feed = deferred<Record<string, unknown>>();
	let slow = true;
	const h = harness({
		op: { operation_id: "slow", task_id: "task_1", state: "RUNNING" },
		onCall: (method) => {
			if (method === "pi.kernel.events" && slow) { entered.resolve(); return feed.promise; }
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("slow");
	const pending = tick(w);
	await entered.promise;
	clock.advance(2_000);
	await tick(w);
	clock.advance(2_000);
	await tick(w);
	assert.equal(h.calls.length, 2, "overlapping polls while bridge is blocked");
	assert.match(latest(h.widgets, "slow"), /UI elapsed since tracking: 4s/);
	assert.match(latest(h.widgets, "slow"), /quiet ≥ 4s since tracking/);
	assert.match(latest(h.widgets, "slow"), /Current liveness\/completion: UNKNOWN/);
	assert.equal(h.sent.length, 0);
	w.stop();
	const count = h.widgets.length;
	feed.resolve({ events: [{ seq: 50, event_type: "operation.output", operation_id: "slow", payload: { text: "stale in-flight output" } }] });
	await pending;
	assert.equal(h.widgets.length, count);
	assert.equal(latest(h.widgets, "slow"), "");
	slow = false;
	await tick(w);
	assert.equal(h.calls.at(-1)?.params.last_seq, 0, "stopped read advanced cursor");
	assert.doesNotMatch(latest(h.widgets, "slow"), /stale in-flight output/);
	assert.equal(h.calls.filter((c) => c.method === "pi.op.get").length, 1);
});

test("stop during terminal ownership read defers notification until a later observation cycle", async (t) => {
	fakeClock(t);
	const entered = deferred<void>();
	const ownership = deferred<Record<string, unknown>>();
	let slow = true;
	const h = harness({
		op: { operation_id: "stop-race", task_id: "task_1", state: "RUNNING", revision: 1 },
		task: { task_id: "task_1", state: "RUNNING", active_revision: 1 }, idle: true,
		eventsByTask: { task_1: [{ seq: 1, event_type: "operation.completed", operation_id: "stop-race", payload: { state: "SUCCEEDED" } }] },
		onCall: (method) => {
			if (method === "pi.kernel.get" && slow) { entered.resolve(); return ownership.promise; }
		},
	});
	const w = new OperationWatcher(h.deps as never);
	w.track("stop-race");
	const pending = tick(w);
	await entered.promise;
	w.stop();
	slow = false;
	ownership.resolve({ task: { task_id: "task_1", state: "RUNNING", active_revision: 1 } });
	await pending;
	assert.equal(h.sent.length, 0);
	await tick(w);
	await tick(w);
	assert.equal(h.sent.length, 1);
	assert.equal(latest(h.widgets, "stop-race"), "");
});
