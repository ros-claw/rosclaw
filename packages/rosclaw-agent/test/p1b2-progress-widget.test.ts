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
import test from "node:test";

import { OperationWatcher } from "../src/native/operation-watcher.js";

interface WidgetCall { key: string; lines: string[] | undefined }

function harness(opts: {
	op?: Record<string, unknown>;
	eventsByTask?: Record<string, Array<Record<string, unknown>>>;
	task?: Record<string, unknown> | null;
	idle?: boolean;
}) {
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	const widgets: WidgetCall[] = [];
	const sent: Array<{ message: unknown; options: unknown }> = [];
	const seqs = new Map<string, number>();
	const deps = {
		call: async (method: string, params: Record<string, unknown>) => {
			calls.push({ method, params });
			if (method === "pi.op.get") return { operation: opts.op ?? {} };
			if (method === "pi.kernel.get") return { task: opts.task ?? null };
			if (method === "pi.kernel.events") {
				const taskId = String(params.task_id);
				const since = Number(params.last_seq ?? 0);
				seqs.set(taskId, since);
				const events = (opts.eventsByTask?.[taskId] ?? []).filter(
					(e) => Number(e.seq) > since,
				);
				return { events };
			}
			throw new Error(`unexpected ${method}`);
		},
		sink: () => ({
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
