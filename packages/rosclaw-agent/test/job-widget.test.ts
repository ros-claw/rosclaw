/** 十一审 PR-C 红测试：TUI 实时 Job 卡。
 *
 * 1. RUNNING job 渲染 phase/工具/耗时/token（来自事件流，不是伪造进度）；
 * 2. liveness 驱动刷新（phase 更新）；stall_warning 标记；
 * 3. 终态 ✓/✗ 渲染；
 * 4. 内容不变不重绘（idle CPU 红线）；
 * 5. renderJobLog 过滤 liveness、渲染工具事件。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function makeWidget(
	orders: Array<Record<string, unknown>>,
	eventsByOrder: Record<string, Array<Record<string, unknown>>>,
) {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { JobsWidget } = await import("../src/workers/job-widget.js");
	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: "mis_1",
		contextRevision: 1,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "FRESH",
		leaseState: "ACTIVE",
		actionsAllowed: true,
	});
	const widgetCalls: Array<string[] | undefined> = [];
	const widget = new JobsWidget({
		active: active as never,
		center: {
			call: async (method: string, params?: unknown) => {
				if (method === "pi.worker.status") return { ok: true, orders };
				if (method === "pi.worker.events") {
					const p = params as { work_order_id: string; after_seq?: number };
					const all = eventsByOrder[p.work_order_id] ?? [];
					const after = p.after_seq ?? 0;
					return { ok: true, events: all.filter((e) => (e.seq as number) > after) };
				}
				return { ok: true };
			},
		} as never,
		setWidget: (lines) => widgetCalls.push(lines),
	});
	return { widget, widgetCalls };
}

test("RUNNING job 卡渲染工具/耗时/token", async () => {
	const { widget, widgetCalls } = await makeWidget(
		[
			{
				work_order_id: "wo_abc123456789",
				assigned_to: "worker:rosclaw:pi",
				status: "RUNNING",
				goal: "实现 MuJoCo rollout",
			},
		],
		{
			wo_abc123456789: [
				{ seq: 1, kind: "attempt_started" },
				{ seq: 2, kind: "model_started" },
				{ seq: 3, kind: "tool_started", tool: "bash" },
				{ seq: 4, kind: "liveness", phase: "RUNNING_TOOL" },
				{ seq: 5, kind: "usage", input_tokens: 38400, output_tokens: 2100, turns: 3 },
			],
		},
	);
	await widget.tick();
	const lines = widgetCalls.at(-1);
	assert.ok(lines, "widget 未渲染");
	const text = lines!.join("\n");
	assert.match(text, /Pi Worker/);
	assert.match(text, /实现 MuJoCo rollout/);
	assert.match(text, /bash/);
	assert.match(text, /41k tok/);
	assert.match(text, /\/job wo_abc123456789/);
});

test("stall_warning 标黄 + 终态 ✓ 渲染", async () => {
	const { widget, widgetCalls } = await makeWidget(
		[
			{ work_order_id: "wo_stall000001", assigned_to: "worker:rosclaw:pi", status: "RUNNING", goal: "长推理" },
			{ work_order_id: "wo_done0000001", assigned_to: "worker:rosclaw:pi", status: "ACCEPTED", goal: "已完成任务" },
		],
		{
			wo_stall000001: [{ seq: 1, kind: "stall_warning" }],
			wo_done0000001: [{ seq: 1, kind: "attempt_finished" }],
		},
	);
	await widget.tick();
	const text = widgetCalls.at(-1)!.join("\n");
	assert.match(text, /静默>90s（仍存活）/);
	assert.match(text, /✓ Pi Worker · 已完成任务/);
});

test("内容不变不重绘（idle CPU 红线）", async () => {
	const { widget, widgetCalls } = await makeWidget(
		[{ work_order_id: "wo_idle0000001", assigned_to: "worker:rosclaw:pi", status: "RUNNING", goal: "x" }],
		{ wo_idle0000001: [] },
	);
	await widget.tick();
	const first = widgetCalls.length;
	await widget.tick();
	// spinner 帧变化会改首行 icon——只有 spinner 行的变化允许重绘。
	const second = widgetCalls.length;
	assert.ok(second - first <= 1, `无谓重绘: ${first} → ${second}`);
});

test("renderJobLog 过滤 liveness、渲染工具与完成", async () => {
	const { renderJobLog } = await import("../src/workers/job-widget.js");
	const text = renderJobLog(
		[
			{ seq: 1, kind: "attempt_started" },
			{ seq: 2, kind: "liveness", phase: "RUNNING_MODEL" },
			{ seq: 3, kind: "tool_started", tool: "write" },
			{ seq: 4, kind: "tool_finished", tool: "write", is_error: false },
			{ seq: 5, kind: "attempt_finished" },
		] as never,
		"wo_x",
	);
	assert.ok(!text.includes("liveness"));
	assert.match(text, /▶ tool write/);
	assert.match(text, /✓ tool write/);
	assert.match(text, /■ finished/);
});
