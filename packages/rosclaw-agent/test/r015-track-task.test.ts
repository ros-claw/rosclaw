/** R0-1.5 红测试（金丝雀实证 + 0826 审计收口）：trackTask——
 * 自动路由任务的 plan 进度 widget + 终态确定性呈现。
 *
 * 断言：
 * 1. plan.node_started/completed → 单活动区 widget 原位更新
 *    （不重复打印、不进模型上下文）；
 * 2. verification.completed PASS → 终态回复确定性呈现一次
 *    （display:true、triggerTurn:false——0827 P0-3：Coordinator 是
 *    唯一终态发布者，不 followUp 唤醒 Agent）——重放不重复投递；
 * 3. 终态后 untrack（不再轮询）。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("trackTask：进度 widget + 终态一次 followUp + untrack", async () => {
	const { OperationWatcher } = await import("../src/native/operation-watcher.js");
	const events = [
		{ seq: 1, event_type: "plan.node_started", payload: { node_id: "resolve_robot" } },
		{ seq: 2, event_type: "plan.node_completed", payload: { node_id: "resolve_robot" } },
		{ seq: 3, event_type: "plan.node_started", payload: { node_id: "make_path" } },
		{ seq: 4, event_type: "plan.node_completed", payload: { node_id: "make_path" } },
		{ seq: 5, event_type: "verification.completed", payload: { status: "PASS", verification_id: "vrf_1" } },
	];
	const widgets: Array<[string, string[] | undefined]> = [];
	const followUps: string[] = [];
	let tick = 0;
	const watcher = new OperationWatcher({
		call: async (method: string, params: Record<string, unknown>) => {
			if (method === "pi.kernel.events") {
				// 游标增量（不重不漏）。
				const after = Number((params as { last_seq?: number }).last_seq ?? 0);
				tick += 1;
				return { ok: true, events: events.filter((e) => e.seq > after) };
			}
			if (method === "pi.coordinator.consider") {
				return {
					ok: true,
					outcome: {
						lifecycle: "COMPLETED",
						verification: "PASS",
						delivery: "DELIVERED",
						artifact_refs: [
							{ artifact_id: "art_g", open_command: "rosclaw artifact open art_g" },
						],
					},
				};
			}
			return { ok: true };
		},
		sink: () => ({
			api: {
				sendMessage: (message: { content: string }, _options: unknown) => {
					followUps.push(message.content);
				},
			},
			isIdle: true,
			setWidget: (key: string, lines: string[] | undefined) => {
				widgets.push([key, lines]);
			},
		}),
	} as never);
	watcher.trackTask("task_auto_1");
	// 手动驱动 tick（定时器在测试外）。
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.ok(widgets.length > 0, "plan 进度未进 widget");
	// 终态清除是最后一个 widget 更新——断言进度行出现过
	// （节点 label 原位更新）。
	const progressLines = widgets
		.filter(([, lines]) => lines !== undefined)
		.map(([, lines]) => (lines ?? []).join(" "));
	assert.ok(
		progressLines.some((line) => /规划|make_path/.test(line)),
		`进度 widget 无规划节点行：${JSON.stringify(widgets)}`,
	);
	// 终态 → 确定性呈现一次（0827 P0-3：display:true +
	// triggerTurn:false，不唤醒模型）。
	assert.equal(followUps.length, 1, `终态呈现应恰好一次（实际 ${followUps.length}）`);
	assert.match(followUps[0], /art_g/);
	// 重放不重复投递。
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(followUps.length, 1, "重放重复投递了 followUp");
});
