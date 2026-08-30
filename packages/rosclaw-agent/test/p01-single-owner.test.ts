/** 0827 体验审计 P0-1/2/3 红测试：单 Owner + 单终态发布者。
 *
 * 0827 实证事故（双控制者）：自动路由认领输入后模型仍收到同一指令
 * 并手工执行，终态由"模型回合"和"watcher followUp"两个发布者给出
 * 相反结论。
 *
 * 闭环断言：
 * 1. TurnDisposition：owner=TASK_ROUTER → suppressModelTurn=true
 *    （input hook 返回 handled，模型整回合不出现）；
 * 2. 任务终态由 watcher 以确定性文本直接呈现（display:true,
 *    triggerTurn:false）——不再 followUp 唤醒 Agent；
 * 3. 终态文本是 TaskOutcome 的确定性呈现：PASS 带交付打开命令；
 *    PARTIAL/MISSING 诚实标注限制，不出现完成宣称；
 * 4. 重放不重复发布。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("TurnDisposition：TASK_ROUTER 认领 → suppress model turn", async () => {
	const { suppressModelTurn } = await import("../src/native/turn-disposition.js");
	assert.equal(
		suppressModelTurn({
			turn_disposition: {
				input_id: "in_1",
				owner: "TASK_ROUTER",
				task_id: "task_1",
				suppress_model_turn: true,
			},
		}),
		true,
	);
	assert.equal(
		suppressModelTurn({
			turn_disposition: {
				input_id: "in_2",
				owner: "PI_CONVERSATION",
				task_id: "",
				suppress_model_turn: false,
			},
		}),
		false,
	);
	assert.equal(suppressModelTurn({}), false);
	// 旧 daemon 无 disposition 字段时回落 auto_task（版本倾斜期不
	// 漏 suppress——双控制者防线不能依赖单点字段）。
	assert.equal(
		suppressModelTurn({ auto_task: { task_id: "task_9" } }),
		true,
	);
});

test("终态呈现器：PASS 带交付命令；PARTIAL 诚实无完成宣称", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const pass = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		artifact_refs: [
			{ artifact_id: "art_g", open_command: "rosclaw artifact open art_g" },
			{ artifact_id: "art_m", open_command: "rosclaw artifact open art_m" },
		],
	});
	assert.match(pass, /任务完成/);
	assert.match(pass, /验收 PASS/);
	assert.match(pass, /rosclaw artifact open art_g/);
	assert.match(pass, /rosclaw artifact open art_m/);
	const partial = renderTerminalReply({
		verification: "PARTIAL",
		delivery: "MISSING",
		artifact_refs: [],
	});
	assert.doesNotMatch(partial, /任务完成/);
	assert.match(partial, /未完全达成|未完成/);
	assert.match(partial, /PARTIAL/);
	assert.match(partial, /MISSING/);
	// P0-4：投影退化如实告知（DELIVERED + DEGRADED 不假装正常）。
	const degraded = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		workspace_projection: "DEGRADED",
		artifact_refs: [
			{ artifact_id: "art_g", open_command: "rosclaw artifact open art_g" },
		],
	});
	assert.match(degraded, /任务完成/);
	assert.match(degraded, /投影退化/);
});

test("watcher 终态：确定性呈现一次（display, 无 triggerTurn）+ 重放不重发", async () => {
	const { OperationWatcher } = await import("../src/native/operation-watcher.js");
	const events = [
		{ seq: 1, event_type: "plan.node_completed", payload: { node_id: "make_path" } },
		{ seq: 2, event_type: "verification.completed", payload: { status: "PASS" } },
	];
	const sent: Array<{ content: string; display: boolean; options: Record<string, unknown> }> = [];
	const watcher = new OperationWatcher({
		call: async (method: string, params: Record<string, unknown>) => {
			if (method === "pi.kernel.events") {
				const after = Number((params as { last_seq?: number }).last_seq ?? 0);
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
				sendMessage: (
					message: { content: string; display: boolean },
					options: Record<string, unknown>,
				) => {
					sent.push({ content: message.content, display: message.display, options });
				},
			},
			isIdle: true,
			setWidget: () => undefined,
		}),
	} as never);
	watcher.trackTask("task_p01");
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 1, `终态呈现应恰好一次（实际 ${sent.length}）`);
	// P0-3：终态呈现绝不唤醒模型（双控制者根治——Coordinator/
	// Presenter 是唯一终态发布者）。
	assert.equal(sent[0].options.triggerTurn ?? false, false,
		"终态呈现不得触发模型回合（followUp triggerTurn 是双控制者根因）");
	assert.notEqual(sent[0].options.deliverAs, "followUp",
		"终态呈现不得以 followUp 注入模型");
	assert.equal(sent[0].display, true, "终态回复必须用户可见");
	assert.match(sent[0].content, /任务完成：验收 PASS/);
	assert.match(sent[0].content, /rosclaw artifact open art_g/);
	// 重放不重发（不重不漏）。
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 1, "重放重复发布了终态");
});
