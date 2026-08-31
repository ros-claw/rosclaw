/** P0-3 补漏红测试（0827 真实 K3 复验实证）：终态呈现空闲门控。
 *
 * 实证：腿2（疑问句→真实模型回答）还在流式输出时，确定性链修订
 * 重跑已终态——watcher 此刻 sendMessage(triggerTurn:false) 被 pi
 * steer 进正在运行的回合：终态回复从屏幕消失（内核 SUCCEEDED 但
 * 用户看不到），且违背"终态后零模型回合"。
 *
 * 闭环断言：Agent 流式中 → 不发布、不标 delivered、保持 tracked；
 * 空闲后下个 tick 独立呈现（display:true triggerTurn:false）。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("终态呈现空闲门控：流式中延迟，空闲后呈现", async () => {
	const { OperationWatcher } = await import("../src/native/operation-watcher.js");
	const events = [
		{ seq: 1, event_type: "task.terminal", payload: { status: "PASS" } },
	];
	const sent: string[] = [];
	let idle = false;
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
						lifecycle: "COMPLETED", verification: "PASS",
						delivery: "DELIVERED", artifact_refs: [],
					},
				};
			}
			return { ok: true };
		},
		sink: () => ({
			api: {
				sendMessage: (message: { content: string }) => {
					sent.push(message.content);
				},
			},
			isIdle: idle,
			setWidget: () => undefined,
		}),
	} as never);
	watcher.trackTask("task_gate");
	// 流式中：终态事件到达但不发布（不能被 steer 吞掉）。
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 0, "流式中竟发布终态（会被 steer 吞掉）");
	// 空闲后：下个 tick 独立呈现。
	idle = true;
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 1, "空闲后终态未呈现");
	assert.match(sent[0], /任务完成/);
});

test("中间态 verification.completed(FAIL) 不呈现；task.terminal 才发布", async () => {
	// 0827 真实 K3 复验实证：修订重跑中 verify_artifacts 抓到瞬态
	// 空文件（同 trace 目录原位重写）→ REPAIR_REQUIRED 的
	// verification.completed 被当成终态呈现——屏幕 FAIL 与内核
	// SUCCEEDED 矛盾（双真相回归）。只有 task.terminal 是发布点。
	const { OperationWatcher } = await import("../src/native/operation-watcher.js");
	const events = [
		{ seq: 1, event_type: "verification.completed", payload: { status: "FAIL" } },
		{ seq: 2, event_type: "task.terminal", payload: { state: "SUCCEEDED" } },
	];
	const sent: string[] = [];
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
						lifecycle: "COMPLETED", verification: "PASS",
						delivery: "DELIVERED", artifact_refs: [],
					},
				};
			}
			return { ok: true };
		},
		sink: () => ({
			api: {
				sendMessage: (message: { content: string }) => {
					sent.push(message.content);
				},
			},
			isIdle: true,
			setWidget: () => undefined,
		}),
	} as never);
	watcher.trackTask("task_midstate");
	await (watcher as never as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 1, `应只在 task.terminal 呈现一次（实际 ${sent.length}）`);
	assert.match(sent[0], /任务完成/, `中间态 FAIL 竟被呈现：${sent[0]}`);
});
