/** 十审 Gate W2 红测试：Worker 完成推送。
 *
 * 1. 终态订单经 custom message（rosclaw.worker.result）注入，
 *    triggerTurn + 不冒充用户输入；
 * 2. 幂等：同一 work_order_id 只投递一次（跨 tick / 账本重载）；
 * 3. 非终态不投递；未绑定 mission 不轮询；
 * 4. 账本持久化——重启（新实例）后不重复投递。
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeDeps(
	orders: Array<Record<string, unknown>>,
	sent: Array<{ message: Record<string, unknown>; options: Record<string, unknown> }>,
	home?: string,
) {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { WorkerCompletionWatcher } = await import("../src/workers/completion-watch.js");
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
	const center = {
		call: async (method: string) => {
			assert.equal(method, "pi.worker.status");
			return { ok: true, orders };
		},
	};
	const rosclawHome = home ?? mkdtempSync(join(tmpdir(), "rh-w2-"));
	const make = () =>
		new WorkerCompletionWatcher({
			rosclawHome,
			active: active as never,
			center: center as never,
			sink: () => ({
				api: {
					sendMessage: (message, options) => {
						sent.push({ message: message as Record<string, unknown>, options: options as Record<string, unknown> });
					},
				},
				isIdle: true,
			}),
		});
	return { make, rosclawHome };
}

test("终态订单注入 custom message（triggerTurn + nextTurn）", async () => {
	const sent: Array<{ message: Record<string, unknown>; options: Record<string, unknown> }> = [];
	const { make } = await makeDeps(
		[
			{
				work_order_id: "wo_done1",
				assigned_to: "worker:rosclaw:pi",
				status: "ACCEPTED",
				accepted: true,
				summary: "分析完成",
			},
			{ work_order_id: "wo_run1", status: "RUNNING" },
		],
		sent,
	);
	const watcher = make();
	await watcher.tick();
	assert.equal(sent.length, 1);
	assert.equal(sent[0].message.customType, "rosclaw.worker.result");
	assert.equal(sent[0].options.triggerTurn, true);
	// idle：不带 deliverAs 才真触发回合（nextTurn 会静默排队）。
	assert.equal(sent[0].options.triggerTurn, true);
	assert.equal(sent[0].options.deliverAs, undefined);
	assert.match(String(sent[0].message.content), /wo_done1/);
	assert.match(String(sent[0].message.content), /untrusted/);
	const details = sent[0].message.details as { workOrderId?: string };
	assert.equal(details.workOrderId, "wo_done1");
});

test("幂等：跨 tick 与账本重载不重复投递", async () => {
	const sent: Array<{ message: Record<string, unknown>; options: Record<string, unknown> }> = [];
	const orders = [
		{ work_order_id: "wo_idem1", status: "ACCEPTED", accepted: true, summary: "x" },
	];
	const home = mkdtempSync(join(tmpdir(), "rh-w2-"));
	const { make } = await makeDeps(orders, sent, home);
	const watcher = make();
	await watcher.tick();
	await watcher.tick();
	assert.equal(sent.length, 1);
	// 模拟重启：新实例读同一账本——不得重投。
	const watcher2 = make();
	await watcher2.tick();
	assert.equal(sent.length, 1);
});

test("未绑定 mission 不轮询", async () => {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { WorkerCompletionWatcher } = await import("../src/workers/completion-watch.js");
	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: undefined,
		contextRevision: 0,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "LOADING",
		leaseState: "NONE",
		actionsAllowed: false,
	});
	let calls = 0;
	const watcher = new WorkerCompletionWatcher({
		rosclawHome: mkdtempSync(join(tmpdir(), "rh-w2-")),
		active: active as never,
		center: { call: async () => { calls += 1; return { ok: true, orders: [] }; } } as never,
		sink: () => undefined,
	});
	await watcher.tick();
	assert.equal(calls, 0);
});
