/** WP-1 红测试（0823 审计 §三.P0-3/§四.WP-1）：终态一致性。
 *
 * 红测试先行——修复前必须红：
 * 1. Operation 终态通知在"owning task 已进入终态"时不得触发模型
 *    回合（只更新账本+TUI）——当前 OperationWatcher 无条件
 *    sendMessage(triggerTurn)；
 * 2. owning task 活跃时 followUp 照常（回归钉住）；
 * 3. 旧 revision 的迟到终态只能存档——不触发回合。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

function makeWatcher(opts: {
	taskState: string;
	revisionMatch?: boolean;
	sent: Array<{ content: string; options: unknown }>;
	notices: string[];
}) {
	return import("../src/native/operation-watcher.js").then(({ OperationWatcher }) => {
		const calls: Array<{ method: string; params: unknown }> = [];
		const watcher = new OperationWatcher({
			call: async (method: string, params: unknown) => {
				calls.push({ method, params });
				if (method === "pi.op.get") {
					return {
						ok: true,
						operation: {
							operation_id: "op_1",
							state: "SUCCEEDED",
							task_id: "task_1",
							revision: 1,
						},
					};
				}
				if (method === "pi.kernel.get") {
					// owning task 状态（终态 SUCCEEDED 或活跃 RUNNING）。
					return {
						ok: true,
						task: {
							task_id: "task_1",
							state: opts.taskState,
							active_revision: opts.revisionMatch === false ? 2 : 1,
						},
					};
				}
				return { ok: true };
			},
			sink: () => ({
				isIdle: true,
				api: {
					sendMessage: (message: { content: string }, options: unknown) => {
						opts.sent.push({ content: message.content, options });
					},
				},
				notify: (text: string) => opts.notices.push(text),
			}),
		});
		return watcher;
	});
}

test("WP-1: owning task 终态 → operation 终态通知不触发模型回合", async () => {
	const sent: Array<{ content: string; options: unknown }> = [];
	const notices: string[] = [];
	const watcher = await makeWatcher({ taskState: "SUCCEEDED", sent, notices });
	watcher.track("op_1");
	await (watcher as unknown as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 0,
		`task 终态后仍触发模型回合: ${JSON.stringify(sent)}`);
	// TUI 通知保留（账本/TUI 更新是合法的）。
	assert.ok(notices.length >= 1);
});

test("WP-1: owning task 活跃 → operation 终态正常 followUp（回归）", async () => {
	const sent: Array<{ content: string; options: unknown }> = [];
	const notices: string[] = [];
	const watcher = await makeWatcher({ taskState: "RUNNING", sent, notices });
	watcher.track("op_1");
	await (watcher as unknown as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 1, "活跃任务的 operation 终态必须照常通知模型");
});

test("WP-1: 旧 revision 的迟到终态只存档（不触发回合）", async () => {
	const sent: Array<{ content: string; options: unknown }> = [];
	const notices: string[] = [];
	// operation 属 revision 1，task 活跃 revision 已是 2。
	const watcher = await makeWatcher({
		taskState: "RUNNING", revisionMatch: false, sent, notices,
	});
	watcher.track("op_1");
	await (watcher as unknown as { tick(): Promise<void> }).tick();
	assert.equal(sent.length, 0,
		"旧 revision 的迟到 operation 结果触发了模型回合");
});
