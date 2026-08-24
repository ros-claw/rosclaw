/** P0-B 红测试（0824 总纲 §19.P0-B）：Terminal Fence 与因果续接。
 *
 * 红测试先行——终态栅栏不存在时必须红。
 *
 * 验收（文档原文）：任务完成后延迟 operation、provider retry、
 * duplicate callback——60 秒内模型请求数增量必须为 0；文件、
 * artifact、revision 不得变化。
 *
 * 1. task 已终态（SUCCEEDED/BLOCKED/FAILED/CANCELLED）→ TurnGuard
 *    不得注入 follow-up（终态后再触发模型回合=幽灵执行）；
 * 2. 合法 nudge 带稳定 causation_id（同一 task+revision 因果唯一，
 *    供内核幂等去重）；
 * 3. 终态后 nudged 集合不再增长（revision 不变）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

interface SentMessage {
	customType: string;
	content: string;
	details: Record<string, unknown>;
}

function makeGuard(task: Record<string, unknown> | null) {
	const sent: SentMessage[] = [];
	const importGuard = import("../src/native/turn-guard.js");
	return importGuard.then(({ TurnGuard }) => ({
		sent,
		guard: new TurnGuard({
			call: async () => ({ task }),
			missionId: () => "mis_1",
			sessionRef: () => "s1",
			sink: () => ({
				isIdle: true,
				api: {
					sendMessage: (msg: SentMessage) => {
						sent.push(msg);
					},
				},
			}),
		}),
	}));
}

describe("P0-B Terminal Fence", () => {
	it("task SUCCEEDED → 不注入 follow-up（终态后模型请求=0）", async () => {
		const { guard, sent } = await makeGuard({
			task_id: "task_1", active_revision: 1, state: "SUCCEEDED",
		});
		guard.noteTool("write");
		await guard.onTurnEnd();
		assert.equal(sent.length, 0, "终态任务被 TurnGuard 触发模型回合");
	});

	it("task FAILED/BLOCKED/CANCELLED → 同样不注入", async () => {
		for (const state of ["FAILED", "BLOCKED", "CANCELLED"]) {
			const { guard, sent } = await makeGuard({
				task_id: "task_1", active_revision: 1, state,
			});
			guard.noteTool("bash");
			await guard.onTurnEnd();
			assert.equal(sent.length, 0, `${state} 终态被触发`);
		}
	});

	it("RUNNING 合法 nudge 带稳定 causation_id（因果唯一）", async () => {
		const { guard, sent } = await makeGuard({
			task_id: "task_1", active_revision: 3, state: "RUNNING",
		});
		guard.noteTool("write");
		await guard.onTurnEnd();
		assert.equal(sent.length, 1);
		const causation = String(sent[0].details.causation_id ?? "");
		assert.ok(causation.length > 0, "缺 causation_id");
		assert.ok(causation.includes("task_1"), "causation 不含 task 身份");
		assert.ok(causation.includes("r3") || causation.includes("3"),
			"causation 不含 revision");
	});

	it("task 查询失败 → 不触发（fail closed 不赌博）", async () => {
		const { TurnGuard } = await import("../src/native/turn-guard.js");
		const sent: SentMessage[] = [];
		const guard = new TurnGuard({
			call: async () => {
				throw new Error("bridge down");
			},
			missionId: () => "mis_1",
			sessionRef: () => "s1",
			sink: () => ({
				isIdle: true,
				api: { sendMessage: (m: SentMessage) => sent.push(m) },
			}),
		});
		guard.noteTool("write");
		await guard.onTurnEnd();
		assert.equal(sent.length, 0);
	});
});
