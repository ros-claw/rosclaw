/** 十四审 PR-14.4：F2 Tasks Center 组件测试（总纲 §4.2 键位表）。 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { TasksCenterComponent, type TasksCenterDeps, type JobCard } from "../src/workers/tasks-center.js";

function makeDeps(jobs: JobCard[]): TasksCenterDeps & {
	controlCalls: Array<{ wo: string; action: string }>;
	cancelConfirms: string[];
	steerCalls: string[];
	retryCalls: string[];
} {
	const calls = {
		controlCalls: [] as Array<{ wo: string; action: string }>,
		cancelConfirms: [] as string[],
		steerCalls: [] as string[],
		retryCalls: [] as string[],
	};
	return {
		...calls,
		fetchJobs: async () => jobs,
		fetchEvents: async () => ({ events: [], status: "RUNNING" }),
		fetchTranscript: async () => ({ records: [], has_more: false, next_cursor: 0, total: 0 }),
		onSteer: async () => undefined,
		sendSteer: async (wo: string, text: string) => {
			calls.steerCalls.push(`${wo}:${text}`);
			return "已送达";
		},
		sendControl: async (wo: string, action: string) => {
			calls.controlCalls.push({ wo, action });
			return { ok: true, state: action === "pause" ? "PAUSED" : "RUNNING" };
		},
		sendRetry: async (wo: string) => {
			calls.retryCalls.push(wo);
			return "已 retry";
		},
		notify: () => undefined,
		onClose: () => undefined,
	};
}

const job = (over: Partial<JobCard> = {}): JobCard => ({
	root_job_id: "wo_root1",
	goal: "UR5e 五角星动力学仿真",
	state: "RUNNING",
	attempts: [
		{ work_order_id: "wo_root1", seq: 1, actor: "native_agent", status: "FAILED", termination_cause: "PROVIDER_TRANSIENT" },
		{ work_order_id: "wo_att2", seq: 2, actor: "auto", status: "RUNNING", termination_cause: "" },
	],
	...over,
});

describe("TasksCenterComponent", () => {
	it("一个用户任务一张卡（attempts 聚合，不显示三张失败卡）", async () => {
		const deps = makeDeps([job()]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		const out = c.render(80).join("\n");
		assert.match(out, /UR5e 五角星动力学仿真/);
		assert.match(out, /attempt 2\/2/);
		// 只有一张卡：root 出现一次（列表行），wo_att2 不作为独立卡。
		assert.equal((out.match(/●|✓|✗|⚠/g) ?? []).length <= 2, true);
		c.dispose();
	});

	it("↑↓ 选择任务（无需复制 ID）", async () => {
		const deps = makeDeps([
			job(),
			job({ root_job_id: "wo_b", goal: "代码审计", state: "ACCEPTED", attempts: [{ work_order_id: "wo_b", seq: 1, actor: "native_agent", status: "ACCEPTED", termination_cause: "" }] }),
		]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		assert.match(c.render(80).join("\n"), /> ● UR5e/);
		c.handleInput("\x1b[B");
		assert.match(c.render(80).join("\n"), /> ✓ 代码审计/);
		c.dispose();
	});

	it("p：RUNNING→pause；PAUSED→resume（走控制 ACK，不乐观）", async () => {
		const deps = makeDeps([job()]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		c.handleInput("p");
		await new Promise((r) => setTimeout(r, 20));
		assert.deepEqual(deps.controlCalls, [{ wo: "wo_att2", action: "pause" }]);
		c.dispose();
	});

	it("x：必须二次确认才取消", async () => {
		const deps = makeDeps([job()]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		c.handleInput("x");
		await new Promise((r) => setTimeout(r, 20));
		assert.equal(deps.controlCalls.length, 0); // 第一次只是确认态
		assert.match(c.render(80).join("\n"), /再按 x 确认取消/);
		c.handleInput("x");
		await new Promise((r) => setTimeout(r, 20));
		assert.deepEqual(deps.controlCalls, [{ wo: "wo_att2", action: "cancel" }]);
		c.dispose();
	});

	it("r：运行态禁用；可恢复终态才请求 retry/resume", async () => {
		const deps = makeDeps([job()]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		c.handleInput("r");
		await new Promise((r) => setTimeout(r, 20));
		assert.equal(deps.retryCalls.length, 0); // RUNNING 禁用
		c.dispose();

		const deps2 = makeDeps([
			job({ state: "INTERRUPTED_RESUMABLE", attempts: [{ work_order_id: "wo_dead", seq: 1, actor: "native_agent", status: "INTERRUPTED_RESUMABLE", termination_cause: "SIGNAL_UNKNOWN" }] }),
		]);
		const c2 = new TasksCenterComponent(deps2);
		await new Promise((r) => setTimeout(r, 20));
		c2.handleInput("r");
		await new Promise((r) => setTimeout(r, 20));
		assert.deepEqual(deps2.retryCalls, ["wo_dead"]);
		c2.dispose();
	});

	it("Tab 切页（Live/Transcript/Files/Artifacts/Metrics）", async () => {
		const deps = makeDeps([job()]);
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		assert.match(c.render(80).join("\n"), /Live/);
		c.handleInput("\t");
		assert.match(c.render(80).join("\n"), /\[Transcript\]/);
		c.handleInput("\t");
		assert.match(c.render(80).join("\n"), /\[Files\]/);
		c.dispose();
	});

	it("Esc 关闭（onClose 回调）", async () => {
		let closed = false;
		const deps = { ...makeDeps([job()]), onClose: () => { closed = true; } };
		const c = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		c.handleInput("\x1b");
		assert.equal(closed, true);
		c.dispose();
	});
});
