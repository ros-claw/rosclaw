/** PR-H9：F2 Task Panel 组件测试（kernel 背板重写版）。
 *
 * 面板只读：↑↓ 选卡、Tab 切 Activity/Artifacts、r 刷新、Esc 关闭；
 * 渲染复用 task-activity（与 /activity /artifacts 同一映射）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
	TasksCenterComponent,
	taskCardLine,
	type TasksCenterDeps,
} from "../src/workers/tasks-center.js";

function makeDeps(tasks: Array<Record<string, unknown>>): TasksCenterDeps & {
	closed: boolean;
	eventCalls: string[];
} {
	const state = { closed: false, eventCalls: [] as string[] };
	return {
		get closed() { return state.closed; },
		eventCalls: state.eventCalls,
		fetchTasks: async () => tasks,
		fetchEvents: async (taskId: string) => {
			state.eventCalls.push(taskId);
			return [
				{ seq: 1, event_type: "task.started", payload: { goal: "画五角星" } },
				{ seq: 2, event_type: "task.terminal", payload: { state: "SUCCEEDED" } },
			];
		},
		fetchArtifacts: async () => [
			{ artifact_id: "art_1", path: "/ws/star.gif", media_type: "image/gif", sha256: "deadbeefcafe", size_bytes: 2048 },
		],
		notify: () => undefined,
		onClose: () => { state.closed = true; },
	};
}

const TASKS = [
	{ task_id: "task_aaa", root_goal: "画五角星", state: "SUCCEEDED", active_revision: 1 },
	{ task_id: "task_bbb", root_goal: "写报告", state: "ACTIVE", active_revision: 2 },
];

describe("H9 Task Panel（kernel 背板）", () => {
	it("taskCardLine 渲染状态图标与 revision", () => {
		assert.match(taskCardLine(TASKS[0], ">"), /> ✓ 画五角星 · SUCCEEDED/);
		assert.match(taskCardLine(TASKS[1]), /● 写报告 · ACTIVE · r2/);
	});

	it("渲染卡列表 + Activity 内容（kernel 事件）", async () => {
		const deps = makeDeps(TASKS);
		const panel = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 30));
		const out = panel.render(80).join("\n");
		assert.match(out, /画五角星/);
		assert.match(out, /写报告/);
		assert.match(out, /任务开始/);
		assert.match(out, /终态：SUCCEEDED/);
		panel.dispose();
	});

	it("Tab 切到 Artifacts 显示产物账本", async () => {
		const deps = makeDeps(TASKS);
		const panel = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 30));
		panel.handleInput("\t");
		await new Promise((r) => setTimeout(r, 30));
		const out = panel.render(80).join("\n");
		assert.match(out, /\[Artifacts\]/);
		assert.match(out, /star\.gif/);
		panel.dispose();
	});

	it("Esc 关闭并停止轮询", async () => {
		const deps = makeDeps(TASKS);
		const panel = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 20));
		panel.handleInput("\x1b");
		assert.equal(deps.closed, true);
	});

	it("空任务诚实空态", async () => {
		const deps = makeDeps([]);
		const panel = new TasksCenterComponent(deps);
		await new Promise((r) => setTimeout(r, 30));
		assert.match(panel.render(80).join("\n"), /无任务/);
		panel.dispose();
	});
});
