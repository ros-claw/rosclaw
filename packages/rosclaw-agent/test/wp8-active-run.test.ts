/** WP-8 红测试（0823 审计 §四.WP-8）：active_run 进具身上下文。
 *
 * 模型每轮必须知道当前任务的运行目录与四区纪律——否则它只会
 * 把交付物写进项目源码树（0823 实测）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { renderTrustedContext, type ContextFetchResult } from "../src/extension/context-injection.js";

function baseResult(): ContextFetchResult {
	return {
		stale: false,
		note: "fresh",
		envelope: {
			schema_version: "rosclaw.embodied_context.v1",
			mission_id: "mis_1",
			context_revision: 3,
			generated_at: "2026-08-23T00:00:00Z",
			expires_at: "2026-08-23T01:00:00Z",
			hash: "sha256:x",
			body: { body_id: "sim/ur5e", effective_body_hash: "sha256:y", summary: "UR5e" },
			safety: { mode: "SIMULATION" },
			pending_approvals: [],
		} as never,
	};
}

describe("WP-8 active_run 上下文渲染", () => {
	it("有活跃任务：task_run 行含运行目录与四区纪律", () => {
		const result = baseResult();
		result.activeRun = {
			task_id: "task_1",
			state: "RUNNING",
			revision: 2,
			run_dir: "/home/u/.rosclaw/runs/task_1/r2",
			zones: {
				scratch: "/home/u/.rosclaw/runs/task_1/r2/scratch",
				outputs: "/home/u/.rosclaw/runs/task_1/r2/outputs",
				evidence: "/home/u/.rosclaw/runs/task_1/r2/evidence",
				logs: "/home/u/.rosclaw/runs/task_1/r2/logs",
			},
		};
		const out = renderTrustedContext(result);
		assert.ok(out.includes("/home/u/.rosclaw/runs/task_1/r2"));
		assert.ok(out.includes("outputs/"), "缺交付区说明");
		assert.ok(out.includes("scratch"), "缺草稿区说明");
		assert.ok(out.includes("不得登记"), "缺 scratch 纪律");
	});

	it("无活跃任务：不渲染 task_run 行", () => {
		const out = renderTrustedContext(baseResult());
		assert.ok(!out.includes("task_run"));
	});
});
