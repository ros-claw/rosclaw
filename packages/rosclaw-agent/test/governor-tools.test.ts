/** PR-H1（ADR-0012）：Native Agent 工具面——主会话工作工具 + 具身集。
 *  模型自己干活（Workspace Pack），task_submit/delegate/work_* 全系
 *  退出模型面（root task 权威在 InputController——PR-H2）。 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";

describe("Native Agent 工具面（PR-H1）", () => {
	it("模型可见工具 = Workspace Pack + Embodiment Pack", () => {
		const workspace = ["read", "grep", "find", "ls", "edit", "write", "bash"];
		const embodiment = [
			"rosclaw_status",
			"rosclaw_capabilities",
			"rosclaw_observe",
			"rosclaw_verify",
			"rosclaw_memory_query",
			"rosclaw_fail_safe",
			"rosclaw_request_action",
		];
		// R0-1.5：rosclaw_task 退出模型面（输入路由自动执行）。
		for (const name of [...workspace, ...embodiment]) {
			assert.ok(MODEL_TOOL_NAMES.includes(name), `缺工具 ${name}`);
		}
	});

	it("N5D：通用能力入口退出模型面（物化精确工具取代）", () => {
		assert.ok(!MODEL_TOOL_NAMES.includes("rosclaw_compute"));
		assert.ok(!MODEL_TOOL_NAMES.includes("rosclaw_execute"));
	});

	it("task 治理/Worker 操控工具不再暴露给模型", () => {
		const banned = [
			"rosclaw_task_submit",
			"rosclaw_task_observe",
			"rosclaw_task_steer",
			"rosclaw_task_answer",
			"rosclaw_task_pause",
			"rosclaw_task_resume",
			"rosclaw_task_cancel",
			"rosclaw_delegate",
			"rosclaw_retry_work",
			"rosclaw_resume_work",
			"rosclaw_extend_work",
			"rosclaw_check_work",
			"rosclaw_list_work",
			"rosclaw_update_work",
			"rosclaw_cancel_work",
			"rosclaw_answer_work",
			"rosclaw_read_work_events",
			"rosclaw_read_work_transcript",
			"rosclaw_list_work_artifacts",
			"rosclaw_read_work_failure",
		];
		for (const name of banned) {
			assert.ok(!MODEL_TOOL_NAMES.includes(name), `${name} 不应暴露给模型`);
		}
	});
});
