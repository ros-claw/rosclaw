/** 十五审 PR-RF-0/RF-1：Native Agent 治理工具面（无为而治）。
 *  模型可见工具只能是治理集 + 只读摘要——不再有 delegate/retry/
 *  check_work 这类底层 Worker 操控入口。 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";

describe("Native Governor 工具面（PR-RF-1）", () => {
	it("模型可见工具 = 治理集 + 只读集", () => {
		const governance = [
			"rosclaw_task_submit",
			"rosclaw_task_observe",
			"rosclaw_task_steer",
			"rosclaw_task_answer",
			"rosclaw_task_pause",
			"rosclaw_task_resume",
			"rosclaw_task_cancel",
		];
		const readOnly = [
			"rosclaw_status",
			"rosclaw_capabilities",
			"rosclaw_observe",
			"rosclaw_compute",
			"rosclaw_verify",
			"rosclaw_memory_query",
			"rosclaw_fail_safe",
			"rosclaw_task",
		];
		for (const name of [...governance, ...readOnly]) {
			assert.ok(MODEL_TOOL_NAMES.includes(name), `缺治理工具 ${name}`);
		}
	});

	it("底层 Worker 操控工具不再暴露给模型", () => {
		const banned = [
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
