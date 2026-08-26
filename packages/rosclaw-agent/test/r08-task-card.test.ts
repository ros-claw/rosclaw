/** R0-8 红测试（0826 体验审计 §5.R0-8）：TUI 任务卡——默认层
 * 不再是 raw JSON 日志窗口。
 *
 * 真实事故（0826 体验旅程）：rosclaw_task 把整段 JSON 直接刷到
 * 终端（resource digest/路径/metrics/receipt 全量）——仓库已有
 * renderResult/summarizeToolResultText 折叠机制但没接到 task tool。
 *
 * 断言：
 * 1. 任务卡渲染：规划/仿真/验证/交付一行一态（✓/△/✗），误差
 *    与目标阈值，交付物可打开命令——不出现 raw JSON 键；
 * 2. FAILED/PARTIAL 诚实：失败节点 ✗、缺失交付 △ + failures；
 * 3. task tool renderResult 输出任务卡（不是 JSON 原文）；
 * 4. 完整 JSON 仍可从模型上下文/activity 获取（模型面不降级）。
 */

import assert from "node:assert/strict";
import test from "node:test";

const VERIFIED_PAYLOAD = {
	state: "VERIFIED",
	goal: "draw_shape",
	task_id: "task_1",
	recipe_id: "recipe:sim.draw_path",
	plan: { refs: ["ResourceRef", "PlanRef", "TraceRef", "RenderRef", "SceneRef", "VerificationRef"], failed_node: "" },
	artifacts: {
		gif: "/home/u/.rosclaw/sim/traces/t1/t1.gif",
		mp4: "/home/u/.rosclaw/sim/traces/t1/t1-scene.mp4",
		metrics: { max_error_m: 0.0196 },
		evidence_level: "SIM_DYN_ROLLOUT",
	},
	artifact_refs: [
		{ artifact_id: "art_g1", media_type: "image/gif", kind: "preview_2d", open_command: "rosclaw artifact open art_g1" },
		{ artifact_id: "art_m1", media_type: "video/mp4", kind: "scene_3d", open_command: "rosclaw artifact open art_m1" },
	],
	failures: [],
	verification: { verdict: "PASS", max_error_m: 0.0196, threshold_m: 0.025, frames: 60, min_frames: 30, verification_id: "vrf_1" },
	evidence_level: "SIM_DYN_ROLLOUT",
};

const FAILED_PAYLOAD = {
	state: "FAILED",
	goal: "draw_shape",
	task_id: "task_2",
	recipe_id: "recipe:sim.draw_path",
	plan: { refs: ["ResourceRef", "PlanRef", "TraceRef", "RenderRef"], failed_node: "render_scene" },
	artifacts: { gif: "/x/t2.gif", metrics: { max_error_m: 0.019 } },
	artifact_refs: [
		{ artifact_id: "art_g2", media_type: "image/gif", kind: "preview_2d", open_command: "rosclaw artifact open art_g2" },
	],
	failures: ["DELIVERABLE_MISSING: required 交付物 scene_video 未在产物账本", "RENDER_BACKEND_UNAVAILABLE: EGL/OSMesa/Xvfb 全部不可用"],
	verification: { verdict: "FAIL", max_error_m: 0.019, threshold_m: 0.025, frames: 60, min_frames: 30, verification_id: "" },
	evidence_level: "SIM_DYN_ROLLOUT",
};

test("VERIFIED 任务卡：一行一态 + 误差/阈值 + 交付打开命令", async () => {
	const { renderTaskCard } = await import("../src/ui/task-card.js");
	const card = renderTaskCard(VERIFIED_PAYLOAD, "zh-CN");
	assert.match(card, /✓.*规划/);
	assert.match(card, /✓.*仿真/);
	assert.match(card, /✓.*验证/);
	assert.match(card, /19\.6\s*mm/);
	assert.match(card, /25\.0\s*mm|25\s*mm/);
	assert.match(card, /rosclaw artifact open art_g1/);
	assert.match(card, /rosclaw artifact open art_m1/);
	// 不得出现 raw JSON 结构键。
	assert.ok(!card.includes('"artifact_refs"'), card);
	assert.ok(!card.includes('"verification"'), card);
	assert.ok(!card.includes("resource_digest"), card);
});

test("FAILED 任务卡：失败节点 ✗ + 缺失交付 △ + failures 可读", async () => {
	const { renderTaskCard } = await import("../src/ui/task-card.js");
	const card = renderTaskCard(FAILED_PAYLOAD, "zh-CN");
	assert.match(card, /✗|△/);
	assert.match(card, /scene_video|场景视频/);
	assert.match(card, /未完成|未通过/);
	// 2D 交付仍在（PARTIAL 诚实——不是全黑）。
	assert.match(card, /art_g2/);
});

test("task tool renderResult 输出任务卡（不是 JSON 原文）", async () => {
	const { renderTaskToolResult } = await import("../src/ui/task-card.js");
	const text = renderTaskToolResult(JSON.stringify(VERIFIED_PAYLOAD));
	assert.match(text, /✓.*规划/);
	assert.ok(!text.includes('"plan"'), text);
	assert.ok(!text.includes('"artifacts"'), text);
});
