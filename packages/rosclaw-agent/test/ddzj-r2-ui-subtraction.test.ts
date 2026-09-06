/** 大道至简 R2 红测试：UI 减法——默认只显示目标/进度/结果。
 *
 * 0905 体验实证（rosclaw体验0905.txt）：rosclaw_artifact_list 把
 * 整段交付物 JSON（8 个 artifact 的 digest/open_command/内部绝对
 * 路径）原样刷屏——用户看到的信息量比模型还多。
 * 方案 R2：默认隐藏 Artifact JSON/trace ID/内部路径（只进
 * /activity）；主交付物直接点击打开。
 *
 * 闭环断言：
 * 1. 三个只读任务工具（task_inspect/artifact_list/artifact_resolve）
 *    挂折叠渲染钩——TUI 单行摘要，模型上下文仍留完整 JSON；
 * 2. artifact list 摘要是计数行（"8 个交付物（video ×2…）"），
 *    不含 digest/open_command/绝对路径；
 * 3. 终态卡交付物文件名带 OSC 8 可点击链接（file:// 绝对路径）
 *    ——终端里直接点开；verbose 的绝对路径行不受影响。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("R2: 只读任务工具挂折叠渲染钩", async () => {
	const { buildReadOnlyTaskTools } = await import("../src/tools/task-read.js");
	const tools = buildReadOnlyTaskTools({
		active: { current: {} },
		center: { call: async () => ({}) },
	} as never);
	for (const name of [
		"rosclaw_task_inspect", "rosclaw_artifact_list", "rosclaw_artifact_resolve",
	]) {
		const tool = tools.find((t) => t.name === name);
		assert.ok(tool, `缺工具 ${name}`);
		assert.ok(
			typeof (tool as { renderResult?: unknown }).renderResult === "function",
			`${name} 未挂折叠渲染钩——原始 JSON 会刷屏（0905 实证）`,
		);
	}
});

test("R2: artifact list 摘要是计数行（无 digest/路径/open_command）", async () => {
	const { summarizeToolResultText } = await import("../src/ui/tool-display.js");
	const raw = JSON.stringify({
		ok: true,
		task_id: "task_61c6a75e",
		artifacts: [
			{ artifact_id: "art_1", kind: "data", media_type: "application/json",
			  path: "/home/ubuntu/.rosclaw/sim/traces/trace_x/trace.json",
			  size_bytes: 219467, digest: "sha256:76cc",
			  open_command: "rosclaw artifact open art_1" },
			{ artifact_id: "art_2", kind: "data", media_type: "video/mp4",
			  path: "/home/ubuntu/.rosclaw/sim/traces/trace_x/trace_x.mp4",
			  size_bytes: 79262, digest: "sha256:19cc",
			  open_command: "rosclaw artifact open art_2" },
			{ artifact_id: "art_3", kind: "data", media_type: "image/gif",
			  path: "/home/ubuntu/.rosclaw/sim/traces/trace_x/trace_x.gif",
			  size_bytes: 15407, digest: "sha256:7725",
			  open_command: "rosclaw artifact open art_3" },
		],
	});
	const line = summarizeToolResultText(raw);
	assert.match(line, /3 个交付物|3 个产物|交付物.*3/, line);
	assert.ok(!line.includes("sha256"), line);
	assert.ok(!line.includes("rosclaw artifact open"), line);
	assert.ok(!line.includes("/home/"), line);
});

test("R2: 终态卡交付物文件名可点击（OSC 8 file:// 链接）", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		lifecycle: "COMPLETED",
		artifact_refs: [{
			artifact_id: "art_1",
			open_command: "rosclaw artifact open art_1",
			path: "/home/u/.rosclaw/runs/task_1/r1/outputs/trace.gif",
			media_type: "image/gif",
			size_bytes: 15407,
		}],
	});
	// OSC 8 超链接：\x1b]8;;file://<path>\x07<文件名>\x1b]8;;\x07。
	assert.ok(
		reply.includes("\x1b]8;;file:///home/u/.rosclaw/runs/task_1/r1/outputs/trace.gif\x07"),
		`缺可点击链接：${JSON.stringify(reply)}`,
	);
	assert.ok(reply.includes("trace.gif"), "文件名仍可见");
});
