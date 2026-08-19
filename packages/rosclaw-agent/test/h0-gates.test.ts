/** PR-H0 红灯 Gate（总纲 v2 §20 PR-H0/§21 Gate A）：
 *
 * 这些断言在当前 main 必须是红的——它们固化真实用户问题的根因：
 * 1. 主会话被剥夺工作工具（只剩治理/路由工具）→ Native Agent 不能
 *    自己干活，一切任务被甩给"第二个 Pi Session 伪装 Worker"；
 * 2. task_submit 在模型面 → 模型可以创建 root task（裂变之源）；
 * 3. NativeHarnessBackend SPI 不存在 → 产品语义直接绑死 Pi 私有类型。
 *
 * PR-H1 把它们逐个转绿。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";

test("Gate A: 主会话拥有策略包装的工作工具（Workspace Pack）", () => {
	for (const name of ["read", "grep", "find", "ls", "edit", "write", "bash"]) {
		assert.ok(
			MODEL_TOOL_NAMES.includes(name),
			`模型面缺工作工具 ${name}——Native Agent 没有手`,
		);
	}
});

test("Gate A: 模型不能创建/操控 root task（task_submit 等退出模型面）", () => {
	for (const name of [
		"rosclaw_task_submit",
		"rosclaw_task_pause",
		"rosclaw_task_resume",
		"rosclaw_task_cancel",
		"rosclaw_task_steer",
		"rosclaw_task_answer",
	]) {
		assert.ok(
			!MODEL_TOOL_NAMES.includes(name),
			`${name} 仍在模型面——root task 权威必须属于 InputController`,
		);
	}
});

test("Gate A: NativeHarnessBackend SPI 存在且 Pi 是唯一默认实现", async () => {
	const port = await import("../src/harness/port.js");
	assert.ok(port, "harness/port.ts 必须存在");
	const pi = await import("../src/harness/pi/pi-backend.js");
	assert.equal(pi.PI_BACKEND_ID, "pi");
});

test("Gate A: 用户可见文案无内部品牌泄漏（engine=pi / pi --session）", async () => {
	const { readFileSync } = await import("node:fs");
	const surface = readFileSync(
		new URL("../../src/tools/surface.ts", import.meta.url),
		"utf-8",
	);
	assert.ok(!surface.includes("engine=pi"), "surface 泄漏引擎品牌");
});
