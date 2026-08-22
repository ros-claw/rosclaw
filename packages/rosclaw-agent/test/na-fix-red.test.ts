/** 红测试（二次审计 P0-4/P0-5/P1-1/P0-3）：先红后修。 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("P0-4: rosclaw_request_action must be registered in the runtime tool list", () => {
	// 直接静态检查：pi-runtime 必须有 request_action 工具的注册。
	const runtime = readFileSync(
		new URL("../../src/harness/pi/pi-runtime.ts", import.meta.url),
		"utf-8",
	);
	assert.match(runtime, /buildRequestActionTool|rosclaw_request_action/,
		"pi-runtime 没有注册 rosclaw_request_action（P0-4）");
});

test("P0-4: native_agent_v2.md 不得声明未注册的工具", () => {
	const prompt = readFileSync(
		new URL("../../dist/prompts/native_agent_v2.md", import.meta.url),
		"utf-8",
	);
	// 当前实际注册：status/observe/verify/memory_query/fail_safe/delegate/request_action
	for (const name of ["plan_patch", "team_coordinate"]) {
		assert.ok(!prompt.includes(`rosclaw_${name}`),
			`prompt 声明了未注册的 rosclaw_${name}（P1-3）`);
	}
});

test("P1-1: raw thinking must be hidden by default", () => {
	const runtime = readFileSync(
		new URL("../../src/harness/pi/pi-runtime.ts", import.meta.url),
		"utf-8",
	);
	assert.match(runtime, /[Hh]ideThinking(Block)?/,
		"runtime 未隐藏 raw thinking（P1-1）");
});

test("P0-3: exit resume hint must be a ROSClaw command, never 'pi --session'", () => {
	// 产品表面扫描：我们维护的上游 patch 必须覆盖退出 resume hint。
	const patch = readFileSync(
		new URL("../../patches/README.md", import.meta.url),
		"utf-8",
	);
	assert.match(patch, /resumeCommandFormatter|AppIdentity/,
		"缺少退出 resume hint 的 ROSClaw 覆盖（P0-3）");
});
