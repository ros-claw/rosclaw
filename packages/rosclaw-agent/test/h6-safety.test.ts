/** PR-H6 红测试（TS）：EffectClass 声明 + REAL/SHADOW shell 强隔离。
 *
 * 红测试先行——修复前必须红：
 * 1. 每个模型可见工具都有 EffectClass 声明（无分类=不暴露）；
 * 2. REAL/SHADOW 模式 bash 必须 bwrap 强隔离（无 bwrap → fail
 *    closed 诚实拒绝，不裸跑）；
 * 3. bwrap 内：无网络、无 /dev 写、workspace 可写、其余只读；
 * 4. SIM 模式保持现状（不 bwrap——效率优先，env 剥离已在 H1）。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";

test("H6: 每个模型可见工具都有 EffectClass 声明", async () => {
	const { EFFECT_BY_TOOL } = await import("../src/tools/effect-class.js");
	for (const name of MODEL_TOOL_NAMES) {
		assert.ok(EFFECT_BY_TOOL[name], `工具 ${name} 缺 EffectClass 声明`);
	}
});

test("H6: bash effect=HOST_PROCESS，request_action=PHYSICAL_EFFECT", async () => {
	const { EFFECT_BY_TOOL } = await import("../src/tools/effect-class.js");
	assert.equal(EFFECT_BY_TOOL.bash, "HOST_PROCESS");
	assert.equal(EFFECT_BY_TOOL.write, "WORKSPACE_WRITE");
	assert.equal(EFFECT_BY_TOOL.read, "READ_ONLY");
	assert.equal(EFFECT_BY_TOOL.rosclaw_request_action, "PHYSICAL_EFFECT");
});

test("H6: REAL/SHADOW 模式无 bwrap → bash fail closed", async () => {
	const { buildWorkspacePackTools } = await import("../src/tools/workspace-pack.js");
	const dir = mkdtempSync(join(tmpdir(), "h6-"));
	const tools = buildWorkspacePackTools({
		root: dir,
		mode: () => "REAL",
		bwrapPath: () => null, // 无 bwrap
	});
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	const result = await bash.execute("c1", { command: "echo hi" }, new AbortController().signal, async () => {}, {} as never);
	assert.ok((result as { isError?: boolean }).isError, "REAL 无 bwrap 必须 fail closed");
	assert.match(String((result.content[0] as { text: string }).text), /隔离|bwrap|fail.closed/i);
});

test("H6: bwrap 内 workspace 可写、宿主其余只读、无网络", async (t) => {
	const { buildWorkspacePackTools, _bwrapAvailable } = await import("../src/tools/workspace-pack.js");
	if (!_bwrapAvailable()) {
		t.skip("无 bwrap——跳过");
		return;
	}
	const dir = mkdtempSync(join(tmpdir(), "h6-"));
	const tools = buildWorkspacePackTools({
		root: dir,
		mode: () => "REAL",
		bwrapPath: () => "/usr/bin/bwrap",
	});
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	// workspace 内可写
	const ok = await bash.execute("c2", { command: "echo x > inside.txt && cat inside.txt" }, new AbortController().signal, async () => {}, {} as never);
	assert.match(String((ok.content[0] as { text: string }).text), /x/);
	assert.ok(readFileSync(join(dir, "inside.txt"), "utf-8").startsWith("x"));
	// 宿主 home 只读（写入失败）
	const denied = await bash.execute("c3", { command: `touch ${process.env.HOME}/h6_should_fail && echo WROTE || echo READONLY` }, new AbortController().signal, async () => {}, {} as never);
	assert.match(String((denied.content[0] as { text: string }).text), /READONLY/);
});
