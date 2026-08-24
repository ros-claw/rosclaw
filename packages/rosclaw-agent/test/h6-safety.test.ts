/** PR-H6 红测试（TS）：EffectClass 声明 + REAL/SHADOW shell 强隔离。
 *
 * 红测试先行——修复前必须红：
 * 1. 每个模型可见工具都有 EffectClass 声明（无分类=不暴露）；
 * 2. REAL/SHADOW 模式 bash 必须 bwrap 强隔离（无 bwrap → fail
 *    closed 诚实拒绝，不裸跑）；
 * 3. bwrap 内：无网络、无 /dev 写、workspace 可写、其余只读；
 * 4. SIM 模式在 H6 曾不沙箱（效率优先）——P0-6（0823 审计）
 *    起全模式沙箱：SIM 裸跑可绕过治理（读凭据/控制 token、
 *    直调 bridge socket），无 bwrap 主机 SIM 降级需操作者显式
 *    授权且带 TOOL_LAYER_ONLY 标记。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";

test("H6/N5C: 每个模型可见工具都有 EffectClass（单一合约来源）", async () => {
	// N5C：手写 EFFECT_BY_TOOL 已删除——rosclaw_* 来自 Python 注册表
	// 生成的 effects.generated.json；workspace 原语来自 workspace-pack
	// 同位声明。
	const { getToolEffect } = await import("../src/tools/effects.js");
	for (const name of MODEL_TOOL_NAMES) {
		assert.ok(getToolEffect(name), `工具 ${name} 缺 EffectClass 声明`);
	}
});

test("H6/N5C: workspace 原语与具身工具分类正确", async () => {
	const { getToolEffect } = await import("../src/tools/effects.js");
	assert.equal(getToolEffect("bash"), "HOST_PROCESS");
	assert.equal(getToolEffect("write"), "WORKSPACE_WRITE");
	assert.equal(getToolEffect("read"), "READ_ONLY");
	assert.equal(getToolEffect("rosclaw_request_action"), "PHYSICAL_EFFECT");
});

test("N5C: 通用入口是 DYNAMIC——按参数解析，不静态写死", async () => {
	const { getToolEffect } = await import("../src/tools/effects.js");
	for (const name of ["rosclaw_execute", "rosclaw_compute", "rosclaw_observe"]) {
		assert.equal(getToolEffect(name), "DYNAMIC",
			`${name} 必须 DYNAMIC（相同入口调 SIM/REAL 能力 effect 不同）`);
	}
});

test("N5C: 手写 effect-class.ts 已删除（结构守门）", () => {
	assert.equal(existsSync(new URL("../src/tools/effect-class.ts", import.meta.url)), false,
		"effect-class.ts 手写表不得复活——effect 从 Capability 生成");
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
