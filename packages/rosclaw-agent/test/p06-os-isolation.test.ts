/** P0-6 红测试（0823 审计 §三.P0-6）：Harness Shell OS 隔离实证。
 *
 * 红测试先行——全模式 bwrap + 敏感路径遮蔽不存在时必须红。
 *
 * 0823 审计：SIM auto 下 Harness Shell 可绕过治理——bash 在宿主
 * 裸跑（无 bwrap），能读 ~/.rosclaw/agent 凭据与 run/ 控制 token
 * （可直接调 bridge socket 绕过全部治理链）、能写项目源码树。
 *
 * 修复契约（全部实证，不是配置断言）：
 * 1. 全模式（SIM 同 REAL/SHADOW）bash 必须 bwrap 强隔离——无
 *    bwrap fail closed，任何模式不裸跑；
 * 2. 沙箱内敏感路径被遮蔽：~/.ssh/.gnupg/.aws + rosclawHome 的
 *    agent/agentd/run（凭据/控制 token/socket 不可读——治理链
 *    不可经 shell 绕过）；
 * 3. 沙箱内无网络（OS 级 socket 实证，不是 argv 黑名单）；
 * 4. workspace 可写、其余宿主只读（全模式一致）。
 */

import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

async function makeBash(root: string, rosclawHome: string, mode: string, bwrapPath?: () => string | null) {
	const { buildWorkspacePackTools } = await import("../src/tools/workspace-pack.js");
	const tools = buildWorkspacePackTools({
		root,
		rosclawHome,
		mode: () => mode,
		...(bwrapPath ? { bwrapPath } : {}),
	});
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	return bash;
}

async function run(bash: { execute: Function }, command: string): Promise<{ text: string; isError: boolean }> {
	const result = await bash.execute(
		`c${Math.floor(Math.random() * 1e9)}`,
		{ command },
		new AbortController().signal,
		async () => {},
		{} as never,
	);
	return {
		text: String((result.content[0] as { text: string }).text),
		isError: (result as { isError?: boolean }).isError === true,
	};
}

test("P0-6: SIM 模式无 bwrap → fail closed（任何模式不裸跑）", async () => {
	const dir = mkdtempSync(join(tmpdir(), "p06-"));
	const bash = await makeBash(dir, dir, "SIMULATION", () => null);
	const out = await run(bash, "echo hi");
	assert.ok(out.isError, "SIM 无 bwrap 裸跑了——治理可绕过");
	assert.match(out.text, /隔离|bwrap|fail.closed/i);
});

test("P0-6: SIM 无 bwrap + 显式降级授权 → 运行但带诚实标记", async () => {
	const dir = mkdtempSync(join(tmpdir(), "p06-"));
	const { buildWorkspacePackTools } = await import("../src/tools/workspace-pack.js");
	const tools = buildWorkspacePackTools({
		root: dir,
		rosclawHome: dir,
		mode: () => "SIMULATION",
		bwrapPath: () => null,
		// 操作者显式授权降级（等同 ROSCLAW_ALLOW_UNSANDBOXED_SHELL=1）。
		allowUnsandboxedShell: () => true,
	});
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	const result = await bash.execute(
		"c1", { command: "echo hi" }, new AbortController().signal,
		async () => {}, {} as never,
	);
	const text = String((result.content[0] as { text: string }).text);
	assert.match(text, /hi/, "显式授权后降级 shell 应运行");
	assert.match(text, /TOOL_LAYER_ONLY/, "降级运行必须带无 OS 沙箱标记");
});

test("P0-6: 敏感路径遮蔽清单覆盖凭据/控制面", async () => {
	const { _sensitiveMasks } = await import("../src/tools/workspace-pack.js");
	const masks = _sensitiveMasks("/home/u", "/home/u/.rosclaw");
	const joined = masks.join(" ");
	for (const path of [
		"/home/u/.ssh",
		"/home/u/.gnupg",
		"/home/u/.rosclaw/agent",
		"/home/u/.rosclaw/agentd",
		"/home/u/.rosclaw/run",
	]) {
		assert.ok(joined.includes(path), `遮蔽清单缺 ${path}——治理可经 shell 绕过`);
	}
});

test("P0-6: SIM 沙箱实证——凭据/token 不可读、无网络、越界只读", async (t) => {
	const { _bwrapAvailable } = await import("../src/tools/workspace-pack.js");
	if (!_bwrapAvailable()) {
		t.skip("无 bwrap——跳过（CI 装 bubblewrap 后实证）");
		return;
	}
	const root = mkdtempSync(join(tmpdir(), "p06-ws-"));
	const rosclawHome = mkdtempSync(join(tmpdir(), "p06-home-"));
	// 伪造凭据/控制 token/socket 目录（与真实布局同名）。
	mkdirSync(join(rosclawHome, "agent"), { recursive: true });
	writeFileSync(join(rosclawHome, "agent", "auth.json"), "{\"token\":\"SECRET\"}");
	mkdirSync(join(rosclawHome, "run"), { recursive: true });
	writeFileSync(join(rosclawHome, "run", "control.token"), "SECRET");
	mkdirSync(join(rosclawHome, "agentd"), { recursive: true });
	const bash = await makeBash(root, rosclawHome, "SIMULATION", () => "/usr/bin/bwrap");

	// 凭据不可读（遮蔽后路径不存在——不是"权限拒绝"）。
	const cred = await run(bash, `cat ${rosclawHome}/agent/auth.json 2>&1 || echo MASKED`);
	assert.match(cred.text, /MASKED/, "沙箱内读到了凭据——治理可绕过");
	const tok = await run(bash, `cat ${rosclawHome}/run/control.token 2>&1 || echo MASKED`);
	assert.match(tok.text, /MASKED/, "沙箱内读到了控制 token——可冒充 agent 调 bridge");

	// 无网络（OS 级 socket，不是 argv 黑名单）。
	const net = await run(bash, "python3 -c \"import socket;socket.create_connection(('8.8.8.8',53),2)\" 2>&1 || echo NO_NET");
	assert.match(net.text, /NO_NET/, "沙箱内有网络——隔离不成立");

	// workspace 可写。
	const wr = await run(bash, "echo ok > inside.txt && cat inside.txt");
	assert.match(wr.text, /ok/);
	assert.ok(readFileSync(join(root, "inside.txt"), "utf-8").startsWith("ok"));

	// 越界只读（rosclawHome 非遮蔽部分也不可写）。
	const outside = join(rosclawHome, "p06_escape");
	const esc = await run(bash, `touch ${outside} 2>&1 || echo READONLY`);
	assert.match(esc.text, /READONLY/, "沙箱内越界可写");
	assert.ok(!existsSync(outside), "越界文件真的被写了");
});

test("P0-6: REAL/SHADOW 语义保持（遮蔽同样生效）", async (t) => {
	const { _bwrapAvailable } = await import("../src/tools/workspace-pack.js");
	if (!_bwrapAvailable()) {
		t.skip("无 bwrap——跳过");
		return;
	}
	const root = mkdtempSync(join(tmpdir(), "p06-ws-"));
	const rosclawHome = mkdtempSync(join(tmpdir(), "p06-home-"));
	mkdirSync(join(rosclawHome, "agent"), { recursive: true });
	writeFileSync(join(rosclawHome, "agent", "auth.json"), "SECRET");
	const bash = await makeBash(root, rosclawHome, "REAL", () => "/usr/bin/bwrap");
	const cred = await run(bash, `cat ${rosclawHome}/agent/auth.json 2>&1 || echo MASKED`);
	assert.match(cred.text, /MASKED/);
});
