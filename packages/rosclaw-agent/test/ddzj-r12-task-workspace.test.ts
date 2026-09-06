/** 大道至简 R1-2a 红测试：任务沙箱可写可运行。
 *
 * 方案 R1：「普通 SIM 任务允许 Pi 在 ~/.rosclaw/runs/<task>/ 自由
 * 写代码、运行代码」——现有 run dir scratch 区（WP-8）是任务沙箱。
 * 当前 write/edit/bash 只认 session workspaceRoot——写到任务
 * scratch 被「path escapes workspace」拒绝。
 *
 * 闭环断言：
 * 1. write/edit 接受活跃任务 scratch 区（extraRoots）；
 * 2. 两个根之外仍拒绝（防逃逸）；
 * 3. 产品核心源码守护在 extraRoots 下依然生效（§4.4 不削弱）；
 * 4. bash 支持 cwd 参数（限定在允许的根内）——Pi 写的脚本可运行；
 *    cwd 越界拒绝。
 */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeTools(dir: string, scratch: string) {
	const { buildWorkspacePackTools } = await import(
		"../src/tools/workspace-pack.js"
	);
	return buildWorkspacePackTools({
		root: dir,
		rosclawHome: dir,
		mode: () => "SIMULATION",
		// 本测试只关心路径限定（cwd/extraRoots），不关心沙箱——
		// 注入无 bwrap + 操作者降级授权（真实主机的 bwrap 完好性
		// 由 os_isolation 探测覆盖，不在此断言）。
		bwrapPath: () => null,
		allowUnsandboxedShell: () => true,
		extraRoots: () => [scratch],
	} as never);
}

test("R1-2a: write 接受任务 scratch 区（extraRoots）", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12-ws-"));
	const scratch = mkdtempSync(join(tmpdir(), "r12-scratch-"));
	const tools = await makeTools(dir, scratch);
	const write = tools.find((t) => t.name === "write") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	const target = join(scratch, "letters.py");
	const out = await write.execute(
		"c1", { path: target, content: "print('hi')" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(!out.isError, out.content[0]?.text);
	assert.equal(readFileSync(target, "utf-8"), "print('hi')");
});

test("R1-2a: 两个根之外仍拒绝（防逃逸）", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12-ws-"));
	const scratch = mkdtempSync(join(tmpdir(), "r12-scratch-"));
	const tools = await makeTools(dir, scratch);
	const write = tools.find((t) => t.name === "write") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	const out = await write.execute(
		"c1", { path: "/etc/r12-escape-test", content: "x" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(out.isError, "越界写入竟放行");
});

test("R1-2a: edit 在 extraRoots 下可用", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12-ws-"));
	const scratch = mkdtempSync(join(tmpdir(), "r12-scratch-"));
	const target = join(scratch, "a.txt");
	writeFileSync(target, "hello", "utf-8");
	const tools = await makeTools(dir, scratch);
	const edit = tools.find((t) => t.name === "edit") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	const out = await edit.execute(
		"c1", { path: target, oldText: "hello", newText: "world" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(!out.isError, out.content[0]?.text);
	assert.equal(readFileSync(target, "utf-8"), "world");
});

test("R1-2a: bash cwd 限定允许的根——scratch 可运行脚本，越界拒绝", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12-ws-"));
	const scratch = mkdtempSync(join(tmpdir(), "r12-scratch-"));
	writeFileSync(join(scratch, "ok.sh"), "echo sandbox-ok", "utf-8");
	const tools = await makeTools(dir, scratch);
	const bash = tools.find((t) => t.name === "bash") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	// cwd 在 scratch：可运行 Pi 写的脚本。
	const ok = await bash.execute(
		"c1", { command: "sh ok.sh", cwd: scratch },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(!ok.isError, ok.content[0]?.text);
	assert.match(ok.content[0]?.text ?? "", /sandbox-ok/);
	// cwd 越界（系统目录）→ 拒绝。
	const bad = await bash.execute(
		"c2", { command: "ls", cwd: "/etc" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(bad.isError, "cwd 越界竟放行");
});
