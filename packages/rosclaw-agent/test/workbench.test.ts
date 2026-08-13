/** 十审 Gate W3 红测试：Developer Workbench 约束工具。
 *
 * 1. 路径约束：../、绝对路径、symlink 逃逸一律拒绝；
 * 2. bash argv 白名单：网络/特权/未列出二进制拒绝；逐参数路径检查；
 * 3. env 隔离：KIMI_API_KEY 等凭据不进 bash 子进程；HOME=workspace；
 * 4. write/edit 真实落盘；bash 日志写 artifacts；
 * 5. 超时/abort 杀命令。
 */

import assert from "node:assert/strict";
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeWorkbench(root?: string) {
	const { buildWorkbenchTools, resolveInRoot } = await import("../src/workers/workbench.js");
	const ws = root ?? mkdtempSync(join(tmpdir(), "wb-"));
	const logPath = join(ws, "artifacts", "bash-log.txt");
	const tools = buildWorkbenchTools({ root: ws, bashLogPath: logPath });
	type ToolResult = { isError?: boolean; content: Array<{ type: string; text?: string }> };
	type LooseTool = { execute: (...args: any[]) => Promise<ToolResult> };
	const byName = Object.fromEntries(tools.map((t) => [t.name, t])) as Record<string, LooseTool>;
	return { ws, logPath, byName, resolveInRoot };
}

test("路径约束：../ 与绝对路径与 symlink 逃逸拒绝", async () => {
	const { ws, resolveInRoot } = await makeWorkbench();
	assert.throws(() => resolveInRoot(ws, "../evil.txt"), /escapes/);
	assert.throws(() => resolveInRoot(ws, "/etc/passwd"), /escapes/);
	// symlink 逃逸：workspace 内的链接指向外部。
	const outside = mkdtempSync(join(tmpdir(), "wb-out-"));
	writeFileSync(join(outside, "secret.txt"), "top secret");
	symlinkSync(outside, join(ws, "link-out"), "dir");
	assert.throws(() => resolveInRoot(ws, "link-out/secret.txt"), /symlink/);
	// 合法路径可用。
	assert.equal(resolveInRoot(ws, "src/a.ts"), join(ws, "src/a.ts"));
});

test("write/edit 真实落盘且拒绝越界", async () => {
	const { ws, byName } = await makeWorkbench();
	await byName.write.execute("t1", { path: "src/hello.py", content: "print('hi')\n" }, undefined, undefined, {} as never);
	assert.equal(readFileSync(join(ws, "src/hello.py"), "utf-8"), "print('hi')\n");
	const denied = await byName.write.execute("t2", { path: "/tmp/wb-evil-x", content: "x" }, undefined, undefined, {} as never);
	// resolveInRoot 在 execute 内抛错——必须被工具层拦住（不静默成功）。
	assert.ok(denied.isError || !existsSync("/tmp/wb-evil-x"));
	const edited = await byName.edit.execute(
		"t3",
		{ path: "src/hello.py", old_text: "print('hi')", new_text: "print('hello')" },
		undefined,
		undefined,
		{} as never,
	);
	assert.ok(!edited.isError);
	assert.match(readFileSync(join(ws, "src/hello.py"), "utf-8"), /hello/);
	const missing = await byName.edit.execute(
		"t4",
		{ path: "src/hello.py", old_text: "not there", new_text: "x" },
		undefined,
		undefined,
		{} as never,
	);
	assert.ok(missing.isError);
});

test("bash 白名单：允许 echo/python，拒绝 curl/sudo/pip/未列出", async () => {
	const { byName } = await makeWorkbench();
	const ok = await byName.bash.execute("t1", { argv: ["echo", "hi"] }, undefined, undefined, {} as never);
	assert.match((ok.content[0] as { text: string }).text, /exit 0/);
	for (const bin of ["curl", "sudo", "pip", "docker", "ssh", "rm-rf-notreal"]) {
		const denied = await byName.bash.execute("t2", { argv: [bin, "x"] }, undefined, undefined, {} as never);
		assert.ok(denied.isError, `${bin} 未被拒绝`);
		assert.match((denied.content[0] as { text: string }).text, /DENIED/);
	}
});

test("bash 参数路径检查：/etc/passwd、auth.json、/dev 拒绝", async () => {
	const { byName } = await makeWorkbench();
	for (const argv of [
		["cat", "/etc/passwd"],
		["cat", "/dev/zero"],
		["cat", `${process.env.HOME}/.rosclaw/agent/auth.json`],
		["cat", "../../etc/passwd"],
	]) {
		const denied = await byName.bash.execute("t1", { argv }, undefined, undefined, {} as never);
		assert.ok(denied.isError, `${argv.join(" ")} 未被拒绝`);
	}
});

test("bash env 隔离：API key 不进子进程，HOME=workspace", async () => {
	process.env.KIMI_API_KEY = "sk-test-should-not-leak";
	const { ws, byName } = await makeWorkbench();
	const result = await byName.bash.execute(
		"t1",
		{ argv: ["python3", "-c", "import os; print('key=' + str(os.environ.get('KIMI_API_KEY'))); print('home=' + os.environ.get('HOME', ''))"] },
		undefined,
		undefined,
		{} as never,
	);
	const text = (result.content[0] as { text: string }).text;
	assert.match(text, /key=None/);
	assert.match(text, new RegExp(`home=${ws.replace(/[/.]/g, (c) => `\\${c}`)}`));
	assert.ok(!text.includes("sk-test-should-not-leak"));
	delete process.env.KIMI_API_KEY;
});

test("bash 超时杀掉长命令 + 日志落 artifacts", async () => {
	const { ws, logPath, byName } = await makeWorkbench();
	const result = await byName.bash.execute(
		"t1",
		{ argv: ["sleep", "30"], timeout_sec: 1 },
		undefined,
		undefined,
		{} as never,
	);
	assert.ok(result.isError, "超时命令未被杀掉");
	assert.ok(existsSync(logPath), "bash log 未写");
	const logText = readFileSync(logPath, "utf-8");
	assert.match(logText, /\$ sleep 30/);
});

test("git diff 语义：workspace 内改动可见（供 patch 工件）", async () => {
	const { ws, byName } = await makeWorkbench();
	await byName.bash.execute("t1", { argv: ["git", "init", "-q"] }, undefined, undefined, {} as never);
	await byName.bash.execute("t2", { argv: ["git", "add", "-A"] }, undefined, undefined, {} as never);
	await byName.write.execute("t3", { path: "a.txt", content: "hello" }, undefined, undefined, {} as never);
	const status = await byName.bash.execute("t4", { argv: ["git", "status", "--porcelain"] }, undefined, undefined, {} as never);
	assert.match((status.content[0] as { text: string }).text, /a\.txt/);
});
