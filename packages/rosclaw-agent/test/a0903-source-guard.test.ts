/** 0903 体验复核红测试（rosclaw体验0903.txt 实证 + §4.4）：
 *  普通产品任务不得修改正在运行的产品核心源码——模型在"画立方体"
 *  任务中途决定给规划器加 cube 形状（编辑 sim_trajectory.py 等
 *  产品源码），把产品任务变成开发任务。
 *
 * 闭环断言：write/edit 目标落在运行中产品自身源码树
 * （<产品根>/src/rosclaw 或 <产品根>/packages/rosclaw-agent/src）
 * → 拒绝并指路；普通工作区文件不受影响。
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

async function makeTools(dir: string) {
	const { buildWorkspacePackTools } = await import(
		"../src/tools/workspace-pack.js"
	);
	return buildWorkspacePackTools({
		root: dir, rosclawHome: dir, mode: () => "SIMULATION",
		bwrapPath: () => "/usr/bin/bwrap", // 有沙箱——沙箱过了也不许写产品源码
	} as never);
}

test("0903: write 到产品核心源码被拒绝（§4.4）", async () => {
	// 真实事故形态：用户在产品 checkout 里开 chat——workspace root
	// 就是产品根，resolveInRoot 放行，守护必须是产品源码判定。
	const { productSourceRoots } = await import("../src/tools/workspace-pack.js");
	const roots = productSourceRoots();
	assert.ok(roots.length >= 1, "产品源码根解析为空");
	const productRoot = dirname(dirname(roots[0])); // src/rosclaw → 产品根
	const tools = await makeTools(productRoot);
	const write = tools.find((t) => t.name === "write") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	const target = join(roots[0], "agentd", "sim_trajectory.py");
	const out = await write.execute(
		"c1", { path: target, content: "# hack" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(out.isError, "产品核心源码竟可写");
	assert.match(out.content[0]?.text ?? "", /产品核心|开发流程/);
});

test("0903: 普通工作区文件不受影响", async () => {
	const dir = mkdtempSync(join(tmpdir(), "a0903-"));
	const tools = await makeTools(dir);
	const write = tools.find((t) => t.name === "write") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
	const out = await write.execute(
		"c1", { path: join(dir, "notes.txt"), content: "hello" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(!out.isError, out.content[0]?.text);
});
