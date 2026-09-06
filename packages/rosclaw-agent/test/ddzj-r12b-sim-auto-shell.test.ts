/** 大道至简 R1-2b 红测试：SIM 任务沙箱 bash 自动执行，不弹批准卡。
 *
 * 方案安全节：「SIM：原物理工具和任务沙箱自动执行，不弹批准。
 * Developer：修改项目源码、联网、装包或访问任务目录外时，确认
 * 一次。REAL：Pi 负责理解和规划，rosclawd 负责动作许可。」
 *
 * 0902 R1-a 的降级确认卡在 SIM 下退役——无 bwrap 主机的 SIM
 * 任务代码自动运行（scrubbed env + 灾难命令/危险模式拦截 +
 * TOOL_LAYER_ONLY 诚实标记不变）。REAL/SHADOW 依然 fail closed。
 *
 * 闭环断言：
 * 1. SIM + 无 bwrap + 无操作者授权 → bash 直接执行（无卡、无
 *    shellGate 往返）；
 * 2. REAL 模式无 bwrap → 依然 fail closed（不裸跑）；
 * 3. 灾难命令（DENIED_COMMAND）依然拒绝（自动放行不覆盖第一层
 *    过滤）；
 * 4. 降级标记仍在输出里（诚实）。
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeBash(dir: string, mode: string, gateCalls: { n: number }) {
	const { buildWorkspacePackTools } = await import(
		"../src/tools/workspace-pack.js"
	);
	const tools = buildWorkspacePackTools({
		root: dir,
		rosclawHome: dir,
		mode: () => mode,
		bwrapPath: () => null, // 强制无 bwrap（降级主机）
		shellGate: {
			check: async () => { gateCalls.n += 1; return false; },
			request: async () => { gateCalls.n += 1; return "shg_test"; },
			status: async () => { gateCalls.n += 1; return "PENDING"; },
		},
	} as never);
	return tools.find((t) => t.name === "bash") as unknown as {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		execute: (...a: any[]) => Promise<{ content: Array<{ text: string }>; isError?: boolean }>;
	};
}

test("R1-2b: SIM 无 bwrap → bash 自动执行（零批准卡往返）", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12b-"));
	const gateCalls = { n: 0 };
	const bash = await makeBash(dir, "SIMULATION", gateCalls);
	const out = await bash.execute(
		"c1", { command: "echo sim-auto-ok" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(!out.isError, out.content[0]?.text);
	assert.match(out.content[0]?.text ?? "", /sim-auto-ok/);
	assert.equal(gateCalls.n, 0, "SIM 任务沙箱竟走批准面");
	// 诚实标记仍在（无 OS 沙箱的事实不隐藏）。
	assert.match(out.content[0]?.text ?? "", /TOOL_LAYER_ONLY/);
});

test("R1-2b: REAL 无 bwrap → 依然 fail closed", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12b-"));
	const gateCalls = { n: 0 };
	const bash = await makeBash(dir, "REAL", gateCalls);
	const out = await bash.execute(
		"c1", { command: "echo nope" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(out.isError, "REAL 无沙箱竟放行");
	assert.match(out.content[0]?.text ?? "", /fail closed/);
});

test("R1-2b: SIM 自动执行不覆盖灾难命令第一层过滤", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r12b-"));
	const gateCalls = { n: 0 };
	const bash = await makeBash(dir, "SIMULATION", gateCalls);
	const out = await bash.execute(
		"c1", { command: "sudo echo hi" },
		new AbortController().signal, async () => {}, {},
	);
	assert.ok(out.isError, "sudo 竟放行");
});
