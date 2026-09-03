/** 0902 R1-a 红测试：shell 降级授权走 Approval Broker（会话内确认
 *  卡），删除全局环境变量授权的正式路径。
 *
 * 0902 实证：用户已答"允许！"，系统仍要求 export
 * ROSCLAW_ALLOW_UNSANDBOXED_SHELL=1 并重启——不可接受。
 *
 * 闭环断言：
 * 1. 无 bwrap + 无 grant + 无注入 → fail closed（且拒文不再指向
 *    全局环境变量）；
 * 2. standing grant 命中 → 降级运行带 TOOL_LAYER_ONLY 标记；
 * 3. 无 grant → 确认卡流（request→pending→允许一次）→ 立即继续
 *    原操作；
 * 4. 拒绝 → 不执行；
 * 5. REAL/SHADOW 永远 fail closed（无降级路径）。
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeBash(dir: string, extras: Record<string, unknown>) {
	const { buildWorkspacePackTools } = await import(
		"../src/tools/workspace-pack.js"
	);
	const tools = buildWorkspacePackTools({
		root: dir,
		rosclawHome: dir,
		mode: () => "SIMULATION",
		bwrapPath: () => null, // 强制无 bwrap
		...extras,
	} as never);
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	return bash;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyTool = { execute: (...a: any[]) => Promise<any> };

async function run(bash: AnyTool, command: string, onUpdate?: (...a: never[]) => void) {
	const result = await bash.execute(
		"c1", { command }, new AbortController().signal,
		(onUpdate ?? (async () => {})) as never, {} as never,
	) as { content: Array<{ text: string }>; isError?: boolean };
	return {
		text: String(result.content[0]?.text ?? ""),
		isError: result.isError === true,
	};
}

test("R1-a: 无 bwrap + 无 grant → fail closed，拒文不再指向全局环境变量", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r1a-"));
	const bash = await makeBash(dir, {});
	const out = await run(bash, "echo hi");
	assert.ok(out.isError, "无 grant 裸跑了");
	assert.doesNotMatch(out.text, /ROSCLAW_ALLOW_UNSANDBOXED_SHELL/,
		"拒文仍指向全局环境变量——正式路径没删干净");
	assert.match(out.text, /确认卡|允许一次/);
});

test("R1-a: standing grant 命中 → 降级运行带 TOOL_LAYER_ONLY 标记", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r1a-"));
	const bash = await makeBash(dir, {
		shellGate: {
			check: async () => true,
			request: async () => "shg_x",
			status: async () => "UNKNOWN",
		},
	});
	const out = await run(bash, "echo hi");
	assert.ok(!out.isError, out.text);
	assert.match(out.text, /hi/);
	assert.match(out.text, /TOOL_LAYER_ONLY/);
});

test("R1-a: 确认卡允许一次 → 原操作立即继续", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r1a-"));
	const updates: Array<Record<string, unknown>> = [];
	const bash = await makeBash(dir, {
		shellGate: {
			check: async () => false,
			request: async () => "shg_req1",
			// 第一次轮询即已批准（模拟用户在卡上选了允许一次）。
			status: async () => "APPROVED_ONCE",
		},
	});
	const out = await run(bash, "echo hi", ((partial: unknown) => {
		updates.push(partial as Record<string, unknown>);
	}) as never);
	assert.ok(!out.isError, out.text);
	assert.match(out.text, /hi/);
	// 卡确实弹出过（AWAITING_SHELL_APPROVAL phase 经 onUpdate 发出）。
	assert.ok(
		updates.some((u) =>
			(u.details as { phase?: string })?.phase === "AWAITING_SHELL_APPROVAL"
		),
		"确认卡 phase 未发出",
	);
});

test("R1-a: 拒绝 → 不执行", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r1a-"));
	const bash = await makeBash(dir, {
		shellGate: {
			check: async () => false,
			request: async () => "shg_req2",
			status: async () => "DENIED",
		},
	});
	const out = await run(bash, "echo hi");
	assert.ok(out.isError, "被拒仍执行了");
	assert.doesNotMatch(out.text, /^hi$/m);
});

test("R1-a: REAL 模式无 bwrap 永远 fail closed（无降级路径）", async () => {
	const dir = mkdtempSync(join(tmpdir(), "r1a-"));
	const { buildWorkspacePackTools } = await import(
		"../src/tools/workspace-pack.js"
	);
	const tools = buildWorkspacePackTools({
		root: dir,
		rosclawHome: dir,
		mode: () => "REAL",
		bwrapPath: () => null,
		shellGate: {
			check: async () => true, // 即使有 grant——REAL 也无降级
			request: async () => "shg_x",
			status: async () => "APPROVED_TASK",
		},
	} as never);
	const bash = tools.find((t) => t.name === "bash");
	assert.ok(bash);
	const out = await run(bash as never, "echo hi");
	assert.ok(out.isError, "REAL 模式竟降级裸跑");
});

test("R1-a: 确认卡三选 → decide 正确映射（本任务允许 → allow_task）", async () => {
	// 经扩展注册的 tool_execution_update 处理器驱动（最小 harness）。
	const handlerList: Array<(event: unknown, ctx: unknown) => Promise<unknown>> = [];
	const decideCalls: Array<Record<string, unknown>> = [];
	const { createRosclawExtension } = await import("../src/extension/index.js");
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const { resolveTaskContext } = await import("../src/native/active-task-context.js");

	const active = new ActiveSessionContext({
		sessionId: "pi_test", missionId: undefined, contextRevision: 0,
		mode: "SIMULATION", profile: "developer", contextState: "LOADING",
		leaseState: "NONE", actionsAllowed: false,
	});
	const call = async (_home: string, method: string, params?: Record<string, unknown>) => {
		if (method === "pi.shell_gate.decide") {
			decideCalls.push(params ?? {});
			return { ok: true };
		}
		return { ok: false, error: "no bridge in test" } as Record<string, unknown>;
	};
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh-test", active,
		leaseManager: new SessionLeaseManager("/tmp/rh-test", call),
		notify: () => undefined, call,
	});
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/rh-test", active,
		operatorSocket: "/tmp/rh-test/run/operatord.sock",
		productVersion: "0.1.0", call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager("/tmp/rh-test/agent");
	const factory = createRosclawExtension({
		profile: "developer", version: "0.1.0", systemPrompt: "TEST",
		active, coordinator, center, locale, rosclawHome: "/tmp/rh-test",
		taskContext: resolveTaskContext({
			rosclawHome: "/tmp/rh-test", cwd: "/tmp", mode: "SIMULATION",
		}),
	});
	const pi = {
		on(name: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) {
			if (name === "tool_execution_update") handlerList.push(handler);
		},
		registerCommand() {}, registerShortcut() {},
		registerEntryRenderer() {}, registerMessageRenderer() {},
		appendEntry() {},
	};
	factory(pi as never);
	assert.ok(handlerList.length >= 1, "tool_execution_update 处理器未注册");
	const ctx = {
		hasUI: true,
		ui: {
			setWorkingMessage() {},
			notify() {},
			select: async () => "本任务允许（当前 revision）",
		},
	};
	const event = {
		toolName: "bash",
		partialResult: {
			details: { phase: "AWAITING_SHELL_APPROVAL", request_id: "shg_x1" },
		},
	};
	for (const h of handlerList) await h(event, ctx);
	assert.equal(decideCalls.length, 1, "decide 未被调用");
	assert.equal(decideCalls[0].request_id, "shg_x1");
	assert.equal(decideCalls[0].decision, "allow_task",
		"本任务允许 应映射 allow_task");
});

test("R1-a: 卡上选拒绝/超时未选 → deny（fail closed）", async () => {
	const handlerList: Array<(event: unknown, ctx: unknown) => Promise<unknown>> = [];
	const decideCalls: Array<Record<string, unknown>> = [];
	const { createRosclawExtension } = await import("../src/extension/index.js");
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const { resolveTaskContext } = await import("../src/native/active-task-context.js");
	const active = new ActiveSessionContext({
		sessionId: "pi_test", missionId: undefined, contextRevision: 0,
		mode: "SIMULATION", profile: "developer", contextState: "LOADING",
		leaseState: "NONE", actionsAllowed: false,
	});
	const call = async (_home: string, method: string, params?: Record<string, unknown>) => {
		if (method === "pi.shell_gate.decide") {
			decideCalls.push(params ?? {});
			return { ok: true };
		}
		return { ok: false } as Record<string, unknown>;
	};
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh-test", active,
		leaseManager: new SessionLeaseManager("/tmp/rh-test", call),
		notify: () => undefined, call,
	});
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/rh-test", active,
		operatorSocket: "/tmp/rh-test/run/operatord.sock",
		productVersion: "0.1.0", call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager("/tmp/rh-test/agent");
	const factory = createRosclawExtension({
		profile: "developer", version: "0.1.0", systemPrompt: "TEST",
		active, coordinator, center, locale, rosclawHome: "/tmp/rh-test",
		taskContext: resolveTaskContext({
			rosclawHome: "/tmp/rh-test", cwd: "/tmp", mode: "SIMULATION",
		}),
	});
	const pi = {
		on(name: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) {
			if (name === "tool_execution_update") handlerList.push(handler);
		},
		registerCommand() {}, registerShortcut() {},
		registerEntryRenderer() {}, registerMessageRenderer() {},
		appendEntry() {},
	};
	factory(pi as never);
	const ctx = {
		hasUI: true,
		ui: {
			setWorkingMessage() {},
			notify() {},
			select: async () => undefined, // 超时/取消 = undefined
		},
	};
	const event = {
		toolName: "bash",
		partialResult: {
			details: { phase: "AWAITING_SHELL_APPROVAL", request_id: "shg_x2" },
		},
	};
	for (const h of handlerList) await h(event, ctx);
	assert.equal(decideCalls.length, 1);
	assert.equal(decideCalls[0].decision, "deny", "未选择 = deny（fail closed）");
});
