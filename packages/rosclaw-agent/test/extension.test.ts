import { resolveTaskContext } from "../src/native/active-task-context.js";
import assert from "node:assert/strict";
import test from "node:test";

import { createRosclawExtension } from "../src/extension/index.js";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

async function collectHandlers() {
	const handlers = new Map<string, Handler>();
	const allHandlers = new Map<string, Handler[]>();
	const commands = new Map<string, { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }>();
	const pi = {
		on(name: string, handler: Handler) {
			handlers.set(name, handler);
			allHandlers.set(name, [...(allHandlers.get(name) ?? []), handler]);
		},
		registerCommand(name: string, options: { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }) {
			commands.set(name, options);
		},
		registerShortcut() {},
		// P0-5F：内核结果卡/冲突条目的渲染器与落盘 API（mock 空实现）。
		registerEntryRenderer() {},
		// P0-6（0901）：内核消息卡渲染器注册（mock 空实现）。
		registerMessageRenderer() {},
		appendEntry() {},
	};
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");

	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: undefined,
		contextRevision: 0,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "LOADING",
		leaseState: "NONE",
		actionsAllowed: false,
	});
	const call = async () => ({ ok: false, error: "no bridge in test" });
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh-test",
		active,
		leaseManager: new SessionLeaseManager("/tmp/rh-test", call),
		notify: () => undefined,
		call,
	});
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/rh-test",
		active,
		operatorSocket: "/tmp/rh-test/run/operatord.sock",
		productVersion: "0.1.0",
		call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager("/tmp/rh-test/agent");
	const factory = createRosclawExtension({ profile: "developer", version: "0.1.0", systemPrompt: "TEST PROMPT", active, coordinator, center, locale, rosclawHome: "/tmp/rh-test", taskContext: resolveTaskContext({ rosclawHome: "/tmp/rh-test", cwd: "/tmp", mode: "SIMULATION" }) });
	factory(pi as never);
	return { handlers, allHandlers, commands };
}

test("tool activity clears completed tools and preserves overlapping calls", async () => {
	const { allHandlers } = await collectHandlers();
	const labels: string[] = [];
	const ctx = { hasUI: true, ui: { setWorkingMessage: (s: string) => labels.push(s) } };
	const emit = async (name: string, event: unknown) => {
		for (const handler of allHandlers.get(name) ?? []) await handler(event, ctx);
	};
	await emit("tool_execution_start", { toolCallId: "bash-1", toolName: "bash" });
	assert.equal(labels.at(-1), "调用 bash");
	await emit("tool_execution_start", { toolCallId: "read-1", toolName: "read" });
	await emit("tool_execution_end", { toolCallId: "bash-1", toolName: "bash" });
	assert.equal(labels.at(-1), "调用 read");
	await emit("tool_execution_end", { toolCallId: "read-1", toolName: "read" });
	assert.equal(labels.at(-1), "Working…");
	await emit("tool_execution_start", { toolCallId: "bash-2", toolName: "bash" });
	await emit("tool_execution_end", { toolCallId: "bash-2", toolName: "bash", isError: true });
	assert.equal(labels.at(-1), "Working…");
});

test("user_bash is fully replaced by a policy refusal (PNA-0 safety)", async () => {
	const { handlers } = await collectHandlers();
	const handler = handlers.get("user_bash");
	assert.ok(handler, "user_bash handler must be registered");
	const result = (await handler({}, {})) as {
		result: { output: string; exitCode: number };
	};
	assert.equal(result.result.exitCode, 1);
	assert.match(result.result.output, /disabled by ROSClaw policy/);
});

test("session lifecycle hooks registered (fork veto point)", async () => {
	const { handlers } = await collectHandlers();
	assert.ok(handlers.get("session_start"), "session_start");
	assert.ok(handlers.get("session_before_fork"), "session_before_fork");
});
