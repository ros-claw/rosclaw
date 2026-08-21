import { resolveTaskContext } from "../src/native/active-task-context.js";
import assert from "node:assert/strict";
import test from "node:test";

import { createRosclawExtension } from "../src/extension/index.js";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

async function collectHandlers() {
	const handlers = new Map<string, Handler>();
	const commands = new Map<string, { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }>();
	const pi = {
		on(name: string, handler: Handler) {
			handlers.set(name, handler);
		},
		registerCommand(name: string, options: { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }) {
			commands.set(name, options);
		},
		registerShortcut() {},
		// P0-5F：内核结果卡/冲突条目的渲染器与落盘 API（mock 空实现）。
		registerEntryRenderer() {},
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
	return { handlers, commands };
}

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
