import assert from "node:assert/strict";
import test from "node:test";

import { createRosclawExtension } from "../src/extension/index.js";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

function collectHandlers() {
	const handlers = new Map<string, Handler>();
	const commands = new Map<string, { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }>();
	const pi = {
		on(name: string, handler: Handler) {
			handlers.set(name, handler);
		},
		registerCommand(name: string, options: { description?: string; handler: (args: string, ctx: unknown) => Promise<void> }) {
			commands.set(name, options);
		},
	};
	const factory = createRosclawExtension({ profile: "developer", version: "0.1.0", systemPrompt: "TEST PROMPT", rosclawHome: "/tmp/rh-test" });
	factory(pi as never);
	return { handlers, commands };
}

test("user_bash is fully replaced by a policy refusal (PNA-0 safety)", async () => {
	const { handlers } = collectHandlers();
	const handler = handlers.get("user_bash");
	assert.ok(handler, "user_bash handler must be registered");
	const result = (await handler({}, {})) as {
		result: { output: string; exitCode: number };
	};
	assert.equal(result.result.exitCode, 1);
	assert.match(result.result.output, /disabled by ROSClaw policy/);
});

test("session lifecycle hooks registered (fork veto point)", async () => {
	const { handlers } = collectHandlers();
	assert.ok(handlers.get("session_start"), "session_start");
	assert.ok(handlers.get("session_before_fork"), "session_before_fork");
});

test("worker commands registered (/workers /delegate)", () => {
	const { commands } = collectHandlers();
	assert.ok(commands.get("workers"), "/workers");
	assert.ok(commands.get("delegate"), "/delegate");
});

test("/delegate without mission binding refuses honestly", async () => {
	const { commands } = collectHandlers();
	const notifications: Array<{ message: string; type?: string }> = [];
	const ctx = { ui: { notify: (message: string, type?: string) => notifications.push({ message, type }) } };
	await commands.get("delegate")!.handler("auto 做点事", ctx);
	assert.ok(notifications.some((n) => n.message.includes("未绑定 Mission")));
});
