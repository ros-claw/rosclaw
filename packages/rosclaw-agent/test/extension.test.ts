import assert from "node:assert/strict";
import test from "node:test";

import { createRosclawExtension } from "../src/extension/index.js";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

function collectHandlers() {
	const handlers = new Map<string, Handler>();
	const pi = {
		on(name: string, handler: Handler) {
			handlers.set(name, handler);
		},
	};
	const factory = createRosclawExtension({ profile: "developer", version: "0.1.0", systemPrompt: "TEST PROMPT", rosclawHome: "/tmp/rh-test" });
	factory(pi as never);
	return handlers;
}

test("user_bash is fully replaced by a policy refusal (PNA-0 safety)", async () => {
	const handlers = collectHandlers();
	const handler = handlers.get("user_bash");
	assert.ok(handler, "user_bash handler must be registered");
	const result = (await handler({}, {})) as {
		result: { output: string; exitCode: number };
	};
	assert.equal(result.result.exitCode, 1);
	assert.match(result.result.output, /disabled by ROSClaw policy/);
});

test("session lifecycle hooks registered (fork veto point)", async () => {
	const handlers = collectHandlers();
	assert.ok(handlers.get("session_start"), "session_start");
	assert.ok(handlers.get("session_before_fork"), "session_before_fork");
});
