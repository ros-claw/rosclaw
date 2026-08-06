import assert from "node:assert/strict";
import test from "node:test";
import { CommandRegistry, LOCAL_COMMANDS } from "../src/commands/registry.js";
import type { CommandSpec } from "../src/client/http.js";

function spec(name: string, disabled = ""): CommandSpec {
	return {
		name,
		aliases: [],
		description: `${name} desc`,
		argument_hint: "",
		category: "mission",
		owner: "MISSION_CONTROL",
		availability: [],
		during_turn: false,
		mutability: "NONE",
		confirmation: "NONE",
		required_capabilities: [],
		handler: `h.${name}`,
		disabled_reason: disabled,
	};
}

test("local commands route locally", () => {
	const registry = new CommandRegistry();
	assert.deepEqual(registry.parse("/help").kind, "local");
	assert.deepEqual(registry.parse("/quit").kind, "local");
	assert.deepEqual(registry.parse("/q").kind, "local");
});

test("remote commands route to control api", () => {
	const registry = new CommandRegistry();
	registry.loadRemote([spec("compact"), spec("rename")]);
	const route = registry.parse("/compact dry-run");
	assert.equal(route.kind, "remote");
	if (route.kind === "remote") {
		assert.equal(route.spec.name, "compact");
		assert.equal(route.args, "dry-run");
	}
});

test("unknown command is never sent to the model", () => {
	const registry = new CommandRegistry();
	const route = registry.parse("/nosuchcommand");
	assert.equal(route.kind, "unknown");
});

test("estop routes locally to the operator channel, never the model", () => {
	const registry = new CommandRegistry();
	const route = registry.parse("/estop");
	assert.equal(route.kind, "local");
});

test("plain text is not a command", () => {
	const registry = new CommandRegistry();
	assert.equal(registry.parse("你好").kind, "not_a_command");
});

test("help lists local + remote with disabled reasons", () => {
	const registry = new CommandRegistry();
	registry.loadRemote([spec("compact", "turn 运行中不可用")]);
	const all = registry.all();
	assert.ok(all.length >= LOCAL_COMMANDS.length + 1);
	const compact = all.find((c) => c.name === "compact");
	assert.equal(compact?.disabled, "turn 运行中不可用");
});
