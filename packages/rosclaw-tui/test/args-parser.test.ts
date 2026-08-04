import assert from "node:assert/strict";
import test from "node:test";
import { parseArgs, type ArgsSchema } from "../src/commands/args-parser.js";

const S = (schema: ArgsSchema) => schema;

test("positional rest captures multi-word", () => {
	const schema = S({ positional: [{ name: "goal", type: "rest", required: true }] });
	const out = parseArgs(schema, "把 红色 方块 移到 B 区");
	assert.deepEqual(out, { ok: true, args: { goal: "把 红色 方块 移到 B 区" } });
});

test("required positional missing is an error, never sent", () => {
	const schema = S({ positional: [{ name: "name", type: "rest", required: true }] });
	const out = parseArgs(schema, "");
	assert.equal(out.ok, false);
	assert.match(out.error ?? "", /缺少参数/);
});

test("enum validation", () => {
	const schema = S({
		positional: [
			{ name: "subcommand", type: "enum", enum: ["inspect", "enable"], required: true },
			{ name: "worker_id", type: "string", required: true },
		],
	});
	assert.deepEqual(parseArgs(schema, "inspect worker:native:basic"), {
		ok: true,
		args: { subcommand: "inspect", worker_id: "worker:native:basic" },
	});
	const bad = parseArgs(schema, "delete worker:x");
	assert.equal(bad.ok, false);
	assert.match(bad.error ?? "", /inspect\|enable/);
});

test("flags parsed before positionals", () => {
	const schema = S({
		positional: [{ name: "focus", type: "rest", required: false }],
		flags: { "dry-run": { type: "boolean" } },
	});
	assert.deepEqual(parseArgs(schema, "dry-run"), { ok: true, args: { "dry_run": true } });
	assert.deepEqual(parseArgs(schema, "dry-run 关注授权"), {
		ok: true,
		args: { "dry_run": true, focus: "关注授权" },
	});
	assert.deepEqual(parseArgs(schema, "关注授权"), { ok: true, args: { focus: "关注授权" } });
});

test("no-args command rejects stray args", () => {
	const out = parseArgs(S({ interaction: "none" }), "unexpected");
	assert.equal(out.ok, false);
});

test("optional positionals may be omitted", () => {
	const schema = S({
		positional: [
			{ name: "key", type: "string", required: false },
			{ name: "value", type: "rest", required: false },
		],
	});
	assert.deepEqual(parseArgs(schema, ""), { ok: true, args: {} });
	assert.deepEqual(parseArgs(schema, "agent.language"), { ok: true, args: { key: "agent.language" } });
	assert.deepEqual(parseArgs(schema, "agent.language zh-CN"), {
		ok: true,
		args: { key: "agent.language", value: "zh-CN" },
	});
});
