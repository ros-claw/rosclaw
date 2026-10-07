// Opt-in declared local artifact schema — public tool surface tests.
// schema_path is an optional declared parameter on rosclaw_deliver only;
// absent schema_path must preserve legacy registration behavior, and the
// bounded validation itself lives in the private Python bridge (see
// tests/agentd/test_pi_tool_bridge.py).
import assert from "node:assert/strict";
import { test } from "node:test";
import { validateToolArguments } from "@earendil-works/pi-ai";
import { buildProductPackTools } from "../src/tools/product-pack.js";
import type { BridgeToolContext } from "../src/tools/bridge-tools.js";

function deliverTool(ctx?: Partial<BridgeToolContext>) {
	const tool = buildProductPackTools(ctx as BridgeToolContext).find((t) => t.name === "rosclaw_deliver");
	assert.ok(tool, "rosclaw_deliver tool missing");
	return tool;
}

test("declared schema: schema_path is a public optional parameter", () => {
	const tool = deliverTool();
	const params = (tool as unknown as { parameters?: { properties?: Record<string, unknown> } }).parameters;
	assert.ok(params?.properties?.schema_path, "schema_path must be declared on the public tool parameters");
	// Absent schema_path stays valid (legacy behavior preserved).
	const legacy = validateToolArguments(tool, {
		type: "toolCall", id: "d1", name: tool.name,
		arguments: { path: "delivery.json", role: "progress_report" },
	});
	assert.equal("schema_path" in legacy, false);
	// A local schema path is accepted.
	const declared = validateToolArguments(tool, {
		type: "toolCall", id: "d2", name: tool.name,
		arguments: { path: "delivery.json", role: "progress_report", schema_path: "schema.json" },
	});
	assert.equal(declared.schema_path, "schema.json");
});

test("declared schema: blank or non-string schema_path is rejected by the public validator", () => {
	const tool = deliverTool();
	// Note: scalar numbers/booleans are coerced to strings by the PI layer
	// (same coercion semantics as other string parameters); blank strings
	// and structured values are rejected here, and a coerced but non-existent
	// path is still typed-rejected at the bridge before any registration.
	for (const bad of ["", {}, []]) {
		assert.throws(() => validateToolArguments(tool, {
			type: "toolCall", id: "d3", name: tool.name,
			arguments: { path: "delivery.json", role: "progress_report", schema_path: bad },
		}), `invalid schema_path accepted: ${JSON.stringify(bad)}`);
	}
});

test("declared schema: execute forwards schema_path and session cwd to the bridge", async () => {
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	const ctx = {
		rosclawHome: "/tmp/home",
		workspaceRoot: "/tmp/project",
		active: { current: { sessionId: "s1", missionId: "m1", contextRevision: 3, bodyHash: "b", mode: "SIMULATION" } },
		center: {
			call: async (method: string, params: Record<string, unknown>) => {
				calls.push({ method, params });
				return { ok: true, result: { ok: true, status: "REGISTERED", summary: "ok" } };
			},
		},
	} as unknown as BridgeToolContext;
	const tool = deliverTool(ctx);
	await tool.execute("id", { path: "/tmp/project/delivery.json", role: "progress_report", schema_path: "/tmp/project/schema.json" }, undefined, undefined, {} as never);
	assert.equal(calls.length, 1);
	const request = (calls[0].params as { request: { tool_name: string; arguments: Record<string, unknown> } }).request;
	assert.equal(request.tool_name, "rosclaw_deliver");
	assert.equal(request.arguments.schema_path, "/tmp/project/schema.json");
	assert.equal(request.arguments.cwd, "/tmp/project");
	assert.equal(request.arguments.role, "progress_report");
});

test("declared schema: absent schema_path forwards no schema argument (legacy wire shape)", async () => {
	const calls: Array<{ params: Record<string, unknown> }> = [];
	const ctx = {
		rosclawHome: "/tmp/home",
		workspaceRoot: "/tmp/project",
		active: { current: { sessionId: "s1", missionId: "m1", contextRevision: 3, bodyHash: "b", mode: "SIMULATION" } },
		center: {
			call: async (_method: string, params: Record<string, unknown>) => {
				calls.push({ params });
				return { ok: true, result: { ok: true, status: "REGISTERED", summary: "ok" } };
			},
		},
	} as unknown as BridgeToolContext;
	const tool = deliverTool(ctx);
	await tool.execute("id", { path: "delivery.json", role: "progress_report" }, undefined, undefined, {} as never);
	const request = (calls[0].params as { request: { arguments: Record<string, unknown> } }).request;
	assert.equal("schema_path" in request.arguments, false);
});
