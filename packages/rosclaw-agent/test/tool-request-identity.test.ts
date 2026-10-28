/** Real e344 Native parallel inspection exposed a millisecond identity collision. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { materializeCapabilityTools } from "../src/tools/materialize.js";
import { buildBridgeTools, type BridgeToolContext } from "../src/tools/bridge-tools.js";
import { buildRequestActionTool } from "../src/tools/request-action.js";

function fixture() {
	const requests: Record<string, unknown>[] = [];
	const proposals: Record<string, unknown>[] = [];
	const cache = new Map<string, string>();
	const state = { sessionId: "pi-one", missionId: "mission-one", contextRevision: 1,
		contextLeaseId: "lease-one", bodyHash: "body-hash", mode: "SIMULATION" };
	const ctx = { rosclawHome: "/tmp/fixture", active: { current: state }, center: {
		async actionReadiness() { return { state: "READY", reason_codes: [] }; },
		async call(method: string, parameters: Record<string, unknown>) {
			if (method === "pi.action.propose") {
				proposals.push(parameters);
				return { ok: true, card: { approval_id: "fixture-approval", decision_authority: "POLICY_AUTO" } };
			}
			if (method === "pi.action.execute") return { ok: true, result: { status: "COMPLETED", executed: true } };
			assert.equal(method, "pi.tools.execute");
			const request = parameters.request as Record<string, unknown>;
			requests.push(request);
			const key = String(request.idempotency_key);
			const intent = JSON.stringify([request.pi_session_id, request.mission_id, request.tool_name, request.arguments]);
			if (cache.has(key) && cache.get(key) !== intent) return { ok: false, result: { error_code: "IDEMPOTENCY_CONFLICT" } };
			cache.set(key, intent);
			return { ok: true, result: { summary: JSON.stringify(request.arguments) } };
		},
	} } as unknown as BridgeToolContext;
	const snapshot = { schema_version: "rosclaw.capability_snapshot.v1", generation: 1, digest: "snapshot-one",
		body_id: "fixture", mode: "SIMULATION", excluded: [], active: ["get_body_profile", "get_runtime_status", "ros_inspect_system"].map(name => ({
			tool_name: name, capability_id: `fixture.${name}`, exposure: "direct" as const,
			effect_class: "READ_ONLY", description: name, input_schema: { type: "object" }, output_schema: {},
		})),
	};
	return { ctx, state, requests, proposals, tools: materializeCapabilityTools(snapshot, ctx) };
}

function errorFlag(result: unknown) {
	assert.ok(result !== null && typeof result === "object" && "isError" in result);
	return result.isError;
}

const signal = new AbortController().signal;
const invoke = (tool: ReturnType<typeof materializeCapabilityTools>[number], id: string) =>
	tool.execute(id, {}, signal, async () => {}, {} as never);

test("parallel projected inspections at the same millisecond keep distinct intents", async t => {
	t.mock.method(Date, "now", () => 1793162068785);
	const f = fixture();
	const results = await Promise.all(f.tools.map((tool, i) => invoke(tool, `actual-call-${i}`)));
	assert.equal(new Set(f.requests.map(r => r.request_id)).size, 3);
	assert.equal(new Set(f.requests.map(r => r.idempotency_key)).size, 3);
	assert.ok(results.every(result => errorFlag(result) === false));
});

test("same projected call preserves identity across a clock change", async t => {
	let now = 1000;
	t.mock.method(Date, "now", () => now);
	const f = fixture();
	await invoke(f.tools[0], "one-logical-call"); now = 900;
	await invoke(f.tools[0], "one-logical-call");
	assert.equal(f.requests[0].request_id, f.requests[1].request_id);
	assert.equal(f.requests[0].idempotency_key, f.requests[1].idempotency_key);
});

test("same call in another mission or session cannot inherit the prior cache", async t => {
	t.mock.method(Date, "now", () => 1000);
	const f = fixture();
	await invoke(f.tools[0], "call");
	f.state.missionId = "mission-two"; await invoke(f.tools[0], "call");
	f.state.sessionId = "pi-two"; await invoke(f.tools[0], "call");
	assert.equal(new Set(f.requests.map(r => r.idempotency_key)).size, 3);
});

test("changing intent under the same logical call still conflicts", async t => {
	t.mock.method(Date, "now", () => 1000);
	const f = fixture();
	await invoke(f.tools[0], "reused-call");
	const conflict = await invoke(f.tools[1], "reused-call");
	assert.equal(errorFlag(conflict), true);
	assert.equal((conflict.details as Record<string, unknown>).error_code, "IDEMPOTENCY_CONFLICT");
});

test("legacy bridge tools preserve the supplied logical call identity", async t => {
	let now = 1000;
	t.mock.method(Date, "now", () => now);
	const f = fixture();
	const tool = buildBridgeTools(f.ctx).find(tool => tool.name === "rosclaw_observe");
	assert.ok(tool);
	await tool.execute("one-call", { capability_id: "fixture.read" }, signal, async () => {}, {} as never);
	now = 2000;
	await tool.execute("one-call", { capability_id: "fixture.read" }, signal, async () => {}, {} as never);
	assert.equal(f.requests[0].idempotency_key, f.requests[1].idempotency_key);
});

test("physical proposals keep stable identity and distinct calls do not collide", async t => {
	let now = 1000;
	t.mock.method(Date, "now", () => now);
	const f = fixture(); const tool = buildRequestActionTool(f.ctx);
	const args = { capability_id: "fixture.action", arguments: {} };
	await tool.execute("action-one", args, signal, async () => {}, {} as never);
	now = 2000;
	await Promise.all([tool.execute("action-one", args, signal, async () => {}, {} as never),
		tool.execute("action-two", args, signal, async () => {}, {} as never)]);
	assert.equal(f.proposals[0].idempotency_key, f.proposals[1].idempotency_key);
	assert.notEqual(f.proposals[1].idempotency_key, f.proposals[2].idempotency_key);
});

test("absent caller IDs are independent fresh calls even with a frozen clock", async t => {
	t.mock.method(Date, "now", () => 1000);
	const f = fixture();
	await Promise.all([invoke(f.tools[0], ""), invoke(f.tools[0], "")]);
	assert.notEqual(f.requests[0].idempotency_key, f.requests[1].idempotency_key);
});

test("process start retries retain the same original operation intent", async t => {
	const { buildProcessTools } = await import("../src/tools/process-tools.js");
	let now = 1000;
	t.mock.method(Date, "now", () => now);
	const f = fixture(); const tool = buildProcessTools(f.ctx).find(tool => tool.name === "process_start");
	assert.ok(tool);
	await tool.execute("original-process", { command: "fixture-only" }, signal, async () => {}, {} as never);
	now = 3000;
	await tool.execute("original-process", { command: "fixture-only" }, signal, async () => {}, {} as never);
	assert.equal(f.requests[0].idempotency_key, f.requests[1].idempotency_key);
	assert.equal(f.requests[0].request_id, f.requests[1].request_id);
});
