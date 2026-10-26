import assert from "node:assert/strict";
import test from "node:test";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { envelopeHash } from "../src/extension/context-injection.js";
import { refreshModelContext } from "../src/harness/pi/model-context.js";

function harness(mutation = "") {
	const active = new ActiveSessionContext({ sessionId: "s", missionId: "m", contextRevision: 1,
		bodyId: "b", bodyHash: "hash", bindingId: "binding", mode: "SIMULATION", profile: "developer",
		contextLeaseId: "expired_after_slow_parallel_tool", contextState: "FRESH", leaseState: "ACTIVE", actionsAllowed: true });
	let calls = 0;
	const center = { call: async () => {
		calls++;
		if (mutation === "unreachable") throw Error("transport unavailable");
		const e = { schema_version: "rosclaw.embodied_context.v1", mission_id: "m", context_revision: calls + 1,
			generated_at: new Date().toISOString(), expires_at: new Date(Date.now()+30000).toISOString(),
			body: { body_id: "b", effective_body_hash: "hash" }, safety: { mode: "SIMULATION" },
			pending_approvals: [], hash: "" };
		if (mutation === "body") e.body.effective_body_hash = "other";
		if (mutation === "mode") e.safety.mode = "REAL";
		if (mutation === "mission") e.mission_id = "other";
		if (mutation === "binding") active.patch({ bindingId: "other" });
		e.hash = envelopeHash(e);
		return { ok: true, context: e, context_lease_id: `lease_${calls}`,
			context_lease_expires_at: new Date(Date.now()+(mutation === "expired" ? -1000 : 30000)).toISOString() };
	} };
	return { active, ctx: { active, center: center as never, rosclawHome: "/tmp/model-context-test" }, count: () => calls };
}

test("each model decision receives a fresh lease after tool batches, old trusted facts are replaced", async () => {
	const h = harness();
	const input = [{ role: "user" as const, content: "task", timestamp: 1 }];
	const first = await refreshModelContext(h.ctx, input);
	assert.equal(h.active.current.contextLeaseId, "lease_1");
	// Simulate a long tool batch completing; model context refresh must not be
	// limited to before_agent_start or to the earlier physical tool's end.
	const second = await refreshModelContext(h.ctx, first);
	assert.equal(h.count(),2);
	assert.equal(h.active.current.contextLeaseId,"lease_2");
	assert.equal(second.length,2);
	assert.equal(second[0],input[0]);
	assert.match(JSON.stringify(second[1]),/revision: 3/);
	assert.match(JSON.stringify(second[1]),/each_model_request/);
});

for (const mutation of ["unreachable", "expired", "body", "mode", "mission", "binding"]) {
	test(`model context fails closed on ${mutation}`, async () => {
		const h = harness(mutation); const messages = await refreshModelContext(h.ctx,[]);
		assert.equal(h.active.current.contextState,"STALE");
		assert.equal(h.active.current.actionsAllowed,false);
		assert.match(JSON.stringify(messages),/stale=\\"true/);
	});
}
