import assert from "node:assert/strict";
import test from "node:test";

import { envelopeHash } from "../src/extension/context-injection.js";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { buildRequestActionTool } from "../src/tools/request-action.js";

async function exercise(mutation = "", mode = "SIMULATION", retryRejected = false) {
	const active = new ActiveSessionContext({ sessionId: "pi_1", missionId: "mis_1",
		contextRevision: 1, mode, profile: "developer", contextState: "FRESH",
		leaseState: "ACTIVE", actionsAllowed: true, contextLeaseId: "old_lease",
		bodyId: "fixture", bodyHash: "body_1" });
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	let proposals = 0;
	const center = {
		actionReadiness: async () => ({ state: "READY", reason_codes: [] }),
		call: async (method: string, params: Record<string, unknown>) => {
			calls.push({ method, params });
			if (method === "pi.action.propose") {
				if (++proposals === 1 || retryRejected) return { ok: false, code: "CONTEXT_HASH_MISMATCH" };
				return { ok: true, card: { approval_id: "approved", display_hash: "display",
					expires_at: new Date(Date.now() + 60_000).toISOString(), decision_authority: "POLICY_AUTO" } };
			}
			if (method === "pi.context") {
				const envelope = { schema_version: "rosclaw.embodied_context.v1",
					mission_id: mutation === "mission" ? "different" : "mis_1", context_revision: 2,
					generated_at: new Date().toISOString(), expires_at: new Date(Date.now() + 60_000).toISOString(),
					body: { body_id: "fixture", effective_body_hash: mutation === "body" ? "different" : "body_1" },
					safety: { mode: mutation === "mode" ? "REAL" : mode }, pending_approvals: [], hash: "" };
				envelope.hash = envelopeHash(envelope);
				if (mutation === "session") active.patch({ sessionId: "different" });
				return { ok: true, context: envelope,
					context_lease_id: mutation === "missing_lease" ? undefined : "new_lease",
					context_lease_expires_at: new Date(Date.now() + (mutation === "expired_lease" ? -1000 : 60_000)).toISOString() };
			}
			if (method === "pi.action.execute") return { ok: true, result: { executed: true, status: "COMPLETED" } };
			throw new Error(`Unexpected method ${method}`);
		},
	};
	const tool = buildRequestActionTool({ rosclawHome: "/tmp/proposal-refresh", active, center: center as never });
	const result = await tool.execute("t", { capability_id: "fixture.move", arguments: { bounded: true } },
		undefined, undefined, {} as never);
	return { calls, result, active };
}

test("card-free SIM rejection returns a new observation without replaying the old intent", async () => {
	const { calls, result, active } = await exercise();
	assert.deepEqual(calls.map(c => c.method), ["pi.action.propose", "pi.context"]);
	assert.equal((result.details as { ok: boolean }).ok, false);
	assert.equal((result.details as { replan_required: boolean }).replan_required, true);
	assert.match(JSON.stringify(result.content), /ROSCLAW_TRUSTED_CONTEXT/);
	assert.equal(active.current.contextRevision, 2);
});

for (const mutation of ["body", "mode", "mission", "session", "missing_lease", "expired_lease"]) {
	test(`SIM refresh fails closed on ${mutation} change`, async () => {
		const { calls, result, active } = await exercise(mutation);
		assert.deepEqual(calls.map(c => c.method), ["pi.action.propose", "pi.context"]);
		assert.equal((result.details as { ok: boolean }).ok, false);
		assert.equal(active.current.contextState, "STALE");
	});
}

for (const mode of ["REAL", "SHADOW"]) {
	test(`${mode} proposal is never automatically refreshed or retried`, async () => {
		const { calls, result } = await exercise("", mode);
		assert.deepEqual(calls.map(c => c.method), ["pi.action.propose"]);
		assert.equal((result.details as { ok: boolean }).ok, false);
	});
}

test("a second context rejection settles without an unbounded retry or execution", async () => {
	const { calls, result } = await exercise("", "SIMULATION", true);
	assert.deepEqual(calls.map(c => c.method), ["pi.action.propose", "pi.context"]);
	assert.equal((result.details as { ok: boolean }).ok, false);
});
