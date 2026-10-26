import assert from "node:assert/strict";
import test from "node:test";
import { envelopeHash } from "../src/extension/context-injection.js";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { buildRequestActionTool } from "../src/tools/request-action.js";

for (const auto of [true, false]) {
    for (const failure of ["NEEDS_REPLAN", "ROS_PRECONDITION_UNKNOWN_OR_FAILED", "EXECUTE_FAILED", "MALFORMED", "TRANSPORT"]) {
        test(`post-approval ${failure} remains visible and never replays (auto=${auto})`, async () => {
            const active = new ActiveSessionContext({ sessionId: "pi_1", missionId: "mis_1",
                contextRevision: 1, mode: "SIMULATION", profile: "developer", contextState: "FRESH",
                leaseState: "ACTIVE", actionsAllowed: true, contextLeaseId: "lease", bodyId: "body", bodyHash: "hash" });
            const calls: string[] = [];
            const center = { actionReadiness: async () => ({ state: "READY", reason_codes: [] }),
                call: async (method: string) => {
                    calls.push(method);
                    if (method === "pi.action.propose") return { ok: true, card: { approval_id: "owned-card",
                        decision_authority: auto ? "POLICY_AUTO" : "OPERATOR" } };
                    if (method === "pi.action.status") return { ok: true, status: "APPROVED" };
                    if (method === "pi.action.execute") {
                        if (failure === "TRANSPORT") throw Error("socket disconnected");
                        if (failure === "MALFORMED") return { ok: false, result: {} };
                        return { ok: false, code: failure, error: "required public evidence changed" };
                    }
                    throw Error(method);
                } };
            const tool = buildRequestActionTool({ rosclawHome: "/tmp/post-approval", active, center: center as never });
            const result = await tool.execute("t", { capability_id: "fixture.move", arguments: {} }, undefined, undefined, {} as never);
            const details = result.details as Record<string, unknown>;
            assert.equal(details.status, ["NEEDS_REPLAN", "ROS_PRECONDITION_UNKNOWN_OR_FAILED"].includes(failure) ? "REJECTED" : "UNKNOWN");
            assert.equal(details.approval_id, "owned-card");
            assert.equal(details.reconciliation_required, details.status === "UNKNOWN");
            assert.equal((result as { isError?: boolean }).isError, true);
            assert.ok(JSON.stringify(result.content).length > 80);
            assert.ok(details.error_code);
            assert.equal(calls.filter(m => m === "pi.action.execute").length, 1);
            assert.equal(calls.filter(m => m === "pi.action.propose").length, 1);
        });
    }
}

function harness(mutation = "", mode = "SIMULATION", withCard = false) {
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
				if (++proposals === 1) return { ok: false, code: "NEEDS_REPLAN",
					...(withCard ? { card: { approval_id: "must_not_reuse" } } : {}) };
				return { ok: true, card: { approval_id: "new_card", decision_authority: "POLICY_AUTO" } };
			}
			if (method === "pi.context") {
				const envelope = { schema_version: "rosclaw.embodied_context.v1", mission_id: "mis_1",
					context_revision: 2, generated_at: new Date().toISOString(),
					expires_at: new Date(Date.now() + 60_000).toISOString(),
					body: { body_id: "fixture", effective_body_hash: "body_1" },
					safety: { mode }, pending_approvals: [], hash: "" };
				if (mutation === "body") envelope.body.effective_body_hash = "changed";
				if (mutation === "mode") envelope.safety.mode = "REAL";
				if (mutation === "mission") envelope.mission_id = "other";
				if (mutation === "session") active.patch({ sessionId: "other" });
				envelope.hash = envelopeHash(envelope);
				return { ok: true, context: envelope,
					context_lease_id: mutation === "missing_lease" ? undefined : "new_lease",
					context_lease_expires_at: new Date(Date.now() + (mutation === "expired_lease" ? -1000 : 60_000)).toISOString() };
			}
			if (method === "pi.action.execute") return { ok: true, result: { executed: true, status: "COMPLETED" } };
			throw Error(method);
		},
	};
	const tool = buildRequestActionTool({ rosclawHome: "/tmp/replan-test", active, center: center as never });
	const execute = (x: number) => tool.execute("test", { capability_id: "fixture.move", arguments: { x } }, undefined, undefined, {} as never);
	return { active, calls, execute };
}

test("NEEDS_REPLAN refreshes and returns a real observation, never replays the rejected action", async () => {
	const h = harness(); const result = await h.execute(1);
	assert.deepEqual(h.calls.map(c => c.method), ["pi.action.propose", "pi.context"]);
	assert.equal((result.details as Record<string, unknown>).ok, false);
	assert.equal((result.details as Record<string, unknown>).error_code, "NEEDS_REPLAN");
	assert.equal((result.details as Record<string, unknown>).context_refreshed, true);
	assert.equal((result.details as Record<string, unknown>).replan_required, true);
	assert.match(JSON.stringify(result.content), /ROSCLAW_TRUSTED_CONTEXT/);
	assert.equal(h.active.current.contextRevision, 2);
	await h.execute(2); // A separate model decision with revised arguments.
	assert.deepEqual(h.calls.map(c => c.method), ["pi.action.propose", "pi.context", "pi.action.propose", "pi.action.execute"]);
	assert.deepEqual(h.calls[2].params.arguments, { x: 2 });
	assert.equal(h.calls[2].params.context_lease_id, "new_lease");
});

for (const mutation of ["body", "mode", "mission", "session", "missing_lease", "expired_lease"]) {
	test(`NEEDS_REPLAN cannot refresh across ${mutation}`, async () => {
		const h = harness(mutation); const result = await h.execute(1);
		assert.deepEqual(h.calls.map(c => c.method), ["pi.action.propose", "pi.context"]);
		assert.equal(h.active.current.contextState, "STALE");
		assert.equal((result.details as Record<string, unknown>).context_refreshed, false);
		assert.equal((result.details as Record<string, unknown>).ok, false);
	});
}

for (const mode of ["REAL", "SHADOW"]) {
	test(`NEEDS_REPLAN does not auto-refresh ${mode}`, async () => {
		const h = harness("", mode); await h.execute(1);
		assert.deepEqual(h.calls.map(c => c.method), ["pi.action.propose"]);
	});
}
test("a rejection containing a card is never refreshed or executed", async () => {
	const h = harness("", "SIMULATION", true); await h.execute(1);
	assert.deepEqual(h.calls.map(c => c.method), ["pi.action.propose"]);
});

test("unknown action reconciliation queries the exact approval and never proposes or dispatches", async () => {
	const active = new ActiveSessionContext({ sessionId: "session", missionId: "mission", contextRevision: 1, bodyHash: "body", mode: "SIMULATION", profile: "developer", contextState: "STALE", leaseState: "LOST", actionsAllowed: false });
	const calls: string[] = [];
	const center = { call: async (method: string, params: Record<string, unknown>) => {
		calls.push(method); assert.equal(params.approval_id, "original-approval");
		return { ok: false, result: { status: "UNKNOWN", executed: false, reconciliation_required: true, approval_id: "original-approval" } };
	} };
	const tool = buildRequestActionTool({ rosclawHome: "/tmp/reconcile", active, center: center as never });
	const result = await tool.execute("query", { capability_id: "fixture.move", arguments: {}, approval_id: "original-approval" }, undefined, undefined, {} as never);
	assert.deepEqual(calls, ["pi.action.reconcile"]);
	assert.equal((result.details as Record<string, unknown>).status, "UNKNOWN");
	assert.equal((result.details as Record<string, unknown>).reconciliation_required, true);
});
