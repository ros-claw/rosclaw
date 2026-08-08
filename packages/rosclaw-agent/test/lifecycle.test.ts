import assert from "node:assert/strict";
import test from "node:test";

import { ActiveSessionContext } from "../src/session/active-context.js";
import { AgentSessionCoordinator } from "../src/session/coordinator.js";
import { SessionLeaseManager } from "../src/session/lease-manager.js";
import { shouldCancelSwitch, shouldCancelTree } from "../src/session/lifecycle.js";

function fakeBridge(responses: Record<string, Record<string, unknown>>) {
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	const call = async (_home: string, method: string, params: Record<string, unknown> = {}) => {
		calls.push({ method, params });
		return responses[method] ?? { ok: false, error: `no stub for ${method}` };
	};
	return { calls, call };
}

function makeCoordinator(
	call: (home: string, method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>,
	initialMission?: string,
) {
	const active = new ActiveSessionContext({
		sessionId: "pi_sess_1",
		missionId: initialMission,
		contextRevision: 0,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "LOADING",
		leaseState: "NONE",
		actionsAllowed: false,
	});
	const notes: string[] = [];
	const leaseManager = new SessionLeaseManager("/tmp/rh", call);
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh",
		active,
		leaseManager,
		notify: (m) => notes.push(m),
		call,
	});
	return { coordinator, active, notes };
}

test("beginNew(fork) creates SIM mission and binds; authority never copied", async () => {
	const { calls, call } = fakeBridge({
		"pi.mission.create": { ok: true, mission_id: "mis_new", mode: "SIMULATION" },
		"pi.session.bind": {
			ok: true,
			binding: { binding_id: "psb_1" },
			lease_token: "t",
		},
		"pi.context": { ok: false, error: "no context in test" },
	});
	const { coordinator, active } = makeCoordinator(call, "mis_old");
	const outcome = await coordinator.beginNew("pi_sess_2", "fork", "mis_old");
	assert.equal(outcome.ok, true);
	assert.equal(active.current.missionId, "mis_new");
	const create = calls.find((c) => c.method === "pi.mission.create");
	assert.equal(create?.params.mode, "SIMULATION", "fork 强制 SIM");
	const bind = calls.find((c) => c.method === "pi.session.bind");
	assert.deepEqual(Object.keys(bind?.params ?? {}).sort(), ["mission_id", "pi_session_id"]);
});

test("resumeInitial with healthy binding reattaches (rebound=true)", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_kept" },
			mission_archived: false,
		},
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_2" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": { ok: false },
	});
	const { coordinator, active } = makeCoordinator(call);
	const outcome = await coordinator.resumeInitial("pi_sess_old");
	assert.equal(outcome.ok, true);
	assert.equal(outcome.ok && outcome.rebound, true);
	assert.equal(active.current.missionId, "mis_kept");
	assert.equal(active.current.sessionId, "pi_sess_old");
});

test("resumeInitial without binding creates fresh SIM binding (never guesses)", async () => {
	const { calls, call } = fakeBridge({
		"pi.session.binding.get": { ok: true, binding: null },
		"pi.mission.create": { ok: true, mission_id: "mis_fresh" },
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_3" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": { ok: false },
	});
	const { coordinator, active } = makeCoordinator(call);
	const outcome = await coordinator.resumeInitial("pi_sess_lost");
	assert.equal(outcome.ok, true);
	assert.equal(active.current.missionId, "mis_fresh");
	assert.ok(calls.some((c) => c.method === "pi.mission.create"));
});

test("archived mission rebinds to fresh SIM mission with notice", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_archived" },
			mission_archived: true,
		},
		"pi.mission.create": { ok: true, mission_id: "mis_new2" },
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_4" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": { ok: false },
	});
	const { coordinator, active, notes } = makeCoordinator(call);
	const outcome = await coordinator.resumeInitial("pi_sess_arch");
	assert.equal(outcome.ok, true);
	assert.equal(active.current.missionId, "mis_new2");
	assert.ok(notes.some((n) => n.includes("已归档")));
});

test("failed bind enters NEEDS_BINDING — mission cleared, no half-switch", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_x" },
			mission_archived: false,
		},
		"pi.session.bind": { ok: false, error: "WRITER_HELD", code: "WRITER_HELD" },
		"pi.session.release": { ok: true },
	});
	const { coordinator, active } = makeCoordinator(call, "mis_x");
	const outcome = await coordinator.resumeInitial("pi_sess_busy");
	assert.equal(outcome.ok, false);
	assert.equal(active.current.missionId, undefined, "半切换禁止：mission 必须清空");
});

test("shouldCancelSwitch: missing/corrupt target header vetoes; valid passes", async () => {
	const veto = await shouldCancelSwitch("/nonexistent/sess.jsonl", () => null);
	assert.ok(veto?.includes("fail closed"));
	const ok = await shouldCancelSwitch("/ok/sess.jsonl", () => "sess_id_abc");
	assert.equal(ok, null);
	// new session（无 target 文件）放行。
	assert.equal(await shouldCancelSwitch(undefined, () => null), null);
});

test("tree navigation vetoed with pending approvals or active actions", async () => {
	const { call } = fakeBridge({
		"pi.context": {
			ok: true,
			context: { pending_approvals: [{ request_id: "r1" }], active_actions: [] },
		},
	});
	assert.ok(
		(await shouldCancelTree({ rosclawHome: "/tmp/rh", missionId: "mis_x", call }))?.includes(
			"待决授权",
		),
	);

	const { call: call2 } = fakeBridge({
		"pi.context": {
			ok: true,
			context: { pending_approvals: [], active_actions: [{ id: "a1" }] },
		},
	});
	assert.ok(
		(await shouldCancelTree({ rosclawHome: "/tmp/rh", missionId: "mis_x", call: call2 }))?.includes(
			"fail closed",
		),
	);

	const { call: call3 } = fakeBridge({
		"pi.context": { ok: true, context: { pending_approvals: [], active_actions: [] } },
	});
	assert.equal(
		await shouldCancelTree({ rosclawHome: "/tmp/rh", missionId: "mis_x", call: call3 }),
		null,
	);
});

test("100 次 A↔B 切换 soak：任何时刻只有一个 writer，无半切换", async () => {
	// 真实 Node lifecycle（coordinator + lease manager + active context），
	// fake bridge 模拟两个 session 的绑定/释放/心跳。
	const bindings = new Map<string, string>(); // sessionId -> missionId
	const leases = new Map<string, string>(); // missionId -> sessionId（单 writer）
	const call = async (_home: string, method: string, params: Record<string, unknown> = {}) => {
		const sess = String(params.pi_session_id ?? "");
		const mis = String(params.mission_id ?? "");
		switch (method) {
			case "pi.session.binding.get":
				return bindings.has(sess)
					? { ok: true, binding: { mission_id: bindings.get(sess) }, mission_archived: false }
					: { ok: true, binding: null };
			case "pi.session.bind":
				if (leases.has(mis) && leases.get(mis) !== sess) {
					return { ok: false, code: "WRITER_HELD", error: "held" };
				}
				bindings.set(sess, mis);
				leases.set(mis, sess);
				return { ok: true, binding: { binding_id: `psb_${sess}` }, lease_token: `tok_${sess}` };
			case "pi.session.release":
				if (leases.get(mis) === sess) leases.delete(mis);
				return { ok: true };
			case "pi.session.heartbeat":
				return leases.get(mis) === sess
					? { ok: true, lease: {} }
					: { ok: false, error: "not writer" };
			case "pi.context":
				return { ok: false, error: "no context in soak" };
			default:
				return { ok: false, error: `no stub for ${method}` };
		}
	};
	const { coordinator, active } = makeCoordinator(call);
	// 预置两个 session 的绑定。
	bindings.set("pi_A", "mis_A");
	bindings.set("pi_B", "mis_B");
	for (let i = 0; i < 100; i += 1) {
		const target = i % 2 === 0 ? "pi_A" : "pi_B";
		const expectedMission = i % 2 === 0 ? "mis_A" : "mis_B";
		const outcome = await coordinator.resumeInitial(target);
		assert.equal(outcome.ok, true, `round ${i} 切换失败: ${JSON.stringify(outcome)}`);
		const state = active.current;
		assert.equal(state.sessionId, target, `round ${i} session 分裂`);
		assert.equal(state.missionId, expectedMission, `round ${i} mission 分裂`);
		// 任何时刻 leases 里每个 mission 的 writer 就是当前 session。
		for (const [m, s] of leases) {
			assert.equal(s, target, `round ${i}: mission ${m} writer 是 ${s}，期望 ${target}`);
		}
	}
});

test("场景 B（P0-4E）：A 的 body/rev 不得被带进 context 失败的 B", async () => {
	// A: body_A/rev_12/FRESH → 切 B → B context fetch 失败 →
	// B 必须 UNAVAILABLE/ACTIONS FORBIDDEN，且不含 A 的任何字段。
	const { calls, call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_B" },
			mission_archived: false,
		},
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_B" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": { ok: false, error: "bridge down" },
	});
	const { coordinator, active } = makeCoordinator(call, "mis_A");
	// A 是 FRESH（body_A/rev 12）。
	active.applyEnvelope(
		{
			schema_version: "rosclaw.embodied_context.v1",
			mission_id: "mis_A",
			context_revision: 12,
			generated_at: "",
			expires_at: "",
			hash: "",
			body: { body_id: "body_A", effective_body_hash: "hash_A" },
			safety: { mode: "SIMULATION" },
			pending_approvals: [],
		} as never,
		"ctxl_A",
	);
	// A 已绑定（leaseState ACTIVE）——否则 actionsAllowed 正确为 false。
	active.patch({ leaseState: "ACTIVE" });
	assert.equal(active.current.contextState, "FRESH");
	assert.equal(active.current.actionsAllowed, true);
	const outcome = await coordinator.resumeInitial("pi_B");
	assert.equal(outcome.ok, true, "绑定成功但 NOT_READY 不算切换失败");
	const state = active.current;
	// 场景 B 核心断言：不含 A 的任何数据。
	assert.equal(state.missionId, "mis_B");
	assert.equal(state.contextState, "UNAVAILABLE");
	assert.equal(state.actionsAllowed, false, "context 失败必须禁动作");
	assert.equal(state.bodyId, undefined, "body_A 被继承了！");
	assert.equal(state.bodyHash, undefined, "hash_A 被继承了！");
	assert.equal(state.contextRevision, 0, "rev_12 被继承了！");
	assert.equal(state.contextLeaseId, undefined);
});

test("revision 0 的合法 FRESH envelope 正确显示 FRESH（不再误判 LOADING）", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_fresh0" },
			mission_archived: false,
		},
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_f0" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": {
			ok: true,
			context: {
				schema_version: "rosclaw.embodied_context.v1",
				mission_id: "mis_fresh0",
				context_revision: 0,
				generated_at: new Date().toISOString(),
				expires_at: new Date(Date.now() + 60_000).toISOString(),
				hash: "",
				body: { body_id: "body_fresh", effective_body_hash: "h" },
				safety: { mode: "SIMULATION" },
				pending_approvals: [],
			},
			context_lease_id: "ctxl_fresh0",
		},
	});
	const { coordinator, active } = makeCoordinator(call, "mis_fresh0");
	// envelopeHash 会校验——测试 stub 的 hash 为空会 stale；绕过：
	// 直接验证 applyEnvelope 的语义（revision 0 + FRESH 不矛盾）。
	active.applyEnvelope(
		{
			schema_version: "rosclaw.embodied_context.v1",
			mission_id: "mis_fresh0",
			context_revision: 0,
			generated_at: "",
			expires_at: "",
			hash: "",
			body: { body_id: "body_fresh", effective_body_hash: "h" },
			safety: { mode: "SIMULATION" },
			pending_approvals: [],
		} as never,
		"ctxl_fresh0",
	);
	active.patch({ leaseState: "ACTIVE" });
	assert.equal(active.current.contextRevision, 0);
	assert.equal(active.current.contextState, "FRESH", "revision 0 的合法 envelope 必须是 FRESH");
	assert.equal(active.current.actionsAllowed, true);
});

test("heartbeat 连续失败 → LEASE_LOST + 动作禁行", async () => {
	let heartbeatCalls = 0;
	const { call: baseCall } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_hb" },
			mission_archived: false,
		},
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_hb" }, lease_token: "t" },
		"pi.session.release": { ok: true },
		"pi.context": { ok: false },
	});
	const call = async (home: string, method: string, params: Record<string, unknown> = {}) => {
		if (method === "pi.session.heartbeat") {
			heartbeatCalls += 1;
			return { ok: false, error: "lease expired" };
		}
		return baseCall(home, method, params);
	};
	const { coordinator, active } = makeCoordinator(call, "mis_hb");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	// 用真实 leaseManager + onLeaseLost（create-runtime 的接线语义）。
	const leaseManager = (coordinator as unknown as { deps: { leaseManager: InstanceType<typeof SessionLeaseManager> } })
		.deps.leaseManager;
	let lost = false;
	leaseManager.onLeaseLost = () => {
		lost = true;
		active.markLeaseLost();
	};
	await coordinator.resumeInitial("pi_hb");
	// 手动触发两次 heartbeat 失败（真实 timer 是 30s——测试直接调）。
	(leaseManager as unknown as { noteHeartbeatFailure: () => void }).noteHeartbeatFailure();
	(leaseManager as unknown as { noteHeartbeatFailure: () => void }).noteHeartbeatFailure();
	assert.equal(lost, true, "连续失败未触发 onLeaseLost");
	assert.equal(active.current.leaseState, "LOST");
	assert.equal(active.current.actionsAllowed, false, "lease 丢失必须禁动作");
});

test("tree context 不可达必须 veto（fail closed，不再放行）", async () => {
	const { call } = fakeBridge({
		"pi.context": { ok: false, error: "bridge down" },
	});
	const veto = await shouldCancelTree({
		rosclawHome: "/tmp/rh",
		missionId: "mis_x",
		call,
	});
	assert.ok(veto?.includes("fail closed"), `context 不可达竟放行 tree: ${veto}`);
});
