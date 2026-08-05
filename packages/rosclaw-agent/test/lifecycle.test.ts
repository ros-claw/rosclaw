import assert from "node:assert/strict";
import test from "node:test";

import {
	handleSessionStart,
	shouldCancelSwitch,
	shouldCancelTree,
	type LifecycleDeps,
} from "../src/session/lifecycle.js";

function fakeBridge(responses: Record<string, Record<string, unknown>>) {
	const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
	const call = async (_home: string, method: string, params: Record<string, unknown> = {}) => {
		calls.push({ method, params });
		return responses[method] ?? { ok: false, error: `no stub for ${method}` };
	};
	return { calls, call };
}

function deps(call: LifecycleDeps["call"], missionId?: string) {
	const notes: string[] = [];
	const d: LifecycleDeps = {
		rosclawHome: "/tmp/rh",
		getMissionId: () => missionId,
		setMissionId: (id) => {
			missionId = id;
		},
		notify: (message) => notes.push(message),
		call,
	};
	return { deps: d, notes, getMission: () => missionId };
}

test("new/fork creates SIM mission and binds; fork never copies authority", async () => {
	const { calls, call } = fakeBridge({
		"pi.mission.create": { ok: true, mission_id: "mis_new", mode: "SIMULATION" },
		"pi.session.bind": { ok: true, binding: { binding_id: "psb_1" }, lease_token: "t" },
	});
	const { deps: d, getMission } = deps(call, "mis_old");
	await handleSessionStart(d, "fork", "pi_sess_2");
	assert.equal(getMission(), "mis_new");
	const create = calls.find((c) => c.method === "pi.mission.create");
	assert.equal(create?.params.mode, "SIMULATION", "fork 强制 SIM");
	// 绑定请求不含任何 authority 字段（结构性不复制）。
	const bind = calls.find((c) => c.method === "pi.session.bind");
	assert.deepEqual(Object.keys(bind?.params ?? {}).sort(), ["mission_id", "pi_session_id"]);
});

test("resume with healthy binding just reattaches", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_kept" },
			mission_archived: false,
		},
	});
	const { deps: d, getMission } = deps(call, undefined);
	const veto = await shouldCancelSwitch(d, "pi_sess_old");
	assert.equal(veto, null);
	assert.equal(getMission(), "mis_kept");
});

test("resume without binding creates a fresh SIM binding (never guesses)", async () => {
	const { calls, call } = fakeBridge({
		"pi.session.binding.get": { ok: true, binding: null },
		"pi.mission.create": { ok: true, mission_id: "mis_fresh" },
		"pi.session.bind": { ok: true },
	});
	const { deps: d, getMission, notes } = deps(call, undefined);
	const veto = await shouldCancelSwitch(d, "pi_sess_lost");
	assert.equal(veto, null);
	assert.equal(getMission(), "mis_fresh");
	assert.ok(notes.some((n) => n.includes("无 Mission 绑定")));
	assert.ok(calls.some((c) => c.method === "pi.mission.create"));
});

test("archived mission rebinds to a fresh SIM mission", async () => {
	const { call } = fakeBridge({
		"pi.session.binding.get": {
			ok: true,
			binding: { mission_id: "mis_archived" },
			mission_archived: true,
		},
		"pi.mission.create": { ok: true, mission_id: "mis_new2" },
		"pi.session.bind": { ok: true },
	});
	const { deps: d, getMission, notes } = deps(call, undefined);
	const veto = await shouldCancelSwitch(d, "pi_sess_arch");
	assert.equal(veto, null);
	assert.equal(getMission(), "mis_new2");
	assert.ok(notes.some((n) => n.includes("已归档")));
});

test("tree navigation vetoed with pending approvals or active actions", async () => {
	const { call } = fakeBridge({
		"pi.context": {
			ok: true,
			context: { pending_approvals: [{ request_id: "r1" }], active_actions: [] },
		},
	});
	const { deps: d } = deps(call, "mis_x");
	assert.ok((await shouldCancelTree(d))?.includes("待决授权"));

	const { call: call2 } = fakeBridge({
		"pi.context": {
			ok: true,
			context: { pending_approvals: [], active_actions: [{ id: "a1" }] },
		},
	});
	const { deps: d2 } = deps(call2, "mis_x");
	assert.ok((await shouldCancelTree(d2))?.includes("fail closed"));

	const { call: call3 } = fakeBridge({
		"pi.context": { ok: true, context: { pending_approvals: [], active_actions: [] } },
	});
	const { deps: d3 } = deps(call3, "mis_x");
	assert.equal(await shouldCancelTree(d3), null);
});
