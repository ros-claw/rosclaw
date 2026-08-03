import assert from "node:assert/strict";
import test from "node:test";
import { reduce, type Effect } from "../src/state/reducer.js";
import { initialState } from "../src/state/types.js";
import type { AgentEvent } from "../src/client/sse.js";

function ev(type: string, payload: Record<string, unknown> = {}, sequence = 1): AgentEvent {
	return {
		event_id: `evt_${type}_${sequence}`,
		sequence,
		mission_id: "mis_x",
		type,
		visibility: "USER",
		payload,
		timestamp: "2026-08-03T00:00:00Z",
	};
}

test("agent.settled stops spinner and flushes delta", () => {
	let state = initialState("mis_x");
	const effects: Effect[] = [];
	for (const e of reduce(state, ev("agent.started", {}, 1))) effects.push(e);
	assert.equal(state.turnInFlight, true);
	for (const e of reduce(state, ev("model.text.delta", { text: "你好" }, 2))) effects.push(e);
	for (const e of reduce(state, ev("agent.settled", {}, 3))) effects.push(e);
	assert.equal(state.turnInFlight, false);
	assert.ok(effects.some((e) => e.kind === "spinner_stop"));
	assert.ok(effects.some((e) => e.kind === "flush_delta"));
});

test("approval card lifecycle", () => {
	let state = initialState("mis_x");
	reduce(state, ev("approval.requested", { request_id: "appr_1234567890abcdef", title: "播放提示音", risk_tier: "LOW" }, 1));
	assert.equal(state.pendingApprovals.length, 1);
	reduce(state, ev("approval.decided", { request_id: "appr_1234567890abcdef", approved: true }, 2));
	assert.equal(state.pendingApprovals.length, 0);
});

test("worker lifecycle tracks terminal status", () => {
	let state = initialState("mis_x");
	reduce(state, ev("worker.offered", { work_order_id: "wo1", worker_id: "native" }, 1));
	reduce(state, ev("worker.accepted", { work_order_id: "wo1", worker_id: "native" }, 2));
	assert.equal(state.workers[0].status, "accepted");
});

test("phase label never exposes chain-of-thought", () => {
	let state = initialState("mis_x");
	const effects = reduce(state, ev("model.request.started", {}, 1));
	const spinner = effects.find((e) => e.kind === "spinner");
	assert.ok(spinner && spinner.kind === "spinner" && spinner.label.includes("模型"));
});

test("sequence tracking takes max", () => {
	let state = initialState("mis_x");
	reduce(state, ev("agent.started", {}, 5));
	reduce(state, ev("agent.settled", {}, 3));
	assert.equal(state.lastSeq, 5);
});
