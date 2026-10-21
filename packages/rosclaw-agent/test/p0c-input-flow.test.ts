/** P0-C 红测试（0824 总纲 §19.P0-C）：TS 输入流与 effect 钩子。
 *
 * 红测试先行——persist 路径/ensure 钩子不存在时必须红。
 *
 * 1. 输入先 persist（pi.input.persist），不再逐条 bind 建 task；
 * 2. persist 失败不投递（HP1 输入丢失防线语义不变）；
 * 3. workspace 工作工具（bash/write/edit）执行前触发
 *    beforeEffect（首个 effectful call 的原子 admission）；
 * 4. 活跃 task 查询替代 bind 返回值（/done 等消费的同一事实源）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("P0-C 输入流 persist-only", () => {
	it("输入走 pi.input.persist（不建 task），失败不投递", async () => {
		const { InputController } = await import("../src/native/input-controller.js");
		const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
		const controller = new InputController({
			call: async (method: string, params: Record<string, unknown>) => {
				calls.push({ method, params });
				return { ok: true, input: { input_id: "inp_1" } };
			},
			missionId: () => "mis_1",
			sessionRef: () => "s1",
			backendNativeId: () => "s1",
			cwd: () => "/tmp",
			bodyId: () => "sim/ur5e",
			notify: () => undefined,
		} as never);
		const result = await controller.persist("hello");
		assert.ok(result !== null, "persist 应成功");
		assert.equal(calls[0]?.method, "pi.input.persist", "仍在 bind 建 task");
		assert.ok(!calls.some((c) => c.method === "pi.task.bind"),
			"输入仍在逐条 bind——hello 会建 task");
	});

	it("persist 失败 → null（不投递）", async () => {
		const { InputController } = await import("../src/native/input-controller.js");
		const controller = new InputController({
			call: async () => {
				throw new Error("bridge down");
			},
			missionId: () => "mis_1",
			sessionRef: () => "s1",
			backendNativeId: () => "s1",
			cwd: () => "/tmp",
			bodyId: () => "",
			notify: () => undefined,
		} as never);
		assert.equal(await controller.persist("hello"), null);
	});

	it("activeTaskId 查询 pi.kernel.active（同一事实源）", async () => {
		const { InputController } = await import("../src/native/input-controller.js");
		const controller = new InputController({
			call: async (method: string) => {
				if (method === "pi.kernel.active") {
					return { task: { task_id: "task_42" } };
				}
				return {};
			},
			missionId: () => "mis_1",
			sessionRef: () => "s1",
			backendNativeId: () => "s1",
			cwd: () => "/tmp",
			bodyId: () => "",
			notify: () => undefined,
		} as never);
		assert.equal(await controller.activeTaskId(), "task_42");
	});
});

describe("SOURCE46 original InputController ACK contract", () => {
	// Explicit protocol fixtures, NOT a DB, model or durable acknowledgement.
	const refused: Array<{ name: string; ack: unknown }> = [
		{ name: "undefined", ack: undefined },
		{ name: "null", ack: null },
		{ name: "false", ack: false },
		{ name: "number", ack: 1 },
		{ name: "string", ack: "ok" },
		{ name: "array", ack: [] },
		{ name: "empty-object", ack: {} },
		{ name: "refused", ack: { ok: false, code: "FIXTURE_REFUSED" } },
		{ name: "refused-with-id", ack: { ok: false, input: { input_id: "inp_no" } } },
		{ name: "missing-ok", ack: { input: { input_id: "inp_no" } } },
		{ name: "string-ok", ack: { ok: "true", input: { input_id: "inp_no" } } },
		{ name: "number-ok", ack: { ok: 1, input: { input_id: "inp_no" } } },
		{ name: "missing-input", ack: { ok: true } },
		{ name: "null-input", ack: { ok: true, input: null } },
		{ name: "array-input", ack: { ok: true, input: [] } },
		{ name: "string-input", ack: { ok: true, input: "inp_no" } },
		{ name: "missing-id", ack: { ok: true, input: {} } },
		{ name: "null-id", ack: { ok: true, input: { input_id: null } } },
		{ name: "number-id", ack: { ok: true, input: { input_id: 123 } } },
		{ name: "boolean-id", ack: { ok: true, input: { input_id: true } } },
		{ name: "object-id", ack: { ok: true, input: { input_id: {} } } },
		{ name: "empty-id", ack: { ok: true, input: { input_id: "" } } },
		{ name: "whitespace-id", ack: { ok: true, input: { input_id: " \n\t" } } },
	];
	for (const { name, ack } of refused) {
		it(`SOURCE46 refuses ${name} without dispatch; pending /new survives recovery`, async () => {
			const { InputController } = await import("../src/native/input-controller.js");
			const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
			const notices: Array<{ text: string; kind: string }> = [];
			let reply: unknown = ack;
			const controller = new InputController({
				call: async (method, params) => {
					calls.push({ method, params: structuredClone(params) });
					if (method === "pi.input.persist") return reply as Record<string, unknown>;
					assert.equal(method, "pi.input.dispatched", "no bind, route or hidden retry");
					return { ok: true };
				},
				missionId: () => "mis_ack", sessionRef: () => "s_ack", backendNativeId: () => "native_ack",
				cwd: () => "/tmp", bodyId: () => "sim/ur5e",
				notify: (text, kind) => { notices.push({ text, kind }); },
			});
			controller.forceNewNext = true;
			assert.equal(await controller.persist("refused fixture input"), null);
			assert.deepEqual(calls.map(c => c.method), ["pi.input.persist"]);
			assert.equal(controller.forceNewNext, true, "refusal must not consume /new");
			assert.equal(calls[0].params.force_new, true);
			assert.equal(notices.length, 1);
			assert.equal(notices[0].kind, "error");
			assert.match(notices[0].text, /输入持久化失败/);
			assert.ok(notices[0].text.length <= 200);
			// SAME controller, no rearming: successful retry must still carry /new.
			reply = { ok: true, input: { input_id: "inp_recovered_fixture" } };
			const recovered = await controller.persist("recovered fixture input");
			assert.equal(recovered?.input_id, "inp_recovered_fixture");
			assert.deepEqual(calls.map(c => c.method), ["pi.input.persist", "pi.input.persist", "pi.input.dispatched"]);
			assert.equal(calls[1].params.force_new, true);
			assert.equal(controller.forceNewNext, false);
			assert.deepEqual(calls[2].params, { mission_id: "mis_ack", message_id: calls[1].params.message_id });
			assert.notEqual(calls[1].params.message_id, calls[0].params.message_id);
			assert.equal(notices.length, 1, "valid ACK must not add error notices");
			assert.notEqual(await controller.persist("ordinary next fixture input"), null);
			assert.equal(calls[3].params.force_new, false);
			assert.deepEqual(calls.map(c => c.method), ["pi.input.persist", "pi.input.persist", "pi.input.dispatched", "pi.input.persist", "pi.input.dispatched"]);
			assert.deepEqual(calls[4].params, { mission_id: "mis_ack", message_id: calls[3].params.message_id });
			assert.notEqual(calls[3].params.message_id, calls[1].params.message_id);
		});
	}

	it("SOURCE46 transport refusal retains pending new intent", async () => {
		const { InputController } = await import("../src/native/input-controller.js");
		const calls: string[] = [];
		const notices: string[] = [];
		const controller = new InputController({
			call: async (method) => { calls.push(method); throw new Error("fixture transport refused"); },
			missionId: () => "mis_ack", sessionRef: () => "s_ack", backendNativeId: () => "native_ack",
			cwd: () => "/tmp", notify: (_text, kind) => { notices.push(kind); },
		});
		controller.forceNewNext = true;
		assert.equal(await controller.persist("fixture input"), null);
		assert.deepEqual(calls, ["pi.input.persist"]);
		assert.equal(controller.forceNewNext, true);
		assert.deepEqual(notices, ["error"]);
	});

	it("SOURCE46 valid ACK is required before dispatch and preserves metadata", async () => {
		const { InputController } = await import("../src/native/input-controller.js");
		const calls: Array<{ method: string; params: Record<string, unknown> }> = [];
		const notices: string[] = [];
		let acknowledge!: (value: Record<string, unknown>) => void;
		const acknowledgement = new Promise<Record<string, unknown>>(resolve => { acknowledge = resolve; });
		const controller = new InputController({
			call: async (method, params) => {
				calls.push({ method, params: structuredClone(params) });
				if (method === "pi.input.persist") return await acknowledgement;
				assert.equal(method, "pi.input.dispatched");
				assert.equal(controller.forceNewNext, false, "consume /new only after the valid ACK");
				return { ok: true };
			},
			missionId: () => "mis_ack", sessionRef: () => "s_ack", backendNativeId: () => "native_ack",
			cwd: () => "/tmp", notify: (_text, kind) => { notices.push(kind); },
		});
		controller.forceNewNext = true;
		const pending = controller.persist("fixture 汉🙂 input");
		assert.deepEqual(calls.map(c => c.method), ["pi.input.persist"], "no dispatch while ACK is unresolved");
		assert.equal(controller.forceNewNext, true);
		assert.deepEqual(notices, []);
		const autoTask = { task_id: "task_fixture", replayed: false };
		const explain = { task_id: "task_fixture", state: "SUCCEEDED", artifacts: [] };
		const disposition = { input_id: "inp_valid_fixture", owner: "PI_CONVERSATION", task_id: "", suppress_model_turn: false };
		acknowledge({ ok: true, input: { input_id: "inp_valid_fixture" }, auto_task: autoTask,
			explain, turn_disposition: disposition });
		assert.deepEqual(await pending, { input_id: "inp_valid_fixture", auto_task: autoTask,
			explain, turn_disposition: disposition });
		assert.deepEqual(calls.map(c => c.method), ["pi.input.persist", "pi.input.dispatched"]);
		assert.deepEqual(calls[0].params, { mission_id: "mis_ack", session_ref: "s_ack", backend_native_id: "native_ack",
			message_id: calls[0].params.message_id, text: "fixture 汉🙂 input", force_new: true });
		assert.match(String(calls[0].params.message_id), /^msg_[0-9a-f-]{36}$/);
		assert.deepEqual(calls[1].params, { mission_id: "mis_ack", message_id: calls[0].params.message_id });
		assert.equal(controller.forceNewNext, false);
		assert.equal(controller.currentTaskId, "");
		assert.equal(controller.currentRevision, 0);
		assert.deepEqual(notices, []);
	});
});

describe("P0-C workspace effect 钩子", () => {
	it("bash 执行前触发 beforeEffect", async () => {
		const { buildWorkspacePackTools } = await import("../src/tools/workspace-pack.js");
		const { mkdtempSync } = await import("node:fs");
		const { tmpdir } = await import("node:os");
		const { join } = await import("node:path");
		let ensured = 0;
		const dir = mkdtempSync(join(tmpdir(), "p0c-"));
		const tools = buildWorkspacePackTools({
			root: dir,
			mode: () => "SIMULATION",
			bwrapPath: () => null,
			beforeEffect: async () => {
				ensured += 1;
			},
		});
		const bash = tools.find((t) => t.name === "bash");
		await bash?.execute("c1", { command: "echo hi" },
			new AbortController().signal, async () => {}, {} as never);
		assert.equal(ensured, 1, "bash 未触发 ensure admission");
	});
});
