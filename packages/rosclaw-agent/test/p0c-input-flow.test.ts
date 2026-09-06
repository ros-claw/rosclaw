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
