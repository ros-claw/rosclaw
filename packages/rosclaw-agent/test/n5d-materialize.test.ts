/** PR-N5D 红测试（TS）：动态工具物化——模型直接看到当前真正可调用
 *  的强类型能力工具。
 *
 * 红测试先行——materialize/snapshot 接线不存在时必须红：
 * 1. snapshot.active → 精确强类型工具（工具名 slug + input_schema
 *    provider-facing 深拷贝投影成为 parameters）；
 * 2. PHYSICAL_EFFECT → propose_<slug>（不直接暴露原始 executor）；
 * 3. rosclaw_compute / rosclaw_execute 退出模型面（EMBODIMENT_PACK
 *    不再有）；rosclaw_task 保留为兼容入口；
 * 4. excluded 能力不产生工具；
 * 5. 物化工具的 execute 经 bridge 调用对应能力（capability_id 钉住，
 *    携带 snapshot digest）。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { MODEL_TOOL_NAMES } from "../src/tools/surface.js";
import { providerCompatibleSchema, materializeCapabilityTools } from "../src/tools/materialize.js";

test("provider integer projection preserves safe bounds and data, recursively copies schemas", () => {
	const unsafe = { type: "integer", minimum: -9223372036854775808, maximum: 9223372036854775807, exclusiveMinimum: -1e19, exclusiveMaximum: 1e19, description: "epoch ns" };
	const safe = { type: "integer", minimum: -Number.MAX_SAFE_INTEGER, maximum: Number.MAX_SAFE_INTEGER, exclusiveMinimum: 0, exclusiveMaximum: 100 };
	const schema = {
		type: "object", required: ["ts"], additionalProperties: false,
		properties: { ts: unsafe, safe, list: { type: "array", items: { anyOf: [unsafe, { type: "null" }] } } },
		$defs: { ns: unsafe }, allOf: [{ oneOf: [unsafe], not: { allOf: [unsafe] } }],
		$ref: "#/$defs/ns", enum: [{ type: "integer", maximum: 1e19 }],
		default: { minimum: -1e19 }, title: "Scan",
	};
	const before = structuredClone(schema);
	const freeze = (v: unknown): void => { if (v && typeof v === "object") { Object.values(v).forEach(freeze); Object.freeze(v); } };
	freeze(schema);
	const projected = providerCompatibleSchema(schema);
	const expectedUnsafe = { type: "integer", description: "epoch ns" };
	assert.deepEqual(projected, {
		...before, properties: { ts: expectedUnsafe, safe, list: { type: "array", items: { anyOf: [expectedUnsafe, { type: "null" }] } } },
		$defs: { ns: expectedUnsafe }, allOf: [{ oneOf: [expectedUnsafe], not: { allOf: [expectedUnsafe] } }],
	});
	assert.deepEqual(schema, before);
	assert.notEqual(projected.properties, schema.properties);
	assert.deepEqual(providerCompatibleSchema({ type: "number", minimum: -1e19, maximum: 1e19 }), { type: "number", minimum: -1e19, maximum: 1e19 });
});

test("safe fractional integer bounds remain exact, including integer/null unions", () => {
	for (const type of ["integer", ["integer", "null"]]) {
		const schema = { type, minimum: 1.5, maximum: 5.5, exclusiveMinimum: -0.5, exclusiveMaximum: 0.5, const: { maximum: 1e19 } };
		const before = structuredClone(schema);
		assert.deepEqual(providerCompatibleSchema(schema), before);
		assert.deepEqual(schema, before);
	}
});

test("inherited allOf unsafe bounds fail closed before materialization/provider", () => {
	for (const key of ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]) {
		for (const allOf of [[{ [key]: 1e19 }], [{ allOf: [{ allOf: [{ [key]: -1e19 }] }] }], [{ type: "integer" }, { [key]: 1e19 }]]) {
			const schema = { type: "integer", allOf };
			const before = structuredClone(schema);
			const isUnsupported = (error: unknown) => error instanceof Error && error.message.startsWith("PROVIDER_SCHEMA_UNSUPPORTED:") && (error as Error & { code: string }).code === "PROVIDER_SCHEMA_UNSUPPORTED";
			assert.throws(() => providerCompatibleSchema(schema), isUnsupported);
			const captured = { calls: [] as Array<{ method: string; params: unknown }> };
			assert.throws(() => materializeCapabilityTools({ ...SNAPSHOT, active: [{ ...SNAPSHOT.active[0], input_schema: schema }] }, { center: fakeCenter(captured), active: FAKE_ACTIVE, rosclawHome: "/tmp/x" }), isUnsupported);
			assert.equal(captured.calls.length, 0);
			assert.deepEqual(schema, before);
		}
	}
	assert.deepEqual(providerCompatibleSchema({ type: "integer", allOf: [{ minimum: 1.5, maximum: 5.5 }] }), { type: "integer", allOf: [{ minimum: 1.5, maximum: 5.5 }] });
	assert.deepEqual(providerCompatibleSchema({ type: "number", maximum: 1e19 }), { type: "number", maximum: 1e19 });
});

test("modern epoch ns passes unchanged through direct/propose wire; internal remains hidden", async () => {
	const schema = { type: "object", properties: { reference_time_ns: { type: "integer", minimum: -9223372036854775808, maximum: 9223372036854775807 } } };
	const captured: { calls: Array<{ method: string; params: unknown }> } = { calls: [] };
	const active = (["direct", "propose_only", "internal"] as const).map((exposure) => ({
		...SNAPSHOT.active[0], tool_name: exposure, exposure, effect_class: "READ_ONLY", input_schema: schema,
	}));
	const tools = materializeCapabilityTools({ ...SNAPSHOT, active }, { center: fakeCenter(captured), active: FAKE_ACTIVE, rosclawHome: "/tmp/x" });
	assert.equal(tools.length, 2);
	const params = { reference_time_ns: 1700000000000000000 };
	assert.ok(params.reference_time_ns > Number.MAX_SAFE_INTEGER);
	for (const tool of tools) await tool.execute("epoch", params, new AbortController().signal, async () => {}, {} as never);
	for (const [i, call] of captured.calls.entries()) {
		const { request } = call.params as { request: { tool_name: string; arguments: { arguments: unknown; capability_id: string; snapshot_digest: string } } };
		assert.equal(request.arguments.arguments, params);
		assert.equal(request.tool_name, i === 0 ? "rosclaw_observe" : "rosclaw_execute");
		assert.equal(request.arguments.capability_id, SNAPSHOT.active[0].capability_id);
		assert.equal(request.arguments.snapshot_digest, SNAPSHOT.digest);
	}
	assert.equal(schema.properties.reference_time_ns.maximum, 9223372036854775807);
	// Exact int64 contract arithmetic is BigInt, not a claim of JS Number precision.
	const lower = -(1n << 63n), upper = (1n << 63n) - 1n;
	const canonicalAccepts = (v: bigint) => v >= lower && v <= upper;
	assert.ok(canonicalAccepts(1700000000000000001n));
	assert.ok(canonicalAccepts(lower) && canonicalAccepts(upper));
	assert.ok(!canonicalAccepts(lower - 1n) && !canonicalAccepts(upper + 1n));
});

const SNAPSHOT = {
	schema_version: "rosclaw.capability_snapshot.v1",
	generation: 3,
	digest: "sha256:abc123",
	body_id: "sim/ur5e",
	mode: "SIMULATION",
	active: [
		{
			tool_name: "ur5e__plan_cartesian_path",
			capability_id: "ur5e.plan_cartesian_path",
			exposure: "direct" as const,
			effect_class: "PURE_COMPUTE",
			description: "规划笛卡尔轨迹",
			input_schema: {
				type: "object",
				properties: { shape: { type: "string", enum: ["star5", "circle"] } },
				required: ["shape"],
				additionalProperties: false,
			},
			output_schema: { type: "object" },
		},
		{
			tool_name: "propose_ur5e__move_joints",
			capability_id: "ur5e.move_joints",
			exposure: "propose_only" as const,
			effect_class: "PHYSICAL_EFFECT",
			description: "移动关节",
			input_schema: {
				type: "object",
				properties: { joints: { type: "array", items: { type: "number" } } },
				required: ["joints"],
				additionalProperties: false,
			},
			output_schema: { type: "object" },
		},
	],
	excluded: [
		{ capability_id: "weird", reason: "EFFECT_NOT_EXPOSABLE" },
	],
};

function fakeCenter(captured: { calls: Array<{ method: string; params: unknown }> }) {
	return {
		async call(method: string, params: unknown) {
			captured.calls.push({ method, params });
			return {
				ok: true, status: "COMPLETED",
				summary: JSON.stringify({ status: "SUCCEEDED", value: { ok: true } }),
			};
		},
	} as never;
}

const FAKE_ACTIVE = {
	current: {
		sessionId: "s1", missionId: "m1", contextRevision: 1,
		bodyId: "sim/ur5e", mode: "SIMULATION",
	},
	patch() {},
} as never;

test("N5D: snapshot 物化为精确强类型工具", async () => {
	const { materializeCapabilityTools } = await import("../src/tools/materialize.js");
	const tools = materializeCapabilityTools(SNAPSHOT, {
		center: fakeCenter({ calls: [] }), active: FAKE_ACTIVE, rosclawHome: "/tmp/x",
	});
	const names = tools.map((t) => t.name);
	assert.ok(names.includes("ur5e__plan_cartesian_path"));
	assert.ok(names.includes("propose_ur5e__move_joints"));
	assert.ok(!names.includes("weird"), "excluded 能力不得产生工具");
	const plan = tools.find((t) => t.name === "ur5e__plan_cartesian_path");
	assert.ok(plan);
	// 安全 schema 的 provider-facing 深拷贝保持语义（不再是 Record<string, unknown>）
	const params = plan.parameters as { properties?: Record<string, unknown>; required?: string[] };
	assert.ok(params.properties?.shape, "input_schema 未成为 parameters");
	assert.deepEqual(params.required, ["shape"]);
});

test("N5D: 物化工具 execute 钉住 capability_id 并携带 snapshot digest", async () => {
	const { materializeCapabilityTools } = await import("../src/tools/materialize.js");
	const captured: { calls: Array<{ method: string; params: unknown }> } = { calls: [] };
	const tools = materializeCapabilityTools(SNAPSHOT, {
		center: fakeCenter(captured), active: FAKE_ACTIVE, rosclawHome: "/tmp/x",
	});
	const plan = tools.find((t) => t.name === "ur5e__plan_cartesian_path");
	assert.ok(plan);
	await plan.execute("c1", { shape: "star5" }, new AbortController().signal, async () => {}, {} as never);
	assert.equal(captured.calls.length, 1);
	const params = captured.calls[0].params as {
		request: { tool_name: string; arguments: Record<string, unknown> };
	};
	// wire 仍走内核验证链（compute 路径），capability_id 钉住
	assert.equal(params.request.tool_name, "rosclaw_compute");
	assert.equal(params.request.arguments.capability_id, "ur5e.plan_cartesian_path");
	assert.equal(params.request.arguments.snapshot_digest, "sha256:abc123");
	// 模型参数原样传递
	assert.deepEqual(params.request.arguments.arguments, { shape: "star5" });
});

test("N5D: propose_ 工具走 admission 链（rosclaw_execute 管线）", async () => {
	const { materializeCapabilityTools } = await import("../src/tools/materialize.js");
	const captured: { calls: Array<{ method: string; params: unknown }> } = { calls: [] };
	const tools = materializeCapabilityTools(SNAPSHOT, {
		center: fakeCenter(captured), active: FAKE_ACTIVE, rosclawHome: "/tmp/x",
	});
	const propose = tools.find((t) => t.name === "propose_ur5e__move_joints");
	assert.ok(propose);
	await propose.execute("c1", { joints: [0, 0, 0, 0, 0, 0] }, new AbortController().signal, async () => {}, {} as never);
	const params = captured.calls[0].params as {
		request: { tool_name: string; arguments: Record<string, unknown> };
	};
	assert.equal(params.request.tool_name, "rosclaw_execute");
	assert.equal(params.request.arguments.capability_id, "ur5e.move_joints");
});

test("N5D: rosclaw_compute/rosclaw_execute 退出模型面；R0-1.5 task 亦退出", () => {
	assert.ok(!MODEL_TOOL_NAMES.includes("rosclaw_compute"),
		"rosclaw_compute 应从默认模型面删除");
	assert.ok(!MODEL_TOOL_NAMES.includes("rosclaw_execute"),
		"rosclaw_execute 应从默认模型面删除");
	// R0-1.5（金丝雀实证 + 0826 审计 §6）：已知 recipe 由输入路由
	// 自动执行——rosclaw_task 退出模型面（wire adapter 保留）。
	assert.ok(!MODEL_TOOL_NAMES.includes("rosclaw_task"),
		"rosclaw_task 已退出模型面（输入路由自动执行）");
});
