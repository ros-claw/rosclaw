/** PR-EIGHT-2 红测试（八审 §1.2/P0-2）：compute 桶必须透出给模型。
 *
 * 红测试先行——当前 rosclaw_capabilities 只渲染 action/observation，
 * 后端返回的 compute_capabilities（plan_cartesian_path/verify_drawing）
 * 被静默丢弃，真实模型看不到专门为它准备的规划/验证能力。
 */

import assert from "node:assert/strict";
import test from "node:test";

function makeCtx(capabilitiesResult: Record<string, unknown>) {
	return {
		active: { current: { missionId: "mis_1", sessionId: "pi_1" } },
		center: { call: async () => capabilitiesResult },
		rosclawHome: "/tmp/rh-eight2",
	} as never;
}

const BACKEND_RESPONSE = {
	ok: true,
	body_id: "sim/ur5e",
	mode: "SIMULATION",
	observation_capabilities: [
		{ capability_id: "ur5e.get_joint_state", description: " joints" },
	],
	compute_capabilities: [
		{ capability_id: "ur5e.plan_cartesian_path", description: "plan" },
		{ capability_id: "ur5e.verify_drawing", description: "verify" },
	],
	action_capabilities: [
		{
			capability_id: "ur5e.execute_cartesian_path",
			risk_tier: "LOW",
			side_effect_class: "SIM",
			description: "execute",
		},
	],
	excluded: [],
};

test("capabilities 工具结果含 compute 桶（plan/verify 对模型可见）", async () => {
	const { buildCapabilitiesTool } = await import("../src/tools/capabilities.js");
	const tool = buildCapabilitiesTool(makeCtx(BACKEND_RESPONSE));
	const result = await tool.execute("tc1", {}, undefined, undefined, {} as never);
	const text = (result.content[0] as { text: string }).text;
	assert.ok(
		text.includes("ur5e.plan_cartesian_path"),
		`compute 能力未透出: ${text.slice(0, 400)}`,
	);
	assert.ok(text.includes("ur5e.verify_drawing"));
	const details = result.details as { compute_ids?: string[] };
	assert.ok(details.compute_ids?.includes("ur5e.plan_cartesian_path"));
});

test("工具描述同步四类语义与策略语言（无过时 human-decides）", async () => {
	const { buildCapabilitiesTool } = await import("../src/tools/capabilities.js");
	const tool = buildCapabilitiesTool(makeCtx(BACKEND_RESPONSE));
	const desc = tool.description;
	assert.match(desc, /compute/i);
	assert.match(desc, /rosclaw_compute/);
	// 过时描述清除：不再说所有动作都由人工决定（默认 SIM 走 POLICY_AUTO）。
	assert.ok(!/every action.*human|all actions.*human/i.test(desc), desc);
});
