/** 0902 R3-c 红测试（§6.1 三层界面）：静态只读工具的 renderResult
 *  用户面折叠——模型上下文保留完整 JSON（诚实性不降级），TUI 只
 *  渲染单行摘要。
 *
 * 0902 实证：模型调 rosclaw_status 后整段 JSON 打到 scrollback
 * （journey 实录 16 行原始 JSON）。§6.1：任何 tool output 进入 UI
 * 前都经过 size/row budget 和结构化 renderer——1342 行记录不能再
 * 出现。
 *
 * 闭环断言：
 * 1. rosclaw_status / rosclaw_inspect / rosclaw_capabilities 都有
 *    renderResult（用户面折叠钩）；
 * 2. renderResult 输出单行、无原始 JSON 大括号块、含关键状态词；
 * 3. 模型面不受影响：execute 返回的 content 仍是完整 JSON。
 */

import assert from "node:assert/strict";
import test from "node:test";

type ToolLike = {
	name: string;
	renderResult?: (result: { content?: Array<{ type: string; text?: string }> }) => unknown;
	execute: (...a: never[]) => Promise<{ content: Array<{ text: string }> }>;
};

async function buildTools(): Promise<ToolLike[]> {
	const { buildStatusTool } = await import("../src/tools/status.js");
	const { buildInspectTool } = await import("../src/tools/inspect.js");
	const { buildCapabilitiesTool } = await import("../src/tools/capabilities.js");
	// 只读工具的 center/call 最小桩（execute 不真打桥——renderResult
	// 是本测试主体；execute 只验证模型面文本形状时用假回包）。
	const centerStub = {
		statusReport: async () => ({
			ok: true, agentd: "READY", authorization_profile: "DEV_SIM_ONLY",
			mission: { mission_id: "mis_x", state: "IDLE", mode: "SIMULATION" },
			snapshot: {
				kernel: "READY", context_state: "FRESH", context_revision: 0,
				lease_state: "ACTIVE", operator: "OFFLINE",
				action_readiness: { state: "READY", reason_codes: [] },
				snapshot_seq: 3,
			},
		}),
	};
	const callStub = async () => ({ ok: true, items: [], digest: "sha256:x" });
	return [
		buildStatusTool(centerStub as never) as unknown as ToolLike,
		buildInspectTool(callStub as never) as unknown as ToolLike,
		buildCapabilitiesTool(callStub as never) as unknown as ToolLike,
	];
}

function renderText(rendered: unknown): string {
	// pi-tui Text 组件：render(width) → 行数组。
	const r = rendered as { render?: (w: number) => string[] };
	if (typeof r.render === "function") return r.render(80).join("\n");
	return String(rendered);
}

test("R3-c: 三个只读工具都有 renderResult（用户面折叠钩）", async () => {
	for (const tool of await buildTools()) {
		assert.ok(tool.renderResult, `${tool.name} 缺 renderResult——原始 JSON 会刷屏`);
	}
});

test("R3-c: status 结果渲染为单行摘要（无原始 JSON 块）", async () => {
	const [status] = await buildTools();
	const big = JSON.stringify(
		{ agentd: "READY", kernel: "READY", operator: "OFFLINE", mission: { state: "IDLE" } },
		null, 1,
	);
	const text = renderText(status.renderResult!({
		content: [{ type: "text", text: big }],
	}));
	assert.ok(text.length < 200, `渲染不是单行摘要: ${text.length} 字符`);
	assert.ok(!text.includes("\n"), "多行输出——row budget 失守");
	assert.match(text, /READY/);
});

test("R3-c: 模型面 content 仍是完整 JSON（诚实性不降级）", async () => {
	const [status] = await buildTools();
	const result = await status.execute(...([] as unknown as [never]));
	const text = result.content[0]?.text ?? "";
	const parsed = JSON.parse(text);
	assert.equal(parsed.agentd, "READY");
	assert.ok(parsed.mission, "模型面 JSON 被裁了");
});
