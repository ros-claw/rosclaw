/** 十四审 PR-14.7：SIM 页面 Operator 状态降级（总纲 §1.9）。
 *  纯 SIM 任务中 "Operator Offline" 不应造成紧张或暗示需要人工审批；
 *  只有 REAL request 涉及操作员时才突出。 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { renderFooter, renderHeader } from "../src/ui/product-state.js";

function snap(mode: string, readiness: { state: string; reason_codes: string[] }) {
	return {
		schema_version: "kernel.snapshot.v1",
		mission_id: "mis_sim_test",
		body_id: "sim/ur5e",
		mode,
		model: "kimi/k3",
		operator: "OFFLINE",
		kernel: "READY",
		context_state: "FRESH",
		context_revision: 3,
		action_readiness: { ...readiness, snapshot_seq: 1 },
	} as never;
}

describe("SIM 页面 Operator 降级（十四审 §1.9）", () => {
	it("纯 SIM（无 REAL 请求）：不显示 Operator Offline——SIM 自动执行为次要信息", () => {
		const s = snap("SIMULATION", { state: "READY", reason_codes: [] });
		const zhHeader = renderHeader(s, "zh-CN");
		const enHeader = renderHeader(s, "en-US");
		assert.doesNotMatch(zhHeader, /操作员 离线/);
		assert.doesNotMatch(enHeader, /Operator Offline/);
		assert.match(zhHeader, /SIM 自动/);
		assert.match(enHeader, /SIM auto/i);
		const enFooter = renderFooter(s, "en-US");
		assert.doesNotMatch(enFooter, /Operator Offline/);
	});

	it("REAL 模式：Operator 状态照常突出", () => {
		const s = snap("REAL", { state: "BLOCKED", reason_codes: ["OPERATOR_OFFLINE"] });
		const enHeader = renderHeader(s, "en-US");
		assert.match(enHeader, /Operator Offline/);
	});

	it("SIM 但动作需要操作员（OPERATOR_OFFLINE 阻塞）：仍突出", () => {
		const s = snap("SIMULATION", { state: "BLOCKED", reason_codes: ["OPERATOR_OFFLINE"] });
		const enHeader = renderHeader(s, "en-US");
		assert.match(enHeader, /Operator Offline/);
	});
});
