/** 工具 Effect 查询（PR-N5C，调整方案 §三.N5C）——单一 Effect Contract。
 *
 * 事实源拆分（没有第二份手写分类）：
 * - rosclaw_* 具身/产品工具 → Python Capability Registry 生成的
 *   effects.generated.json（同目录，漂移会被
 *   tests/agentd/test_n5c_effect_contract.py 钉红）；
 * - workspace 原语（read/bash/…）→ 下方 WORKSPACE_TOOL_EFFECTS
 *   同位声明（TS 原生工具，Python 注册表不覆盖它们）。
 *
 * 通用入口（rosclaw_observe/compute/execute）是 DYNAMIC——运行时由
 * EffectResolver 按 capability_id 解析并冻结进 tool.effect_resolved
 * 事件；静态写死 SIMULATED_EFFECT 是调 REAL 能力时的降级谎言。
 */

import { readFileSync } from "node:fs";

export type EffectClass =
	| "READ_ONLY"
	| "PURE_COMPUTE"
	| "WORKSPACE_WRITE"
	| "HOST_PROCESS"
	| "NETWORK_EFFECT"
	| "SIMULATED_EFFECT"
	| "SHADOW_PROPOSAL"
	| "PHYSICAL_EFFECT"
	| "DYNAMIC";

/** TS 原生 workspace 原语的唯一声明（与工具实现同包）。 */
export const WORKSPACE_TOOL_EFFECTS: Record<string, EffectClass> = {
	read: "READ_ONLY",
	grep: "READ_ONLY",
	find: "READ_ONLY",
	ls: "READ_ONLY",
	edit: "WORKSPACE_WRITE",
	write: "WORKSPACE_WRITE",
	bash: "HOST_PROCESS",
	process_start: "HOST_PROCESS",
	process_status: "READ_ONLY",
	process_output: "READ_ONLY",
	process_stop: "HOST_PROCESS",
};

let _generated: Record<string, EffectClass> | undefined;

function generatedEffects(): Record<string, EffectClass> {
	if (_generated) return _generated;
	const url = new URL("./effects.generated.json", import.meta.url);
	_generated = JSON.parse(readFileSync(url, "utf-8")) as Record<string, EffectClass>;
	return _generated;
}

/** 工具的 effect 分类；未分类返回 undefined（无分类 = 不暴露）。 */
export function getToolEffect(name: string): EffectClass | undefined {
	return WORKSPACE_TOOL_EFFECTS[name] ?? generatedEffects()[name];
}
