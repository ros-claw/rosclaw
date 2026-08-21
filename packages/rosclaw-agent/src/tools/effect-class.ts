/** EffectClass（PR-H6，总纲 v2 §10.4/§14.1）——每个模型可见工具的
 *  副作用分类。安全决策按 effect/mode/sandbox，不再只靠
 *  OBSERVE/COMPUTE/ACTION 三分类。
 *
 * 不变量：MODEL_TOOL_NAMES 里的工具必须全部在此声明（无分类=不
 * 暴露——h6-safety 测试强制）。
 */

export type EffectClass =
	| "READ_ONLY"
	| "WORKSPACE_WRITE"
	| "HOST_PROCESS"
	| "NETWORK"
	| "SIMULATED_EFFECT"
	| "SHADOW_PROPOSAL"
	| "PHYSICAL_EFFECT";

export const EFFECT_BY_TOOL: Record<string, EffectClass> = {
	// Workspace Pack
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
	// Embodiment Pack
	rosclaw_status: "READ_ONLY",
	rosclaw_capabilities: "READ_ONLY",
	rosclaw_observe: "READ_ONLY",
	rosclaw_compute: "SIMULATED_EFFECT",
	rosclaw_task: "SIMULATED_EFFECT",
	rosclaw_verify: "READ_ONLY",
	rosclaw_request_action: "PHYSICAL_EFFECT",
	rosclaw_execute: "SIMULATED_EFFECT", // physical 在 admission 链内分流
	rosclaw_wait_operation: "READ_ONLY",
	rosclaw_stop_operation: "HOST_PROCESS",
	rosclaw_memory_query: "READ_ONLY",
	rosclaw_inspect: "READ_ONLY",
	rosclaw_fail_safe: "SHADOW_PROPOSAL",
	// Product Pack
	rosclaw_artifact_register: "WORKSPACE_WRITE",
	rosclaw_task_finish: "WORKSPACE_WRITE",
	rosclaw_task_blocked: "WORKSPACE_WRITE",
};
