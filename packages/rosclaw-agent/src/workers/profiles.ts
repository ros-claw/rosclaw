/** WorkerProfileV1（十审 W1，审计 §6）：通用能力而非任务特化。
 *
 * Profile 决定工具 allowlist 与系统提示，不决定 provider——默认使用
 * Native Agent 当前模型（ModelExecutionSnapshot 经 WorkOrder 下发，
 * 无 secret）。
 *
 * P0 安全边界：
 * - 只读工具集（read/grep/find/ls）——无 write/edit/bash（W3 Workbench
 *   落地前任何内置 Worker 都没有写能力）；
 * - 无 ROSClaw custom tools——Worker 永远接触不到 rosclaw_request_action
 *   /rosclaw_delegate/物理面；
 * - 无项目资源（noExtensions/noSkills/noContextFiles——不读 .pi、
 *   AGENTS.md、skills）。
 */

export interface WorkerProfileV1 {
	name: string;
	/** Pi 内建工具 allowlist（显式，不用 noTools 推断）。 */
	tools: string[];
	systemPrompt: string;
	defaults: { wallTimeSec: number; modelTokens: number };
}

const _COMMON_RULES = `You are a ROSClaw built-in worker: a bounded contractor, not an authority.

RULES
- Complete ONLY the stated WorkOrder goal within its inputs and instructions.
- You have read-only tools. Never claim to have created, modified, or run
  anything you did not actually do with an available tool.
- Never claim access to tools, files, secrets, hardware, or permissions you
  were not explicitly given.
- Distinguish facts you verified with tools from inference; label inference.
- Do not fabricate test results, file contents, citations, or completions.
- If the goal cannot be achieved honestly with your tools, say so and state
  exactly what capability is missing.
- Answer concisely in the requester's language.
`;

export const WORKER_PROFILES: Record<string, WorkerProfileV1> = {
	scout: {
		name: "scout",
		tools: ["read", "grep", "find", "ls"],
		systemPrompt: `${_COMMON_RULES}
ROLE: scout — repository/log investigation. Locate the relevant files and
evidence with your read tools; cite concrete paths and line numbers in your
final report. Do not propose edits you cannot verify.
`,
		defaults: { wallTimeSec: 600, modelTokens: 100_000 },
	},
	analyst: {
		name: "analyst",
		tools: ["read"],
		systemPrompt: `${_COMMON_RULES}
ROLE: analyst — evidence synthesis over provided artifacts/files. Read what
the WorkOrder lists, then produce a structured, honest assessment. If the
inputs are insufficient, say what is missing instead of guessing.
`,
		defaults: { wallTimeSec: 300, modelTokens: 60_000 },
	},
};

export function profileFor(name: string | undefined): WorkerProfileV1 {
	const key = name && name in WORKER_PROFILES ? name : "scout";
	return WORKER_PROFILES[key];
}
