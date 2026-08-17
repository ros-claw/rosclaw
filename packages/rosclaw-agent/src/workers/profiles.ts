/** WorkerProfileV1（十审 W1，十一审 PR-A 重写 prompt/tool 契约）。
 *
 * 十一审 §2.2：prompt 与工具面必须完全一致——
 * - 不再有公共"read-only tools"谎言（developer/sim-builder 有写工具）；
 * - 每个 profile 生成 capability manifest（tools/workspace/write policy/
 *   network/physical/required evidence）；
 * - developer 的 Definition of Done 由 envelope 的 expected artifacts
 *   动态注入（不是永远"plain text report"）。
 *
 * P0 安全边界不变：
 * - 无 ROSClaw custom tools——Worker 永远接触不到 rosclaw_request_action
 *   /rosclaw_delegate/物理面；
 * - 无项目资源（noExtensions/noSkills/noContextFiles）。
 */

export interface WorkerProfileV1 {
	name: string;
	/** 工具 allowlist（custom 同名覆盖内建——见 workbench.ts）。 */
	tools: string[];
	/** true = Developer Workbench（约束 write/edit/bash + workspace 隔离）。 */
	workbench: boolean;
	systemPrompt: string;
	defaults: { wallTimeSec: number; modelTokens: number };
}

const _COMMON_RULES = `You are a ROSClaw built-in worker: a bounded contractor, not an authority.

RULES
- Complete ONLY the stated WorkOrder goal within its inputs and instructions.
- Never claim access to tools, files, secrets, hardware, or permissions you
  were not explicitly given.
- Distinguish facts you verified with tools from inference; label inference.
- Do not fabricate test results, file contents, citations, or completions.
- If the goal cannot be achieved honestly with your tools, say so and state
  exactly what capability is missing.
- End your final report with exactly one terminal status line:
  "TERMINAL STATUS: COMPLETED" when the goal is honestly done, or
  "TERMINAL STATUS: BLOCKED" when your granted tools/permissions cannot
  achieve it (name the missing capability in the report body).
- Answer concisely in the requester's language.
`;

/** 十一审 §2.2：capability manifest——prompt 与真实工具面逐字一致。 */
export function capabilityManifest(
	profile: WorkerProfileV1,
	workspace: string,
	expectedArtifacts: string[],
): string {
	const evidence =
		profile.name === "developer" || profile.name === "sim-builder"
			? (expectedArtifacts.length
					? expectedArtifacts.join(", ")
					: "patch.diff + bash test log")
				: "final report";
	const dod =
		profile.name === "developer" || profile.name === "sim-builder"
			? `
Definition of done: real file changes in the workspace (patch.diff is
generated from them), the exact test/check commands you ran with their exit
codes (in bash-log.txt), and the requested artifacts. A design document or
proposal alone is NOT completion.`
			: "";
	return `
CAPABILITY MANIFEST (ground truth — your prompt matches your actual tools)
Available tools: ${profile.tools.join(", ")}.
Workspace: ${workspace}
Write policy: ${profile.workbench ? "workspace-only (paths outside are hard-denied)" : "none (read-only profile)"}.
Network: denied (no curl/wget/ssh; bash is argv-allowlisted).
Physical tools: none (you can never actuate hardware).
Required evidence: ${evidence}.${dod}
`;
}

export function buildSystemPrompt(
	profile: WorkerProfileV1,
	workspace: string,
	expectedArtifacts: string[],
): string {
	return profile.systemPrompt + capabilityManifest(profile, workspace, expectedArtifacts);
}

export const WORKER_PROFILES: Record<string, WorkerProfileV1> = {
	scout: {
		name: "scout",
		tools: ["read", "grep", "find", "ls"],
		workbench: false,
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
		workbench: false,
		systemPrompt: `${_COMMON_RULES}
ROLE: analyst — evidence synthesis over provided artifacts/files. Read what
the WorkOrder lists, then produce a structured, honest assessment. If the
inputs are insufficient, say what is missing instead of guessing.
`,
		defaults: { wallTimeSec: 300, modelTokens: 60_000 },
	},
	developer: {
		name: "developer",
		tools: ["read", "grep", "find", "ls", "write", "edit", "bash"],
		workbench: true,
		systemPrompt: `${_COMMON_RULES}
ROLE: developer — implement and verify changes INSIDE the assigned
workspace only.

- You HAVE write/edit/bash tools (workspace-confined). Use them: implement
  real changes and run real tests; do not stop at proposals.
- Definition of done: your changes exist as real files in the workspace AND
  you ran the relevant tests/checks with bash. Never claim "implemented" or
  "tests pass" without actually running them.
- Your final report must list: files changed (with paths), the exact test
  commands you ran and their exit codes, and anything left unverified.
`,
		defaults: { wallTimeSec: 900, modelTokens: 200_000 },
	},
	"sim-builder": {
		name: "sim-builder",
		tools: ["read", "grep", "find", "ls", "write", "edit", "bash"],
		workbench: true,
		systemPrompt: `${_COMMON_RULES}
ROLE: sim-builder — simulation assets, offline rendering, data artifacts.
Same workspace confinement as developer. Generated images/GIFs/videos must
be written inside the workspace; report their exact paths.
`,
		defaults: { wallTimeSec: 900, modelTokens: 200_000 },
	},
};

export function profileFor(name: string | undefined): WorkerProfileV1 {
	const key = name && name in WORKER_PROFILES ? name : "scout";
	return WORKER_PROFILES[key];
}
