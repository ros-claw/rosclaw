import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";

/** Operator-supplied restrictions for one runtime lifetime, never permission. */
export interface ToolCallBudget {
	allowedTools: readonly string[];
	maxCalls?: Readonly<Record<string, number>>;
	maxTotalCalls?: number;
}

export function createToolCallBudgetExtension(policy: ToolCallBudget): ExtensionFactory {
	if (!policy || typeof policy !== "object" || Array.isArray(policy)
		|| Object.keys(policy).some(key => !["allowedTools", "maxCalls", "maxTotalCalls"].includes(key))) {
		throw new Error("INVALID_TOOL_CALL_BUDGET");
	}
	if (!policy || !Array.isArray(policy.allowedTools)
		|| policy.allowedTools.some(name => typeof name !== "string" || !name || name !== name.trim())
		|| new Set(policy.allowedTools).size !== policy.allowedTools.length) {
		throw new Error("INVALID_TOOL_CALL_BUDGET_ALLOWLIST");
	}
	const allowed = new Set(policy.allowedTools);
	if (policy.maxCalls !== undefined && (!policy.maxCalls || typeof policy.maxCalls !== "object"
		|| Array.isArray(policy.maxCalls))) throw new Error("INVALID_TOOL_CALL_BUDGET_LIMITS");
	const limits = new Map(Object.entries(policy.maxCalls ?? {}));
	for (const [name, count] of limits) {
		if (!allowed.has(name) || !Number.isSafeInteger(count) || count < 0) {
			throw new Error("INVALID_TOOL_CALL_BUDGET_LIMIT");
		}
	}
	const maxTotal = policy.maxTotalCalls;
	if (maxTotal !== undefined && (!Number.isSafeInteger(maxTotal) || maxTotal < 0)) {
		throw new Error("INVALID_TOOL_CALL_BUDGET_TOTAL");
	}
	const counts = new Map<string, number>();
	let total = 0;
	return pi => {
		pi.on("tool_call", async event => {
			const name = event.toolName;
			if (!allowed.has(name)) return { block: true, reason: "TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED" };
			const used = counts.get(name) ?? 0;
			const limit = limits.get(name);
			if ((limit !== undefined && used >= limit) || (maxTotal !== undefined && total >= maxTotal)) {
				return { block: true, reason: "TOOL_CALL_BUDGET_EXHAUSTED" };
			}
			// Count admission before execution; failures and parallel/nested calls consume it.
			counts.set(name, used + 1);
			total++;
			return undefined;
		});
	};
}
