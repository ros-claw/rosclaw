import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";

/** Operator-supplied restrictions for one runtime lifetime, never permission. */
export interface ToolCallBudget {
	allowedTools: readonly string[];
	maxCalls?: Readonly<Record<string, number>>;
	maxTotalCalls?: number;
	/**
	 * Optional exact command allowlist per allowed tool. A configured tool's
	 * `input.command` string must exactly equal one declared command; no
	 * parsing, normalization, trimming, execution, path resolution or shell
	 * sandbox claim is made.
	 */
	exactCommands?: Readonly<Record<string, readonly string[]>>;
	/** Opt-in model-visible policy snapshot notices. Default false keeps old notice-free output. */
	visibleBudget?: boolean;
}

const POLICY_MARKER = "ROSCLAW_TOOL_POLICY_JSON:";
const KNOWN_KEYS = ["allowedTools", "maxCalls", "maxTotalCalls", "exactCommands", "visibleBudget"];

function stripMarkerLines(text: string): string {
	return text.split("\n").filter(line => !line.startsWith(POLICY_MARKER)).join("\n");
}

export function createToolCallBudgetExtension(policy: ToolCallBudget): ExtensionFactory {
	if (!policy || typeof policy !== "object" || Array.isArray(policy)
		|| Object.keys(policy).some(key => !KNOWN_KEYS.includes(key))) {
		throw new Error("INVALID_TOOL_CALL_BUDGET");
	}
	if (!policy || !Array.isArray(policy.allowedTools)
		|| policy.allowedTools.some(name => typeof name !== "string" || !name || name !== name.trim())
		|| new Set(policy.allowedTools).size !== policy.allowedTools.length) {
		throw new Error("INVALID_TOOL_CALL_BUDGET_ALLOWLIST");
	}
	const allowed = new Set(policy.allowedTools);
	const allowedNames = Object.freeze([...policy.allowedTools]);
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
	if (policy.exactCommands !== undefined && (!policy.exactCommands || typeof policy.exactCommands !== "object"
		|| Array.isArray(policy.exactCommands))) throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_COMMANDS");
	const exact = new Map<string, readonly string[]>();
	for (const [name, commands] of Object.entries(policy.exactCommands ?? {})) {
		if (!allowed.has(name) || !Array.isArray(commands) || commands.length === 0
			|| commands.some(command => typeof command !== "string" || command.length === 0)
			|| new Set(commands).size !== commands.length) {
			throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_COMMAND");
		}
		exact.set(name, Object.freeze([...commands]));
	}
	if (policy.visibleBudget !== undefined && typeof policy.visibleBudget !== "boolean") {
		throw new Error("INVALID_TOOL_CALL_BUDGET_VISIBLE");
	}
	const visible = policy.visibleBudget === true;
	const counts = new Map<string, number>();
	let total = 0;
	const snapshot = () => {
		// Object.fromEntries defines own keys (CreateDataProperty), so reserved
		// names like __proto__/constructor/prototype stay own data keys and no
		// prototype is ever mutated.
		const usedCalls: Record<string, number> = Object.fromEntries(allowedNames.map(name => [name, counts.get(name) ?? 0]));
		const usedTotal = Object.values(usedCalls).reduce((sum, used) => sum + used, 0);
		const remainingCalls: Record<string, number | null> = Object.fromEntries(allowedNames.map(name => {
			const limit = limits.get(name);
			return [name, limit === undefined ? null : Math.max(0, limit - (counts.get(name) ?? 0))];
		}));
		return {
			allowedTools: [...allowedNames],
			exactCommands: Object.fromEntries([...exact].map(([name, commands]) => [name, [...commands]])),
			maxCalls: Object.fromEntries(limits),
			maxTotalCalls: maxTotal ?? null,
			usedCalls,
			usedTotal,
			remainingCalls,
			remainingTotal: maxTotal === undefined ? null : Math.max(0, maxTotal - usedTotal),
		};
	};
	const notice = () => POLICY_MARKER + JSON.stringify(snapshot());
	const blocked = (code: string) => visible
		? { block: true as const, reason: `${code}\n${notice()}` }
		: { block: true as const, reason: code };
	return pi => {
		pi.on("tool_call", async event => {
			const name = event.toolName;
			if (!allowed.has(name)) return blocked("TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED");
			const commands = exact.get(name);
			if (commands !== undefined) {
				const command = (event.input as Record<string, unknown> | undefined)?.command;
				if (typeof command !== "string" || !commands.includes(command)) {
					return blocked("TOOL_CALL_BUDGET_EXACT_COMMAND_REJECTED");
				}
			}
			const used = counts.get(name) ?? 0;
			const limit = limits.get(name);
			if ((limit !== undefined && used >= limit) || (maxTotal !== undefined && total >= maxTotal)) {
				return blocked("TOOL_CALL_BUDGET_EXHAUSTED");
			}
			// Count admission before execution; failures and parallel/nested calls consume it.
			counts.set(name, used + 1);
			total++;
			return undefined;
		});
		if (!visible) return;
		pi.on("before_agent_start", event => ({
			systemPrompt: `${stripMarkerLines(event.systemPrompt)}\n${notice()}`,
		}));
		pi.on("tool_result", event => {
			// Replace a prior own marker line (for example on an already-blocked
			// result) instead of duplicating it; preserve all other text.
			const content = event.content.map(part => part.type === "text"
				? { ...part, text: stripMarkerLines(part.text) }
				: part);
			return {
				content: [...content, { type: "text" as const, text: notice() }],
				structuredContent: event.structuredContent,
			};
		});
	};
}
