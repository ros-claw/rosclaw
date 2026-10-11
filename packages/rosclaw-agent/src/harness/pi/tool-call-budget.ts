import type { ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { createHash, randomUUID } from "node:crypto";
export interface ModelUsageAwareness {
    version: 1;
    lifetime: "runtime";
    inclusiveInputLimit: number;
    outputLimit: number;
    deliveryReserveInput: number;
    deliveryReserveOutput: number;
}
/** Pure, strict validation shared with the pre-SDK CLI. */
export function validateModelUsageAwareness(value: unknown): asserts value is ModelUsageAwareness {
    const keys = ["version", "lifetime", "inclusiveInputLimit", "outputLimit", "deliveryReserveInput", "deliveryReserveOutput"];
    if (!value || typeof value !== "object" || Array.isArray(value))
        throw new Error("INVALID_MODEL_USAGE_AWARENESS");
    const v = value as Record<string, unknown>;
    if (Object.keys(v).length !== keys.length || keys.some(k => !Object.hasOwn(v, k))
        || v.version !== 1 || v.lifetime !== "runtime")
        throw new Error("INVALID_MODEL_USAGE_AWARENESS");
    for (const k of keys.slice(2)) {
        if (typeof v[k] !== "number" || !Number.isSafeInteger(v[k]) || (v[k] as number) < 0)
            throw new Error("INVALID_MODEL_USAGE_AWARENESS");
    }
    if ((v.inclusiveInputLimit as number) <= 0 || (v.outputLimit as number) <= 0
        || (v.deliveryReserveInput as number) > (v.inclusiveInputLimit as number)
        || (v.deliveryReserveOutput as number) > (v.outputLimit as number))
        throw new Error("INVALID_MODEL_USAGE_AWARENESS");
}
const USAGE_MARKER = "ROSCLAW_MODEL_USAGE_JSON:";
const stripUsage = (text: string) => text.split("\n").filter(line => !line.startsWith(USAGE_MARKER)).join("\n");
/** Serial public assistant lifecycle, not start/end object identity (SDK clones starts).
 * Final-object identity is used ONLY to reject replayed ends, never values/timestamps. */
function usageObserver(config: ModelUsageAwareness) {
    const runtimeGeneration = randomUUID();
    let sequence = 0, completedMainRequests = 0, knownInclusiveInput = 0, knownOutput = 0;
    let pendingMainRequests = 0, unknownUsageRequests = 0, unaccountedCompaction = false;
    let sessionId: string | undefined;
    let pendingSession: string | undefined;
    const settled = new WeakSet<object>();
    const add = (a: number, b: number) => {
        if (!Number.isSafeInteger(a + b)) {
            unknownUsageRequests++;
            return a;
        }
        return a + b;
    };
    const notice = () => {
        const remainingInputUpperBound = Math.max(0, config.inclusiveInputLimit - knownInclusiveInput);
        const remainingOutputUpperBound = Math.max(0, config.outputLimit - knownOutput);
        const reserveState = pendingMainRequests || unknownUsageRequests || unaccountedCompaction ? "unknown"
            : remainingInputUpperBound <= config.deliveryReserveInput || remainingOutputUpperBound <= config.deliveryReserveOutput ? "finish_delivery" : "normal";
        return USAGE_MARKER + JSON.stringify({ version: 1, runtimeGeneration, sequence, completedMainRequests,
            knownInclusiveInput, knownOutput, remainingInputUpperBound, remainingOutputUpperBound,
            pendingMainRequests, unknownUsageRequests, unaccountedCompaction, reserveState });
    };
    const factory: ExtensionFactory = pi => {
        pi.on("session_start", (_event, ctx) => {
            if (pendingMainRequests) {
                unknownUsageRequests += pendingMainRequests;
                pendingMainRequests = 0;
            }
            pendingSession = undefined;
            sessionId = ctx.sessionManager.getSessionId();
            sequence++;
        });
        pi.on("message_start", (event, ctx) => {
            if (event.message.role !== "assistant" || ctx.sessionManager.getSessionId() !== sessionId)
                return;
            if (pendingMainRequests)
                unknownUsageRequests += pendingMainRequests;
            pendingMainRequests = 1;
            pendingSession = sessionId;
            sequence++;
        });
        pi.on("message_end", (event, ctx) => {
            const message = event.message;
            if (message.role !== "assistant" || settled.has(message) || !pendingMainRequests
                || pendingSession !== ctx.sessionManager.getSessionId() || pendingSession !== sessionId)
                return;
            settled.add(message);
            pendingMainRequests = 0;
            pendingSession = undefined;
            completedMainRequests++;
            sequence++;
            const usage = message.usage;
            const failed = message.stopReason === "error" || message.stopReason === "aborted";
            let unknown = failed;
            let input = 0, output = 0;
            for (const key of ["input", "cacheRead", "cacheWrite", "output"] as const) {
                const n = usage?.[key];
                if (typeof n !== "number" || !Number.isSafeInteger(n) || n < 0) {
                    unknown = true;
                    continue;
                }
                // Error/abort SDK defaults are not measurements; positive partials are lowerbounds.
                if (failed && n === 0)
                    continue;
                if (key === "output")
                    output = n;
                else if (Number.isSafeInteger(input + n))
                    input += n;
                else
                    unknown = true;
            }
            knownInclusiveInput = add(knownInclusiveInput, input);
            knownOutput = add(knownOutput, output);
            if (unknown)
                unknownUsageRequests++;
        });
        pi.on("session_before_compact", () => { unaccountedCompaction = true; sequence++; });
    };
    return { factory, notice };
}
import { lstatSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
const PATH_TOOLS = new Set(["read", "write", "edit", "rosclaw_deliver"]);
function validPath(value: unknown): value is string {
    return typeof value === "string" && value.trim().length > 0 && !value.includes("\0")
        && !value.split(/[\\/]/).includes("..") && !value.endsWith("/");
}
/** Shared early schema check; no filesystem, SDK, model or authentication access. */
export function validateExactPaths(value: unknown, allowed: ReadonlySet<string>): void {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_PATHS");
    }
    for (const [name, paths] of Object.entries(value)) {
        if (!allowed.has(name) || !PATH_TOOLS.has(name) || !Array.isArray(paths) || paths.length === 0
            || paths.some(p => !validPath(p)) || new Set(paths).size !== paths.length) {
            throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_PATH");
        }
    }
}
function inside(root: string, target: string): boolean {
    const rel = relative(root, target);
    return rel === "" || (!isAbsolute(rel) && rel !== ".." && !rel.startsWith(`..${sep}`));
}
/** Resolve existing ancestors without creating anything. Check each symlink boundary,
 * including the leaf; dangling links and non-directory parents fail closed. */
function canonicalFile(root: string, canonicalRoot: string, value: unknown): string {
    if (!validPath(value))
        throw new Error("PATH_INPUT");
    const target = resolve(root, value);
    const base = inside(root, target) ? root : canonicalRoot;
    if (!inside(base, target) || target === base)
        throw new Error("PATH_OUTSIDE");
    let current = base;
    let canonical = canonicalRoot;
    const parts = relative(base, target).split(sep);
    let missing = false;
    for (let i = 0; i < parts.length; i++) {
        current = join(current, parts[i]);
        if (missing) {
            canonical = join(canonical, parts[i]);
            continue;
        }
        try {
            lstatSync(current);
        }
        catch (err) {
            if ((err as NodeJS.ErrnoException).code !== "ENOENT")
                throw err;
            missing = true;
            canonical = join(canonical, parts[i]);
            continue;
        }
        canonical = realpathSync(current);
        if (!inside(canonicalRoot, canonical))
            throw new Error("PATH_SYMLINK_ESCAPE");
        const stat = statSync(current);
        if (i < parts.length - 1 && !stat.isDirectory())
            throw new Error("PATH_PARENT");
        if (i === parts.length - 1 && stat.isDirectory())
            throw new Error("PATH_DIRECTORY");
    }
    return canonical;
}
export type ToolCallBudgetExtension = ExtensionFactory & {
    /** Native execution seam: recheck final args and count once, before tool effects. */
    wrapTools(tools: ToolDefinition<any, any>[]): ToolDefinition<any, any>[];
    hasPath(toolName: string): boolean;
};
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
    /** Exact canonical files for read/write/edit/rosclaw_deliver; never recursive authority. */
    exactPaths?: Readonly<Record<string, readonly string[]>>;
    /** Opt-in model-visible policy snapshot notices. Default false keeps old notice-free output. */
    visibleBudget?: boolean;
    /** Result presentation only; omitted means full. Never changes admission. */
    visibleBudgetMode?: "full" | "compact";
    /** Advisory observer only, never permissions or a provider hard cap. */
    modelUsageAwareness?: ModelUsageAwareness;
}
const POLICY_MARKER = "ROSCLAW_TOOL_POLICY_JSON:";
const COMPACT_MARKER = "ROSCLAW_TOOL_BUDGET_COMPACT_JSON:";
const KNOWN_KEYS = ["allowedTools", "maxCalls", "maxTotalCalls", "exactCommands", "visibleBudget", "exactPaths", "visibleBudgetMode", "modelUsageAwareness"];
function stripMarkerLines(text: string, compact = false): string {
    return text.split("\n").filter(line => !line.startsWith(POLICY_MARKER)
        && !(compact && line.startsWith(COMPACT_MARKER))).join("\n");
}
// Recursive sorted-key JSON; arrays retain their effective order.
function canonicalJSON(value: unknown): string {
    if (Array.isArray(value))
        return `[${value.map(canonicalJSON).join(",")}]`;
    if (value !== null && typeof value === "object") {
        const record = value as Record<string, unknown>;
        return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${canonicalJSON(record[key])}`).join(",")}}`;
    }
    return JSON.stringify(value)!;
}
export function createToolCallBudgetExtension(policy: ToolCallBudget, workspaceRoot?: string): ToolCallBudgetExtension {
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
        || Array.isArray(policy.maxCalls)))
        throw new Error("INVALID_TOOL_CALL_BUDGET_LIMITS");
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
        || Array.isArray(policy.exactCommands)))
        throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_COMMANDS");
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
    if (policy.visibleBudgetMode !== undefined && policy.visibleBudgetMode !== "full" && policy.visibleBudgetMode !== "compact") {
        throw new Error("INVALID_TOOL_CALL_BUDGET_VISIBLE_MODE");
    }
    if (Object.hasOwn(policy, "modelUsageAwareness"))
        validateModelUsageAwareness(policy.modelUsageAwareness);
    const usage = policy.modelUsageAwareness ? usageObserver({ ...policy.modelUsageAwareness }) : undefined;
    const compact = policy.visibleBudgetMode === "compact";
    const paths = new Map<string, readonly string[]>();
    let root: string | undefined;
    let canonicalRoot: string | undefined;
    if (policy.exactPaths !== undefined) {
        validateExactPaths(policy.exactPaths, allowed);
        if (typeof workspaceRoot !== "string" || !isAbsolute(workspaceRoot)) {
            throw new Error("INVALID_TOOL_CALL_BUDGET_PATH_WORKSPACE_REQUIRED");
        }
        root = resolve(workspaceRoot);
        try {
            canonicalRoot = realpathSync(root);
            if (!statSync(root).isDirectory())
                throw new Error("not a directory");
            for (const [name, declared] of Object.entries(policy.exactPaths)) {
                const bound = declared.map(p => canonicalFile(root!, canonicalRoot!, p));
                if (new Set(bound).size !== bound.length)
                    throw new Error("duplicate canonical file");
                paths.set(name, Object.freeze(bound));
            }
        }
        catch {
            throw new Error("INVALID_TOOL_CALL_BUDGET_EXACT_PATH_BINDING");
        }
    }
    const hasPaths = policy.exactPaths !== undefined;
    const wrapped = new Set<string>();
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
            ...(hasPaths ? { workspaceRoot: canonicalRoot,
                exactPaths: Object.fromEntries([...paths].map(([name, files]) => [name, [...files]])) } : {}),
            maxCalls: Object.fromEntries(limits),
            maxTotalCalls: maxTotal ?? null,
            usedCalls,
            usedTotal,
            remainingCalls,
            remainingTotal: maxTotal === undefined ? null : Math.max(0, maxTotal - usedTotal),
        };
    };
    const { usedCalls: _used, usedTotal: _total, remainingCalls: _remaining, remainingTotal: _remainingTotal, ...staticPolicy } = snapshot();
    const policyDigest = createHash("sha256").update(canonicalJSON(staticPolicy), "utf8").digest("hex");
    const fullNotice = () => POLICY_MARKER + JSON.stringify(snapshot());
    const notice = () => {
        if (!compact)
            return fullNotice();
        const { usedCalls, usedTotal, remainingCalls, remainingTotal } = snapshot();
        return COMPACT_MARKER + JSON.stringify({ policyDigest, usedCalls, usedTotal, remainingCalls, remainingTotal });
    };
    const blocked = (code: string) => visible
        ? { block: true as const, reason: `${code}\n${notice()}` }
        : { block: true as const, reason: code };
    const check = (name: string, input: Record<string, unknown> | undefined, consume: boolean) => {
        if (!allowed.has(name))
            return blocked("TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED");
        const commands = exact.get(name);
        if (commands !== undefined) {
            const command = input?.command;
            if (typeof command !== "string" || !commands.includes(command)) {
                return blocked("TOOL_CALL_BUDGET_EXACT_COMMAND_REJECTED");
            }
        }
        const files = paths.get(name);
        if (files !== undefined) {
            try {
                if (!files.includes(canonicalFile(root!, canonicalRoot!, input?.path))) {
                    return blocked("TOOL_CALL_BUDGET_EXACT_PATH_REJECTED");
                }
            }
            catch {
                return blocked("TOOL_CALL_BUDGET_EXACT_PATH_REJECTED");
            }
        }
        const used = counts.get(name) ?? 0;
        const limit = limits.get(name);
        if ((limit !== undefined && used >= limit) || (maxTotal !== undefined && total >= maxTotal)) {
            return blocked("TOOL_CALL_BUDGET_EXHAUSTED");
        }
        if (consume) {
            // Synchronous admission, shared by direct, parallel and nested calls.
            counts.set(name, used + 1);
            total++;
        }
        return undefined;
    };
    const extension: ExtensionFactory = pi => {
        pi.on("tool_call", async (event) => check(event.toolName, event.input as Record<string, unknown> | undefined, !wrapped.has(event.toolName)));
        usage?.factory(pi);
        if (!visible && !usage)
            return;
        pi.on("before_agent_start", event => {
            let text = visible ? stripMarkerLines(event.systemPrompt, compact) : event.systemPrompt;
            if (usage)
                text = stripUsage(text);
            return { systemPrompt: text + (visible ? `\n${fullNotice()}` : "") + (usage ? `\n${usage.notice()}` : "") };
        });
        // before_agent_start's forced prompt is scoped to a manual run and is
        // cleared at settlement. Idle custom triggerTurn bypasses that hook.
        // This public post-restoration hook owns the actual request transcript,
        // including its system prompt; never persist this advisory projection.
        if (usage)
            pi.on("context_with_system", event => ({
                messages: event.messages.map(message => message.role === "system"
                    ? { ...message, content: typeof message.content === "string"
                            ? `${stripUsage(message.content)}\n${usage.notice()}`
                            : [...message.content.map(part => ({ ...part, text: stripUsage(part.text) })),
                                { type: "text" as const, text: usage.notice() }] }
                    : message),
            }));
        pi.on("tool_result", event => {
            // Replace a prior own marker line (for example on an already-blocked
            // result) instead of duplicating it; preserve all other text.
            const content = event.content.map(part => {
                if (part.type !== "text")
                    return part;
                let text = visible ? stripMarkerLines(part.text, compact) : part.text;
                if (usage)
                    text = stripUsage(text);
                return { ...part, text };
            });
            return {
                content: [...content, ...(visible ? [{ type: "text" as const, text: notice() }] : []),
                    ...(usage ? [{ type: "text" as const, text: usage.notice() }] : [])],
                structuredContent: event.structuredContent,
            };
        });
    };
    return Object.assign(extension, {
        hasPath: (name: string) => paths.has(name),
        wrapTools(tools: ToolDefinition<any, any>[]): ToolDefinition<any, any>[] {
            return tools.map(tool => {
                if (!paths.has(tool.name))
                    return tool;
                wrapped.add(tool.name);
                return { ...tool, execute(id, params, signal, onUpdate, ctx) {
                        // PI tool_call input is mutable and handlers run sequentially. Do not
                        // trust the earlier hook: snapshot/revalidate the actual execute args.
                        const finalInput = { ...(params as Record<string, unknown>) };
                        const denial = check(tool.name, finalInput, false);
                        if (denial)
                            throw new Error(denial.reason);
                        finalInput.path = canonicalFile(root!, canonicalRoot!, finalInput.path);
                        const admission = check(tool.name, finalInput, true);
                        if (admission)
                            throw new Error(admission.reason);
                        return tool.execute(id, finalInput, signal, onUpdate, ctx);
                    } };
            });
        },
    });
}
