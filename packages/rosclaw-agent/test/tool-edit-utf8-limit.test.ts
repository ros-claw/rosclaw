import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, type ToolCallBudget } from "../src/harness/pi/tool-call-budget.js";

const marker = "ROSCLAW_TOOL_POLICY_JSON:";
const ctx = {} as never;
function notice(text: string) { return JSON.parse(text.split("\n").find(line => line.startsWith(marker))!.slice(marker.length)); }
function fixture(run: (root: string) => unknown) {
    const root = mkdtempSync(join(tmpdir(), "edit-utf8-"));
    writeFileSync(join(root, "allowed"), "seed");
    return Promise.resolve().then(async () => { await run(root); }).finally(() => rmSync(root, { recursive: true, force: true }));
}
function make(root: string, cap: number | null = 24576, paths = true, maximum = 5) {
    const policy: ToolCallBudget = { allowedTools: ["edit", "write"], visibleBudget: true,
        maxTotalCalls: maximum, maxCalls: { edit: maximum },
        ...(cap === null ? {} : { maxEditUtf8BytesExclusive: cap }),
        ...(paths ? { exactPaths: { edit: ["allowed"] } } : {}) };
    const extension = createToolCallBudgetExtension(policy, root);
    let effects = 0;
    const tool = (name: string): ToolDefinition<any, any> => ({ name, label: name, description: "inert IO fixture",
        parameters: Type.Object({ path: Type.String(), oldText: Type.String(), newText: Type.String() }),
        async execute(_id, args: any) {
            effects++;
            readFileSync(args.path); // Rejection must never reach either IO operation.
            writeFileSync(args.path, args.newText);
            return { content: [{ type: "text", text: "fixture effect" }], details: {} };
        } });
    const tools = extension.wrapTools([tool("edit"), tool("write")]);
    let hook: any;
    extension({ on(name: string, fn: unknown) { if (name === "tool_call") hook = fn; } } as unknown as ExtensionAPI);
    return { policy, tools, effects: () => effects,
        call: (args: Record<string, unknown>) => tools[0].execute("id", args, undefined, undefined, ctx),
        hook: (args: Record<string, unknown>) => hook({ toolName: "edit", input: args }, {}) };
}

for (const [label, oldText, newText] of [
    ["ASCII below", "x", "a".repeat(24574)],
    ["ASCII exact", "x", "a".repeat(24575)],
    ["multibyte below", "中".repeat(8191), "aa"],
    ["multibyte exact", "中".repeat(8191), "界"],
    ["combined halves", "x".repeat(12288), "y".repeat(12288)],
    ["astral UTF8", "😀".repeat(6143), "😀"],
] as const) {
    test(`strict old+new UTF8 limit: ${label}`, () => fixture(async root => {
        const f = make(root);
        const bytes = Buffer.byteLength(oldText) + Buffer.byteLength(newText);
        const args = { path: "allowed", oldText, newText };
        assert.equal(await f.hook(args), undefined); // Hook does not charge wrapped calls.
        if (bytes < 24576) { await f.call(args); assert.equal(f.effects(), 1); }
        else {
            assert.throws(() => f.call(args), error => {
                assert.match((error as Error).message, /EDIT_UTF8_LIMIT_REJECTED.*split/);
                assert.ok((error as Error).message.includes(`oldText=${Buffer.byteLength(oldText)}B newText=${Buffer.byteLength(newText)}B total=${bytes}B; total must be < 24576B`));
                assert.match((error as Error).message, /charged once \(used edit calls=1, used total calls=1\); no edit executed and no call refunded/);
                assert.match((error as Error).message, /current file, each with UTF8\(oldText\)\+UTF8\(newText\) <= 24575B/);
                assert.equal(notice((error as Error).message).usedTotal, 1);
                return true;
            });
            assert.equal(f.effects(), 0);
            assert.equal(readFileSync(join(root, "allowed"), "utf8"), "seed");
        }
    }));
}

test("refused attempt consumes exactly once and agent can split/retry until exhaustion", () => fixture(async root => {
    const f = make(root, 24576, true, 2);
    assert.throws(() => f.call({ path: "allowed", oldText: "x", newText: "a".repeat(24575) }), /EDIT_UTF8_LIMIT_REJECTED/);
    assert.equal(f.effects(), 0);
    assert.equal(await f.hook({ path: "allowed" }), undefined);
    await f.call({ path: "allowed", oldText: "seed", newText: "small" });
    assert.equal(f.effects(), 1);
    assert.throws(() => f.call({ path: "allowed", oldText: "small", newText: "third" }), /EXHAUSTED/);
    assert.equal(f.effects(), 1);
    assert.equal(readFileSync(join(root, "allowed"), "utf8"), "small");
}));

test("no exactPaths needed for argument guard; other tools and omitted option unchanged", () => fixture(async root => {
    const f = make(root, 10, false);
    const args = { path: join(root, "allowed"), oldText: "seed", newText: "long content" };
    assert.throws(() => f.call(args), /EDIT_UTF8_LIMIT_REJECTED/);
    await f.tools[1].execute("write", args, undefined, undefined, ctx);
    assert.equal(f.effects(), 1);
    const legacy = make(root, null);
    await legacy.call({ ...args, path: "allowed", newText: "x".repeat(30000) });
    assert.equal(legacy.effects(), 1);
}));

test("invalid text types consume an attempt and never enter the tool body", () => fixture(root => {
    const f = make(root);
    for (const args of [{ oldText: 1, newText: "x" }, { oldText: "x", newText: null }]) {
        assert.throws(() => f.call({ path: "allowed", ...args }), /EDIT_UTF8_INPUT_REJECTED/);
    }
    assert.equal(f.effects(), 0);
}));

test("post-hook mutation rechecked; execute snapshots params and immutable configured cap", () => fixture(async root => {
    const f = make(root);
    const args = { path: "allowed", oldText: "seed", newText: "small" };
    assert.equal(await f.hook(args), undefined);
    args.newText = "x".repeat(24576);
    assert.throws(() => f.call(args), /EDIT_UTF8_LIMIT_REJECTED/);
    f.policy.maxEditUtf8BytesExclusive = 100000;
    assert.throws(() => f.call(args), /EDIT_UTF8_LIMIT_REJECTED/);
    const original = { path: "allowed", oldText: "seed", newText: "snapshot" };
    const pending = f.call(original);
    original.newText = "x".repeat(30000);
    await pending;
    assert.equal(readFileSync(join(root, "allowed"), "utf8"), "snapshot");
    assert.equal(f.effects(), 1);
}));

test("constructor rejects invalid or inapplicable opt-in policy", () => {
    for (const cap of [undefined, null, true, "24576", 0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
        assert.throws(() => createToolCallBudgetExtension({ allowedTools: ["edit"], maxEditUtf8BytesExclusive: cap } as ToolCallBudget), /EDIT_UTF8_LIMIT/);
    }
    assert.throws(() => createToolCallBudgetExtension({ allowedTools: ["read"], maxEditUtf8BytesExclusive: 24576 }), /EDIT_UTF8_LIMIT/);
    assert.doesNotThrow(() => createToolCallBudgetExtension({ allowedTools: ["edit"], maxEditUtf8BytesExclusive: 1 }));
});

test("policy notice and digest bind cap; default notice shape stays unchanged", () => fixture(root => {
    function snapshot(cap?: number) {
        const f = createToolCallBudgetExtension({ allowedTools: ["edit"], visibleBudget: true, visibleBudgetMode: "compact",
            ...(cap === undefined ? {} : { maxEditUtf8BytesExclusive: cap }) });
        let before: any, resultHook: any;
        f({ on(name: string, fn: unknown) { if (name === "before_agent_start") before = fn; if (name === "tool_result") resultHook = fn; } } as unknown as ExtensionAPI);
        const result = before({ systemPrompt: "fixture", systemPromptOptions: {} });
        const compact = resultHook({ content: [], structuredContent: undefined }).content[0].text;
        return { ...notice(result.systemPrompt), policyDigest: JSON.parse(compact.slice("ROSCLAW_TOOL_BUDGET_COMPACT_JSON:".length)).policyDigest };
    }
    const a = snapshot(24576), b = snapshot(24577), old = snapshot();
    assert.equal(a.maxEditUtf8BytesExclusive, 24576);
    assert.notEqual(a.policyDigest, b.policyDigest);
    assert.equal(Object.hasOwn(old, "maxEditUtf8BytesExclusive"), false);
    assert.equal(a.usedTotal, 0);
}));

test("real offline PI runner returns refusal as tool error then continues with smaller edit", { timeout: 10000 }, () => fixture(async root => {
    const { InMemoryCredentialStore, createAssistantMessageEventStream } = await import("@earendil-works/pi-ai");
    const { ModelRuntime, DefaultResourceLoader, SettingsManager, SessionManager, createAgentSession } = await import("@earendil-works/pi-coding-agent");
    const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
    const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
        modelsStorePath: join(root, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
    const model: import("@earendil-works/pi-ai").Model<"openai-completions"> = {
        id: "fixture", name: "fixture", api: "openai-completions", provider: "fixture", baseUrl: "http://invalid.local",
        reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100 };
    runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "OFFLINE_FIXTURE", models: [model] });
    const factory = createToolCallBudgetExtension({ allowedTools: ["edit"], exactPaths: { edit: ["allowed"] },
        maxCalls: { edit: 2 }, maxTotalCalls: 2, maxEditUtf8BytesExclusive: 24576, visibleBudget: true }, root);
    const loader = new DefaultResourceLoader({ cwd: root, agentDir: root, settingsManager: settings,
        noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
        extensionFactories: [{ name: "budget", factory }] });
    await loader.reload();
    let effects = 0;
    const tools = factory.wrapTools([{ name: "edit", label: "edit", description: "inert file fixture",
        parameters: Type.Object({ path: Type.String(), oldText: Type.String(), newText: Type.String() }),
        async execute(_id, params: any) {
            effects++; assert.equal(readFileSync(params.path, "utf8"), params.oldText);
            writeFileSync(params.path, params.newText);
            return { content: [{ type: "text" as const, text: "edited" }], details: {} };
        } }]);
    const { session } = await createAgentSession({ cwd: root, agentDir: root, model: runtime.getModel(model.provider, model.id)!,
        modelRuntime: runtime, sessionManager: SessionManager.inMemory(root), settingsManager: settings, resourceLoader: loader,
        tools: ["edit"], customTools: tools, thinkingLevel: "off" });
    let requests = 0;
    session.agent.streamFunction = ((_m: unknown, context: any) => {
        requests++;
        assert.ok(requests <= 3);
        if (requests === 2) {
            assert.equal(effects, 0); assert.equal(readFileSync(join(root, "allowed"), "utf8"), "seed");
            const result = context.messages.findLast((m: any) => m.role === "toolResult");
            assert.equal(result.isError, true);
            assert.match(result.content.map((p: any) => p.text ?? "").join(""), /EDIT_UTF8_LIMIT_REJECTED/);
        }
        const calls = requests < 3;
        const message = { role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
            content: calls ? [{ type: "toolCall" as const, id: `c${requests}`, name: "edit", arguments: {
                path: "allowed", oldText: "seed", newText: requests === 1 ? "a".repeat(24572) : "small" } }] : [],
            usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: calls ? "toolUse" as const : "stop" as const, timestamp: Date.now() };
        const stream = createAssistantMessageEventStream(); stream.push({ type: "done", reason: message.stopReason, message }); return stream;
    }) as typeof session.agent.streamFunction;
    try {
        await session.prompt("offline boundary fixture");
        assert.equal(requests, 3); assert.equal(effects, 1);
        const results = session.messages.filter(m => m.role === "toolResult");
        assert.equal(results.length, 2); assert.equal(results[0].isError, true); assert.equal(results[1].isError, false);
        const last = notice(results[1].content.filter(p => p.type === "text").map(p => p.text).join("\n"));
        assert.equal(last.usedTotal, 2); assert.equal(last.usedCalls.edit, 2); assert.equal(last.remainingTotal, 0);
        assert.equal(readFileSync(join(root, "allowed"), "utf8"), "small");
    } finally { session.dispose(); }
}));


test("async wrapped body receives the admitted snapshot after caller mutates original args", () => fixture(async root => {
    const f = createToolCallBudgetExtension({ allowedTools: ["edit"], maxEditUtf8BytesExclusive: 20 });
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const [tool] = f.wrapTools([{ name: "edit", label: "edit", description: "deferred fixture", parameters: Type.Object({}),
        async execute(_id, params: any) {
            await gate;
            assert.equal(params.newText, "small");
            assert.notEqual(params, args);
            writeFileSync(params.path, params.newText);
            return { content: [], details: {} };
        } }]);
    const args = { path: join(root, "allowed"), oldText: "seed", newText: "small" };
    const pending = tool.execute("id", args, undefined, undefined, ctx);
    args.newText = "x".repeat(30000);
    release(); await pending;
    assert.equal(readFileSync(join(root, "allowed"), "utf8"), "small");
}));


test("configured edit cap fails closed if caller forgets final-execute wrapping", async () => {
    const f = createToolCallBudgetExtension({ allowedTools: ["edit"], maxEditUtf8BytesExclusive: 24576, visibleBudget: true });
    let hook: any;
    f({ on(name: string, fn: unknown) { if (name === "tool_call") hook = fn; } } as unknown as ExtensionAPI);
    const result = await hook({ toolName: "edit", input: { oldText: "x", newText: "small" } }, {});
    assert.equal(result.block, true);
    assert.match(result.reason, /EDIT_EXECUTION_GUARD_REQUIRED/);
    assert.equal(notice(result.reason).usedTotal, 0);
});
