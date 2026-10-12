import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile, readFile, readdir, access } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { createAssistantMessageEventStream, InMemoryCredentialStore, Type, type Model, type AssistantMessage } from "@earendil-works/pi-ai";
import { createAgentSession, createAgentSessionServices, createAgentSessionRuntime, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { ModelRequestAdmission, ModelRequestAdmissionError, validateModelRequestAdmission, type ModelRequestAdmissionPolicy } from "../src/harness/pi/model-request-admission.js";
import { installModelRequestAdmission, modelAdmissionSettings } from "../src/harness/pi/pi-model-runtime.js";

// Common wire cases: Python reads this exact JSON, never launches Node.
const ADMISSION_PARITY_JSON = String.raw`{
  "base":{"version":1,"lifetime":"runtime","guarantee":"local_reservation","inclusiveInputLimit":100,"outputLimit":100,"requestReserveInput":80,"requestReserveOutput":10},
  "cases":[
    {"caseID":"valid","accepted":true},
    {"caseID":"float-integer","field":"requestReserveInput","raw":"80.0","accepted":true},
    {"caseID":"float-version","field":"version","raw":"1.0","accepted":true},
    {"caseID":"null","raw":"null","accepted":false},
    {"caseID":"bool","raw":"true","accepted":false},
    {"caseID":"array","raw":"[]","accepted":false},
    {"caseID":"empty","raw":"{}","accepted":false},
    {"caseID":"extra","field":"extra","raw":"1","accepted":false},
    {"caseID":"version-bool","field":"version","raw":"true","accepted":false},
    {"caseID":"version-wrong","field":"version","raw":"2","accepted":false},
    {"caseID":"lifetime","field":"lifetime","raw":"\"session\"","accepted":false},
    {"caseID":"strict","field":"guarantee","raw":"\"strict_billing\"","accepted":false},
    {"caseID":"guarantee","field":"guarantee","raw":"\"unknown\"","accepted":false},
    {"caseID":"input-inconsistent","field":"requestReserveInput","raw":"101","accepted":false},
    {"caseID":"output-inconsistent","field":"requestReserveOutput","raw":"101","accepted":false}
  ],
  "integerFields":["inclusiveInputLimit","outputLimit","requestReserveInput","requestReserveOutput"],
  "invalidNumbers":[
    {"id":"null","raw":"null"},{"id":"bool","raw":"true"},
    {"id":"negative","raw":"-1"},{"id":"zero","raw":"0"},
    {"id":"fraction","raw":"0.5"},{"id":"unsafe","raw":"9007199254740992"},
    {"id":"nonfinite","raw":"1e999"},{"id":"string","raw":"\"80\""}
  ],
  "missingFields":["version","lifetime","guarantee","inclusiveInputLimit","outputLimit","requestReserveInput","requestReserveOutput"]
}`;
function parityCases(): { caseID: string; value: unknown; accepted: boolean }[] {
    const data = JSON.parse(ADMISSION_PARITY_JSON);
    return [
        ...data.cases.map((c: any) => ({ caseID: c.caseID, accepted: c.accepted,
            value: c.field ? { ...data.base, [c.field]: JSON.parse(c.raw) } : c.raw ? JSON.parse(c.raw) : { ...data.base } })),
        ...data.integerFields.flatMap((field: string) => data.invalidNumbers.map((n: any) => ({
            caseID: `${field}-${n.id}`, accepted: false, value: { ...data.base, [field]: JSON.parse(n.raw) } }))),
        ...data.missingFields.map((field: string) => {
            const value = { ...data.base }; delete value[field]; return { caseID: `missing-${field}`, value, accepted: false };
        }),
    ];
}
import { createToolCallBudgetExtension } from "../src/harness/pi/tool-call-budget.js";

const policy: ModelRequestAdmissionPolicy = { version: 1, lifetime: "runtime", guarantee: "local_reservation", inclusiveInputLimit: 100, outputLimit: 100, requestReserveInput: 80, requestReserveOutput: 10 };
const model: Model<"openai-completions"> = { id: "offline", name: "offline", api: "openai-completions", provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100 };
function message(reason: AssistantMessage["stopReason"] = "stop", input = 21): AssistantMessage {
    return { role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: 1,
        content: [{ type: "text", text: "INERT" }], usage: { input: input - 10, cacheRead: 7, cacheWrite: 3, output: 5, totalTokens: input + 5,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: reason };
}
type Mode = "normal" | "tool" | "error" | "abort" | "missing" | "throw" | "partial" | "duplicate" | "hold";
async function fixture(opts: { policy?: ModelRequestAdmissionPolicy | false; advisory?: boolean; mode?: Mode; warmEligible?: boolean } = {}) {
    const home = await mkdtemp(join(tmpdir(), "admission-sdk-"));
    const loadedSettings = SettingsManager.inMemory({ compaction: { enabled: false, keepRecentTokens: 32 }, retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 7 } }, cacheWarming: "streaming" });
    const settings = opts.policy === false ? loadedSettings : modelAdmissionSettings(loadedSettings);
    const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
    let fixtureModel = model;
    if (opts.warmEligible) {
        const eligible = ["claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "claude-sonnet-4-6", "claude-opus-4-6"]
            .map(id => runtime.getPhysicalModel("anthropic", id)).find(m => m?.promptCache);
        assert.ok(eligible?.promptCache, "use actual installed prompt-cache eligible catalog metadata");
        fixtureModel = { ...model, promptCache: eligible.promptCache };
    }
    runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [fixtureModel] });
    if (opts.warmEligible) assert.ok(runtime.getModel(model.provider, model.id)?.promptCache, "registered fixture remains eligible");
    const gate = opts.policy === false ? undefined : new ModelRequestAdmission(opts.policy ?? policy);
    const factory = createToolCallBudgetExtension({ allowedTools: ["diagnostic"], maxCalls: { diagnostic: 2 }, visibleBudget: true,
        ...(opts.advisory ? { modelUsageAwareness: { version: 1, lifetime: "runtime", inclusiveInputLimit: 100, outputLimit: 100, deliveryReserveInput: 10, deliveryReserveOutput: 5 } as const } : {}) });
    const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings, noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
        extensionFactories: [{ name: "budget", factory }] });
    await loader.reload();
    let requests = 0, effects = 0;
    const payloads: any[] = [], events: any[] = [];
    let held: ReturnType<typeof createAssistantMessageEventStream> | undefined;
    let mode = opts.mode ?? "normal";
    const host = await createAgentSessionRuntime(async target => {
        const services = await createAgentSessionServices({ cwd: home, agentDir: home, modelRuntime: runtime, settingsManager: settings });
        const result = await createAgentSession({ ...target, model: runtime.getModel(model.provider, model.id)!, modelRuntime: runtime, settingsManager: settings, resourceLoader: loader,
            tools: ["diagnostic"], customTools: [{ name: "diagnostic", label: "diagnostic", description: "inert", parameters: Type.Object({}), execute: async () => { effects++; return { content: [{ type: "text", text: "BODY" }], details: { preserved: true } }; } }], thinkingLevel: "off" });
        runtime.streamSimple = ((_m, context, options) => {
            assert.ok(++requests <= 12, "bounded offline invocation counter");
            // Actual SDK buildRequestOptions must forward the independent provider setting.
            assert.equal(options?.maxRetries, opts.policy === false ? 7 : 0);
            payloads.push(context);
            if (mode === "throw") throw Error("ORIGINAL_INERT_ERROR");
            const source = createAssistantMessageEventStream();
            if (mode === "hold") { held = source; return source; }
            if (mode === "missing") { source.end(); return source; }
            const msg = message(mode === "error" || mode === "partial" ? "error" : mode === "abort" ? "aborted" : mode === "tool" && requests === 1 ? "toolUse" : "stop");
            if (mode === "partial") { msg.usage.input = 9; (msg.usage as any).cacheRead = -1; (msg.usage as any).cacheWrite = undefined; msg.usage.output = 0; }
            if (msg.stopReason === "toolUse") msg.content = [{ type: "toolCall", id: "c1", name: "diagnostic", arguments: {} }];
            if (msg.stopReason === "error" || msg.stopReason === "aborted") { msg.errorMessage = "ORIGINAL_INERT_ERROR"; source.push({ type: "error", reason: msg.stopReason, error: msg }); }
            else { const end = { type: "done" as const, reason: msg.stopReason as "stop" | "toolUse", message: msg }; source.push(end); if (mode === "duplicate") source.push(end); }
            return source;
        }) as typeof runtime.streamSimple;
        if (gate) installModelRequestAdmission(result.session, gate); // SAME production installer, not a mocked gate
        result.session.subscribe(event => events.push(event));
        await result.session.bindExtensions({});
        return { ...result, services, diagnostics: [] };
    }, { cwd: home, agentDir: home, sessionManager: SessionManager.create(home, join(home, "sessions")) });
    return { home, host, gate, settings, loadedSettings, runtime, payloads, events, get session() { return host.session; }, get requests() { return requests; }, get effects() { return effects; }, setMode(value: Mode) { mode = value; }, finish() { held!.push({ type: "done", reason: "stop", message: message() }); },
        async close() { await host.dispose(); await rm(home, { recursive: true, force: true }); } };
}
function assistantError(f: Awaited<ReturnType<typeof fixture>>) {
    return (f.session.messages.filter((m: any) => m.role === "assistant").at(-1) as AssistantMessage)?.errorMessage;
}

// Prompt failure is SDK assistant error, not a successful answer; thrown callback errors
// remain errors. Check both canonical nonzero stopReason and bounded code.
async function deniedPrompt(f: Awaited<ReturnType<typeof fixture>>, code: string) {
    await f.session.prompt("inert denial");
    assert.match(assistantError(f) ?? "", new RegExp(code));
    assert.equal((f.session.messages.at(-1) as AssistantMessage).stopReason, "error");
    assert.ok(f.session.isIdle);
}

test("A01 absent/advisory retains actual SDK followups, payloads, tool events", { timeout: 10000 }, async () => {
    const plain = await fixture({ policy: false, mode: "tool" });
    const advisory = await fixture({ policy: false, mode: "tool", advisory: true });
    try {
        for (const f of [plain, advisory]) { await f.session.prompt("inert"); assert.equal(f.requests, 2); assert.equal(f.effects, 1); assert.ok(f.session.isIdle); assert.ok(f.events.some(e => e.type === "tool_execution_end" && e.result.details.preserved)); }
        assert.deepEqual(plain.events.map(e => e.type), advisory.events.map(e => e.type));
        assert.match(JSON.stringify(advisory.payloads[1]), /ROSCLAW_MODEL_USAGE_JSON/);
        assert.doesNotMatch(JSON.stringify(plain.payloads), /ROSCLAW_MODEL_USAGE_JSON/);
        assert.match(JSON.stringify(plain.payloads[1]), /ROSCLAW_TOOL_POLICY_JSON/);
    } finally { await plain.close(); await advisory.close(); }
});

test("A02 pre-dispatch exhaustion and unsupported strict guarantee invoke nothing", { timeout: 10000 }, async () => {
    assert.throws(() => new ModelRequestAdmission({ ...policy, guarantee: "strict_billing" } as any), e => e instanceof ModelRequestAdmissionError && e.code === "MODEL_REQUEST_BILLING_GUARANTEE_UNSUPPORTED");
    const f = await fixture(); try {
        // Previously settled known spend leaves insufficient allowance for reservation.
        await f.session.prompt("one"); assert.equal(f.requests, 1);
        await deniedPrompt(f, "MODEL_REQUEST_ALLOWANCE_EXHAUSTED"); assert.equal(f.requests, 1);
        assert.equal(f.gate!.snapshot().pendingRequests, 0);
    } finally { await f.close(); }
});

test("A03 real immediate tool followup denied before underlying call", { timeout: 10000 }, async () => {
    const f = await fixture({ mode: "tool" }); try {
        await f.session.prompt("tool"); assert.equal(f.requests, 1); assert.equal(f.effects, 1);
        assert.match(assistantError(f) ?? "", /MODEL_REQUEST_ALLOWANCE_EXHAUSTED/);
        assert.deepEqual(f.gate!.snapshot(), { knownInclusiveInput: 21, knownOutput: 5, unknownUsageRequests: 0, dispatchedRequests: 1, settledRequests: 1, pendingRequests: 0, reservedInput: 0, reservedOutput: 0, lastDenial: "MODEL_REQUEST_ALLOWANCE_EXHAUSTED", providerCostBoundVerified: false });
        assert.ok(f.session.isIdle);
    } finally { await f.close(); }
});

test("A04 cached inclusive, duplicate ends, equal-valued requests, UNKNOWN failures", { timeout: 10000 }, async () => {
    const f = await fixture({ policy: { ...policy, inclusiveInputLimit: 1000 }, mode: "duplicate" }); try {
        await f.session.prompt("one"); await f.session.prompt("two");
        assert.equal(f.requests, 2); assert.equal(f.gate!.snapshot().knownInclusiveInput, 42); assert.equal(f.gate!.snapshot().settledRequests, 2);
    } finally { await f.close(); }
    for (const mode of ["error", "abort", "partial", "missing", "throw"] as const) {
        const f = await fixture({ mode, policy: { ...policy, inclusiveInputLimit: 1000 } }); try {
            await f.session.prompt("failed");
            assert.equal(f.gate!.snapshot().unknownUsageRequests, 1);
            assert.equal(f.gate!.snapshot().knownInclusiveInput, mode === "partial" ? 9 : mode === "missing" || mode === "throw" ? 0 : 21);
            await deniedPrompt(f, "MODEL_REQUEST_USAGE_UNKNOWN"); assert.equal(f.requests, 1);
            assert.equal(f.gate!.snapshot().unknownUsageRequests, 1, "refusal must not count fake SDK zero usage");
        } finally { await f.close(); }
    }
});

test("A05 actual SDK main + summary concurrent reservation; replacement retains ledger", { timeout: 10000 }, async () => {
    const f = await fixture({ mode: "hold" }); try {
        const pending = f.session.prompt("hold");
        for (let i = 0; f.requests === 0 && i < 100; i++) await new Promise(r => setTimeout(r, 5));
        assert.equal(f.requests, 1); assert.equal(f.gate!.snapshot().pendingRequests, 1);
        // Real public compact route with canonical persisted history; not extracted hooks.
        for (let i = 0; i < 6; i++) { f.session.sessionManager.appendMessage({ role: "user", content: "history ".repeat(200), timestamp: i }); f.session.sessionManager.appendMessage(message()); }
        await assert.rejects(f.session.compact(), /MODEL_REQUEST_SUMMARY_UNSUPPORTED/);
        assert.equal(f.requests, 1); assert.equal(f.gate!.snapshot().pendingRequests, 1);
        f.finish(); await pending; assert.ok(f.session.isIdle);
        f.setMode("normal");
        assert.equal((await f.host.newSession()).cancelled, false);
        await deniedPrompt(f, "MODEL_REQUEST_ALLOWANCE_EXHAUSTED"); assert.equal(f.requests, 1);
    } finally { await f.close(); }
});

test("A06 never-dispatched abort releases; local recovery/save/close, UNKNOWN never resets", { timeout: 10000 }, async () => {
    const f = await fixture(); try {
        let calls = 0; const controller = new AbortController();
        const pending = f.gate!.wrap(() => { calls++; throw Error("unreachable"); })(model, {} as never, { signal: controller.signal });
        controller.abort(); await assert.rejects(Promise.resolve(pending), /CANCELLED_BEFORE_DISPATCH/);
        assert.equal(calls, 0); assert.equal(f.gate!.snapshot().pendingRequests, 0); assert.equal(f.gate!.snapshot().unknownUsageRequests, 0);
        await f.session.prompt("known"); await deniedPrompt(f, "ALLOWANCE_EXHAUSTED");
        f.gate!.reconfigure({ ...policy, inclusiveInputLimit: 200 });
        await f.session.prompt("supported local recovery"); assert.equal(f.requests, 2); assert.equal(f.gate!.snapshot().knownInclusiveInput, 42);
        await access(f.session.sessionFile!); assert.ok(f.session.isIdle);
        f.setMode("error"); await f.session.prompt("failed");
        assert.match(assistantError(f) ?? "", /ORIGINAL_INERT_ERROR/);
        f.gate!.reconfigure({ ...policy, inclusiveInputLimit: 300 });
        await deniedPrompt(f, "USAGE_UNKNOWN"); assert.equal(f.requests, 3); assert.equal(f.gate!.snapshot().unknownUsageRequests, 1);
    } finally { await f.close(); }
    await assert.rejects(access(f.home));
});

test("A07 Node runtime + CLI strict shape pre-home rejection", { timeout: 20000 }, async () => {
    const home = await mkdtemp(join(tmpdir(), "admission-parser-")); try {
        validateModelRequestAdmission(policy);
        const results = parityCases().map(({ caseID, value, accepted }) => {
            let actual = true;
            try { validateModelRequestAdmission(value); } catch { actual = false; }
            assert.equal(actual, accepted, caseID);
            if (accepted) createToolCallBudgetExtension({ allowedTools: [], modelRequestAdmission: value as any });
            return { caseID, result: actual ? "ACCEPT" : "REJECT" };
        });
        console.log("ADMISSION_PARITY_Node=" + JSON.stringify(results));
        for (const { value } of parityCases().filter(c => !c.accepted)) {
            assert.throws(() => validateModelRequestAdmission(value));
            assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], modelRequestAdmission: value as any }));
            const file = join(home, "policy.json"); await writeFile(file, JSON.stringify({ allowedTools: [], modelRequestAdmission: value }));
            const result = spawnSync(process.execPath, [resolve("dist/src/main.js"), "--tool-call-policy", file], { encoding: "utf8", timeout: 3000, env: { ...process.env, ROSCLAW_HOME: join(home, "unused") } });
            assert.notEqual(result.status, 0); assert.match(result.stderr, /INVALID_TOOL_CALL_POLICY/); await assert.rejects(access(join(home, "unused")));
        }
    } finally { await rm(home, { recursive: true, force: true }); }
});

test("A08 source-bound Codex capability boundary and real SDK refusal cleanup", { timeout: 10000 }, async () => {
    const sdk = await readFile(new URL("./core/sdk.js", import.meta.resolve("@earendil-works/pi-coding-agent")), "utf8");
    assert.match(sdk, /getProviderRetrySettings/);
    const apiSources = new URL("./api/", import.meta.resolve("@earendil-works/pi-ai"));
    const files = await readdir(apiSources, { recursive: true });
    const codexFiles = files.filter(f => f.includes("openai-codex-responses") && f.endsWith(".js"));
    assert.ok(codexFiles.length > 0, "installed Codex serializer source must exist");
    const source = (await Promise.all(codexFiles.map(f => readFile(new URL(f, apiSources), "utf8")))).join("\n");
    assert.match(source, /function buildRequestBody/);
    assert.doesNotMatch(source, /max_output_tokens\s*[:=]/, "installed Codex serializer has no output cap");
    assert.throws(() => validateModelRequestAdmission({ ...policy, guarantee: "strict_billing" }), /BILLING_GUARANTEE_UNSUPPORTED/);
    const f = await fixture({ mode: "partial" }); const home = f.home;
    try {
        await f.session.prompt("partial failure"); await deniedPrompt(f, "USAGE_UNKNOWN");
        assert.equal(f.requests, 1); assert.equal(f.gate!.snapshot().knownInclusiveInput, 9);
        assert.equal(f.gate!.snapshot().unknownUsageRequests, 1); assert.ok(f.session.isIdle); await access(f.session.sessionFile!);
        assert.ok(f.events.some(e => e.type === "agent_end"));
    } finally { await f.close(); }
    await assert.rejects(access(home));
    const eligible = await fixture({ warmEligible: true });
    try {
        assert.equal(eligible.settings.getSettings().cacheWarming, "off");
        assert.equal(eligible.settings.getProviderRetrySettings().maxRetries, 0);
        assert.equal(eligible.loadedSettings.getProviderRetrySettings().maxRetries, 7, "loaded settings stay unchanged");
        assert.equal(eligible.loadedSettings.getSettings().cacheWarming, "streaming");
        await eligible.session.prompt("eligible inert cache context");
        await new Promise(r => setTimeout(r, 100));
        assert.equal(eligible.requests, 1, "zero cache-warmer bypass for otherwise eligible model");
        assert.equal(eligible.gate!.snapshot().dispatchedRequests, 1);
    } finally { await eligible.close(); }
});
