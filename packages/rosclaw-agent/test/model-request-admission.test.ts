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
async function fixture(opts: { policy?: ModelRequestAdmissionPolicy | false; advisory?: boolean; mode?: Mode; warmEligible?: boolean;
    linkAdmission?: boolean; savedEdits?: boolean;
    awareness?: { version: 1; lifetime: "runtime"; inclusiveInputLimit: number; outputLimit: number; deliveryReserveInput: number; deliveryReserveOutput: number } } = {}) {
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
        ...(opts.advisory ? { modelUsageAwareness: opts.awareness ?? { version: 1, lifetime: "runtime", inclusiveInputLimit: 100, outputLimit: 100, deliveryReserveInput: 10, deliveryReserveOutput: 5 } as const } : {}) }, home, opts.linkAdmission === false ? undefined : gate);
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
            tools: ["diagnostic"], customTools: [{ name: "diagnostic", label: "diagnostic", description: "inert", parameters: Type.Object({}), execute: async () => { effects++;
                if (opts.savedEdits) {
                    await writeFile(join(home, "partial-producer.txt"), "ACKNOWLEDGED_PRODUCER");
                    await writeFile(join(home, "partial-test.txt"), "ACKNOWLEDGED_TEST");
                }
                return { content: [{ type: "text", text: "BODY" }], details: { preserved: true } }; } }], thinkingLevel: "off" });
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

// SOURCE-only feasibility regressions: actual SDK lifecycle and production
// installer, synthetic streams only. These cases have NOT been executed here.
// No transport receipt, external usage reconciler or new policy is simulated.
test("A09 public in-flight abort settles SDK but v1 UNKNOWN still blocks, including repeated cancel", { timeout: 10000 }, async () => {
    for (const partialInput of [0, 9]) {
        const f = await fixture({ policy: { ...policy, inclusiveInputLimit: 1000 } });
        let callbacks = 0, admissions = 0, abortEvents = 0;
        let source: ReturnType<typeof createAssistantMessageEventStream> | undefined;
        let running: Promise<void> | undefined;
        try {
            await f.session.prompt("known spend before cancellation");
            assert.equal(f.requests, 1);
            assert.equal(f.gate!.snapshot().knownInclusiveInput, 21);
            assert.equal(f.gate!.snapshot().knownOutput, 5);
            f.runtime.streamSimple = ((_m, _context, options) => {
                callbacks++;
                const stream = createAssistantMessageEventStream();
                const aborted = () => {
                    const msg = message("aborted");
                    msg.content = [];
                    // Placeholder zeros are unavailable usage, not measured zero.
                    msg.usage = { input: partialInput, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: partialInput,
                        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
                    msg.errorMessage = "SYNTHETIC_ABORT_USAGE_UNAVAILABLE";
                    stream.push({ type: "error", reason: "aborted", error: msg });
                    stream.end();
                };
                assert.ok(options?.signal, "use actual SDK cancellation signal");
                if (options.signal.aborted) { aborted(); return stream; }
                admissions++;
                // An unexpected followup must finish so assertions fail rather
                // than hanging on a second held stream.
                if (admissions !== 1) { stream.push({ type: "done", reason: "stop", message: message() }); return stream; }
                source = stream;
                stream.push({ type: "start", partial: message() });
                options.signal.addEventListener("abort", () => { abortEvents++; aborted(); }, { once: true });
                return stream;
            }) as typeof f.runtime.streamSimple;
            running = f.session.prompt("start then cancel in flight");
            for (let i = 0; admissions === 0 && i < 100; i++) await new Promise(r => setTimeout(r, 5));
            assert.equal(admissions, 1);
            assert.equal(f.gate!.snapshot().pendingRequests, 1);
            assert.ok(f.session.isStreaming);
            await Promise.all([f.session.abort(), f.session.abort(), running]);
            assert.equal(abortEvents, 1, "repeated public cancellation owns one stream");
            assert.ok(f.session.isIdle);
            assert.equal((f.session.messages.at(-1) as AssistantMessage).stopReason, "aborted");
            assert.match(assistantError(f) ?? "", /SYNTHETIC_ABORT_USAGE_UNAVAILABLE/);
            assert.ok(f.events.some(e => e.type === "agent_start"));
            assert.ok(f.events.some(e => e.type === "agent_end"));
            const settled = f.gate!.snapshot();
            assert.equal(settled.knownInclusiveInput, 21 + partialInput);
            assert.equal(settled.knownOutput, 5);
            assert.equal(settled.dispatchedRequests, 2);
            assert.equal(settled.settledRequests, 2);
            assert.equal(settled.pendingRequests, 0);
            assert.equal(settled.unknownUsageRequests, 1);
            assert.equal(settled.reservedInput, 0, "v1 releases pending slots but blocks on UNKNOWN; it is not a retained-reservation v2");
            // A later terminal cannot replace this stream's first abort terminal.
            // This is not a provider usage reconciliation endpoint.
            const late = message("toolUse");
            late.content = [{ type: "toolCall", id: "late-after-cancel", name: "diagnostic", arguments: {} }];
            source!.push({ type: "done", reason: "toolUse", message: late });
            source!.end(late);
            assert.equal((await source!.result()).stopReason, "aborted");
            assert.deepEqual(f.gate!.snapshot(), settled, "no duplicate settlement or late refund");
            await deniedPrompt(f, "MODEL_REQUEST_USAGE_UNKNOWN");
            f.gate!.reconfigure({ ...policy, inclusiveInputLimit: 2000 });
            await deniedPrompt(f, "MODEL_REQUEST_USAGE_UNKNOWN");
            await f.session.abort();
            assert.equal(callbacks, 1);
            assert.equal(admissions, 1, "no postcancel synthetic admission");
            assert.equal(f.effects, 0, "no postcancel tool execution");
            assert.equal(f.gate!.snapshot().unknownUsageRequests, 1);
            assert.equal(f.gate!.snapshot().knownInclusiveInput, 21 + partialInput);
            assert.equal(f.gate!.snapshot().settledRequests, 2);
            assert.equal(f.gate!.snapshot().providerCostBoundVerified, false);
        } finally {
            await f.session.abort();
            await running;
            await f.close();
        }
    }
});

test("A10 actual SDK startup abort never reaches v1 underlying stream or creates UNKNOWN", { timeout: 10000 }, async () => {
    const home = await mkdtemp(join(tmpdir(), "admission-start-cancel-"));
    let release!: () => void;
    let entered!: () => void;
    const holding = new Promise<void>(resolve => { release = resolve; });
    const startup = new Promise<void>(resolve => { entered = resolve; });
    const gate = new ModelRequestAdmission(policy);
    let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
    let running: Promise<void> | undefined;
    let callbacks = 0;
    try {
        const settings = modelAdmissionSettings(SettingsManager.inMemory({ compaction: { enabled: false } }));
        const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
            modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
        runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [model] });
        const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
            noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
            extensionFactories: [{ name: "held-startup", factory: pi => {
                pi.on("before_agent_start", async () => { entered(); await holding; });
            } }] });
        await loader.reload();
        assert.deepEqual(loader.getExtensions().errors, []);
        ({ session } = await createAgentSession({ cwd: home, agentDir: home, modelRuntime: runtime,
            model: runtime.getModel(model.provider, model.id)!, settingsManager: settings, resourceLoader: loader,
            sessionManager: SessionManager.inMemory(home), tools: [], thinkingLevel: "off" }));
        session.agent.streamFunction = () => {
            callbacks++;
            const stream = createAssistantMessageEventStream();
            stream.push({ type: "done", reason: "stop", message: message() });
            return stream;
        };
        installModelRequestAdmission(session, gate);
        await session.bindExtensions({});
        running = session.prompt("cancel while genuine before_agent_start is held");
        await startup;
        assert.ok(session.isStreaming);
        const cancelling = session.abort();
        release();
        await Promise.all([running, cancelling]);
        assert.ok(session.isIdle);
        assert.equal(callbacks, 0, "the production wrapper rejects before synthetic underlying invocation");
        assert.equal(gate.snapshot().dispatchedRequests, 0);
        assert.equal(gate.snapshot().pendingRequests, 0);
        assert.equal(gate.snapshot().unknownUsageRequests, 0);
        assert.equal(gate.snapshot().knownInclusiveInput, 0);
        assert.equal(gate.snapshot().knownOutput, 0);
        // SDK may stop before reaching the wrapper; neither route incurs a reservation.
        assert.ok([undefined, "MODEL_REQUEST_CANCELLED_BEFORE_DISPATCH"].includes(gate.snapshot().lastDenial));
    } finally {
        release();
        await session?.abort();
        await running;
        session?.dispose();
        await rm(home, { recursive: true, force: true });
    }
});

// Source50: warning opportunity, not escrow, cap, or guaranteed FINAL.
const deliveryPolicy: ModelRequestAdmissionPolicy = { version: 1, lifetime: "runtime", guarantee: "local_reservation",
    inclusiveInputLimit: 4000000, outputLimit: 24576, requestReserveInput: 100000, requestReserveOutput: 2048 };
const deliveryAwareness = { version: 1 as const, lifetime: "runtime" as const, inclusiveInputLimit: 4000000,
    outputLimit: 24576, deliveryReserveInput: 100000, deliveryReserveOutput: 4096 };
function deliveryNotes(context: any) {
    return context.messages.filter((m: any) => m.role === "system").flatMap((m: any) => {
        const text = typeof m.content === "string" ? m.content : m.content.map((p: any) => p.text ?? "").join("\n");
        return text.split("\n").filter((line: string) => line.startsWith("ROSCLAW_MODEL_USAGE_JSON:"))
            .map((line: string) => JSON.parse(line.slice("ROSCLAW_MODEL_USAGE_JSON:".length)));
    });
}
async function deliveryCounterexample(linked: boolean) {
    const f = await fixture({ policy: deliveryPolicy, advisory: true, awareness: deliveryAwareness,
        linkAdmission: linked, savedEdits: true });
    let calls = 0;
    const notices: any[] = [], optionsSeen: any[] = [];
    f.runtime.streamSimple = ((_m, context, options) => {
        assert.ok(++calls <= 2, "no retry/extra provider request on continuation refusal");
        notices.push(deliveryNotes(context).at(-1)); optionsSeen.push(options);
        const msg = message(calls === 1 ? "stop" : "toolUse");
        msg.usage.output = calls === 1 ? 19345 : 4340;
        msg.usage.totalTokens = msg.usage.input + msg.usage.cacheRead + msg.usage.cacheWrite + msg.usage.output;
        msg.content = calls === 1 ? [{ type: "text", text: "INERT_PRIOR_RESPONSE" }]
            : [{ type: "toolCall", id: "saved-edit", name: "diagnostic", arguments: {} }];
        const stream = createAssistantMessageEventStream();
        stream.push({ type: "done", reason: msg.stopReason as "stop" | "toolUse", message: msg });
        return stream;
    }) as typeof f.runtime.streamSimple;
    return { f, notices, optionsSeen, get calls() { return calls; } };
}

test("F01 exact Source47 counterexample: earlier notice, completed effects, missing FINAL and original refusal", { timeout: 15000 }, async () => {
    for (const linked of [false, true]) {
        const c = await deliveryCounterexample(linked), f = c.f;
        try {
            await f.session.prompt("prior known spend");
            assert.equal(f.gate!.snapshot().knownOutput, 19345);
            assert.equal(f.gate!.deliveryContract().availableOutput, 5231);
            await f.session.prompt("tool output followed immediately by FINAL attempt");
            assert.equal(c.calls, 2); assert.equal(f.effects, 1);
            assert.equal(c.notices[1].remainingOutputUpperBound, 5231);
            assert.equal(c.notices[1].reserveState, linked ? "finish_delivery" : "normal");
            assert.equal(c.notices[1].deliveryNoticeContract, "advisory_only_no_final_guarantee");
            if (linked) {
                assert.equal(c.notices[1].admission.availableOutput, 5231);
                assert.equal(c.notices[1].admission.nextRequestAdmissible, true);
                assert.equal(c.notices[1].admission.finalDeliveryGuaranteed, false);
                assert.equal(c.notices[1].admission.responseOutputLimit, "not_enforced");
            } else assert.equal(c.notices[1].admission, undefined);
            assert.equal(f.gate!.snapshot().knownOutput, 23685);
            assert.equal(f.gate!.deliveryContract().availableOutput, 891);
            assert.equal(f.gate!.deliveryContract().nextRequestAdmissible, false);
            assert.equal(f.gate!.snapshot().unknownUsageRequests, 0, "SDK refusal is not a dispatched unknown request");
            assert.equal(f.gate!.snapshot().settledRequests, 2);
            assert.equal((f.session.messages.at(-1) as AssistantMessage).stopReason, "error");
            assert.match(assistantError(f) ?? "", /MODEL_REQUEST_ALLOWANCE_EXHAUSTED/);
            assert.equal(await readFile(join(f.home, "partial-producer.txt"), "utf8"), "ACKNOWLEDGED_PRODUCER");
            assert.equal(await readFile(join(f.home, "partial-test.txt"), "utf8"), "ACKNOWLEDGED_TEST");
            const toolEnd = f.events.find(e => e.type === "tool_execution_end");
            assert.equal(toolEnd.result.details.preserved, true);
            assert.match(JSON.stringify(toolEnd.result.content), /BODY/);
            assert.ok(f.events.indexOf(toolEnd) < f.events.findIndex(e => e.type === "message_end" && e.message.errorMessage?.includes("MODEL_REQUEST_ALLOWANCE_EXHAUSTED")));
            assert.equal(f.session.messages.some((m: any) => m.role === "assistant" && m.content.some((p: any) => p.type === "text" && p.text.includes('"status":"SUCCESS"'))), false);
            const { classifyAssistantFailure, ProviderErrorGate } = await import("../src/native/model-errors.js");
            const { mapPiEvent } = await import("../src/harness/pi/pi-session-adapter.js");
            const last = f.session.messages.at(-1) as AssistantMessage;
            const classified = classifyAssistantFailure(last)!;
            assert.equal(classified.code, "MODEL_REQUEST_ALLOWANCE_EXHAUSTED");
            assert.equal(classified.taskRecoverable, false);
            const card = new ProviderErrorGate().onError(classified, { hasActiveTask: true });
            assert.match(card.cardText, /未完成的交付仍未完成/);
            assert.doesNotMatch(card.cardText, /自动重试中|任务保持可恢复|\/model/);
            const mapped = mapPiEvent({ type: "message_end", message: last });
            assert.equal(mapped?.type, "turn.failed");
            const ledger = f.gate!.snapshot();
            assert.equal((await f.host.newSession()).cancelled, false);
            await deniedPrompt(f, "MODEL_REQUEST_ALLOWANCE_EXHAUSTED");
            assert.equal(c.calls, 2);
            assert.equal(f.gate!.snapshot().knownOutput, ledger.knownOutput);
            assert.equal(f.gate!.snapshot().dispatchedRequests, ledger.dispatchedRequests);
            assert.equal(f.gate!.snapshot().providerCostBoundVerified, false);
            assert.ok(f.session.isIdle);
        } finally { await f.close(); }
    }
});

test("F02 exact warning and reservation boundaries, no arbitrary response clamp", { timeout: 15000 }, async () => {
    for (const [remaining, expected] of [[6145, "normal"], [6144, "finish_delivery"], [2048, "finish_delivery"]] as const) {
        const f = await fixture({ policy: deliveryPolicy, advisory: true, awareness: deliveryAwareness });
        let calls = 0;
        const notices: any[] = [];
        try {
            f.runtime.streamSimple = ((_m, context) => {
                assert.ok(++calls <= 2);
                notices.push(deliveryNotes(context).at(-1));
                const msg = message(); msg.usage.output = calls === 1 ? 24576 - remaining : remaining - 2047;
                const stream = createAssistantMessageEventStream(); stream.push({ type: "done", reason: "stop", message: msg }); return stream;
            }) as typeof f.runtime.streamSimple;
            await f.session.prompt("settle prior response");
            await f.session.prompt("exact boundary permitted");
            assert.equal(calls, 2);
            assert.equal(notices[1].reserveState, expected);
            assert.equal(notices[1].admission.availableOutput, remaining);
            assert.equal(notices[1].admission.nextRequestAdmissible, true);
            assert.equal(f.gate!.deliveryContract().availableOutput, 2047);
            await deniedPrompt(f, "ALLOWANCE_EXHAUSTED");
            assert.equal(calls, 2);
            assert.equal(f.gate!.snapshot().knownOutput, 22529);
        } finally { await f.close(); }
    }
});

test("F03 simultaneous local reservations and provider overspend are disclosed, never refunded", { timeout: 10000 }, async () => {
    const gate = new ModelRequestAdmission({ ...deliveryPolicy, outputLimit: 4096 });
    const sources: ReturnType<typeof createAssistantMessageEventStream>[] = [];
    const wrapped = gate.wrap(() => { assert.ok(sources.length < 2); const s = createAssistantMessageEventStream(); sources.push(s); return s; });
    const first = await wrapped(model, {} as never), second = await wrapped(model, {} as never);
    assert.equal(gate.deliveryContract().availableOutput, 0);
    assert.equal(gate.deliveryContract().pendingRequests, 2);
    assert.equal(gate.deliveryContract().nextRequestAdmissible, false);
    await assert.rejects(Promise.resolve(wrapped(model, {} as never)), /ALLOWANCE_EXHAUSTED/);
    for (const [index, output] of [[1, 3000], [0, 1200]]) {
        const msg = message(); msg.usage.output = output;
        sources[index].push({ type: "done", reason: "stop", message: msg });
    }
    await Promise.all([first.result(), second.result()]);
    assert.equal(gate.snapshot().knownOutput, 4200, "real reported response may exceed a local slot/total");
    assert.equal(gate.snapshot().settledRequests, 2);
    assert.equal(gate.deliveryContract().availableOutput, 0);
    assert.equal(gate.deliveryContract().providerCostBoundVerified, false);
    assert.equal(gate.deliveryContract().finalDeliveryProtection, "none");
    await assert.rejects(Promise.resolve(wrapped(model, {} as never)), /ALLOWANCE_EXHAUSTED/);
    assert.equal(sources.length, 2);
});

test("F04 UNKNOWN contract remains fail-closed across replacement; original abort/duplicate tests retained", { timeout: 15000 }, async () => {
    for (const mode of ["error", "abort", "partial", "missing", "throw", "duplicate"] as const) {
        const f = await fixture({ mode, policy: deliveryPolicy });
        try {
            await f.session.prompt("one");
            const unknown = mode !== "duplicate";
            assert.equal(f.gate!.deliveryContract().unknownUsageRequests, unknown ? 1 : 0);
            assert.equal(f.gate!.deliveryContract().nextRequestAdmissible, !unknown);
            assert.equal(f.gate!.snapshot().settledRequests, 1);
            assert.equal(f.gate!.snapshot().pendingRequests, 0);
            assert.equal((await f.host.newSession()).cancelled, false);
            if (unknown) {
                await deniedPrompt(f, "USAGE_UNKNOWN");
                assert.equal(f.requests, 1);
                assert.equal(f.gate!.deliveryContract().unknownUsageRequests, 1);
            } else {
                await f.session.prompt("separate equal valued request");
                assert.equal(f.gate!.snapshot().settledRequests, 2);
                assert.equal(f.gate!.snapshot().knownInclusiveInput, 42);
            }
            assert.equal(f.gate!.deliveryContract().finalDeliveryGuaranteed, false);
        } finally { await f.close(); }
    }
});

// Controlled HTTP 400 responses inspect ORIGINAL adapter request bytes and
// exercise actual error terminals. They are not token/billing measurements.
async function originalWireFixture(selected: Model<any>, apiKey: string, maxTokens = 23) {
    const home = await mkdtemp(join(tmpdir(), "delivery-wire-"));
    const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
        modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
    runtime.registerProvider(selected.provider, { api: selected.api, baseUrl: selected.baseUrl,
        apiKey, models: [selected] });
    await runtime.refresh({ allowNetwork: false });
    const bodies: any[] = [];
    const fetch: typeof globalThis.fetch = async (input, init) => {
        assert.equal(bodies.length, 0, "no hidden transport retry");
        const request = new Request(input, init);
        let bytes = new Uint8Array(await request.arrayBuffer());
        if (request.headers.get("content-encoding") === "zstd") {
            const zlib = await import("node:zlib") as unknown as { zstdDecompressSync(b: Uint8Array): Uint8Array };
            bytes = new Uint8Array(zlib.zstdDecompressSync(bytes));
        }
        bodies.push(JSON.parse(new TextDecoder().decode(bytes)));
        return new Response(JSON.stringify({ error: { message: "INERT_WIRE_REFUSAL", type: "invalid_request_error" } }),
            { status: 400, headers: { "content-type": "application/json" } });
    };
    return { home, runtime, bodies, selected, options: { apiKey, fetch, transport: "sse" as const, maxRetries: 0, maxTokens },
        async close() { await rm(home, { recursive: true, force: true }); } };
}
function syntheticCodexCredential() {
    // Deliberately synthetic credential required by ORIGINAL account-id parser.
    return "inert." + Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "offline" } })).toString("base64url") + ".inert";
}
async function pinnedKimiModel(): Promise<Model<any>> {
    const path = new URL("./providers/data/kimi-coding.json", import.meta.resolve("@earendil-works/pi-ai"));
    const raw = await readFile(path);
    const { createHash } = await import("node:crypto");
    assert.equal(createHash("sha256").update(raw).digest("hex"), "728d5ddb63ef52130552022d5ae2fd138fdc131ff854ac8c3b10ccaad355ab39");
    const selected = JSON.parse(raw.toString("utf8"))["openai-completions"]["chat:k3"];
    assert.equal(selected.api, "openai-completions");
    assert.equal(selected.compat.maxTokensField, "max_tokens");
    return selected;
}

test("F05 original ModelRuntime and actual wire bodies distinguish Codex, Responses exceptions, and pinned Kimi", { timeout: 20000 }, async () => {
    const kimi = await pinnedKimiModel();
    const ordinary: Model<any> = { ...model, provider: "openai", api: "openai-responses", baseUrl: "https://api.openai.com/v1" };
    const cases = [
        { selected: { ...ordinary, provider: "openai-codex", api: "openai-codex-responses", baseUrl: "https://chatgpt.com/backend-api" }, key: syntheticCodexCredential(), max: 23, field: null, value: undefined, capability: "omitted" },
        { selected: ordinary, key: "sk-offline-inert", max: 23, field: "max_output_tokens", value: 23, capability: "conditional" },
        { selected: ordinary, key: "sk-offline-inert", max: 7, field: "max_output_tokens", value: 16, capability: "conditional" },
        { selected: { ...ordinary, compat: { supportsMaxOutputTokens: false } }, key: "sk-offline-inert", max: 23, field: null, value: undefined, capability: "conditional" },
        { selected: ordinary, key: syntheticCodexCredential(), max: 23, field: null, value: undefined, capability: "conditional" },
        { selected: kimi, key: "offline-inert", max: 23, field: "max_tokens", value: 23, capability: "compat_field" },
    ];
    const { modelRequestOutputCapability } = await import("../src/harness/pi/model-request-admission.js");
    for (const c of cases) {
        const f = await originalWireFixture(c.selected, c.key, c.max);
        const gate = new ModelRequestAdmission(deliveryPolicy);
        try {
            // No runtime/adapter monkeypatch. Original ModelRuntime resolves auth,
            // normalizes transcript, composes provider and invokes original API.
            const wrapped = gate.wrap(f.runtime.streamSimple.bind(f.runtime));
            const stream = await wrapped(f.selected, { messages: [{ role: "user", content: "inert", timestamp: 1 }] } as never, f.options);
            const result = await stream.result();
            assert.equal(result.stopReason, "error");
            assert.match(result.errorMessage ?? "", /INERT_WIRE_REFUSAL/);
            assert.equal(f.bodies.length, 1);
            const body = f.bodies[0];
            assert.equal(body.model, c.selected.id);
            for (const field of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) {
                assert.equal(body[field], field === c.field ? c.value : undefined, c.selected.api + ":" + field);
            }
            assert.equal(modelRequestOutputCapability(c.selected).genericMaxTokens, c.capability);
            assert.equal(gate.deliveryContract().lastRoute?.api, c.selected.api);
            assert.equal(gate.deliveryContract().responseOutputLimit, "not_enforced");
            assert.equal(gate.snapshot().dispatchedRequests, 1);
            assert.equal(gate.snapshot().unknownUsageRequests, 1, "adapter's default zero is not a usage receipt");
            assert.equal(gate.deliveryContract().nextRequestAdmissible, false);
            assert.equal(gate.deliveryContract().providerCostBoundVerified, false);
            await assert.rejects(Promise.resolve(wrapped(f.selected, {} as never, f.options)), /USAGE_UNKNOWN/);
            assert.equal(f.bodies.length, 1);
            assert.throws(() => validateModelRequestAdmission({ ...deliveryPolicy, guarantee: "strict_billing" }), /BILLING_GUARANTEE_UNSUPPORTED/);
        } finally { await f.close(); }
    }
});

test("F06 original AgentSession -> production installer -> original ModelRuntime -> Kimi adapter, absent-option control", { timeout: 20000 }, async () => {
    const kimi = await pinnedKimiModel();
    for (const enabled of [false, true]) {
        const f = await originalWireFixture(kimi, "offline-inert");
        let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
        try {
            const loaded = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 0 } }, cacheWarming: "off" });
            const settings = enabled ? modelAdmissionSettings(loaded) : loaded;
            const loader = new DefaultResourceLoader({ cwd: f.home, agentDir: f.home, settingsManager: settings,
                noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true });
            await loader.reload();
            ({ session } = await createAgentSession({ cwd: f.home, agentDir: f.home, modelRuntime: f.runtime,
                model: f.runtime.getModel(kimi.provider, kimi.id)!, settingsManager: settings, resourceLoader: loader,
                sessionManager: SessionManager.inMemory(f.home), tools: [], thinkingLevel: "low" }));
            // Controlled transport seam only; original runtime and adapters remain
            // intact. Capture cancellation signal forwarded by the real SDK.
            const signals: AbortSignal[] = [];
            session.agent.streamFunction = (m, context, options) => {
                assert.ok(options?.signal); signals.push(options.signal);
                return f.runtime.streamSimple(m, context, { ...options, ...f.options, signal: options.signal });
            };
            const gate = enabled ? new ModelRequestAdmission(deliveryPolicy) : undefined;
            if (gate) installModelRequestAdmission(session, gate);
            const ends: AssistantMessage[] = [];
            session.subscribe(e => { if (e.type === "message_end" && e.message.role === "assistant") ends.push(e.message); });
            await session.bindExtensions({});
            await session.prompt("one original offline serialized request");
            assert.equal(f.bodies.length, 1);
            assert.equal(f.bodies[0].max_tokens, 23);
            assert.equal(f.bodies[0].max_output_tokens, undefined);
            assert.equal(signals.length, 1);
            assert.equal(ends.at(-1)?.stopReason, "error");
            assert.match(ends.at(-1)?.errorMessage ?? "", /INERT_WIRE_REFUSAL/);
            assert.equal(session.isIdle, true);
            if (gate) {
                await session.prompt("UNKNOWN refusal, never a network retry");
                assert.equal(f.bodies.length, 1);
                assert.match(ends.at(-1)?.errorMessage ?? "", /MODEL_REQUEST_USAGE_UNKNOWN/);
                assert.equal(gate.snapshot().dispatchedRequests, 1);
                assert.equal(gate.snapshot().unknownUsageRequests, 1);
                assert.equal(gate.snapshot().providerCostBoundVerified, false);
            } else assert.equal(ends.length, 1, "absence adds no accounting/automatic continuation");
        } finally { await session?.abort(); await session?.dispose(); await f.close(); }
    }
});

// Independently configured policies are legal. Exercise actual SDK lifecycle,
// production admission installer and request-transcript notices, not copied math.
test("F07 higher-cap admission never suppresses lower-cap awareness on either axis", { timeout: 20000 }, async () => {
    const highPolicy: ModelRequestAdmissionPolicy = { version: 1, lifetime: "runtime", guarantee: "local_reservation",
        inclusiveInputLimit: 1000000, outputLimit: 1000, requestReserveInput: 1000, requestReserveOutput: 10 };
    for (const axis of ["output", "input"] as const) {
        for (const [remaining, expected] of (axis === "output"
            ? [[6, "normal"], [5, "finish_delivery"], [4, "finish_delivery"]] as const
            : [[1001, "normal"], [1000, "finish_delivery"], [999, "finish_delivery"]] as const)) {
            for (const linked of [false, true]) {
                const awareness = { version: 1 as const, lifetime: "runtime" as const,
                    inclusiveInputLimit: axis === "input" ? 10000 : 1000000, outputLimit: 100,
                    deliveryReserveInput: 1000, deliveryReserveOutput: 5 };
                const input = axis === "input" ? awareness.inclusiveInputLimit - remaining : 100;
                const output = axis === "output" ? awareness.outputLimit - remaining : 0;
                const f = await fixture({ policy: highPolicy, advisory: true, awareness, linkAdmission: linked });
                const notices: any[] = [];
                let calls = 0;
                try {
                    f.runtime.streamSimple = ((_m, context, options) => {
                        assert.ok(++calls <= 2, "two finite requests, no retry or automatic continuation");
                        assert.equal(options?.maxRetries, 0);
                        const current = deliveryNotes(context);
                        assert.equal(current.length, 1, "one fresh leading system notice");
                        notices.push(current[0]);
                        const msg = message("stop", calls === 1 ? input : 21);
                        msg.usage.output = calls === 1 ? output : 0;
                        msg.usage.totalTokens = msg.usage.input + msg.usage.cacheRead + msg.usage.cacheWrite + msg.usage.output;
                        const stream = createAssistantMessageEventStream();
                        stream.push({ type: "done", reason: "stop", message: msg });
                        return stream;
                    }) as typeof f.runtime.streamSimple;
                    await f.session.prompt("settle independently observed usage");
                    const settled = f.gate!.snapshot();
                    assert.equal(settled.knownInclusiveInput, input);
                    assert.equal(settled.knownOutput, output);
                    assert.equal(settled.pendingRequests, 0);
                    assert.equal(settled.unknownUsageRequests, 0);
                    assert.equal(f.gate!.deliveryContract().nextRequestAdmissible, true);
                    // Even when awareness warns, it must not deny this real request.
                    await f.session.prompt("observe lower-cap advisory without changing admission");
                    assert.equal(calls, 2);
                    const n = notices[1];
                    assert.equal(notices[0].reserveState, "normal");
                    assert.equal(n.completedMainRequests, 1);
                    assert.equal(n.knownInclusiveInput, input);
                    assert.equal(n.knownOutput, output);
                    assert.equal(n.remainingInputUpperBound, awareness.inclusiveInputLimit - input);
                    assert.equal(n.remainingOutputUpperBound, awareness.outputLimit - output);
                    assert.equal(n.pendingMainRequests, 0);
                    assert.equal(n.unknownUsageRequests, 0);
                    assert.equal(n.unaccountedCompaction, false);
                    assert.equal(n.reserveState, expected, `${axis}:${remaining}:linked=${linked}`);
                    assert.equal(n.deliveryNoticeContract, "advisory_only_no_final_guarantee");
                    if (linked) {
                        assert.equal(n.admission.availableInput, highPolicy.inclusiveInputLimit - input);
                        assert.equal(n.admission.availableOutput, highPolicy.outputLimit - output);
                        assert.ok(n.admission.availableInput - n.admission.requestReserveInput > awareness.deliveryReserveInput);
                        assert.ok(n.admission.availableOutput - n.admission.requestReserveOutput > awareness.deliveryReserveOutput);
                        assert.equal(n.admission.nextRequestAdmissible, true);
                        assert.equal(n.admission.finalDeliveryGuaranteed, false);
                        assert.equal(n.admission.responseOutputLimit, "not_enforced");
                        assert.equal(n.admission.providerCostBoundVerified, false);
                    } else assert.equal(n.admission, undefined);
                    assert.equal(f.gate!.snapshot().knownInclusiveInput, input + 21);
                    assert.equal(f.gate!.snapshot().knownOutput, output);
                    assert.equal(f.gate!.snapshot().dispatchedRequests, 2);
                    assert.equal(f.gate!.snapshot().settledRequests, 2);
                    assert.equal(f.gate!.snapshot().unknownUsageRequests, 0);
                    assert.equal(f.gate!.snapshot().pendingRequests, 0);
                    assert.equal(f.effects, 0);
                    assert.equal((f.session.messages.at(-1) as AssistantMessage).stopReason, "stop");
                    assert.ok(f.session.isIdle);
                } finally { await f.close(); }
            }
        }
    }
});
