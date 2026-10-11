import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createRosclawRuntime } from "../src/harness/pi/pi-runtime.js";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryCredentialStore, Type, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, gateModelUsageStream, validateModelUsagePreadmission } from "../src/harness/pi/tool-call-budget.js";
import { observeCompactionStream } from "../src/harness/pi/compaction-stream.js";
import type { RosclawRuntimeOptions } from "../src/harness/pi/pi-runtime.js";
import { defaultOperatorSocket } from "../src/bridge/operatord-client.js";
import { envelopeHash, type EmbodiedContextEnvelope } from "../src/extension/context-injection.js";

/** Feasibility probes, NOT the seven acceptance cases. The provider SDK is real;
 * only HTTP is replaced with an in-process rejecting transport. Never contact
 * an endpoint or patch the installed SDK. No token bound is invented here. */
async function probe(refuse: boolean, strict = false, retry = false) {
    const home = await mkdtemp(join(tmpdir(), "preadmission-probe-"));
    const originalFetch = globalThis.fetch;
    let transportCalls = 0, hookCalls = 0;
    const payloads: unknown[] = [], ends: any[] = [];
    let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
    globalThis.fetch = async () => {
        transportCalls++;
        return new Response(JSON.stringify({ error: { message: "OFFLINE_FIXTURE_REJECTED", type: "invalid_request_error" } }),
            { status: 400, headers: { "content-type": "application/json" } });
    };
    try {
        const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: retry } });
        const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
            modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
        const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
            provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 100000, maxTokens: 30 };
        runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [model] });
        const factory = createToolCallBudgetExtension({ allowedTools: ["diagnostic"], maxTotalCalls: 1,
            modelUsageAwareness: { version: 1, lifetime: "runtime", inclusiveInputLimit: 800000, outputLimit: 100,
                deliveryReserveInput: 0, deliveryReserveOutput: 0 } });
        const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
            noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
            extensionFactories: [{ name: "original-awareness", factory }, { name: "feasibility", factory: pi => {
                pi.on("before_agent_start", event => ({ systemPrompt: event.systemPrompt + "\nSYSTEM_BOUNDARY_🧪" }));
                pi.on("before_provider_request", event => {
                    hookCalls++; payloads.push(event.payload);
                    if (refuse) throw new Error("MODEL_USAGE_PREADMISSION_UNKNOWN_TOKEN_BOUND");
                });
                pi.on("message_end", event => { if (event.message.role === "assistant") ends.push(event.message); });
            } }] });
        await loader.reload();
        ({ session } = await createAgentSession({ cwd: home, agentDir: home, model: runtime.getModel("fixture", "fixture")!,
            modelRuntime: runtime, settingsManager: settings, resourceLoader: loader, sessionManager: SessionManager.inMemory(home),
            tools: ["diagnostic"], customTools: [{ name: "diagnostic", label: "diagnostic", description: "TOOL_BOUNDARY_" + "大".repeat(1024),
                parameters: Type.Object({ value: Type.String() }), execute: async () => { throw new Error("UNEXPECTED_TOOL_EFFECT"); } }], thinkingLevel: "off" }));
        await session.bindExtensions({});
        // Same composition as production: gate outside the summary observer.
        // No extension hook is treated as a cancellation authority.
        const observed = observeCompactionStream(session.agent.streamFunction, {
            enabled: () => session!.isCompacting,
            owner: () => ({ session_id: session!.sessionId, pid: process.pid, activity: "session_summary" }),
            record: () => {},
        });
        const policy = strict ? { version: 1 as const, mode: "strict" as const } : undefined;
        session.agent.streamFunction = gateModelUsageStream(observed, policy);
        if (!strict) assert.equal(session.agent.streamFunction, observed);
        await session.prompt("USER_BOUNDARY_é🧪");
        if (strict) await session.prompt("SECOND_EXPLICIT_REQUEST_é🧪");
        assert.equal(session.isIdle, true);
        return { transportCalls, hookCalls, payloads, ends };
    } finally {
        session?.dispose();
        globalThis.fetch = originalFetch;
        await rm(home, { recursive: true, force: true });
    }
}

test("feasibility-original-provider-reaches-inprocess-fetch", async () => {
    const result = await probe(false);
    assert.ok(result.transportCalls > 0, "Control must reach actual provider HTTP boundary");
    assert.ok(result.hookCalls > 0);
    assert.equal(result.ends.length, 1);
    assert.equal(result.ends[0].stopReason, "error");
});

test("counterexample-extension-throw-does-not-prevent-actual-fetch", async () => {
    const result = await probe(true);
    assert.ok(result.hookCalls > 0, "Must exercise installed provider hook, not manual callback");
    assert.equal(result.transportCalls, 1, "Extension throw is not a safe gate if SDK continues to fetch");
});

test("feasibility-full-provider-envelope-is-not-a-token-bound", async () => {
    const result = await probe(false);
    const serialized = JSON.stringify(result.payloads);
    assert.ok(serialized.includes("SYSTEM_BOUNDARY_🧪"));
    assert.ok(serialized.includes("USER_BOUNDARY_é🧪"));
    assert.ok(serialized.includes("TOOL_BOUNDARY_"));
    // This proves envelope visibility only. There is deliberately no conversion
    // from bytes/characters to tokens and no acceptance-test name on this probe.
    assert.ok(result.transportCalls > 0);
});

// Production composition, not a separately assembled SDK session. The only
// transport is an in-process HTTP rejection; observing the installed stream
// delegates to the runtime's existing gate without attaching a second gate.
async function productionFixture(strict = true, reserve = 0, limit = 100, persistFails = false) {
    const home = await mkdtemp(join(tmpdir(), "preadmission-native-"));
    const originalFetch = globalThis.fetch;
    // Per-instance dependencies only: genuine coordinator, lease heartbeat,
    // context validation, InputController and SDK lifecycle remain intact.
    const dependencyCalls: Array<{ method: string; input: unknown }> = [];
    const rejectedMethods: string[] = [];
    const missions = new Set<string>();
    const bindings = new Map<string, string>();
    const leases = new Map<string, { mission: string; token: string }>();
    const persisted = new Map<string, { mission: string; session: string; inputId: string }>();
    const mirrored = new Map<string, string>();
    const identity = (value: unknown): string => {
        assert.equal(typeof value, "string");
        assert.ok((value as string).length > 0);
        return value as string;
    };
    const validateIdentities = (input: Record<string, unknown>, allowUnbound = false) => {
        const unbound = allowUnbound && input.mission_id === "";
        if (input.mission_id !== undefined && !unbound) assert.ok(missions.has(identity(input.mission_id)));
        const sessionValue = input.pi_session_id ?? input.session_ref;
        if (sessionValue !== undefined) {
            const session = identity(sessionValue);
            if (unbound) {
                assert.equal(bindings.has(session), false, "Empty mission is only valid for an unbound session");
                assert.equal(leases.has(session), false);
            } else {
                assert.ok(bindings.has(session));
                if (input.mission_id !== undefined) assert.equal(bindings.get(session), input.mission_id);
            }
        }
    };
    const rejectMethod = (method: string): never => {
        rejectedMethods.push(method);
        throw new Error(`UNDECLARED_FIXTURE_RPC:${method}`);
    };
    const rpcDependencies: NonNullable<RosclawRuntimeOptions["rpcDependencies"]> = {
        call: async (rpcHome, method, input = {}) => {
            dependencyCalls.push({ method, input: structuredClone(input) });
            assert.equal(rpcHome, home);
            switch (method) {
                case "pi.mission.create": {
                    identity(input.goal);
                    assert.equal(input.mode, "SIMULATION");
                    const mission = `fixture-mission-${missions.size + 1}`;
                    missions.add(mission);
                    return { ok: true, mission_id: mission };
                }
                case "pi.session.binding.get": {
                    const session = identity(input.pi_session_id);
                    const mission = bindings.get(session);
                    return { ok: true, binding: mission ? { mission_id: mission } : null, mission_archived: false };
                }
                case "pi.session.bind": {
                    const session = identity(input.pi_session_id);
                    const mission = identity(input.mission_id);
                    assert.ok(missions.has(mission));
                    assert.equal(leases.size, 0, "Original release must precede new bind");
                    const token = `fixture-token-${dependencyCalls.length}`;
                    bindings.set(session, mission);
                    leases.set(session, { mission, token });
                    return { ok: true, binding: { binding_id: `fixture-binding-${session}` }, lease_token: token };
                }
                case "pi.session.heartbeat":
                case "pi.session.release": {
                    validateIdentities(input);
                    const session = identity(input.pi_session_id);
                    const lease = leases.get(session);
                    assert.ok(lease);
                    assert.equal(input.mission_id, lease.mission);
                    assert.equal(input.lease_token, lease.token);
                    if (method === "pi.session.release") leases.delete(session);
                    return { ok: true };
                }
                case "pi.context": {
                    validateIdentities(input);
                    const expires = new Date(Date.now() + 60_000).toISOString();
                    const context: EmbodiedContextEnvelope = {
                        schema_version: "rosclaw.embodied_context.v1",
                        mission_id: identity(input.mission_id), context_revision: 1,
                        generated_at: new Date().toISOString(), expires_at: expires,
                        body: { body_id: "fixture/offline", effective_body_hash: "fixture-body",
                            summary: "In-process SOURCE fixture; not physical evidence" },
                        safety: { mode: "SIMULATION" }, pending_approvals: [], hash: "",
                    };
                    context.hash = envelopeHash(context);
                    return { ok: true, context, context_lease_id: `fixture-context-${input.mission_id}`,
                        context_lease_expires_at: expires };
                }
                case "pi.status":
                    validateIdentities(input);
                    return { ok: true, sim_policy: "auto", body_display: "Offline fixture", robot_kit: null };
                case "pi.turn.record":
                    validateIdentities(input);
                    identity(input.text);
                    assert.equal(input.source, "interactive");
                    return { ok: true };
                case "pi.turn.latest":
                    validateIdentities(input);
                    return { ok: true, turn: null };
                case "pi.capability.snapshot":
                    validateIdentities(input);
                    return { ok: true, snapshot: null };
                case "pi.capabilities":
                    validateIdentities(input);
                    return { ok: true, action_capabilities: [] };
                case "pi.input.persist": {
                    validateIdentities(input, true);
                    const message = identity(input.message_id);
                    const session = identity(input.session_ref);
                    identity(input.backend_native_id);
                    identity(input.text);
                    assert.equal(typeof input.force_new, "boolean");
                    if (persistFails) throw new Error("FIXTURE_PERSISTENCE_REJECTED");
                    const inputId = `fixture-input-${message}`;
                    persisted.set(message, { mission: input.mission_id as string, session, inputId });
                    return { input: { input_id: inputId },
                        turn_disposition: { input_id: inputId, owner: "model", suppress_model_turn: false } };
                }
                case "pi.input.dispatched": {
                    validateIdentities(input, true);
                    const saved = persisted.get(identity(input.message_id));
                    assert.ok(saved);
                    assert.equal(input.mission_id, saved.mission);
                    validateIdentities({ mission_id: saved.mission, session_ref: saved.session }, true);
                    return { ok: true };
                }
                case "pi.kernel.active":
                    validateIdentities(input, true);
                    return { task: null };
                case "pi.events.batch": {
                    assert.ok(Array.isArray(input.events));
                    assert.ok(input.events.length > 0 && input.events.length <= 256);
                    let stored = 0;
                    for (const event of input.events as Array<Record<string, unknown>>) {
                        validateIdentities(event, true);
                        identity(event.pi_session_id);
                        const mirrorId = identity(event.mirror_id);
                        identity(event.event_type);
                        identity(event.identity_protocol);
                        identity(event.occurred_at);
                        const payload = JSON.stringify(event);
                        const previous = mirrored.get(mirrorId);
                        if (previous !== undefined) assert.equal(previous, payload, "Mirror replay must preserve exact payload");
                        else { mirrored.set(mirrorId, payload); stored++; }
                    }
                    return { ok: true, stored };
                }
                default: return rejectMethod(method);
            }
        },
        operatorCallFn: async (socket, method, input = {}) => {
            dependencyCalls.push({ method, input: structuredClone(input) });
            assert.equal(socket, defaultOperatorSocket(home));
            if (method !== "approvals.list") return rejectMethod(method);
            validateIdentities(input);
            return { ok: true, approvals: [] };
        },
    };
    let transportCalls = 0;
    globalThis.fetch = async () => {
        transportCalls++;
        return new Response(JSON.stringify({ error: { message: "OFFLINE_FIXTURE_REJECTED", type: "invalid_request_error" } }),
            { status: 400, headers: { "content-type": "application/json" } });
    };
    let native: Awaited<ReturnType<typeof createRosclawRuntime>> | undefined;
    const contexts: any[] = [], ends: any[] = [];
    const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
        provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 100000, maxTokens: 30 };
    try {
        await mkdir(join(home, "agent"), { recursive: true });
        await writeFile(join(home, "agent", "settings.json"), JSON.stringify({
            compaction: { enabled: false, keepRecentTokens: 1 }, retry: { enabled: false }
        }));
        const modelRuntime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
            modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
        modelRuntime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [model] });
        const policy = { allowedTools: [], maxTotalCalls: 1, visibleBudget: true, visibleBudgetMode: "compact" as const,
            modelUsageAwareness: { version: 1 as const, lifetime: "runtime" as const, inclusiveInputLimit: limit,
                outputLimit: limit, deliveryReserveInput: reserve, deliveryReserveOutput: reserve },
            ...(strict ? { modelUsagePreadmission: { version: 1 as const, mode: "strict" as const } } : {}) };
        native = await createRosclawRuntime({ cwd: home, rosclawHome: home, profile: "robot", version: "SOURCE_FIXTURE",
            taskContext: { workspaceRoot: home, productRoot: home, workspaceSource: "explicit" } as any,
            explicitModel: { modelRuntime, model: modelRuntime.getModel("fixture", "fixture")! },
            explicitThinking: "off", toolCallBudget: policy, rpcDependencies });
        // Headless host must bind the same public extension lifecycle that the
        // interactive host binds, on initial creation and every replacement.
        await native.runtime.session.bindExtensions({});
        native.runtime.setRebindSession(session => session.bindExtensions({}));
        const attachObserver = () => {
            const session = native!.runtime.session;
            const original = session.agent.streamFunction;
            session.agent.streamFunction = function (...args: Parameters<typeof original>) {
                contexts.push(structuredClone(args[1]));
                return original.apply(this, args);
            };
            session.subscribe(event => {
                if (event.type === "message_end" && event.message.role === "assistant") ends.push(event.message);
            });
        };
        attachObserver();
        return { native, contexts, ends, modelRuntime, model, attachObserver, dependencyCalls,
            get session() { return native!.runtime.session; },
            get transportCalls() { return transportCalls; },
            async close() {
                try {
                    try { await native!.leaseManager.release(); }
                    finally { await native!.runtime.dispose(); }
                } finally {
                    try { native!.ownership.releaseAll(); }
                    finally {
                        globalThis.fetch = originalFetch;
                        await rm(home, { recursive: true, force: true });
                    }
                }
                assert.equal(leases.size, 0, "Original release cleared fixture leases");
                assert.deepEqual(rejectedMethods, [], "No undeclared fixture RPC was swallowed");
                console.log("INPUT_DEPENDENCY_CALLS", JSON.stringify(dependencyCalls));
            } };
    } catch (error) {
        try {
            try { await native?.leaseManager.release(); }
            finally { await native?.runtime.dispose(); }
        } finally {
            try { native?.ownership.releaseAll(); }
            finally {
                globalThis.fetch = originalFetch;
                await rm(home, { recursive: true, force: true });
            }
        }
        throw error;
    }
}
const usageNotice = (context: any) => {
    const system = context.messages.find((m: any) => m.role === "system");
    const text = typeof system.content === "string" ? system.content : system.content.filter((p: any) => p.type === "text").map((p: any) => p.text).join("\n");
    const lines = text.split("\n").filter((l: string) => l.startsWith("ROSCLAW_MODEL_USAGE_JSON:"));
    assert.equal(lines.length, 1);
    return JSON.parse(lines[0].slice("ROSCLAW_MODEL_USAGE_JSON:".length));
};
const assertDenied = (f: Awaited<ReturnType<typeof productionFixture>>) => {
    assert.equal(f.transportCalls, 0);
    assert.ok(f.ends.length > 0);
    for (const end of f.ends) {
        assert.equal(end.stopReason, "error");
        assert.match(end.errorMessage, /MODEL_USAGE_PREADMISSION_UNKNOWN_TOKEN_BOUND/);
    }
};

// Outer-wrapper acceptance is separate from the unchanged unsafe-hook failure
// above (CLOSED30794). All bounds remain UNKNOWN: no synthetic estimator/hold.
test("preadmission-real-sdk-denies-before-transport", async () => {
    const productionControl = await productionFixture(false);
    try {
        await productionControl.session.prompt("production omitted control");
        assert.equal(productionControl.transportCalls, 1, "Production runtime reaches installed provider HTTP");
        assert.equal(productionControl.contexts.length, 1);
        assert.equal(productionControl.ends[0].stopReason, "error");
        assert.equal(productionControl.ends.length, 1);
        assert.ok(productionControl.dependencyCalls.some(c => c.method === "pi.input.persist"));
        assert.ok(productionControl.dependencyCalls.some(c => c.method === "pi.input.dispatched"));
    } finally { await productionControl.close(); }
    const persistenceRejected = await productionFixture(false, 0, 100, true);
    try {
        await persistenceRejected.session.prompt("negative persistence control");
        assert.ok(persistenceRejected.dependencyCalls.some(c => c.method === "pi.input.persist"));
        assert.equal(persistenceRejected.contexts.length, 0, "Persistence failure handles input before stream");
        assert.equal(persistenceRejected.transportCalls, 0);
        assert.equal(persistenceRejected.ends.length, 0, "Handled input is not a fabricated SDK refusal");
    } finally { await persistenceRejected.close(); }
    const productionDenied = await productionFixture();
    try {
        await productionDenied.session.prompt("production UNKNOWN");
        await productionDenied.session.prompt("explicit next request");
        assertDenied(productionDenied);
        assert.equal(productionDenied.ends.length, 2);
        assert.equal(usageNotice(productionDenied.contexts[1]).unknownUsageRequests, 1);
    } finally { await productionDenied.close(); }
    const denied = await probe(false, true, true);
    assert.equal(denied.transportCalls, 0);
    assert.equal(denied.hookCalls, 0, "No provider delegation or unsafe hook on denial");
    assert.equal(denied.ends.length, 2, "Only the two explicit prompts; retry adds no turn");
    for (const end of denied.ends) {
        assert.equal(end.stopReason, "error");
        assert.match(end.errorMessage, /MODEL_USAGE_PREADMISSION_UNKNOWN_TOKEN_BOUND/);
        // SDK default zeros on locally rejected error messages are not evidence
        // of measured usage; the unchanged observer treats errors as UNKNOWN.
    }
});

test("preadmission-bound-provenance-and-unknown", async () => {
    validateModelUsagePreadmission({ version: 1, mode: "strict" });
    for (const invalid of [undefined, null, [], true, { version: 1 },
        { version: 1, mode: "awareness" }, { version: 2, mode: "strict" },
        { version: 1, mode: "strict", inputBound: 1 },
        { version: 1, mode: "strict", estimator: "chars" },
        { version: 1, mode: "strict", tokenizer: "unsupported" }]) {
        assert.throws(() => validateModelUsagePreadmission(invalid), /INVALID_MODEL_USAGE_PREADMISSION/);
        assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], modelUsagePreadmission: invalid } as any),
            /INVALID_MODEL_USAGE_PREADMISSION/);
    }
    // No caller-supplied bound, prior usage, envelope size or tokenizer name
    // qualifies this actual Unicode/history/tool request. Always UNKNOWN.
    const result = await probe(false, true);
    assert.equal(result.transportCalls, 0);
    assert.equal(result.ends.length, 2);
});

test("preadmission-reserve-and-exact-boundaries", async () => {
    // Remaining vs reserve is advisory only, including exact equality. None of
    // these numbers supplies a qualified bound for the next Unicode request.
    for (const [limit, reserve, state] of [[99, 99, "finish_delivery"], [100, 100, "finish_delivery"], [101, 100, "normal"]] as const) {
        const f = await productionFixture(true, reserve, limit);
        try {
            await f.session.prompt("UNKNOWN even at an exact reserve boundary 🧪");
            const n = usageNotice(f.contexts[0]);
            assert.equal(n.remainingInputUpperBound, limit);
            assert.equal(n.remainingOutputUpperBound, limit);
            assert.equal(n.reserveState, state);
            assertDenied(f);
            await f.session.prompt("prior refusal is not measured zero");
            assert.equal(usageNotice(f.contexts[1]).reserveState, "unknown");
            assertDenied(f);
        } finally { await f.close(); }
    }
});

test("preadmission-concurrent-reservations", async () => {
    const f = await productionFixture();
    try {
        await f.session.prompt("establish an actual SDK context");
        const context = f.contexts[0];
        const results = await Promise.allSettled(Array.from({ length: 8 }, () => Promise.resolve().then(() =>
            f.session.agent.streamFunction(f.model, context, {}))));
        assert.equal(results.length, 8);
        for (const result of results) {
            assert.equal(result.status, "rejected");
            if (result.status === "rejected") {
                assert.equal(result.reason.code, "MODEL_USAGE_PREADMISSION_UNKNOWN_TOKEN_BOUND");
                assert.equal(result.reason.status, "UNKNOWN");
            }
        }
        // Explicit re-entry still cannot reach the provider. These direct
        // runtime-boundary probes create no fake assistant lifecycle/holds.
        assert.throws(() => f.session.agent.streamFunction(f.model, context, {}), /UNKNOWN_TOKEN_BOUND/);
        await f.session.prompt("next real lifecycle");
        assertDenied(f);
        const n = usageNotice(f.contexts.at(-1));
        assert.equal(n.completedMainRequests, 1);
        assert.equal(n.unknownUsageRequests, 1);
        assert.equal(n.pendingMainRequests, 0);
        assert.equal(f.ends.length, 2);
        assert.deepEqual(f.session.getActiveToolNames(), []);
    } finally { await f.close(); }
});

test("preadmission-settlement-and-error-unknown", async () => {
    const f = await productionFixture();
    try {
        await f.session.prompt("SDK rejected stream");
        await f.session.abort();
        await f.session.prompt("explicit next request");
        assertDenied(f);
        const n = usageNotice(f.contexts[1]);
        assert.equal(n.completedMainRequests, 1);
        assert.equal(n.unknownUsageRequests, 1);
        assert.equal(n.pendingMainRequests, 0);
        assert.equal(n.reserveState, "unknown");
    } finally { await f.close(); }
    // Artificial lifecycle controls ONLY: valid settled cache-inclusive usage,
    // error defaults, partial abort, duplicate final and explicit next start.
    // They do not grant production bounds, transport or measured-zero claims.
    const handlers: Record<string, any> = {};
    createToolCallBudgetExtension({ allowedTools: [], modelUsageAwareness: {
        version: 1, lifetime: "runtime", inclusiveInputLimit: 100, outputLimit: 100,
        deliveryReserveInput: 0, deliveryReserveOutput: 0
    } })({ on: (name: string, handler: any) => { handlers[name] = handler; } } as any);
    const ctx = { sessionManager: { getSessionId: () => "artificial-settlement-negative-control" } };
    await handlers.session_start({}, ctx);
    const snapshot = async () => {
        const patch = await handlers.before_agent_start({ systemPrompt: "CONTROL" }, ctx);
        return usageNotice({ messages: [{ role: "system", content: patch.systemPrompt }] });
    };
    const settle = async (message: any) => {
        await handlers.message_start({ message: { ...message } }, ctx);
        await handlers.message_end({ message }, ctx);
    };
    const complete = { role: "assistant", stopReason: "stop", usage: { input: 11, cacheRead: 7, cacheWrite: 3, cacheWrite1h: 2, output: 5 } };
    await settle(complete);
    assert.equal((await snapshot()).knownInclusiveInput, 21);
    assert.equal((await snapshot()).knownOutput, 5);
    const error = { role: "assistant", stopReason: "error", usage: { input: 0, cacheRead: 0, cacheWrite: 0, output: 0 } };
    await settle(error);
    const partial = { role: "assistant", stopReason: "aborted", usage: { input: 9, cacheRead: -1, output: 0 } };
    await settle(partial);
    const before = await snapshot();
    await handlers.message_end({ message: partial }, ctx);
    assert.deepEqual(await snapshot(), before);
    assert.equal(before.completedMainRequests, 3);
    assert.equal(before.knownInclusiveInput, 30);
    assert.equal(before.knownOutput, 5);
    assert.equal(before.unknownUsageRequests, 2);
    await handlers.message_start({ message: { role: "assistant" } }, ctx);
    const pending = await snapshot();
    assert.equal(pending.pendingMainRequests, 1);
    assert.equal(pending.knownInclusiveInput, 30);
    assert.equal(pending.reserveState, "unknown");
});

test("preadmission-resume-and-managed-wakeup", async () => {
    const f = await productionFixture();
    try {
        await f.session.prompt("history 🧪 ".repeat(512));
        const saved = f.session.sessionFile!;
        for (const state of ["SUCCEEDED", "FAILED", "CANCELLED"]) {
            assert.equal(f.session.isIdle, true);
            const details = { operation_id: `source-owned-${state}`, state };
            await f.session.sendCustomMessage({ customType: "rosclaw.operation.result", content: `Owned terminal ${state}`,
                display: false, details }, { triggerTurn: true });
            const persisted = f.session.messages.find((m: any) => m.role === "custom" && m.details?.operation_id === details.operation_id) as any;
            assert.deepEqual(persisted.details, details);
            assert.equal(persisted.display, false);
            assertDenied(f);
        }
        assert.equal((await f.native.runtime.newSession()).cancelled, false);
        f.attachObserver();
        await f.session.setModel(f.modelRuntime.getModel("fixture", "fixture")!);
        await f.session.prompt("new session retains strict policy");
        assertDenied(f);
        assert.equal((await f.native.runtime.switchSession(saved)).cancelled, false);
        f.attachObserver();
        await f.session.prompt("resumed history 🧪 ".repeat(512));
        assertDenied(f);
        const alternate = { ...f.model, id: "alternate", name: "alternate" };
        f.modelRuntime.registerProvider("fixture", { api: alternate.api, baseUrl: alternate.baseUrl, apiKey: "INERT", models: [f.model, alternate] });
        await f.session.setModel(f.modelRuntime.getModel("fixture", "alternate")!);
        await f.session.prompt("switched model is not a qualified estimator");
        assertDenied(f);
        // Real preparation + summary request route, not cancellation by an
        // extension or direct invocation of a manually attached gate.
        const before = f.contexts.length;
        await assert.rejects(f.session.compact(), /MODEL_USAGE_PREADMISSION_UNKNOWN_TOKEN_BOUND/);
        assert.ok(f.contexts.length > before, "Compaction must reach the production outer stream");
        assert.equal(f.transportCalls, 0);
        await f.session.prompt("explicit recovery after summary refusal");
        const n = usageNotice(f.contexts.at(-1));
        assert.equal(n.unaccountedCompaction, true);
        assert.equal(n.reserveState, "unknown");
        assertDenied(f);
    } finally { await f.close(); }
});

test("preadmission-default-policy-tool-budget-compatibility", async () => {
    // Omitted strict mode retains original awareness and actual provider flow.
    // The existing protected tool-budget regression suite is still required.
    const result = await probe(false);
    assert.equal(result.transportCalls, 1);
    assert.equal(result.hookCalls, 1);
    assert.equal(result.ends.length, 1);
    assert.equal(result.ends[0].stopReason, "error");
});
