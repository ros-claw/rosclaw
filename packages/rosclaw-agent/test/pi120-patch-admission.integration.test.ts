import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { createToolCallBudgetExtension } from "../src/harness/pi/tool-call-budget.js";

test("1.2 real thin patches retain exact anchors, behavior, idempotence and missing-anchor refusal", () => {
    const root = mkdtempSync(join(tmpdir(), "rosclaw-pi120-patch-"));
    const agentRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"))), "..");
    const aiRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))), "..");
    const paths = [
        [agentRoot, "@earendil-works/pi-coding-agent", "dist/modes/interactive/interactive-mode.js"],
        [agentRoot, "@earendil-works/pi-coding-agent", "dist/core/session-manager.js"],
        [aiRoot, "@earendil-works/pi-ai", "dist/api/openai-completions.js"],
        [agentRoot, "@earendil-works/pi-coding-agent", "dist/core/agent-session.js"],
        [agentRoot, "@earendil-works/pi-coding-agent", "package.json"],
    ];
    try {
        const scriptDir = join(root, "patches");
        mkdirSync(scriptDir, { recursive: true });
        const sourceScript = new URL("../../patches/apply-upstream-patches.mjs", import.meta.url);
        writeFileSync(join(scriptDir, "apply.mjs"), readFileSync(sourceScript));
        for (const [installed, name, file] of paths) {
            const dest = join(root, "node_modules", name, file);
            mkdirSync(dirname(dest), { recursive: true });
            writeFileSync(dest, readFileSync(join(installed, file)));
        }
        const run = () => spawnSync(process.execPath, [join(scriptDir, "apply.mjs")], { encoding: "utf8", timeout: 5000 });
        const first = run();
        assert.equal(first.status, 0, first.stderr + first.stdout);
        const interactive = join(root, "node_modules", paths[0][1], paths[0][2]);
        const patched = readFileSync(interactive, "utf8");
        const snapshots = paths.map(([, name, file]) => readFileSync(join(root, "node_modules", name, file)));
        const second = run();
        assert.equal(second.status, 0, second.stderr);
        for (const [index, [, name, file]] of paths.entries()) {
            assert.deepEqual(readFileSync(join(root, "node_modules", name, file)), snapshots[index], "repeat application must be byte-identical for every target");
        }
        assert.equal(readFileSync(interactive, "utf8"), patched);
        assert.ok(patched.includes("rosclaw continue"));
        assert.ok(patched.includes("rosclaw sessions"));
        const guardStart = patched.indexOf('{ const cmd = text.split(/\\s/, 1)[0].toLowerCase();');
        const guardEnd = patched.indexOf("            // Handle commands", guardStart);
        assert.ok(guardStart >= 0 && guardEnd > guardStart);
        const guard = patched.slice(guardStart, guardEnd);
        let dispatches = 0;
        let cleared = 0;
        const warnings: unknown[] = [];
        const ui = { editor: { setText: (text: string) => { assert.equal(text, ""); cleared++; } },
            chatContainer: { addChild: (value: unknown) => warnings.push(value) } };
        const globals = globalThis as typeof globalThis & { __rosclawBuiltinPolicy?: { disabled: Set<string> } };
        const prior = globals.__rosclawBuiltinPolicy;
        try {
            globals.__rosclawBuiltinPolicy = { disabled: new Set(["/trust", "/share", "/import", "/reload"]) };
            const execute = new Function("text", "Text", "theme", "dispatch", guard + "\ndispatch();");
            class Text { constructor(public text: string) {} }
            for (const command of ["/TRUST path", "/share", "/import foo", "/reload"]) {
                execute.call(ui, command, Text, { fg: (_color: string, text: string) => text }, () => dispatches++);
            }
            assert.equal(dispatches, 0, "disabled commands cannot reach builtin dispatch");
            assert.equal(cleared, 4);
            assert.equal(warnings.length, 4);
            for (const warning of warnings) {
                assert.ok(warning instanceof Text);
                assert.match(warning.text, /is disabled by ROSClaw policy in this profile/);
            }
            execute.call(ui, "/help", Text, { fg: (_color: string, text: string) => text }, () => dispatches++);
            assert.equal(dispatches, 1, "unblocked command still dispatches");
        } finally { globals.__rosclawBuiltinPolicy = prior; }
        // Mandatory corruption, not an optional retired patch: must hard fail.
        writeFileSync(interactive, patched.replace(guard, "").replace("            // Handle commands", "            // CORRUPTED_MANDATORY_ANCHOR"));
        const broken = run();
        assert.equal(broken.status, 1);
        assert.match(broken.stderr, /PATCH ANCHOR MISSING: patch-02/);
    } finally { rmSync(root, { recursive: true, force: true }); }
});

test("1.2.0 startRun literal admits only reviewed bodies and preserves the constructor and every outside byte", async () => {
    const ts = await import("typescript");
    const { chmodSync, statSync } = await import("node:fs");
    const script = readFileSync(new URL("../../patches/apply-upstream-patches.mjs", import.meta.url), "utf8");
    const ast = ts.createSourceFile("patcher.mjs", script, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
    const literal = (name: string) => {
        const declarations = ast.statements.filter(ts.isVariableStatement).flatMap(s => [...s.declarationList.declarations]);
        const declaration = declarations.find(d => d.name.getText(ast) === name);
        assert.ok(declaration?.initializer && ts.isObjectLiteralExpression(declaration.initializer), `${name} is mandatory`);
        const result: Record<string, string> = {};
        for (const property of declaration.initializer.properties) {
            assert.ok(ts.isPropertyAssignment(property), "compat entry must be a pure literal");
            assert.ok(ts.isStringLiteral(property.initializer) || ts.isNoSubstitutionTemplateLiteral(property.initializer));
            result[property.name.getText(ast)] = property.initializer.text;
        }
        return result;
    };
    const patch = literal("PI120_START_RUN_COMPAT_PATCH");
    const forced = literal("PI120_FORCED_PROMPT_COMPAT_PATCH");
    assert.deepEqual(Object.keys(patch).sort(), ["anchor", "method", "replacement", "target", "version"]);
    assert.equal(patch.version, "1.4.0");
    assert.equal(patch.method, "_startRun");
    assert.equal(patch.target, "@earendil-works/pi-coding-agent/dist/core/agent-session.js");
    // Independent official 1.2.0 _startRun body (verbatim, not derived from the replacement).
    const officialBody = `{
        this._isAgentRunActive = true;
        this._agentRunAbortRequested = false;
        let messages;
        try {
            messages = await this._prepareRun(trigger);
        }
        catch (error) {
            this._isAgentRunActive = false;
            this._flushPendingBashMessages();
            this._flushPendingCustomMessages();
            this._resolveIdleWaitIfIdle();
            throw error;
        }
        // Messages held while the run was starting go before its starting messages, as they would
        // have while idle. No run exists yet, so they cannot split a tool call from its result.
        this._flushPendingBashMessages();
        this._flushPendingCustomMessages();
        onStarted?.();
        // Compaction before the prompt may have scheduled a retry; the new prompt replaces it.
        this._failedResponse = undefined;
        this._recordSelection();
        // The run records the loadout in the transcript; restored tools that did not register by now
        // are dropped, so a tool that never registers does not stay pending.
        this._pendingToolNames.clear();
        try {
            await this.agent.prompt(messages);
            while (!this._agentRunAbortRequested) {
                if (await this._handlePostAgentRun()) {
                    if (this._agentRunAbortRequested)
                        break;
                    await this.agent.continue();
                    continue;
                }
                if (this._agentRunAbortRequested || !(await this._runBeforeSettleBoundary()))
                    break;
                if (this._agentRunAbortRequested)
                    break;
                await this.agent.continue();
            }
        }
        finally {
            if (this._agentRunAbortRequested)
                this._finishCancelledRetry();
            this._failedResponse = undefined;
            this._runSystemPromptOptions = undefined;
            this._flushPendingBashMessages();
            this._flushPendingCustomMessages();
            await this._emitAgentSettled();
        }
    }`;
    assert.equal(patch.anchor, officialBody);
    const body = (text: string, name: string) => {
        const file = ts.createSourceFile("agent-session.js", text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
        const cls = file.statements.find(s => ts.isClassDeclaration(s) && s.name?.text === "AgentSession");
        assert.ok(cls && ts.isClassDeclaration(cls));
        const method = cls.members.find(m => ts.isMethodDeclaration(m) && m.name.getText(file) === name);
        assert.ok(method && ts.isMethodDeclaration(method) && method.body);
        const constructor = cls.members.find(ts.isConstructorDeclaration);
        assert.ok(constructor);
        return { start: method.body.getStart(file), end: method.body.end, text: method.body.getText(file), constructor: constructor.getText(file) };
    };
    const root = mkdtempSync(join(tmpdir(), "rosclaw-pi120-start-patch-"));
    try {
        const agentRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"))), "..");
        const aiRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))), "..");
        const scriptDir = join(root, "patches");
        mkdirSync(scriptDir, { recursive: true });
        const runner = join(scriptDir, "apply.mjs");
        writeFileSync(runner, script);
        for (const [installed, name, file] of [
            [agentRoot, "@earendil-works/pi-coding-agent", "dist/modes/interactive/interactive-mode.js"],
            [agentRoot, "@earendil-works/pi-coding-agent", "dist/core/session-manager.js"],
            [aiRoot, "@earendil-works/pi-ai", "dist/api/openai-completions.js"],
            [agentRoot, "@earendil-works/pi-coding-agent", "package.json"],
        ]) {
            const target = join(root, "node_modules", name, file);
            mkdirSync(dirname(target), { recursive: true });
            writeFileSync(target, readFileSync(join(installed, file)));
        }
        const installed = readFileSync(join(agentRoot, "dist/core/agent-session.js"), "utf8");
        const installedBody = body(installed, patch.method);
        assert.ok(installedBody.text === officialBody || installedBody.text === patch.replacement,
            "actual installed SDK must have one of the two exact reviewed bodies");
        // Normalize only the already-reviewed forced projection so all other bytes,
        // including real constructor instrumentation, are compared unchanged below.
        let pristine = installed.slice(0, installedBody.start) + officialBody + installed.slice(installedBody.end);
        const forcedBody = body(pristine, forced.method);
        assert.ok(forcedBody.text === forced.anchor || forcedBody.text === forced.replacement);
        pristine = pristine.slice(0, forcedBody.start) + forced.replacement + pristine.slice(forcedBody.end);
        const target = join(root, "node_modules", patch.target);
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, pristine);
        const run = () => spawnSync(process.execPath, [runner], { encoding: "utf8", timeout: 5000 });
        const first = run();
        assert.equal(first.status, 0, first.stderr + first.stdout);
        const patched = readFileSync(target, "utf8");
        const before = body(pristine, patch.method), after = body(patched, patch.method);
        assert.equal(after.text, patch.replacement);
        assert.equal(patched.slice(0, after.start), pristine.slice(0, before.start));
        assert.equal(patched.slice(after.end), pristine.slice(before.end), "all outside-method bytes remain exact");
        assert.equal(after.constructor, installedBody.constructor, "actual constructor instrumentation is untouched");
        assert.equal(patched, pristine.replace(officialBody, patch.replacement));
        chmodSync(target, 0o400); // Fixture copy only; never chmod the installed SDK.
        const statBefore = statSync(target, { bigint: true });
        const second = run();
        assert.equal(second.status, 0, second.stderr + second.stdout);
        assert.equal(readFileSync(target, "utf8"), patched);
        const statAfter = statSync(target, { bigint: true });
        assert.equal(statAfter.mtimeNs, statBefore.mtimeNs, "idempotence must not open the SDK for write");
        assert.equal(statAfter.ctimeNs, statBefore.ctimeNs);
        assert.equal(statAfter.mode, statBefore.mode);
        chmodSync(target, 0o600);
        for (const corrupted of [
            pristine.replace("onStarted?.();", "onStarted?.(true);"),
            pristine.replace("async _startRun(", "async _missingStartRun("),
            pristine + "\n    async _startRun(trigger, onStarted) " + officialBody,
        ]) {
            writeFileSync(target, corrupted);
            const rejected = run();
            assert.equal(rejected.status, 1, rejected.stderr + rejected.stdout);
            assert.match(rejected.stderr, /PATCH ANCHOR MISSING: patch-06/);
            assert.equal(readFileSync(target, "utf8"), corrupted, "unknown or missing preimages are never rewritten");
        }
        writeFileSync(target, pristine);
        const manifest = join(root, "node_modules/@earendil-works/pi-coding-agent/package.json");
        writeFileSync(manifest, JSON.stringify({ version: "1.3.1" }));
        const wrongVersion = run();
        assert.equal(wrongVersion.status, 1);
        assert.match(wrongVersion.stderr, /PATCH VERSION MISMATCH: patch-06/);
        assert.equal(readFileSync(target, "utf8"), pristine);
        rmSync(manifest);
        const missingVersion = run();
        assert.notEqual(missingVersion.status, 0, "missing version evidence fails closed");
        assert.equal(readFileSync(target, "utf8"), pristine);
    } finally { rmSync(root, { recursive: true, force: true }); }
});

test("invalid admission and usage policies refuse locally before SDK or auth construction", () => {
    assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], modelRequestAdmission: {} as never }), /INVALID_MODEL_REQUEST_ADMISSION/);
    assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], modelUsageAwareness: {
        version: 1, lifetime: "runtime", inclusiveInputLimit: 1, outputLimit: 1,
        deliveryReserveInput: 2, deliveryReserveOutput: 0,
    } }), /INVALID_MODEL_USAGE_AWARENESS/);
    assert.throws(() => createToolCallBudgetExtension({ allowedTools: ["read"], exactPaths: { read: ["../outside"] } }, "/tmp"), /INVALID_TOOL_CALL_BUDGET_EXACT_PATH/);
});
