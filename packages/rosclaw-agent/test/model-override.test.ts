/** Native-owned tests: real offline model authority + SDK, synthetic histories only. */
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, statSync, chmodSync, rmSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SessionManager, createAgentSession, SettingsManager, DefaultResourceLoader } from "@earendil-works/pi-coding-agent";
import { resolveExplicitModel, requireSupportedThinking } from "../src/harness/pi/pi-runtime.js";
import { forkPiSessionForOverride, inspectOverrideSource } from "../src/harness/pi/pi-sessions.js";
import { SessionWriterOwnership, SessionInUseError } from "../src/harness/pi/session-writer-ownership.js";
import { EventMirror } from "../src/extension/event-mirror.js";

function fixture(legacy = false) {
	const root = mkdtempSync(join(tmpdir(), "native-override-"));
	const agent = join(root, "agent");
	const sessions = join(agent, "sessions");
	mkdirSync(sessions, { recursive: true });
	const auth = { "kimi-coding": { type: "api_key", key: "SYNTHETIC_NO_REQUEST" },
		"openai-codex": { type: "oauth", access: "SYNTHETIC_NO_REQUEST", refresh: "SYNTHETIC_NO_REFRESH", expires: Date.now() + 3600000 } };
	writeFileSync(join(agent, "auth.json"), JSON.stringify(auth), { mode: 0o600 });
	writeFileSync(join(agent, "settings.json"), JSON.stringify({ defaultProvider: "openai-codex", defaultModel: "gpt-6.1-sol", defaultThinkingLevel: "high", retry: { enabled: false, maxRetries: 0 } }));
	const source = SessionManager.create(root, sessions);
	source.appendModelChange("kimi-coding", "kimi-for-coding");
	source.appendThinkingLevelChange("high");
	source.appendMessage({ role: "user", content: "APPROVED_SYNTHETIC_CONTEXT", timestamp: 1 });
	source.appendMessage({ role: "assistant", content: [
		{ type: "thinking", thinking: "SYNTHETIC_THINKING_NOT_ORIGINAL" },
		{ type: "toolCall", id: "synthetic-history-call", name: "bash", arguments: { command: "echo HISTORICAL_MUST_NOT_EXECUTE" } },
	], api: "anthropic-messages", provider: "kimi-coding", model: "kimi-for-coding",
		usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "toolUse", timestamp: 2 });
	const branch = source.appendMessage({ role: "toolResult", toolCallId: "synthetic-history-call", toolName: "bash",
		content: [{ type: "text", text: "INERT_PRIOR_RESULT" }], isError: false, timestamp: 3 });
	source.appendMessage({ role: "user", content: "DISCARDED_SYNTHETIC_BRANCH", timestamp: 2 });
	source.branch(branch);
	source.appendMessage({ role: "user", content: "SELECTED_BRANCH", timestamp: 3 });
	source.appendCustomEntry("old-task-authority", { task_id: "oldTask", current_ref: "oldAuthority" });
	const path = source.getSessionFile()!;
	if (legacy) {
		const entries = readFileSync(path, "utf8").trim().split("\n").map(x => JSON.parse(x));
		entries[0].version = 1;
		for (const e of entries.slice(1)) { delete e.id; delete e.parentId; }
		writeFileSync(path, entries.map(e => JSON.stringify(e)).join("\n") + "\n");
	}
	chmodSync(path, 0o400);
	return { root, agent, sessions, source, path };
}
function snapshot(path: string) {
	return { bytes: readFileSync(path), mtime: statSync(path).mtimeMs, size: statSync(path).size };
}

// Bounded malformed graphs go ONLY to the pure inspector, never SDK open,
// getBranch or fork. All data is tiny, synthetic and created in this test.
const boundedEntry = (id: string, parentId: string | null, thinkingLevel = "high") =>
	({ type: "thinking_level_change", id, parentId, thinkingLevel, timestamp: "2026-01-01T00:00:00.000Z" });
const boundedControls = [
	{ name: "empty history", entries: [], expected: undefined },
	{ name: "nearest selected thinking", entries: [boundedEntry("a", null, "low"), boundedEntry("b", "a")], expected: "high" },
	{ name: "discarded thinking ignored", entries: [boundedEntry("a", null, "low"), boundedEntry("b", "a", "impossible-effort"), boundedEntry("c", "a")], expected: "high" },
	{ name: "legacy in-memory migration", entries: [{ type: "thinking_level_change", thinkingLevel: "low", timestamp: "2026-01-01T00:00:00.000Z" }], expected: "low", legacy: true },
	{ name: "duplicate IDs", entries: [boundedEntry("a", null), boundedEntry("a", null)], invalid: true },
	{ name: "dangling discarded parent", entries: [boundedEntry("a", "missing"), boundedEntry("b", null)], invalid: true },
	{ name: "selected self cycle", entries: [boundedEntry("a", "a")], invalid: true },
	{ name: "cycle behind nearest thinking", entries: [boundedEntry("a", "b", "low"), boundedEntry("b", "a", "low"), boundedEntry("c", "b")], invalid: true },
];
for (const control of boundedControls) test(`bounded direct source inspection: ${control.name}`, { timeout: 2000 }, () => {
	const root = mkdtempSync(join(tmpdir(), "native-bounded-inspect-"));
	const path = join(root, "synthetic.jsonl");
	const fetchBefore = globalThis.fetch;
	let fetches = 0;
	globalThis.fetch = async () => { fetches++; throw new Error("NO_PROVIDER_FETCH"); };
	try {
		const header = { type: "session", version: control.legacy ? 1 : 3, id: "synthetic-bounded", cwd: root, timestamp: "2026-01-01T00:00:00.000Z" };
		writeFileSync(path, [header, ...control.entries].map(e => JSON.stringify(e)).join("\n") + "\n");
		chmodSync(path, 0o400);
		const before = snapshot(path), modeBefore = statSync(path).mode, filesBefore = readdirSync(root);
		if (control.invalid) {
			assert.throws(() => inspectOverrideSource(path), (error: unknown) =>
				error instanceof Error && error.message === "INVALID_OVERRIDE_SOURCE");
		} else {
			assert.equal(inspectOverrideSource(path).thinking, control.expected);
		}
		assert.deepEqual(snapshot(path), before);
		assert.equal(statSync(path).mode, modeBefore);
		assert.deepEqual(readdirSync(root), filesBefore);
		assert.equal(fetches, 0);
	} finally { globalThis.fetch = fetchBefore; rmSync(root, { recursive: true, force: true }); }
});

// Real SDK trees, not surrogate entry arrays: selected ancestry and discarded
// siblings coexist in the persisted file. Never prompt or replay any history.
const branchControls: { name: string; selected: string[]; discarded: string; default: string; reject?: boolean }[] = [
	{ name: "absent / discarded low / default high", selected: [], discarded: "low", default: "high" },
	{ name: "absent / discarded off / default high", selected: [], discarded: "off", default: "high" },
	{ name: "absent / discarded unsupported / default high", selected: [], discarded: "impossible-effort", default: "high" },
	{ name: "selected low / discarded high", selected: ["low"], discarded: "high", default: "high" },
	{ name: "selected high / discarded low", selected: ["high"], discarded: "low", default: "low" },
	{ name: "selected low / discarded unsupported", selected: ["low"], discarded: "impossible-effort", default: "high" },
	{ name: "selected high / discarded off / unsupported unused default", selected: ["high"], discarded: "off", default: "impossible-effort" },
	{ name: "latest selected high supersedes selected low", selected: ["low", "high"], discarded: "low", default: "low" },
	{ name: "latest selected low supersedes selected high", selected: ["high", "low"], discarded: "high", default: "high" },
	{ name: "selected off rejects before fork", selected: ["off"], discarded: "high", default: "high", reject: true },
	{ name: "selected unsupported rejects before fork", selected: ["impossible-effort"], discarded: "high", default: "high", reject: true },
	{ name: "absent / unsupported default rejects before fork", selected: [], discarded: "low", default: "impossible-effort", reject: true },
	{ name: "absent / off default rejects before fork", selected: [], discarded: "high", default: "off", reject: true },
];
for (const control of branchControls) test(`selected SDK ancestry: ${control.name}`, async () => {
	const root = mkdtempSync(join(tmpdir(), "native-selected-effort-"));
	const agent = join(root, "agent"), sessions = join(agent, "sessions");
	mkdirSync(sessions, { recursive: true });
	writeFileSync(join(agent, "auth.json"), JSON.stringify({ "openai-codex": {
		type: "oauth", access: "SYNTHETIC_NO_REQUEST", refresh: "SYNTHETIC_NO_REFRESH", expires: Date.now() + 3600000,
	} }), { mode: 0o600 });
	const owner = new SessionWriterOwnership();
	const fetchBefore = globalThis.fetch;
	let fetches = 0;
	globalThis.fetch = async () => { fetches++; throw new Error("NO_PROVIDER_FETCH"); };
	try {
		const source = SessionManager.create(root, sessions);
		source.appendModelChange("kimi-coding", "kimi-for-coding");
		const appendEffort = (value: string) => source.appendThinkingLevelChange(value as Parameters<SessionManager["appendThinkingLevelChange"]>[0]);
		for (const value of control.selected) appendEffort(value);
		const base = source.appendMessage({ role: "user", content: "SYNTHETIC_BASE", timestamp: 1 });
		appendEffort(control.discarded);
		source.appendMessage({ role: "user", content: "SYNTHETIC_DISCARDED", timestamp: 2 });
		source.branch(base);
		source.appendMessage({ role: "user", content: "SYNTHETIC_SELECTED", timestamp: 3 });
		// The SDK persists the real tree on the first assistant append.
		source.appendMessage({ role: "assistant", content: [{ type: "text", text: "SYNTHETIC_PERSIST" }],
			api: "anthropic-messages", provider: "kimi-coding", model: "kimi-for-coding",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "stop", timestamp: 4 });
		const path = source.getSessionFile()!;
		chmodSync(path, 0o400);
		const before = snapshot(path), modeBefore = statSync(path).mode;
		const authBefore = snapshot(join(agent, "auth.json"));
		const filesBefore = readdirSync(sessions);
		assert.equal(source.getBranch().filter(e => e.type === "thinking_level_change").length, control.selected.length);
		assert.equal(source.getEntries().filter(e => e.type === "thinking_level_change").length, control.selected.length + 1);
		const recorded = control.selected.at(-1);
		assert.equal(inspectOverrideSource(path).thinking, recorded);
		const selection = await resolveExplicitModel(agent, "developer", "openai-codex", "gpt-6.1-sol");
		const expected = recorded ?? control.default;
		if (control.reject) {
			assert.throws(() => forkPiSessionForOverride(path, sessions, owner, selection, control.default),
				(error: unknown) => error instanceof Error && error.message.includes(`MODEL_THINKING_UNSUPPORTED: ${expected}`));
			assert.deepEqual(readdirSync(sessions), filesBefore);
			assert.equal(owner.claimCount, 0);
		} else {
			const fork = forkPiSessionForOverride(path, sessions, owner, selection, control.default);
			assert.equal(fork.thinking, expected);
			assert.equal(fork.session.buildSessionContext().thinkingLevel, expected);
			assert.notEqual(fork.session.getSessionId(), source.getSessionId());
			assert.equal(owner.claimCount, 1);
			assert.equal(owner.has(fork.session.getSessionFile()!), true);
			assert.equal(readdirSync(sessions).filter(x => x.endsWith(".jsonl")).length, 2);
			assert.doesNotMatch(readFileSync(fork.session.getSessionFile()!, "utf8"), /SYNTHETIC_DISCARDED/);
		}
		assert.equal(owner.has(path), false);
		assert.deepEqual(snapshot(path), before);
		assert.equal(statSync(path).mode, modeBefore);
		assert.deepEqual(snapshot(join(agent, "auth.json")), authBefore);
		assert.equal(fetches, 0);
	} finally { globalThis.fetch = fetchBefore; owner.releaseAll(); rmSync(root, { recursive: true, force: true }); }
});

for (const legacy of [false, true]) test(`immutable ${legacy ? "legacy" : "current"} source: new branch/model/effort before SDK restoration`, async () => {
	const f = fixture(legacy);
	const owner = new SessionWriterOwnership();
	const before = snapshot(f.path);
	const settingsBefore = snapshot(join(f.agent, "settings.json"));
	const authBefore = snapshot(join(f.agent, "auth.json"));
	const fetchBefore = globalThis.fetch;
	globalThis.fetch = async () => { throw new Error("NO_PROVIDER_FETCH"); };
	try {
		const selected = await resolveExplicitModel(f.agent, "developer", "openai-codex", "gpt-6.1-sol");
		const fork = forkPiSessionForOverride(f.path, f.sessions, owner, selected, "high");
		assert.notEqual(fork.session.getSessionId(), f.source.getSessionId());
		assert.equal(fork.session.getHeader()?.parentSession, f.path);
		assert.equal(owner.claimCount, 1);
		assert.equal(owner.has(f.path), false);
		const context = fork.session.buildSessionContext();
		assert.deepEqual(context.model, { provider: "openai-codex", modelId: "gpt-6.1-sol" });
		assert.equal(context.thinkingLevel, "high");
		assert.match(JSON.stringify(context.messages), /APPROVED_SYNTHETIC_CONTEXT/);
		if (!legacy) assert.doesNotMatch(JSON.stringify(context.messages), /DISCARDED_SYNTHETIC_BRANCH/);
		assert.doesNotMatch(readFileSync(fork.session.getSessionFile()!, "utf8"), /oldTask|oldAuthority/);
		const settings = SettingsManager.inMemory({ defaultProvider: "kimi-coding", defaultModel: "kimi-for-coding", defaultThinkingLevel: "high" });
		const loader = new DefaultResourceLoader({ cwd: f.root, agentDir: f.agent, settingsManager: settings,
			noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
		await loader.reload();
		const result = await createAgentSession({ cwd: f.root, agentDir: f.agent, sessionManager: fork.session,
			modelRuntime: selected.modelRuntime, settingsManager: settings, resourceLoader: loader,
			model: selected.model, thinkingLevel: fork.thinking, tools: [] });
		assert.equal(result.session.model?.provider, "openai-codex");
		assert.equal(result.session.model?.id, "gpt-6.1-sol");
		assert.equal(result.session.thinkingLevel, "high");
		assert.equal(result.session.isIdle, true); // no prompt / historical command execution
		await result.session.dispose();
		assert.deepEqual(snapshot(f.path), before);
		assert.deepEqual(snapshot(join(f.agent, "settings.json")), settingsBefore);
		assert.deepEqual(snapshot(join(f.agent, "auth.json")), authBefore);
	} finally { globalThis.fetch = fetchBefore; owner.releaseAll(); rmSync(f.root, { recursive: true, force: true }); }
});

test("ordinary resume retains recorded Kimi despite global ChatGPT", async () => {
	const f = fixture();
	try {
		assert.deepEqual(f.source.buildSessionContext().model, { provider: "kimi-coding", modelId: "kimi-for-coding" });
		assert.equal(inspectOverrideSource(f.path).thinking, "high");
		const runtime = await resolveExplicitModel(f.agent, "developer", "kimi-coding", "kimi-for-coding");
		const settings = SettingsManager.create(f.root, f.agent);
		const loader = new DefaultResourceLoader({ cwd: f.root, agentDir: f.agent, settingsManager: settings,
			noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
		await loader.reload();
		const result = await createAgentSession({ sessionManager: f.source, modelRuntime: runtime.modelRuntime,
			settingsManager: settings, resourceLoader: loader, tools: [] });
		assert.equal(result.session.model?.provider, "kimi-coding");
		await result.session.dispose();
	} finally { rmSync(f.root, { recursive: true, force: true }); }
});

test("unknown provider/model and absent auth reject without a fork or network", async () => {
	const f = fixture();
	try {
		const before = snapshot(f.path);
		await assert.rejects(resolveExplicitModel(f.agent, "developer", "not-provider", "not-model"), /UNKNOWN_MODEL_PROVIDER/);
		await assert.rejects(resolveExplicitModel(f.agent, "developer", "openai-codex", "not-model"), /UNKNOWN_MODEL_TARGET/);
		writeFileSync(join(f.agent, "auth.json"), "{}", { mode: 0o600 });
		await assert.rejects(resolveExplicitModel(f.agent, "developer", "openai-codex", "gpt-6.1-sol"), /MODEL_AUTH_UNAVAILABLE/);
		assert.equal(readdirSync(f.sessions).filter(x => x.endsWith(".jsonl")).length, 1);
		assert.deepEqual(snapshot(f.path), before);
	} finally { rmSync(f.root, { recursive: true, force: true }); }
});

test("unsupported effort and live source fail closed without altering owner/source", async () => {
	const f = fixture();
	const live = new SessionWriterOwnership();
	const forkOwner = new SessionWriterOwnership();
	try {
		const selected = await resolveExplicitModel(f.agent, "developer", "openai-codex", "gpt-6.1-sol");
		assert.throws(() => requireSupportedThinking(selected, "impossible-effort"), /MODEL_THINKING_UNSUPPORTED/);
		const before = snapshot(f.path);
		live.acquire(f.path);
		assert.throws(() => forkPiSessionForOverride(f.path, f.sessions, forkOwner, selected, "high"), SessionInUseError);
		assert.equal(live.claimCount, 1);
		assert.equal(forkOwner.claimCount, 0);
		assert.deepEqual(snapshot(f.path), before);
	} finally { live.releaseAll(); forkOwner.releaseAll(); rmSync(f.root, { recursive: true, force: true }); }
});

test("compiled public parser: partial/duplicate overrides reject before any session", () => {
	const root = mkdtempSync(join(tmpdir(), "native-override-parser-"));
	try {
		for (const flags of [["--provider", "openai-codex"], ["--model", "gpt-6.1-sol"], ["--provider", "a", "--provider", "b", "--model", "c"]]) {
			const p = spawnSync(process.execPath, [fileURLToPath(new URL("../src/main.js", import.meta.url)), ...flags],
				{ env: { ...process.env, ROSCLAW_HOME: root, PI_OFFLINE: "1" }, encoding: "utf8", timeout: 5000 });
			assert.equal(p.status, 2);
			assert.match(p.stderr, /MODEL_OVERRIDE_PAIR_REQUIRED|DUPLICATE_MODEL_OVERRIDE/);
			assert.equal(readdirSync(root).length, 0);
		}
	} finally { rmSync(root, { recursive: true, force: true }); }
});

test("fork mirror identities belong to new session, not copied old task authority", async () => {
	const f = fixture();
	const owner = new SessionWriterOwnership();
	try {
		const selection = await resolveExplicitModel(f.agent, "developer", "openai-codex", "gpt-6.1-sol");
		const fork = forkPiSessionForOverride(f.path, f.sessions, owner, selection, "high");
		const events: Record<string, unknown>[] = [];
		const mirror = new EventMirror(f.root, fork.session.getSessionId(), "new-mission", async (_h, _m, params = {}) => {
			events.push(...params.events as Record<string, unknown>[]); return { ok: true };
		});
		mirror.push("message_end", { entryId: "new-response", text: "synthetic", model: "gpt-6.1-sol" });
		await mirror.flush();
		assert.equal(events.length, 1);
		assert.doesNotMatch(JSON.stringify(events), /oldTask|oldAuthority/);
		assert.match(JSON.stringify(events), new RegExp(fork.session.getSessionId()));
	} finally { owner.releaseAll(); rmSync(f.root, { recursive: true, force: true }); }
});
