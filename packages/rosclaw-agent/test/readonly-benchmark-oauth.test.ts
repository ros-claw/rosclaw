import assert from "node:assert/strict";
import test from "node:test";
import { chmodSync, linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir, homedir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { assertReadonlyOAuthOutsideWritableRoots, BenchmarkOAuthError, createReadonlyBenchmarkModelRuntime, OAUTH_MIN_VALIDITY_MS, ReadOnlyBenchmarkOAuthStore, READONLY_OAUTH_ENV } from "../src/harness/pi/readonly-benchmark-oauth.js";
import { createNativeBenchmarkRuntime } from "../src/harness/pi/benchmark-native.js";
import { createSharedModelRuntime } from "../src/harness/pi/pi-model-runtime.js";

const access = "offline-dummy-access-never-use-for-a-request";
const refresh = "offline-dummy-refresh-never-use-for-a-request";
function fixture(t: { after: (fn: () => void) => void }, expires = Date.now() + 3600000): string {
	const dir = mkdtempSync(join(tmpdir(), "rosclaw-readonly-oauth-test-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const path = join(dir, "dedicated-auth.json");
	writeFileSync(path, JSON.stringify({ "openai-codex": { type: "oauth", access, refresh, expires } }), { mode: 0o600 });
	return path;
}
function networkForbidden(t: { after: (fn: () => void) => void }): void {
	const original = globalThis.fetch;
	let calls = 0;
	globalThis.fetch = async () => { calls++; assert.fail("network attempted in offline credential test"); };
	t.after(() => { globalThis.fetch = original; assert.equal(calls, 0, "no network attempt, including caught failures"); });
}

test("PI resolves one readonly source for A and B with no token logs, fallback or source mutation", async (t) => {
	networkForbidden(t);
	const path = fixture(t);
	const before = readFileSync(path);
	const logs: unknown[] = [];
	const originalLog = console.log, originalWarn = console.warn, originalError = console.error;
	console.log = console.warn = console.error = (...values: unknown[]) => { logs.push(values); };
	t.after(() => { console.log = originalLog; console.warn = originalWarn; console.error = originalError; });
	const oldSource = process.env[READONLY_OAUTH_ENV];
	const oldKey = process.env.OPENAI_API_KEY;
	process.env[READONLY_OAUTH_ENV] = path;
	process.env.OPENAI_API_KEY = "offline-key-must-not-be-used-as-fallback";
	t.after(() => {
		if (oldSource === undefined) delete process.env[READONLY_OAUTH_ENV]; else process.env[READONLY_OAUTH_ENV] = oldSource;
		if (oldKey === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = oldKey;
	});
	const a = await createReadonlyBenchmarkModelRuntime(path);
	const b = await createSharedModelRuntime(join(path, "..", "unused-trial-agent"), "developer");
	const modelId = a.getModels("openai-codex")[0]?.id;
	assert.ok(modelId, "pinned PI builtin catalog must have a selectable model");
	for (const runtime of [a, b]) {
		const model = runtime.getModel("openai-codex", modelId);
		assert.ok(model);
		const auth = await runtime.getAuth(model);
		assert.equal(auth?.source, "OAuth");
		assert.ok(auth?.auth.apiKey); // never print the value
		assert.deepEqual(runtime.getRegisteredProviderIds(), []);
		await assert.rejects(() => runtime.getAuth("openai"));
	}
	const model = a.getModels("openai-codex")[0]!;
	await Promise.all(Array.from({ length: 8 }, (_, i) => (i % 2 ? a : b).getAuth(model)));
	await assert.rejects(() => createSharedModelRuntime("/unused", "robot"), /BENCH_OAUTH_ROBOT_PROFILE_FORBIDDEN/);
	assert.deepEqual(readFileSync(path), before);
	assert.deepEqual(logs, []);
});

test("the readonly store refuses writes without invoking a refresh callback", async (t) => {
	const path = fixture(t);
	const before = readFileSync(path);
	const store = new ReadOnlyBenchmarkOAuthStore(path);
	let refreshed = false;
	await assert.rejects(() => store.modify("openai-codex", async () => { refreshed = true; throw new Error("must not run"); }), /REFRESH_OR_LOGIN_FORBIDDEN/);
	await assert.rejects(() => store.delete(), /LOGOUT_FORBIDDEN/);
	await assert.rejects(() => store.read("openai"), /PROVIDER_FORBIDDEN/);
	assert.equal(refreshed, false);
	assert.deepEqual(await store.list(), [{ providerId: "openai-codex", type: "oauth" }]);
	assert.deepEqual(readFileSync(path), before);
});

for (const remaining of [-1, 1000, OAUTH_MIN_VALIDITY_MS - 1]) {
	test(`expired or near-expiry source fails before refresh (${remaining}ms)`, async (t) => {
		networkForbidden(t);
		const path = fixture(t, Date.now() + remaining);
		const before = readFileSync(path);
		await assert.rejects(() => createReadonlyBenchmarkModelRuntime(path), /RENEW_EXTERNALLY_REQUIRED/);
		assert.deepEqual(readFileSync(path), before);
	});
}

test("a trial-specific validity horizon is enforced, not just the PI minimum", async (t) => {
	const path = fixture(t, Date.now() + 600000);
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path, 1500000).read("openai-codex"), /RENEW_EXTERNALLY_REQUIRED/);
});

test("unsafe, missing, linked, global and non-OAuth sources fail without token/command exposure", async (t) => {
	const path = fixture(t);
	const dir = join(path, "..");
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(join(dir, "missing")).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	for (const mode of [0o644, 0o400]) {
		chmodSync(path, mode);
		await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	}
	chmodSync(path, 0o600);
	const alias = join(dir, "linked.json");
	symlinkSync(path, alias);
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(alias).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	rmSync(alias);
	linkSync(path, alias);
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	rmSync(alias);
	const parentLink = join(dir, "linked-dir");
	symlinkSync(dir, parentLink);
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(join(parentLink, "dedicated-auth.json")).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	for (const location of [".pi", ".rosclaw"]) {
		assert.throws(() => new ReadOnlyBenchmarkOAuthStore(join(homedir(), location, "agent", "auth.json")), /DEDICATED_SOURCE_REQUIRED/);
	}
	writeFileSync(path, JSON.stringify({ "openai-codex": { type: "api_key", key: "!must-not-run-a-command" } }));
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path).read("openai-codex"), /CREDENTIAL_MISSING_OR_INVALID/);
	writeFileSync(path, "invalid-json-with-a-secret-marker");
	await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path).read("openai-codex"), (err: unknown) => err instanceof BenchmarkOAuthError && err.message === "BENCH_OAUTH_CREDENTIAL_MISSING_OR_INVALID");
});

test("store re-reads an externally renewed source; it never writes or refreshes it", async (t) => {
	const path = fixture(t);
	const store = new ReadOnlyBenchmarkOAuthStore(path);
	await store.read("openai-codex");
	writeFileSync(path, JSON.stringify({ "openai-codex": { type: "oauth", access, refresh, expires: Date.now() - 1 } }));
	await assert.rejects(() => store.read("openai-codex"), /RENEW_EXTERNALLY_REQUIRED/);
});

test("the bridge check emits only safe readiness metadata and no session or auth copies", async (t) => {
	const path = fixture(t);
	const before = readFileSync(path);
	const runtime = await createReadonlyBenchmarkModelRuntime(path);
	const id = runtime.getModels("openai-codex")[0]!.id;
	const trial = join(path, "..", "isolated-agent");
	mkdirSync(trial);
	const entry = fileURLToPath(new URL("../src/harness/pi/benchmark-native.js", import.meta.url));
	const result = spawnSync(process.execPath, [entry, "--check", path, id, "900000"], {
		encoding: "utf8", env: { ...process.env, PI_CODING_AGENT_DIR: trial, PI_SKIP_VERSION_CHECK: "1" }, timeout: 15000,
	});
	assert.equal(result.status, 0);
	assert.deepEqual(JSON.parse(result.stdout), { status: "READY", provider: "openai-codex", model_id: id, credential_mode: "EXPLICIT_READONLY_NO_REFRESH", remote_auth_verified: false });
	assert.equal(result.stdout.includes(access) || result.stdout.includes(refresh), false);
	assert.equal(result.stderr.includes(access) || result.stderr.includes(refresh), false);
	assert.deepEqual(readFileSync(path), before);
	const { readdirSync } = await import("node:fs");
	assert.deepEqual(readdirSync(trial), []);
	const expired = spawnSync(process.execPath, [entry, "--check", path, id, "7200000"], { encoding: "utf8", timeout: 15000 });
	assert.equal(expired.status, 2);
	assert.deepEqual(JSON.parse(expired.stdout), { status: "NOT_RUN", code: "BENCH_OAUTH_RENEW_EXTERNALLY_REQUIRED" });
});


test("native PI SDK session factory keeps separate trials and original coding tools without a model turn", async (t) => {
	networkForbidden(t);
	const path = fixture(t);
	const sourceBefore = readFileSync(path);
	const modelRuntime = await createReadonlyBenchmarkModelRuntime(path);
	const modelId = modelRuntime.getModels("openai-codex")[0]!.id;
	const dirs: string[] = [];
	for (const name of ["trial-a", "trial-b"]) {
		const cwd = join(path, "..", name);
		const agentDir = join(cwd, "agent");
		mkdirSync(agentDir, { recursive: true });
		writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ defaultProvider: "openai-codex", defaultModel: modelId }));
		const runtime = await createNativeBenchmarkRuntime(cwd, agentDir, path, modelId);
		try {
			assert.equal(runtime.session.model?.provider, "openai-codex");
			assert.equal(runtime.session.model?.id, modelId);
			const tools = runtime.session.getActiveToolNames();
			for (const name of ["read", "bash", "edit", "write"]) assert.ok(tools.includes(name));
			assert.equal(tools.some((name) => name.startsWith("rosclaw")), false);
			assert.deepEqual(runtime.session.state.messages, []);
			dirs.push(runtime.session.sessionManager.getSessionDir());
		} finally { await runtime.dispose(); }
	}
	assert.notEqual(dirs[0], dirs[1]);
	assert.deepEqual(readFileSync(path), sourceBefore);
});


test("a source owned by another UID is refused without credential reads", async (t) => {
	const path = fixture(t);
	const original = process.getuid;
	if (!original) throw new Error("UID controls require the supported Unix benchmark host");
	process.getuid = () => original() + 1;
	try {
		await assert.rejects(() => new ReadOnlyBenchmarkOAuthStore(path).read("openai-codex"), /SOURCE_UNAVAILABLE_OR_UNSAFE/);
	} finally { process.getuid = original; }
});

test("PI request auth refuses a source that expires after preflight without refreshing it", async (t) => {
	networkForbidden(t);
	const path = fixture(t);
	const runtime = await createReadonlyBenchmarkModelRuntime(path);
	const model = runtime.getModels("openai-codex")[0]!;
	writeFileSync(path, JSON.stringify({ "openai-codex": { type: "oauth", access, refresh, expires: Date.now() + 1000 } }));
	const before = readFileSync(path);
	await assert.rejects(() => runtime.getAuth(model));
	assert.deepEqual(readFileSync(path), before);
});


test("direct SDK entry rejects source within workspace or agent writable namespace before session setup", async (t) => {
	const path = fixture(t);
	const before = readFileSync(path);
	const { readdirSync } = await import("node:fs");
	const root = join(path, "..");
	const beforeNames = readdirSync(root);
	await assert.rejects(() => createNativeBenchmarkRuntime(root, join(root, "future-agent"), path, "unused"), /SOURCE_OVERLAPS_WRITABLE_SCOPE/);
	await assert.rejects(() => createNativeBenchmarkRuntime(join(root, "future-work"), root, path, "unused"), /SOURCE_OVERLAPS_WRITABLE_SCOPE/);
	assert.deepEqual(readFileSync(path), before);
	assert.deepEqual(readdirSync(root), beforeNames);
});

test("direct shared B runtime rejects source/agent namespace overlap without source writes", async (t) => {
	const path = fixture(t);
	const before = readFileSync(path);
	const old = process.env[READONLY_OAUTH_ENV];
	process.env[READONLY_OAUTH_ENV] = path;
	try {
		await assert.rejects(() => createSharedModelRuntime(join(path, ".."), "developer"), /SOURCE_OVERLAPS_WRITABLE_SCOPE/);
	} finally {
		if (old === undefined) delete process.env[READONLY_OAUTH_ENV]; else process.env[READONLY_OAUTH_ENV] = old;
	}
	assert.deepEqual(readFileSync(path), before);
});

test("writable namespace aliases cannot bypass the readonly source boundary", async (t) => {
	const path = fixture(t);
	const root = join(path, "..");
	const alias = join(root, "aliased-work");
	symlinkSync(root, alias);
	const before = readFileSync(path);
	assert.throws(() => assertReadonlyOAuthOutsideWritableRoots(path, [alias]), /SOURCE_OVERLAPS_WRITABLE_SCOPE/);
	assert.doesNotThrow(() => assertReadonlyOAuthOutsideWritableRoots(path, [join(root, "fresh", "nested")]));
	assert.deepEqual(readFileSync(path), before);
});
