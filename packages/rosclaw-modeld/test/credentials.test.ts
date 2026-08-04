import assert from "node:assert/strict";
import { existsSync, mkdtempSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { UnifiedCredentialStore } from "../src/credentials.js";

test("opt-in file store writes 0600 and lists fingerprints only", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new UnifiedCredentialStore(dir, { allowFileCredentials: true });
	assert.equal(store.policy.file_credentials, true);
	assert.ok(store.policy.warning);
	const { scope } = store.set("moonshot", "sk-test-secret-value");
	assert.equal(scope, "file");
	const mode = statSync(join(dir, "modeld-credentials.json")).mode & 0o777;
	assert.equal(mode, 0o600);
	const list = store.list();
	assert.equal(list.length, 1);
	assert.equal(list[0].provider, "moonshot");
	assert.notEqual(list[0].fingerprint, "sk-test-secret-value");
	assert.equal(JSON.stringify(list).includes("sk-test-secret-value"), false);
	// resolve 只在内存路径返回 key。
	assert.equal(store.resolve("moonshot"), "sk-test-secret-value");
	assert.equal(store.delete("moonshot"), true);
	assert.equal(store.resolve("moonshot"), undefined);
});

test("missing file reads as empty, never fabricates", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new UnifiedCredentialStore(dir, { allowFileCredentials: true });
	assert.deepEqual(store.list(), []);
	assert.equal(store.resolve("moonshot"), undefined);
});

test("default policy is env-only: login is session-scoped, no file written", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new UnifiedCredentialStore(dir);
	assert.equal(store.policy.file_credentials, false);
	assert.equal(store.policy.warning, null);
	const { scope } = store.set("moonshot", "sk-test-secret-value");
	assert.equal(scope, "session");
	assert.equal(existsSync(join(dir, "modeld-credentials.json")), false);
	assert.equal(store.resolve("moonshot"), "sk-test-secret-value");
	assert.ok(!JSON.stringify(store.list()).includes("sk-test-secret-value"));
});

test("corrupt file throws quarantined error, never silent-empty", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new UnifiedCredentialStore(dir, { allowFileCredentials: true });
	store.set("moonshot", "sk-x");
	writeFileSync(join(dir, "modeld-credentials.json"), "not json");
	assert.throws(() => store.list(), /corrupt/);
});
