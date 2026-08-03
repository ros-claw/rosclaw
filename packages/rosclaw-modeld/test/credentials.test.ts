import assert from "node:assert/strict";
import { mkdtempSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { FileCredentialStore } from "../src/credentials.js";

test("store writes 0600 and lists fingerprints only", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new FileCredentialStore(dir);
	store.set("moonshot", "sk-test-secret-value");
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
	const store = new FileCredentialStore(dir);
	assert.deepEqual(store.list(), []);
	assert.equal(store.resolve("moonshot"), undefined);
});

test("corrupt file reads as empty (fail honest)", () => {
	const dir = mkdtempSync(join(tmpdir(), "modeld-cred-"));
	const store = new FileCredentialStore(dir);
	store.set("moonshot", "sk-x");
	writeFileSync(join(dir, "modeld-credentials.json"), "not json");
	assert.deepEqual(store.list(), []);
});
