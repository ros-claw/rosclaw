import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { migrateProviders, parseRosclawConfig } from "../src/credentials/migration.js";
import { EnvOnlyCredentialStore, HardenedFileCredentialStore } from "../src/credentials/store.js";

const CONFIG = `agent:
  enabled: true
  default_profile: embodied_default
models:
  backend: modeld
  profiles:
    embodied_default:
      provider: kimi_code
      model: k3
      base_url: https://api.kimi.com/coding/v1
      api_key_ref: env:ROSCLAW_KIMI_API_KEY
      capabilities: [llm.chat]
`;

test("config.yaml parses into provider/model/key-ref", () => {
	const parsed = parseRosclawConfig(CONFIG);
	assert.equal(parsed.defaultProfile, "embodied_default");
	assert.equal(parsed.profiles.embodied_default.provider, "kimi_code");
	assert.equal(parsed.profiles.embodied_default.model, "k3");
	assert.equal(parsed.profiles.embodied_default.apiKeyRef, "env:ROSCLAW_KIMI_API_KEY");
});

test("migration maps kimi_code -> kimi-coding and bridges the env key in-process", () => {
	const home = mkdtempSync(join(tmpdir(), "rh-mig-"));
	writeFileSync(join(home, "config.yaml"), CONFIG);
	mkdirSync(join(home, "agent"), { recursive: true });
	const env = { ROSCLAW_KIMI_API_KEY: "sk-test-dummy" } as NodeJS.ProcessEnv;
	const result = migrateProviders(home, env);
	assert.equal(result?.provider, "kimi-coding");
	assert.equal(result?.model, "k3");
	assert.equal(env.KIMI_API_KEY, "sk-test-dummy");
	assert.equal(result?.envKeySet, true);
	const settings = JSON.parse(readFileSync(join(home, "agent", "settings.json"), "utf8"));
	assert.equal(settings.defaultProvider, "kimi-coding");
	// 不重复迁移。
	assert.equal(migrateProviders(home, env), null);
});

test("robot profile credential store refuses writes (env-only)", async () => {
	const store = new EnvOnlyCredentialStore();
	assert.equal(await store.read("kimi-coding"), undefined);
	await assert.rejects(() => store.modify("kimi-coding", async () => ({ type: "api_key" })), /env-only/);
	await assert.rejects(() => store.delete("kimi-coding"), /env-only/);
});

test("developer file store: 0600, symlink rejected, roundtrip", async () => {
	const dir = mkdtempSync(join(tmpdir(), "rh-cred-"));
	const store = new HardenedFileCredentialStore(dir);
	await store.modify("kimi-coding", async () => ({ type: "api_key", key: "x" }));
	const stat = readFileSync(join(dir, "auth.json"), "utf8");
	assert.ok(stat.includes("kimi-coding"));
	const mode = (await import("node:fs")).statSync(join(dir, "auth.json")).mode & 0o777;
	assert.equal(mode, 0o600);
	assert.equal((await store.read("kimi-coding"))?.type, "api_key");
	await store.delete("kimi-coding");
	assert.equal(await store.read("kimi-coding"), undefined);
	// symlink → 拒
	(await import("node:fs")).rmSync(join(dir, "auth.json"));
	symlinkSync("/etc/passwd", join(dir, "auth.json"));
	await assert.rejects(() => store.read("kimi-coding"), /regular single-link/);
});
