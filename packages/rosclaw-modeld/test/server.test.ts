import assert from "node:assert/strict";
import { mkdtempSync, statSync, existsSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { startModeld } from "../src/server.js";

const TOKEN = "test-token-123";

async function uds(
	socketPath: string,
	method: string,
	path: string,
	body?: unknown,
	token: string | null = TOKEN,
): Promise<{ code: number; body: string }> {
	return new Promise((resolve, reject) => {
		const req = request(
			{
				socketPath,
				method,
				path,
				headers: {
					...(body ? { "content-type": "application/json" } : {}),
					...(token ? { authorization: `Bearer ${token}` } : {}),
				},
			},
			(res) => {
				const chunks: Buffer[] = [];
				res.on("data", (c) => chunks.push(c));
				res.on("end", () => resolve({ code: res.statusCode ?? 0, body: Buffer.concat(chunks).toString() }));
			},
		);
		req.on("error", reject);
		if (body) req.write(JSON.stringify(body));
		req.end();
	});
}

async function withModeld(fn: (socketPath: string, home: string) => Promise<void>): Promise<void> {
	const home = mkdtempSync(join(tmpdir(), "modeld-test-"));
	const socketPath = join(home, "run", "modeld.sock");
	const server = await startModeld({ socketPath, token: TOKEN, homeDir: home });
	try {
		await fn(socketPath, home);
	} finally {
		server.close();
	}
}

test("socket dir 0700 and socket 0600", async () => {
	await withModeld(async (socketPath, home) => {
		assert.equal(statSync(join(home, "run")).mode & 0o700, 0o700);
		assert.equal(statSync(socketPath).mode & 0o777, 0o600);
	});
});

test("requests without bearer token get 401", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "GET", "/v1/providers", undefined, null);
		assert.equal(res.code, 401);
	});
});

test("providers listed with auth status, never secrets", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "GET", "/v1/providers");
		assert.equal(res.code, 200);
		const body = JSON.parse(res.body);
		const ids = body.providers.map((p: { id: string }) => p.id);
		assert.ok(ids.includes("moonshot") && ids.includes("kimi-code") && ids.includes("ollama"));
		assert.ok(!res.body.includes("sk-"));
	});
});

test("login stores credential; auth lists fingerprint only; logout deletes", async () => {
	await withModeld(async (socketPath) => {
		const login = await uds(socketPath, "POST", "/v1/auth/moonshot/login", {
			mode: "api_key",
			api_key: "sk-test-value-12345",
		});
		assert.equal(login.code, 200);
		const auth = await uds(socketPath, "GET", "/v1/auth");
		const creds = JSON.parse(auth.body).credentials;
		assert.equal(creds.length, 1);
		assert.ok(!auth.body.includes("sk-test-value-12345"), "fingerprint only, no secret");
		const logout = await uds(socketPath, "POST", "/v1/auth/moonshot/logout", {});
		assert.equal(JSON.parse(logout.body).deleted, true);
	});
});

test("oauth login honestly reports not-implemented", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "POST", "/v1/auth/moonshot/login", { mode: "oauth" });
		assert.equal(res.code, 501);
	});
});

test("stream without credential yields error event, never fabricates", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "POST", "/v1/stream", {
			provider: "moonshot",
			model: "kimi-k3",
			messages: [{ role: "user", content: "hi" }],
		});
		assert.equal(res.code, 200);
		assert.ok(res.body.includes("no_credential"));
		assert.ok(!res.body.includes('"done"'));
	});
});

test("unknown model yields error event", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "POST", "/v1/stream", {
			provider: "ollama",
			model: "no-such-model",
			messages: [{ role: "user", content: "hi" }],
		});
		assert.ok(res.body.includes("unknown_model"));
	});
});

test("models endpoint lists catalog", async () => {
	await withModeld(async (socketPath) => {
		const res = await uds(socketPath, "GET", "/v1/models?provider=kimi-code");
		const models = JSON.parse(res.body).models;
		assert.ok(models.some((m: { id: string }) => m.id === "k3"));
	});
});
