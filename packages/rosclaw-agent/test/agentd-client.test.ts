import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { fetchAgentdStatus } from "../src/bridge/agentd-client.js";

test("status unreachable is honest (no fabricated robot state)", async () => {
	const home = mkdtempSync(join(tmpdir(), "rh-test-"));
	const result = await fetchAgentdStatus(home, "http://127.0.0.1:1");
	assert.equal(result.reachable, false);
	assert.ok(result.error);
});

test("status sends control token from 0600 file", async () => {
	const home = mkdtempSync(join(tmpdir(), "rh-test-"));
	writeFileSync(join(home, "agentd-control.token"), "tok_abc");
	const server = createServer((req, res) => {
		if (req.headers["x-rosclaw-token"] === "tok_abc") {
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify({ missions: 1 }));
		} else {
			res.writeHead(401);
			res.end("{}");
		}
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	const port = typeof address === "object" && address ? address.port : 0;
	// token 文件路径是 run/agentd-control.token
	writeFileSync(join(home, "x"), "");
	const { mkdirSync } = await import("node:fs");
	mkdirSync(join(home, "run"), { recursive: true });
	writeFileSync(join(home, "run", "agentd-control.token"), "tok_abc");
	try {
		const ok = await fetchAgentdStatus(home, `http://127.0.0.1:${port}`);
		assert.equal(ok.reachable, true);
		assert.deepEqual(ok.status, { missions: 1 });
		const denied = await fetchAgentdStatus(
			mkdtempSync(join(tmpdir(), "rh-empty-")),
			`http://127.0.0.1:${port}`,
		);
		assert.equal(denied.reachable, false);
		assert.match(denied.error ?? "", /401/);
	} finally {
		server.close();
	}
});
