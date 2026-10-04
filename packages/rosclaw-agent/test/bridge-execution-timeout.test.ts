import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { bridgeCall } from "../src/bridge/bridge-client.js";

test("both execution RPCs retain their receipts beyond the query timeout", async () => {
	const home = mkdtempSync(join(tmpdir(), "rosclaw-bridge-timeout-"));
	mkdirSync(join(home, "run"));
	const received: string[] = [];
	const server = createServer((socket) => {
		let buffer = "";
		socket.on("data", (chunk) => {
			buffer += chunk.toString();
			if (!buffer.includes("\n")) return;
			const request = JSON.parse(buffer.trim());
			received.push(request.method);
			setTimeout(() => socket.end(JSON.stringify({ status: "COMPLETED" }) + "\n"), 5200);
		});
	});
	try {
		await new Promise<void>((resolve, reject) => {
			server.once("error", reject);
			server.listen(join(home, "run/pi-bridge.sock"), resolve);
		});
		const receipts = await Promise.all([
			bridgeCall(home, "pi.tools.execute"),
			bridgeCall(home, "pi.action.execute"),
		]);
		assert.deepEqual(received.sort(), ["pi.action.execute", "pi.tools.execute"]);
		assert.ok(receipts.every((receipt) => receipt.status === "COMPLETED"));
	} finally {
		await new Promise<void>((resolve) => server.close(() => resolve()));
		rmSync(home, { recursive: true, force: true });
	}
});
