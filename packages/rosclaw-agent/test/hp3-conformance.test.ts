/** PR-HP3 红测试：Backend Conformance Suite（调整方案 §四.HP3）。
 *
 * 任何 Harness Backend 都必须通过同一组测试。本文件驱动 Pi backend
 * 跑第一遍；未来 Codex/其他 backend 复用同一套件（不许为其单写）。
 *
 * 红测试先行——harness/conformance 不存在时必须红。
 */
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

/** 最小假 OpenAI 服务端点（确定性 SSE；模型腿用）。 */
async function startFakeModel(): Promise<{ baseUrl: string; close(): void; requests: unknown[] }> {
	const { createServer } = await import("node:http");
	const requests: unknown[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => { body += c; });
		req.on("end", () => {
			requests.push(JSON.parse(body || "{}"));
			const chunk = (content: string, finish: string | null) =>
				`data: ${JSON.stringify({
					id: "c", object: "chat.completion.chunk", created: 1, model: "fake-k3",
					choices: [{ index: 0, delta: { content }, finish_reason: finish }],
				})}\n\n`;
			const payload = `${chunk("pong", null)}${chunk("", "stop")}data: [DONE]\n\n`;
			res.writeHead(200, { "Content-Type": "text/event-stream" });
			res.end(payload);
		});
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const port = (server.address() as { port: number }).port;
	return {
		baseUrl: `http://127.0.0.1:${port}/v1`,
		close: () => server.close(),
		requests,
	};
}

/** 准备带假模型的 rosclawHome（settings + models + 空工作区）。 */
function prepareHome(baseUrl: string): { rosclawHome: string; cwd: string } {
	const root = mkdtempSync(join(tmpdir(), "hp3-"));
	const rosclawHome = join(root, "rh");
	const cwd = join(root, "ws");
	mkdirSync(join(rosclawHome, "agent"), { recursive: true });
	mkdirSync(cwd, { recursive: true });
	writeFileSync(
		join(rosclawHome, "agent", "settings.json"),
		JSON.stringify({ defaultProvider: "fake", defaultModel: "fake-k3" }),
	);
	writeFileSync(
		join(rosclawHome, "agent", "models.json"),
		JSON.stringify({
			providers: {
				fake: {
					name: "Fake", baseUrl, api: "openai-completions",
					apiKey: "sk-fake",
					models: [{ id: "fake-k3", name: "Fake K3", contextWindow: 8192, maxTokens: 2048 }],
				},
			},
		}),
	);
	return { rosclawHome, cwd };
}

test("HP3 conformance: Pi backend 通过共享套件", async () => {
	const fake = await startFakeModel();
	const { rosclawHome, cwd } = prepareHome(fake.baseUrl);
	const { runBackendConformance } = await import(
		"../src/harness/conformance/suite.js"
	);
	const { createPiBackend } = await import("../src/harness/pi/pi-backend.js");
	const report = await runBackendConformance("pi", () => createPiBackend(), {
		rosclawHome,
		cwd,
		version: "0.0.0-test",
	});
	fake.close();
	const failed = report.results.filter((r) => !r.ok);
	assert.deepEqual(
		failed.map((r) => `${r.name}: ${r.detail ?? ""}`),
		[],
		"conformance 失败项",
	);
	// 套件覆盖面钉死（方案 §四.HP3 清单）。
	const names = report.results.map((r) => r.name);
	for (const required of [
		"create", "resume", "prompt", "cancel", "compact", "session-close",
		"no-lost-input", "event-replay", "crash-recovery",
	]) {
		assert.ok(names.includes(required), `套件缺 ${required}`);
	}
});
